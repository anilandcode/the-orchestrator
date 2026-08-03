import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { FEATURE_DIMENSION, LinUcbBandit } from "@orchestrator/router";
import type { CallEvent, TaskType } from "@orchestrator/shared";
import {
  RollingNormalizer,
  SqliteCallEventRepository,
  SqliteRoutingDecisionRepository,
  aggregateByModel,
  computeReward,
  openDatabase,
  summarize,
} from "@orchestrator/telemetry";

/**
 * Offline evaluation against REAL logged traffic.
 *
 * This is the report to run before promoting `ROUTER_MODE` from `shadow` to `adaptive`.
 *
 * What it can and cannot tell you, stated plainly:
 *
 *   - **Can**: what actually happened per model (win rate, cost per success, latency), and how often
 *     the bandit disagreed with the model that ran.
 *   - **Can (with an assumption)**: an off-policy reward estimate via *replay matching* — walk the
 *     log, ask the bandit to choose, and score only the rounds where its choice equals the logged
 *     one. This is unbiased only if the logging policy explored uniformly over the candidate set.
 *     Shadow mode does NOT explore uniformly (it runs static rules), so treat the estimate as
 *     directional, not decisive. It is most trustworthy on traffic logged with a high exploration
 *     floor.
 *   - **Cannot**: tell you the reward a model would have earned on a request it never served. No
 *     amount of log analysis recovers that; only running the model does.
 */

const DB_PATH = process.env.ORCHESTRATOR_DB_PATH ?? "./data/orchestrator.sqlite";

interface ReplayResult {
  matched: number;
  matchedReward: number;
  loggedRewardOnMatched: number;
}

function replayMatch(events: CallEvent[], candidatesByTask: Map<TaskType, string[]>): ReplayResult {
  const bandit = new LinUcbBandit({
    dimension: FEATURE_DIMENSION,
    alpha: 0.35,
    explorationFloor: 0,
    random: () => 0.5,
  });

  let matched = 0;
  let matchedReward = 0;

  for (const event of events) {
    if (event.features.length !== FEATURE_DIMENSION) continue;

    const candidates = candidatesByTask.get(event.taskType) ?? [];
    if (candidates.length < 2) continue;

    const reward = event.reward ?? computeReward(event);
    const choice = bandit.select(candidates, event.features);

    // Rejection sampling: only rounds where the policy under evaluation agrees with the logged
    // action carry information about that policy's value.
    if (choice.armId === event.modelId) {
      matched += 1;
      matchedReward += reward;
    }

    // The bandit always learns, matched or not — otherwise it would never accumulate history.
    bandit.update(event.modelId, event.features, reward);
  }

  return { matched, matchedReward, loggedRewardOnMatched: matchedReward };
}

function main(): void {
  const db = openDatabase(DB_PATH);
  const events = new SqliteCallEventRepository(db);
  const decisions = new SqliteRoutingDecisionRepository(db);

  const allEvents = events.query();
  const allDecisions = decisions.query();

  if (allEvents.length === 0) {
    console.error(
      `No CallEvent rows in ${DB_PATH}.

This tool analyses real traffic. Run the API and send some requests first, or use
\`pnpm replay:simulate\` for the synthetic-world evaluation that needs no data.`,
    );
    process.exit(1);
  }

  // Warm the normalizer on history so rewards are scored against the same percentiles the live
  // system would have used.
  const normalizer = new RollingNormalizer();
  normalizer.observeAll(allEvents);

  const scored = allEvents.map((event) => ({
    event,
    reward: event.reward ?? computeReward(event, { stats: normalizer.statsFor(event.taskType) }),
  }));

  const candidatesByTask = new Map<TaskType, string[]>();
  for (const event of allEvents) {
    const seen = candidatesByTask.get(event.taskType) ?? [];
    if (!seen.includes(event.modelId)) seen.push(event.modelId);
    candidatesByTask.set(event.taskType, seen);
  }

  const disagreements = allDecisions.filter(
    (decision) => decision.shadowModelId !== null && decision.shadowModelId !== decision.modelId,
  );

  const replay = replayMatch(allEvents, candidatesByTask);
  const summary = summarize(allEvents);
  const perModel = aggregateByModel(allEvents);

  const lines: string[] = [];
  lines.push("# Router Replay — Real Traffic");
  lines.push("");
  lines.push(`Source: \`${DB_PATH}\`  ·  Generated: ${new Date().toISOString()}`);
  lines.push("");

  lines.push("## Traffic");
  lines.push("");
  lines.push(
    `- Attempts: **${summary.events.toLocaleString()}** across ${summary.requests.toLocaleString()} requests`,
  );
  lines.push(`- Success rate: **${(summary.successRate * 100).toFixed(2)}%**`);
  lines.push(`- Retry/fallback rate: **${(summary.retryRate * 100).toFixed(2)}%** of requests`);
  lines.push(`- Total cost: **$${summary.totalCostUsd.toFixed(4)}**`);
  lines.push(
    `- Latency p50 / p95: **${summary.p50LatencyMs.toFixed(0)}ms / ${summary.p95LatencyMs.toFixed(0)}ms**`,
  );
  lines.push("");

  lines.push("## Per-model");
  lines.push("");
  lines.push("| Model | Attempts | Success | p95 latency | Cost/success | Avg reward |");
  lines.push("|---|---:|---:|---:|---:|---:|");
  for (const stats of perModel) {
    const costPerSuccess = Number.isFinite(stats.costPerSuccessUsd)
      ? `$${stats.costPerSuccessUsd.toFixed(6)}`
      : "n/a";
    lines.push(
      `| \`${stats.modelId}\` | ${stats.attempts} | ${(stats.successRate * 100).toFixed(1)}% | ${stats.p95LatencyMs.toFixed(0)}ms | ${costPerSuccess} | ${stats.avgReward.toFixed(4)} |`,
    );
  }
  lines.push("");

  lines.push("## Shadow-mode disagreement");
  lines.push("");
  if (allDecisions.length === 0) {
    lines.push(
      "No routing decisions persisted yet. Shadow analysis needs `routing_decisions` rows —",
    );
    lines.push("check that the API is recording decisions alongside call events.");
  } else {
    const rate = (disagreements.length / allDecisions.length) * 100;
    lines.push(`- Decisions recorded: **${allDecisions.length.toLocaleString()}**`);
    lines.push(
      `- Bandit disagreed with what ran: **${disagreements.length.toLocaleString()}** (${rate.toFixed(1)}%)`,
    );
    lines.push("");
    lines.push("A disagreement rate near 0% means the bandit has learned the static rules and");
    lines.push(
      "promoting it would change little. A high rate means it has found something different —",
    );
    lines.push(
      "which is either an opportunity or a bug, and the replay estimate below is the first check.",
    );

    const byModel = new Map<string, number>();
    for (const decision of disagreements) {
      const key = `${decision.modelId} -> ${decision.shadowModelId}`;
      byModel.set(key, (byModel.get(key) ?? 0) + 1);
    }
    if (byModel.size > 0) {
      lines.push("");
      lines.push("| Ran -> bandit preferred | Count |");
      lines.push("|---|---:|");
      for (const [pair, count] of [...byModel.entries()].sort((a, b) => b[1] - a[1]).slice(0, 15)) {
        lines.push(`| \`${pair}\` | ${count} |`);
      }
    }
  }
  lines.push("");

  lines.push("## Off-policy estimate (replay matching)");
  lines.push("");
  const loggedAvg = scored.reduce((total, row) => total + row.reward, 0) / scored.length;
  if (replay.matched === 0) {
    lines.push("No matched rounds — not enough overlap between the bandit's choices and the log.");
  } else {
    const banditAvg = replay.matchedReward / replay.matched;
    lines.push(
      `- Matched rounds: **${replay.matched.toLocaleString()}** of ${allEvents.length.toLocaleString()}`,
    );
    lines.push(`- Bandit avg reward on matched rounds: **${banditAvg.toFixed(4)}**`);
    lines.push(`- Logged policy avg reward (all rounds): **${loggedAvg.toFixed(4)}**`);
    lines.push("");
    lines.push(
      "> **Read this carefully.** Replay matching is unbiased only when the logging policy",
    );
    lines.push(
      "> explored uniformly. Shadow mode runs static rules, which do not — so a favourable",
    );
    lines.push(
      "> number here is encouraging, not conclusive. The stronger evidence is a period of",
    );
    lines.push("> traffic logged with a raised exploration floor.");
  }
  lines.push("");

  const outputPath = resolve(dirname(fileURLToPath(import.meta.url)), "../out/replay.md");
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, lines.join("\n"), "utf8");

  console.log(
    `Events: ${allEvents.length}  Decisions: ${allDecisions.length}  Disagreements: ${disagreements.length}`,
  );
  console.log(`Logged avg reward: ${loggedAvg.toFixed(4)}`);
  if (replay.matched > 0) {
    console.log(
      `Replay-matched: ${replay.matched} rounds, avg reward ${(replay.matchedReward / replay.matched).toFixed(4)}`,
    );
  }
  console.log(`\nReport written to ${outputPath}`);

  events.close();
}

main();
