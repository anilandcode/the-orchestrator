import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  AdaptiveRouter,
  FEATURE_DIMENSION,
  LinUcbBandit,
  type Router,
  type RoutingContext,
  StaticRouter,
  ThompsonBandit,
  extractFeatures,
  selectCandidates,
} from "@orchestrator/router";
import {
  type RouteMode,
  TASK_TYPES,
  type TaskType,
  UnifiedChatRequestSchema,
  createSequentialIds,
} from "@orchestrator/shared";
import { bestArm, seededRandom, simulateCall, simulationModels } from "./world.js";

/**
 * Offline evaluation against a known-ground-truth world.
 *
 * This is the artifact that answers "is the adaptive router actually working?" on day one, before any
 * production traffic exists. It is NOT evidence about your traffic — that is what `replay.ts` is for
 * once real events accumulate. What it does prove is that the algorithm, the feature vector, and the
 * reward function work together to beat the static baseline when a better choice genuinely exists.
 */

const ROUNDS = 6_000;
const ROUTE_MODES: RouteMode[] = ["cheap", "balanced", "best"];

interface ArmRun {
  name: string;
  router: Router;
  totalReward: number;
  totalRegret: number;
  regretCurve: number[];
  picks: Map<string, number>;
  failures: number;
  totalCostUsd: number;
  /** Rounds where the strategy chose the arm with the highest expected reward. */
  oracleHits: number;
  /** Rounds where the bandit was still deferring to the baseline. */
  deferrals: number;
}

function makeContext(random: () => number): {
  context: RoutingContext;
  taskType: TaskType;
  routeMode: RouteMode;
  promptTokens: number;
  completionTokens: number;
} {
  const taskType = TASK_TYPES[Math.floor(random() * TASK_TYPES.length)] as TaskType;
  const routeMode = ROUTE_MODES[Math.floor(random() * ROUTE_MODES.length)] as RouteMode;
  const promptTokens = Math.floor(200 + random() * 4_000);
  const completionTokens = Math.floor(50 + random() * 800);

  const request = UnifiedChatRequestSchema.parse({
    tenantId: "sim",
    messages: [{ role: "user", content: "x" }],
    route: { mode: routeMode, taskType },
  });

  return {
    context: { request, available: simulationModels(), estimatedPromptTokens: promptTokens },
    taskType,
    routeMode,
    promptTokens,
    completionTokens,
  };
}

function buildRuns(): ArmRun[] {
  const baseline = () => new StaticRouter({ ids: createSequentialIds() });

  const linucb = new AdaptiveRouter({
    bandit: new LinUcbBandit({
      dimension: FEATURE_DIMENSION,
      alpha: 0.35,
      explorationFloor: 0.03,
      random: seededRandom(99),
    }),
    baseline: baseline(),
    mode: "adaptive",
    coldStartPulls: 15,
    ids: createSequentialIds(),
  });

  const thompson = new AdaptiveRouter({
    bandit: new ThompsonBandit({ random: seededRandom(123) }),
    baseline: baseline(),
    mode: "adaptive",
    coldStartPulls: 15,
    ids: createSequentialIds(),
  });

  const empty = (): Omit<ArmRun, "name" | "router"> => ({
    totalReward: 0,
    totalRegret: 0,
    regretCurve: [],
    picks: new Map(),
    failures: 0,
    totalCostUsd: 0,
    oracleHits: 0,
    deferrals: 0,
  });

  return [
    { name: "static (baseline)", router: baseline(), ...empty() },
    { name: "adaptive (LinUCB)", router: linucb, ...empty() },
    { name: "adaptive (Thompson)", router: thompson, ...empty() },
  ];
}

function run(): { runs: ArmRun[]; oracleReward: number } {
  const runs = buildRuns();
  let oracleReward = 0;

  // Each router sees the identical request stream, so differences are strategy, not luck.
  for (let round = 0; round < ROUNDS; round++) {
    const scenario = makeContext(seededRandom(round * 7 + 1));
    const candidates = selectCandidates(scenario.context);

    const oracle = bestArm(
      candidates,
      scenario.taskType,
      scenario.routeMode,
      scenario.promptTokens,
      scenario.completionTokens,
    );
    oracleReward += oracle.reward;

    for (const armRun of runs) {
      const decision = armRun.router.select(scenario.context);
      const spec = candidates.find((candidate) => candidate.modelId === decision.modelId);
      if (!spec) continue;

      // Common random numbers: every strategy faces the identical noise draw for this round, so a
      // difference in outcome is a difference in judgement rather than in luck.
      const outcomeRandom = seededRandom(round * 31 + 17);
      const { event, reward } = simulateCall({
        spec,
        taskType: scenario.taskType,
        routeMode: scenario.routeMode,
        promptTokens: scenario.promptTokens,
        completionTokens: scenario.completionTokens,
        features: extractFeatures(scenario.context),
        random: outcomeRandom,
      });

      armRun.router.observe({
        decisionId: decision.decisionId,
        modelId: decision.modelId,
        features: decision.features,
        taskType: scenario.taskType,
        reward,
      });

      armRun.totalReward += reward;
      armRun.totalCostUsd += event.costUsd;
      if (event.status === "error") armRun.failures += 1;
      armRun.picks.set(decision.modelId, (armRun.picks.get(decision.modelId) ?? 0) + 1);
      if (decision.modelId === oracle.spec.modelId) armRun.oracleHits += 1;
      if (decision.reason.includes("bandit deferred")) armRun.deferrals += 1;

      // Regret against the best *expected* arm, so bad luck is not scored as a bad decision.
      const chosenExpected = bestArm(
        [spec],
        scenario.taskType,
        scenario.routeMode,
        scenario.promptTokens,
        scenario.completionTokens,
      ).reward;
      armRun.totalRegret += Math.max(0, oracle.reward - chosenExpected);
      armRun.regretCurve.push(armRun.totalRegret);
    }
  }

  return { runs, oracleReward };
}

function sparkline(values: number[], buckets = 40): string {
  if (values.length === 0) return "";
  const chars = "▁▂▃▄▅▆▇█";
  const step = Math.max(1, Math.floor(values.length / buckets));
  const sampled: number[] = [];
  for (let i = 0; i < values.length; i += step) sampled.push(values[i] as number);

  const max = Math.max(...sampled) || 1;
  return sampled
    .map(
      (value) => chars[Math.min(chars.length - 1, Math.floor((value / max) * (chars.length - 1)))],
    )
    .join("");
}

function report(runs: ArmRun[], oracleReward: number): string {
  const lines: string[] = [];
  const baseline = runs[0] as ArmRun;

  lines.push("# Adaptive Router — Simulated Evaluation");
  lines.push("");
  lines.push(`Rounds: **${ROUNDS.toLocaleString()}**  ·  Generated: ${new Date().toISOString()}`);
  lines.push("");
  lines.push(
    "Synthetic traffic against a known-ground-truth world. This proves the algorithm, features, and",
  );
  lines.push(
    "reward function work together — it says nothing about your production traffic. Run `pnpm replay`",
  );
  lines.push("against real `CallEvent` data before promoting `ROUTER_MODE` to `adaptive`.");
  lines.push("");

  lines.push("## Results");
  lines.push("");
  lines.push(
    "| Strategy | Total reward | vs baseline | Optimal pick | Gap to oracle closed | Cumulative regret | Total cost |",
  );
  lines.push("|---|---:|---:|---:|---:|---:|---:|");

  const baselineGap = oracleReward - baseline.totalReward;

  for (const armRun of runs) {
    const delta =
      armRun === baseline
        ? "—"
        : `${(((armRun.totalReward - baseline.totalReward) / baseline.totalReward) * 100).toFixed(2)}%`;
    const gapClosed =
      armRun === baseline || baselineGap <= 0
        ? "—"
        : `${(((armRun.totalReward - baseline.totalReward) / baselineGap) * 100).toFixed(1)}%`;
    lines.push(
      `| ${armRun.name} | ${armRun.totalReward.toFixed(1)} | ${delta} | ${((armRun.oracleHits / ROUNDS) * 100).toFixed(1)}% | ${gapClosed} | ${armRun.totalRegret.toFixed(1)} | $${armRun.totalCostUsd.toFixed(2)} |`,
    );
  }

  lines.push(
    `| *oracle (upper bound)* | ${oracleReward.toFixed(1)} | ${((baselineGap / baseline.totalReward) * 100).toFixed(2)}% | 100.0% | 100% | 0.0 | — |`,
  );
  lines.push("");
  lines.push(
    `The **gap to oracle** is only ${((baselineGap / baseline.totalReward) * 100).toFixed(2)}% wide — the static rules are already a strong baseline in this world,`,
  );
  lines.push(
    "which is the honest framing. Published bandit-routing gains are usually quoted against *random*",
  );
  lines.push(
    "routing; against tuned rules there is far less room, and what matters is the share of the",
  );
  lines.push("remaining gap the router recovers.");
  lines.push("");

  lines.push("## Cumulative regret");
  lines.push("");
  lines.push("Flattening means the router has stopped making avoidable mistakes.");
  lines.push("");
  for (const armRun of runs) {
    lines.push(`- \`${armRun.name.padEnd(20)}\` ${sparkline(armRun.regretCurve)}`);
  }
  lines.push("");

  lines.push("## Model selection share");
  lines.push("");
  for (const armRun of runs) {
    lines.push(`### ${armRun.name}`);
    lines.push("");
    const sorted = [...armRun.picks.entries()].sort((a, b) => b[1] - a[1]);
    for (const [modelId, count] of sorted) {
      lines.push(`- \`${modelId}\` — ${((count / ROUNDS) * 100).toFixed(1)}%`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

const { runs, oracleReward } = run();
const baseline = runs[0] as ArmRun;
const linucb = runs[1] as ArmRun;

const outputPath = resolve(dirname(fileURLToPath(import.meta.url)), "../out/simulation.md");
mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, report(runs, oracleReward), "utf8");

const gap = oracleReward - baseline.totalReward;

for (const armRun of runs) {
  const delta = ((armRun.totalReward - baseline.totalReward) / baseline.totalReward) * 100;
  const closed = gap > 0 ? ((armRun.totalReward - baseline.totalReward) / gap) * 100 : 0;
  console.log(
    `${armRun.name.padEnd(22)} reward=${armRun.totalReward.toFixed(1).padStart(8)}  ` +
      `optimal=${((armRun.oracleHits / ROUNDS) * 100).toFixed(1).padStart(5)}%  ` +
      `regret=${armRun.totalRegret.toFixed(1).padStart(7)}  ` +
      `vs baseline=${delta >= 0 ? "+" : ""}${delta.toFixed(2)}%  gap closed=${closed.toFixed(1)}%  ` +
      `deferrals=${armRun.deferrals}`,
  );
}
console.log(
  `oracle upper bound     reward=${oracleReward.toFixed(1).padStart(8)}  optimal=100.0%  ` +
    `(baseline is already within ${((gap / oracleReward) * 100).toFixed(2)}% of optimal)`,
);
console.log(`\nReport written to ${outputPath}`);

// CI runs this. The exit criterion for Phase 4 is that the bandit beats the rules it replaced;
// failing the build is how that stays true as the feature vector and reward function evolve.
if (linucb.totalReward <= baseline.totalReward) {
  console.error(
    `\nFAIL: LinUCB (${linucb.totalReward.toFixed(1)}) did not beat static (${baseline.totalReward.toFixed(1)}).`,
  );
  process.exit(1);
}
if (linucb.totalRegret >= baseline.totalRegret) {
  console.error(
    `\nFAIL: LinUCB regret (${linucb.totalRegret.toFixed(1)}) is not below static (${baseline.totalRegret.toFixed(1)}).`,
  );
  process.exit(1);
}

console.log("\nPASS: adaptive routing beats the static baseline on reward and regret.");
