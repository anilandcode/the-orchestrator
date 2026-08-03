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
import {
  type Observability,
  VALIDATOR_COVERED,
  bestArm,
  seededRandom,
  simulateCall,
  simulationModels,
} from "./world.js";

/**
 * Offline evaluation against a known-ground-truth world.
 *
 * The central question this answers is not "does a bandit beat rules" in the abstract — it is
 * **"does the quality signal carry enough information for the bandit to learn from?"**
 *
 * So the run includes a control: the same LinUCB, on the same traffic, with quality observable only
 * through the pre-Phase-4.5 heuristic. The gap between that arm and the validated one *is* the value
 * of the quality work. If they perform the same, the validators are not earning their place and the
 * honest response is to say so and reweight the reward toward cost and latency.
 *
 * Regret is always measured against latent ground truth, never against what the scorers could see.
 * A router that cannot observe quality is still wrong when it picks a worse model — it just has no
 * way to know.
 */

const ROUNDS = Number(process.env.SIM_ROUNDS ?? 8_000);
/**
 * Exploration settings are env-overridable because they are the parameters most worth sweeping.
 *
 * An exploration budget only makes sense relative to the headroom it is chasing. If the static rules
 * are already within 2% of optimal, spending 3% of traffic on deliberate exploration costs more than
 * perfect routing could ever return — the sweep in `SIM_ALPHA`/`SIM_FLOOR` is how you find that out
 * rather than assuming it.
 */
const ALPHA = Number(process.env.SIM_ALPHA ?? 0.35);
const EXPLORATION_FLOOR = Number(process.env.SIM_FLOOR ?? 0);
const ROUTE_MODES: RouteMode[] = ["cheap", "balanced", "best"];

interface ArmRun {
  name: string;
  router: Router;
  observability: Observability;
  /**
   * Sum of *observed* reward — what this arm's scorers could see.
   *
   * NOT comparable across arms with different observability: the validated arm observes binary 0/1
   * quality while the heuristic arm observes a constant 0.8, so their sums are drawn from different
   * distributions. Kept only to show what each arm was learning from.
   */
  observedReward: number;
  /**
   * Sum of the ground-truth expected value of the models actually chosen. This *is* comparable
   * across arms, and is the headline metric.
   */
  trueValue: number;
  totalRegret: number;
  regretCurve: number[];
  picks: Map<string, number>;
  oracleHits: number;
  /** Optimal picks split by whether a validator could see quality on that task. */
  coveredHits: number;
  coveredRounds: number;
  uncoveredHits: number;
  uncoveredRounds: number;
  totalCostUsd: number;
}

interface Scenario {
  context: RoutingContext;
  taskType: TaskType;
  routeMode: RouteMode;
  promptTokens: number;
  completionTokens: number;
}

function makeScenario(random: () => number): Scenario {
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

function adaptiveRouter(seed: number, minQualityConfidence = 0): AdaptiveRouter {
  return new AdaptiveRouter({
    minQualityConfidence,
    bandit: new LinUcbBandit({
      dimension: FEATURE_DIMENSION,
      alpha: ALPHA,
      explorationFloor: EXPLORATION_FLOOR,
      random: seededRandom(seed),
    }),
    baseline: new StaticRouter({ ids: createSequentialIds() }),
    mode: "adaptive",
    coldStartPulls: 15,
    ids: createSequentialIds(),
  });
}

function buildRuns(): ArmRun[] {
  const empty = () => ({
    observedReward: 0,
    trueValue: 0,
    totalRegret: 0,
    regretCurve: [] as number[],
    picks: new Map<string, number>(),
    oracleHits: 0,
    coveredHits: 0,
    coveredRounds: 0,
    uncoveredHits: 0,
    uncoveredRounds: 0,
    totalCostUsd: 0,
  });

  return [
    {
      name: "static (baseline)",
      router: new StaticRouter({ ids: createSequentialIds() }),
      observability: "validated",
      ...empty(),
    },
    {
      name: "LinUCB (heuristic only)",
      router: adaptiveRouter(99),
      observability: "heuristic-only",
      ...empty(),
    },
    {
      name: "LinUCB (validated)",
      router: adaptiveRouter(99),
      observability: "validated",
      ...empty(),
    },
    {
      name: "LinUCB (validated + gated)",
      router: adaptiveRouter(99, 0.5),
      observability: "validated",
      ...empty(),
    },
    {
      name: "Thompson (validated)",
      router: new AdaptiveRouter({
        bandit: new ThompsonBandit({ random: seededRandom(123) }),
        baseline: new StaticRouter({ ids: createSequentialIds() }),
        mode: "adaptive",
        coldStartPulls: 15,
        ids: createSequentialIds(),
      }),
      observability: "validated",
      ...empty(),
    },
  ];
}

function run(): { runs: ArmRun[]; oracleReward: number } {
  const runs = buildRuns();
  let oracleReward = 0;

  for (let round = 0; round < ROUNDS; round++) {
    const scenario = makeScenario(seededRandom(round * 7 + 1));
    const candidates = selectCandidates(scenario.context);
    const covered = VALIDATOR_COVERED.has(scenario.taskType);

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

      // Common random numbers: every arm faces the identical noise draw, so a difference in outcome
      // is a difference in judgement rather than in luck.
      const { event, reward, observed } = simulateCall({
        spec,
        taskType: scenario.taskType,
        routeMode: scenario.routeMode,
        promptTokens: scenario.promptTokens,
        completionTokens: scenario.completionTokens,
        features: extractFeatures(scenario.context),
        random: seededRandom(round * 31 + 17),
        observability: armRun.observability,
      });

      armRun.router.observe({
        decisionId: decision.decisionId,
        modelId: decision.modelId,
        features: decision.features,
        taskType: scenario.taskType,
        reward,
        qualityConfidence: observed.confidence,
      });

      armRun.observedReward += reward;
      armRun.totalCostUsd += event.costUsd;
      armRun.picks.set(decision.modelId, (armRun.picks.get(decision.modelId) ?? 0) + 1);

      const hit = decision.modelId === oracle.spec.modelId;
      if (hit) armRun.oracleHits += 1;
      if (covered) {
        armRun.coveredRounds += 1;
        if (hit) armRun.coveredHits += 1;
      } else {
        armRun.uncoveredRounds += 1;
        if (hit) armRun.uncoveredHits += 1;
      }

      // Regret against the best *expected* arm under ground truth, so bad luck is not scored as a
      // bad decision.
      const chosenExpected = bestArm(
        [spec],
        scenario.taskType,
        scenario.routeMode,
        scenario.promptTokens,
        scenario.completionTokens,
      ).reward;
      armRun.trueValue += chosenExpected;
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
  const baseline = runs[0] as ArmRun;
  const gap = oracleReward - baseline.trueValue;
  const lines: string[] = [];

  const closed = (armRun: ArmRun) =>
    gap > 0 ? ((armRun.trueValue - baseline.trueValue) / gap) * 100 : 0;

  lines.push("# Adaptive Router — Simulated Evaluation");
  lines.push("");
  lines.push(`Rounds: **${ROUNDS.toLocaleString()}**  ·  Generated: ${new Date().toISOString()}`);
  lines.push("");
  lines.push("Synthetic traffic against a known-ground-truth world. This proves the algorithm,");
  lines.push(
    "features, and reward function work together — it says nothing about production traffic.",
  );
  lines.push("Run `pnpm replay` against real `CallEvent` data before promoting `ROUTER_MODE`.");
  lines.push("");
  lines.push(
    "**Quality is only partially observable, by design.** Validators can grade extraction,",
  );
  lines.push(
    "code, and classification; open-ended prose falls back to the heuristic constant. The",
  );
  lines.push(
    "`heuristic only` arm is the pre-Phase-4.5 control — the difference between it and the",
  );
  lines.push("validated arm is what the quality work bought.");
  lines.push("");

  lines.push("## Results");
  lines.push("");
  lines.push(
    "| Strategy | True value | Optimal pick | Gap to oracle closed | Regret | Observed reward | Cost |",
  );
  lines.push("|---|---:|---:|---:|---:|---:|---:|");
  for (const armRun of runs) {
    const gapCell = armRun === baseline ? "—" : `${closed(armRun).toFixed(1)}%`;
    lines.push(
      `| ${armRun.name} | ${armRun.trueValue.toFixed(1)} | ${((armRun.oracleHits / ROUNDS) * 100).toFixed(1)}% | ${gapCell} | ${armRun.totalRegret.toFixed(1)} | ${armRun.observedReward.toFixed(1)} | $${armRun.totalCostUsd.toFixed(2)} |`,
    );
  }
  lines.push(
    `| *oracle (upper bound)* | ${oracleReward.toFixed(1)} | 100.0% | 100% | 0.0 | — | — |`,
  );
  lines.push("");

  lines.push("## Where the quality signal pays off");
  lines.push("");
  lines.push("Optimal-pick rate split by whether a deterministic validator could grade the task.");
  lines.push("");
  lines.push("| Strategy | Validator-covered tasks | Heuristic-only tasks |");
  lines.push("|---|---:|---:|");
  for (const armRun of runs) {
    const coveredPct = armRun.coveredRounds
      ? ((armRun.coveredHits / armRun.coveredRounds) * 100).toFixed(1)
      : "—";
    const uncoveredPct = armRun.uncoveredRounds
      ? ((armRun.uncoveredHits / armRun.uncoveredRounds) * 100).toFixed(1)
      : "—";
    lines.push(`| ${armRun.name} | ${coveredPct}% | ${uncoveredPct}% |`);
  }
  lines.push("");
  lines.push(
    "Covered tasks: `extraction`, `code`, `classification`. Everything else is graded only by",
  );
  lines.push('"the call did not error", which is a constant and therefore carries no signal about');
  lines.push("*which* model to prefer.");
  lines.push("");

  lines.push("## Cumulative regret");
  lines.push("");
  lines.push("Flattening means the router has stopped making avoidable mistakes.");
  lines.push("");
  for (const armRun of runs) {
    lines.push(`- \`${armRun.name.padEnd(24)}\` ${sparkline(armRun.regretCurve)}`);
  }
  lines.push("");

  lines.push("## Model selection share");
  lines.push("");
  for (const armRun of runs) {
    lines.push(`### ${armRun.name}`);
    lines.push("");
    for (const [modelId, count] of [...armRun.picks.entries()].sort((a, b) => b[1] - a[1])) {
      lines.push(`- \`${modelId}\` — ${((count / ROUNDS) * 100).toFixed(1)}%`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

const { runs, oracleReward } = run();
const baseline = runs[0] as ArmRun;
const heuristicOnly = runs[1] as ArmRun;
const validated = runs[2] as ArmRun;
const gated = runs[3] as ArmRun;
const gap = oracleReward - baseline.trueValue;

const outputPath = resolve(dirname(fileURLToPath(import.meta.url)), "../out/simulation.md");
mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, report(runs, oracleReward), "utf8");

for (const armRun of runs) {
  const closed = gap > 0 ? ((armRun.trueValue - baseline.trueValue) / gap) * 100 : 0;
  const coveredPct = armRun.coveredRounds ? (armRun.coveredHits / armRun.coveredRounds) * 100 : 0;
  console.log(
    `${armRun.name.padEnd(24)} true=${armRun.trueValue.toFixed(1).padStart(8)}  ` +
      `optimal=${((armRun.oracleHits / ROUNDS) * 100).toFixed(1).padStart(5)}%  ` +
      `covered=${coveredPct.toFixed(1).padStart(5)}%  ` +
      `regret=${armRun.totalRegret.toFixed(1).padStart(7)}  gap closed=${closed.toFixed(1).padStart(6)}%`,
  );
}
console.log(
  `oracle upper bound       true=${oracleReward.toFixed(1).padStart(8)}  ` +
    `(baseline is already within ${((gap / oracleReward) * 100).toFixed(2)}% of optimal)`,
);

const validatedClosed = gap > 0 ? ((validated.trueValue - baseline.trueValue) / gap) * 100 : 0;
const heuristicClosed = gap > 0 ? ((heuristicOnly.trueValue - baseline.trueValue) / gap) * 100 : 0;
console.log(
  `\nQuality signal contribution: ${(validatedClosed - heuristicClosed).toFixed(1)} percentage points ` +
    `of the oracle gap (${heuristicClosed.toFixed(1)}% -> ${validatedClosed.toFixed(1)}%)`,
);

const validatedCovered = validated.coveredHits / validated.coveredRounds;
const heuristicCovered = heuristicOnly.coveredHits / heuristicOnly.coveredRounds;
const staticCovered = baseline.coveredHits / baseline.coveredRounds;

console.log(
  `\nOn validator-covered tasks: static ${(staticCovered * 100).toFixed(1)}%  ->  ` +
    `bandit+heuristic ${(heuristicCovered * 100).toFixed(1)}%  ->  ` +
    `bandit+validators ${(validatedCovered * 100).toFixed(1)}% optimal picks`,
);

if (validated.trueValue <= baseline.trueValue) {
  console.log(
    `\nNOTE: the UNGATED bandit trails static on total value (${validated.trueValue.toFixed(1)} vs ${baseline.trueValue.toFixed(1)}). The static rules sit within ${((gap / oracleReward) * 100).toFixed(2)}% of optimal, and the bandit's mistakes on tasks no validator can grade cost more than that headroom is worth. Gating is the answer to this, not a workaround.`,
  );
}

/*
 * CI exit criteria.
 *
 * These assert what the measurement supports. The UNGATED bandit does not beat static on total
 * value in this world, and asserting that it does would either fail forever or invite quietly
 * reshaping the world until it passed. What the evidence supports is sharper:
 *
 *   1. Where quality is observable, the bandit routes better than the rules.
 *   2. Observable quality beats the heuristic it replaced.
 *   3. Gated to observable tasks, the bandit beats the rules OVERALL — on value and on regret.
 *
 * (3) is the product claim. If it breaks, adaptive routing is not earning its place.
 */
if (validatedCovered <= staticCovered) {
  console.error(
    "\nFAIL: on validator-covered tasks the bandit no longer beats the static baseline.\n" +
      "That is the narrowest claim this system rests on; if it breaks, the wedge is gone.",
  );
  process.exit(1);
}
if (validatedCovered <= heuristicCovered) {
  console.error(
    "\nFAIL: observable quality did not improve optimal-pick rate over the heuristic.\n" +
      "That is the premise of Phase 4.5 — if it does not hold, the validators are not carrying signal\n" +
      "and the honest response is to reweight the reward toward cost and latency.",
  );
  process.exit(1);
}
if (gated.trueValue <= baseline.trueValue) {
  console.error(
    `\nFAIL: the gated router (${gated.trueValue.toFixed(1)}) did not beat static (${baseline.trueValue.toFixed(1)}) on total value.\n` +
      "Routing adaptively only where quality is observable is the product claim; without it there is\n" +
      "no measured reason to run a bandit at all.",
  );
  process.exit(1);
}
if (gated.totalRegret >= baseline.totalRegret) {
  console.error("\nFAIL: gated router regret is not below the static baseline.");
  process.exit(1);
}

console.log(
  `\nGating: ${gated.trueValue.toFixed(1)} vs static ${baseline.trueValue.toFixed(1)} true value, ` +
    `regret ${gated.totalRegret.toFixed(1)} vs ${baseline.totalRegret.toFixed(1)}, ` +
    `${(((gated.trueValue - baseline.trueValue) / gap) * 100).toFixed(1)}% of the oracle gap closed.`,
);
console.log(
  "\nPASS: validators beat the heuristic, and gated adaptive routing beats the static baseline.",
);
