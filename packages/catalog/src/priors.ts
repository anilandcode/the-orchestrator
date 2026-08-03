import {
  CallEventSchema,
  type ModelRegistry,
  ROUTE_MODES,
  type RouteMode,
  TASK_TYPES,
  type TaskType,
  computeCostUsd,
} from "@orchestrator/shared";
import { computeReward } from "@orchestrator/telemetry";
import { mappingFor, percentileRank } from "./mapping.js";
import type { BenchmarkScore, TaskCapability } from "./schema.js";

/**
 * Turning published benchmark results into router priors.
 *
 * Two properties matter more than the arithmetic:
 *
 *   1. **Priors are computed with the real reward function.** A prior that lived on its own scale
 *      would be incomparable to what the bandit learns from live traffic, and the seeding would be
 *      quietly meaningless. `computeReward` from telemetry is used verbatim.
 *   2. **A prior is scoped to (taskType, routeMode).** Reward weights differ by mode, so "excellent
 *      at code" is not one claim but three: in `best` mode an expensive strong model is the right
 *      answer, and in `cheap` mode the same model is the wrong one. Collapsing them would seed the
 *      router with an averaged opinion it can never act on correctly.
 */

/** Prompt/completion sizes used for the synthetic pricing of a prior. Mid-range, not adversarial. */
const REFERENCE_PROMPT_TOKENS = 1_500;
const REFERENCE_COMPLETION_TOKENS = 400;

/**
 * Ceiling on prior strength, in pseudo-observations.
 *
 * Sized against `DEFAULT_COLD_START_PULLS` (25): a full-coverage prior is worth roughly half of what
 * it takes to open the cold-start gate, so real traffic overtakes external evidence quickly. Raising
 * this trades faster convergence-to-benchmark for slower correction when a benchmark is wrong, and
 * benchmarks are wrong often enough that the conservative side is the right default.
 */
export const MAX_PRIOR_WEIGHT = 12;

export interface DerivedPrior {
  modelId: string;
  taskType: TaskType;
  routeMode: RouteMode;
  /** Expected reward, on the same 0..1 scale live outcomes produce. */
  reward: number;
  weight: number;
  capability: number;
  source: string;
}

/**
 * Collapse raw benchmark scores into a per-(model, task) capability in [0,1].
 *
 * Returns nothing for task types the mapping declines to cover, and nothing for models with no
 * contributing benchmark. Both are abstentions, and both must stay abstentions: a default capability
 * would be indistinguishable, to the bandit, from a measured one.
 */
export function deriveCapabilities(scores: BenchmarkScore[]): TaskCapability[] {
  // Population per benchmark, for percentile normalization.
  const populations = new Map<string, number[]>();
  for (const score of scores) {
    const population = populations.get(score.benchmarkId) ?? [];
    population.push(score.score);
    populations.set(score.benchmarkId, population);
  }

  const byModelBenchmark = new Map<string, number>();
  for (const score of scores) {
    byModelBenchmark.set(`${score.modelId}|${score.benchmarkId}`, score.score);
  }

  const modelIds = [...new Set(scores.map((score) => score.modelId))];
  const capabilities: TaskCapability[] = [];

  for (const taskType of TASK_TYPES) {
    const mapping = mappingFor(taskType);
    if (!mapping) continue; // Abstention: no defensible benchmark for this task type.

    for (const modelId of modelIds) {
      let weighted = 0;
      let coveredWeight = 0;
      const contributing: string[] = [];

      for (const { benchmarkId, weight } of mapping) {
        const raw = byModelBenchmark.get(`${modelId}|${benchmarkId}`);
        if (raw === undefined) continue;

        const population = populations.get(benchmarkId) ?? [];
        weighted += weight * percentileRank(raw, population);
        coveredWeight += weight;
        contributing.push(benchmarkId);
      }

      // No contributing benchmark: say nothing rather than default to a middling capability.
      if (coveredWeight === 0) continue;

      const totalWeight = mapping.reduce((total, entry) => total + entry.weight, 0);
      capabilities.push({
        modelId,
        taskType,
        // Renormalized over covered weight, so partial coverage does not read as low capability.
        capability: weighted / coveredWeight,
        contributingBenchmarks: contributing,
        // Coverage is reported separately and drives prior strength instead.
        coverage: coveredWeight / totalWeight,
      });
    }
  }

  return capabilities;
}

/**
 * Convert capabilities into priors the router can seed.
 *
 * Only models the registry actually knows are included: a prior for a model the gateway cannot reach
 * is dead weight, and seeding it would put an unreachable arm into the bandit's state.
 */
export function derivePriors(
  capabilities: TaskCapability[],
  registry: ModelRegistry,
  options: { maxWeight?: number } = {},
): DerivedPrior[] {
  const maxWeight = options.maxWeight ?? MAX_PRIOR_WEIGHT;
  const priors: DerivedPrior[] = [];

  for (const capability of capabilities) {
    const spec = registry.get(capability.modelId);
    if (!spec) continue;

    for (const routeMode of ROUTE_MODES) {
      const usage = {
        promptTokens: REFERENCE_PROMPT_TOKENS,
        completionTokens: REFERENCE_COMPLETION_TOKENS,
        totalTokens: REFERENCE_PROMPT_TOKENS + REFERENCE_COMPLETION_TOKENS,
        cachedPromptTokens: 0,
      };

      // A synthetic event, scored by the real reward function so the prior lands on the same scale
      // as everything the bandit will later learn from.
      const event = CallEventSchema.parse({
        id: "prior",
        tenantId: "prior",
        requestId: "prior",
        attempt: 1,
        provider: spec.provider,
        modelId: spec.modelId,
        taskType: capability.taskType,
        routeMode,
        promptTokens: usage.promptTokens,
        completionTokens: usage.completionTokens,
        costUsd: computeCostUsd(spec, usage),
        latencyMs: spec.typicalLatencyMs,
        status: "success",
        finishReason: "stop",
        createdAt: 0,
      });

      priors.push({
        modelId: spec.modelId,
        taskType: capability.taskType,
        routeMode,
        reward: computeReward(event, { quality: capability.capability }),
        // Thin benchmark coverage yields a weaker prior, which is the honest response to knowing
        // less rather than pretending otherwise.
        weight: Math.round(maxWeight * capability.coverage * 100) / 100,
        capability: capability.capability,
        source: `benchmarks:${capability.contributingBenchmarks.join("+")}`,
      });
    }
  }

  return priors;
}
