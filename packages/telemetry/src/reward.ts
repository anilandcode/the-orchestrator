import type { CallEvent, RouteMode, TaskType } from "@orchestrator/shared";
import type { CallEventRepository } from "./store/repository.js";

/**
 * The reward function. Everything the adaptive router learns comes through here, so a mistake in this
 * file does not produce a bug — it produces a router that confidently optimizes for the wrong thing.
 */

export interface RewardWeights {
  quality: number;
  cost: number;
  latency: number;
}

/**
 * Weights per route mode. Each set sums to 1 so rewards stay comparable across modes — otherwise a
 * `best`-mode arm would look better than a `cheap`-mode arm purely from the scale of the sum.
 *
 * This is the preference-conditioned routing knob: exposing these as a customer-facing slider is what
 * Phase 8 turns them into.
 */
export const REWARD_WEIGHTS: Readonly<Record<RouteMode, RewardWeights>> = Object.freeze({
  cheap: { quality: 0.35, cost: 0.5, latency: 0.15 },
  balanced: { quality: 0.6, cost: 0.25, latency: 0.15 },
  best: { quality: 0.85, cost: 0.05, latency: 0.1 },
});

export interface NormalizationStats {
  /** Cost at which the cost term bottoms out. */
  costScaleUsd: number;
  /** Latency at which the latency term bottoms out. */
  latencyScaleMs: number;
}

/** Used only until enough real traffic exists to compute percentiles. */
export const DEFAULT_NORMALIZATION: NormalizationStats = {
  costScaleUsd: 0.05,
  latencyScaleMs: 10_000,
};

/** Quality assumed for a clean success when nothing better is available. */
export const DEFAULT_SUCCESS_QUALITY = 0.8;

/**
 * The fallback quality signal, used when no scorer or explicit feedback has supplied one.
 *
 * It is intentionally coarse. The honest position is that we do not know how good an answer was
 * without evaluating it — this only separates the cases we can read off the response envelope.
 */
export function heuristicQuality(event: CallEvent): number {
  if (event.status === "error") return 0;

  switch (event.finishReason) {
    case "content_filter":
      return 0;
    case "length":
      // Truncated output is a partial answer, not a good one.
      return 0.5;
    default:
      return DEFAULT_SUCCESS_QUALITY;
  }
}

export interface RewardOptions {
  stats?: NormalizationStats;
  /** Overrides both the stored score and the heuristic. */
  quality?: number | null;
}

export function computeReward(event: CallEvent, options: RewardOptions = {}): number {
  // A failed attempt scores exactly zero.
  //
  // This short-circuit is load-bearing. A failure costs $0 and returns in milliseconds, so running it
  // through the weighted sum below would score it ABOVE a slow, expensive, correct answer — and the
  // bandit would dutifully learn to prefer models that fail fast. Do not remove this.
  if (event.status === "error") return 0;

  const weights = REWARD_WEIGHTS[event.routeMode];
  const stats = options.stats ?? DEFAULT_NORMALIZATION;

  const quality = clamp01(options.quality ?? event.qualityScore ?? heuristicQuality(event));
  const normalizedCost = clamp01(event.costUsd / stats.costScaleUsd);
  const normalizedLatency = clamp01(event.latencyMs / stats.latencyScaleMs);

  return clamp01(
    weights.quality * quality +
      weights.cost * (1 - normalizedCost) +
      weights.latency * (1 - normalizedLatency),
  );
}

/**
 * Rolling per-task-type percentiles for cost and latency.
 *
 * Fixed constants would go stale the moment the model pool changes: a new cheap model shifts what
 * "expensive" means. Normalizing against recent traffic of the same task type keeps the cost and
 * latency terms meaningful as the pool evolves.
 */
export class RollingNormalizer {
  private readonly costs = new Map<TaskType, number[]>();
  private readonly latencies = new Map<TaskType, number[]>();

  constructor(
    private readonly windowSize = 500,
    private readonly percentile = 0.95,
    private readonly minSamples = 20,
    private readonly fallback: NormalizationStats = DEFAULT_NORMALIZATION,
  ) {}

  /** Only successful calls inform the scale — failures have no meaningful cost or latency. */
  observe(event: CallEvent): void {
    if (event.status !== "success") return;
    push(this.costs, event.taskType, event.costUsd, this.windowSize);
    push(this.latencies, event.taskType, event.latencyMs, this.windowSize);
  }

  observeAll(events: Iterable<CallEvent>): void {
    for (const event of events) this.observe(event);
  }

  statsFor(taskType: TaskType): NormalizationStats {
    const costs = this.costs.get(taskType) ?? [];
    const latencies = this.latencies.get(taskType) ?? [];

    if (costs.length < this.minSamples) return this.fallback;

    return {
      // A degenerate scale of 0 would make every call look infinitely expensive.
      costScaleUsd: percentileOf(costs, this.percentile) || this.fallback.costScaleUsd,
      latencyScaleMs: percentileOf(latencies, this.percentile) || this.fallback.latencyScaleMs,
    };
  }
}

/** Ties the normalizer to the store: score an event, persist the score, return the reward. */
export class RewardService {
  constructor(
    private readonly repository: CallEventRepository,
    private readonly normalizer: RollingNormalizer = new RollingNormalizer(),
  ) {}

  settle(event: CallEvent, quality?: number | null): number {
    this.normalizer.observe(event);

    const stats = this.normalizer.statsFor(event.taskType);
    const reward = computeReward(event, {
      stats,
      ...(quality !== undefined ? { quality } : {}),
    });

    const qualityScore =
      quality ?? event.qualityScore ?? (event.status === "error" ? 0 : heuristicQuality(event));

    this.repository.scoreEvent(event.id, qualityScore, reward);
    return reward;
  }

  statsFor(taskType: TaskType): NormalizationStats {
    return this.normalizer.statsFor(taskType);
  }

  /** Warm the rolling window from history, so a restart does not reset the scale. */
  warmUp(events: Iterable<CallEvent>): void {
    this.normalizer.observeAll(events);
  }
}

// --- helpers ----------------------------------------------------------------

function clamp01(value: number): number {
  if (Number.isNaN(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function push(
  map: Map<TaskType, number[]>,
  key: TaskType,
  value: number,
  windowSize: number,
): void {
  const list = map.get(key) ?? [];
  list.push(value);
  if (list.length > windowSize) list.shift();
  map.set(key, list);
}

export function percentileOf(values: number[], percentile: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(percentile * sorted.length) - 1));
  return sorted[index] ?? 0;
}
