import {
  type CallEvent,
  CallEventSchema,
  type ModelSpec,
  type RouteMode,
  type TaskType,
  computeCostUsd,
  defaultRegistry,
} from "@orchestrator/shared";
import { computeReward } from "@orchestrator/telemetry";

/**
 * A synthetic world with known ground truth.
 *
 * The point of simulating rather than only replaying: on real logs we observe the outcome of the arm
 * that was chosen and nothing about the others, so "did the router pick well?" is unanswerable. Here
 * the true quality of every model on every task is fixed in advance, so regret against the optimal
 * choice is exactly computable.
 *
 * The quality table below is the load-bearing assumption. It is deliberately *contextual* — cheap
 * models are competitive on classification and extraction but weak on code and reasoning — because a
 * world where one model dominates everywhere needs no router at all, and would flatter the bandit
 * without proving anything.
 */

export interface ModelProfile {
  /** Mean quality in [0,1] per task type. */
  quality: Record<TaskType, number>;
  /** Probability that a call fails outright. */
  failureRate: number;
  /** Multiplicative latency noise. */
  latencyJitter: number;
}

export const MODEL_PROFILES: Record<string, ModelProfile> = {
  "openai/gpt-4o-mini": {
    quality: {
      general: 0.72,
      code: 0.55,
      extraction: 0.86,
      summarization: 0.82,
      classification: 0.9,
      reasoning: 0.48,
      creative: 0.66,
    },
    failureRate: 0.02,
    latencyJitter: 0.3,
  },
  "openai/gpt-4.1": {
    quality: {
      general: 0.84,
      code: 0.83,
      extraction: 0.9,
      summarization: 0.87,
      classification: 0.91,
      reasoning: 0.8,
      creative: 0.79,
    },
    failureRate: 0.015,
    latencyJitter: 0.25,
  },
  "openai/gpt-4o": {
    quality: {
      general: 0.83,
      code: 0.78,
      extraction: 0.88,
      summarization: 0.86,
      classification: 0.9,
      reasoning: 0.75,
      creative: 0.81,
    },
    failureRate: 0.015,
    latencyJitter: 0.25,
  },
  "anthropic/claude-haiku-4-5": {
    quality: {
      general: 0.78,
      code: 0.7,
      extraction: 0.87,
      summarization: 0.86,
      classification: 0.89,
      reasoning: 0.6,
      creative: 0.74,
    },
    failureRate: 0.012,
    latencyJitter: 0.28,
  },
  "anthropic/claude-sonnet-5": {
    quality: {
      general: 0.88,
      code: 0.9,
      extraction: 0.9,
      summarization: 0.89,
      classification: 0.9,
      reasoning: 0.88,
      creative: 0.86,
    },
    failureRate: 0.01,
    latencyJitter: 0.22,
  },
  "anthropic/claude-opus-5": {
    quality: {
      general: 0.91,
      code: 0.94,
      extraction: 0.91,
      summarization: 0.9,
      classification: 0.9,
      reasoning: 0.95,
      creative: 0.9,
    },
    failureRate: 0.008,
    latencyJitter: 0.2,
  },
};

export interface SimulatedCall {
  event: CallEvent;
  reward: number;
  /** The model's true quality on this draw, which the router does not get to see directly. */
  latentQuality: number;
  /** What a scorer was actually able to measure, and how confident it was. */
  observed: { quality: number; source: string; confidence: number };
}

/**
 * Task types a deterministic validator can actually grade.
 *
 * This is the crux of Phase 4.5. Extraction has a declarable output schema, code has checkable
 * structure, and classification has an enum to validate against — a validator reads real quality off
 * those. Open-ended prose has none of that, so absent a judge the only signal is "it did not error",
 * which is a constant.
 *
 * Modelling that split is what makes the simulation honest: the earlier version handed the router
 * true quality on every call, which no deployment will ever have.
 */
export const VALIDATOR_COVERED: ReadonlySet<TaskType> = new Set<TaskType>([
  "extraction",
  "code",
  "classification",
]);

export type Observability = "validated" | "heuristic-only";

/** The constant a clean success scores when nothing can actually grade it. */
const HEURISTIC_SUCCESS_QUALITY = 0.8;

export interface SimulationInputs {
  spec: ModelSpec;
  taskType: TaskType;
  routeMode: RouteMode;
  promptTokens: number;
  completionTokens: number;
  features: number[];
  random: () => number;
  /**
   * `validated` — validators grade covered task types; everything else falls to the heuristic.
   * `heuristic-only` — nothing grades anything. This is the pre-Phase-4.5 world, kept as the control.
   */
  observability?: Observability;
}

/** Draws one outcome and scores it with the real reward function, not a stand-in. */
export function simulateCall(inputs: SimulationInputs): SimulatedCall {
  const profile = MODEL_PROFILES[inputs.spec.modelId];
  if (!profile) throw new Error(`No simulation profile for ${inputs.spec.modelId}`);

  const failed = inputs.random() < profile.failureRate;
  const usage = {
    promptTokens: inputs.promptTokens,
    completionTokens: failed ? 0 : inputs.completionTokens,
    totalTokens: inputs.promptTokens + (failed ? 0 : inputs.completionTokens),
    cachedPromptTokens: 0,
  };

  const latencyMs = failed
    ? inputs.spec.typicalLatencyMs * 0.3
    : inputs.spec.typicalLatencyMs * (1 + (inputs.random() - 0.5) * 2 * profile.latencyJitter);

  const latentQuality = failed
    ? 0
    : clamp01(profile.quality[inputs.taskType] + (inputs.random() - 0.5) * 0.15);

  // What a scorer can actually see. A validator on a covered task reads latent quality closely; on
  // anything else the router is left with a constant that carries no information about the model.
  const observability = inputs.observability ?? "validated";
  const covered = observability === "validated" && VALIDATOR_COVERED.has(inputs.taskType);

  let observedQuality: number;
  let source: string;
  // Mirrors the CONFIDENCE bands in packages/quality: deterministic 0.9, heuristic 0.2.
  let confidence: number;

  if (failed) {
    observedQuality = 0;
    source = "finish-reason";
    confidence = 0.9; // "it errored" is a definitive observation
  } else if (covered) {
    // Validators are binary in practice — the JSON parses or it does not, the code balances or it
    // does not — so a latent quality of 0.9 shows up as a 90% chance of scoring 1.
    observedQuality = inputs.random() < latentQuality ? 1 : 0;
    source = inputs.taskType === "code" ? "code-structure" : "json-schema";
    confidence = 0.9;
  } else {
    observedQuality = HEURISTIC_SUCCESS_QUALITY;
    source = "finish-reason";
    confidence = 0.2;
  }

  const quality = observedQuality;

  const event = CallEventSchema.parse({
    id: "sim",
    tenantId: "sim",
    requestId: "sim",
    attempt: 1,
    provider: inputs.spec.provider,
    modelId: inputs.spec.modelId,
    taskType: inputs.taskType,
    routeMode: inputs.routeMode,
    features: inputs.features,
    promptTokens: usage.promptTokens,
    completionTokens: usage.completionTokens,
    costUsd: failed ? 0 : computeCostUsd(inputs.spec, usage),
    latencyMs,
    status: failed ? "error" : "success",
    errorClass: failed ? "provider_unavailable" : null,
    finishReason: failed ? null : "stop",
    qualityScore: quality,
    qualitySource: source,
    createdAt: Date.now(),
  });

  return {
    event,
    reward: computeReward(event, { quality }),
    latentQuality,
    observed: { quality, source, confidence },
  };
}

/**
 * The expected reward of an arm, with the noise averaged out.
 *
 * Regret is measured against this rather than against sampled outcomes: a router that picked the best
 * available model and got unlucky did not make a mistake, and scoring it as one would just measure
 * variance.
 */
export function expectedReward(
  spec: ModelSpec,
  taskType: TaskType,
  routeMode: RouteMode,
  promptTokens: number,
  completionTokens: number,
): number {
  const profile = MODEL_PROFILES[spec.modelId];
  if (!profile) throw new Error(`No simulation profile for ${spec.modelId}`);

  const usage = {
    promptTokens,
    completionTokens,
    totalTokens: promptTokens + completionTokens,
    cachedPromptTokens: 0,
  };

  const successEvent = CallEventSchema.parse({
    id: "sim",
    tenantId: "sim",
    requestId: "sim",
    attempt: 1,
    provider: spec.provider,
    modelId: spec.modelId,
    taskType,
    routeMode,
    promptTokens,
    completionTokens,
    costUsd: computeCostUsd(spec, usage),
    latencyMs: spec.typicalLatencyMs,
    status: "success",
    finishReason: "stop",
    createdAt: 0,
  });

  const rewardIfSuccessful = computeReward(successEvent, { quality: profile.quality[taskType] });
  // A failure contributes exactly 0, so the expectation is just the success branch scaled down.
  return (1 - profile.failureRate) * rewardIfSuccessful;
}

export function bestArm(
  candidates: ModelSpec[],
  taskType: TaskType,
  routeMode: RouteMode,
  promptTokens: number,
  completionTokens: number,
): { spec: ModelSpec; reward: number } {
  let best = candidates[0] as ModelSpec;
  let bestReward = Number.NEGATIVE_INFINITY;

  for (const spec of candidates) {
    const reward = expectedReward(spec, taskType, routeMode, promptTokens, completionTokens);
    if (reward > bestReward) {
      bestReward = reward;
      best = spec;
    }
  }

  return { spec: best, reward: bestReward };
}

export function simulationModels(): ModelSpec[] {
  return defaultRegistry.list().filter((spec) => MODEL_PROFILES[spec.modelId]);
}

/** Mulberry32 — small, fast, and reproducible across runs. */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d_2b_79_f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}
