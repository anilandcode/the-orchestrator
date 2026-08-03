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
}

export interface SimulationInputs {
  spec: ModelSpec;
  taskType: TaskType;
  routeMode: RouteMode;
  promptTokens: number;
  completionTokens: number;
  features: number[];
  random: () => number;
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

  const quality = failed
    ? 0
    : clamp01(profile.quality[inputs.taskType] + (inputs.random() - 0.5) * 0.15);

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
    qualityScore: failed ? 0 : quality,
    createdAt: Date.now(),
  });

  return { event, reward: computeReward(event, { quality: failed ? 0 : quality }) };
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
