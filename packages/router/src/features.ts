import { TASK_TYPES } from "@orchestrator/shared";
import { REWARD_WEIGHTS } from "@orchestrator/telemetry";
import type { RoutingContext } from "./router.js";

/**
 * The context vector the bandit conditions on.
 *
 * Deliberately cheap: extraction runs on every request, ahead of the model call, so nothing here may
 * tokenize, embed, or call out. Every component is already known by the time routing happens.
 *
 * Components are kept roughly in [0,1] so no single one dominates the ridge regression by scale.
 */

const FEATURE_NAMES = [
  "bias",
  "log_prompt_tokens",
  "has_tools",
  "has_images",
  "turn_depth",
  "prior_attempt_failed",
  ...TASK_TYPES.map((task) => `task_${task}`),
  "weight_quality",
  "weight_cost",
  "weight_latency",
] as const;

export const FEATURE_DIMENSION = FEATURE_NAMES.length;

export function featureNames(): readonly string[] {
  return FEATURE_NAMES;
}

export function extractFeatures(context: RoutingContext): number[] {
  const { request, estimatedPromptTokens } = context;
  const route = request.route;
  const weights = REWARD_WEIGHTS[route.mode];

  const hasTools = request.tools?.length ? 1 : 0;
  const hasImages = request.messages.some(
    (message) =>
      Array.isArray(message.content) && message.content.some((part) => part.type === "image"),
  )
    ? 1
    : 0;

  const features: number[] = [
    1, // bias
    // log-scaled: the difference between 100 and 1,000 tokens matters more than 100k vs 101k.
    Math.min(1, Math.log10(1 + estimatedPromptTokens) / 6),
    hasTools,
    hasImages,
    Math.min(1, (context.turnIndex ?? 0) / 10),
    context.priorAttemptFailed ? 1 : 0,
  ];

  for (const task of TASK_TYPES) {
    features.push(route.taskType === task ? 1 : 0);
  }

  // Preference conditioning: the bandit learns which model wins *given* how this caller trades off
  // quality, cost, and speed — not one global ranking applied to everyone.
  features.push(weights.quality, weights.cost, weights.latency);

  return features;
}
