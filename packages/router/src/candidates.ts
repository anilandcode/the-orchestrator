import {
  type ModelSpec,
  OrchestratorError,
  blendedCostPerMTok,
  worstCaseCostUsd,
} from "@orchestrator/shared";
import type { RoutingContext, TenantPolicy } from "./router.js";

/**
 * Narrows the model pool to those that can actually serve this request.
 *
 * Hard filters (a call would fail, or would break an explicit promise):
 *   - missing a required capability
 *   - context window too small
 *   - worst-case cost above the caller's `maxCostUsd`
 *
 * `maxLatencyMs` is deliberately NOT a hard filter. `typicalLatencyMs` is a rough prior, and
 * eliminating every candidate on a guess serves nobody — it biases ordering instead.
 */
export function selectCandidates(context: RoutingContext, policy?: TenantPolicy): ModelSpec[] {
  const { request, available, estimatedPromptTokens } = context;
  const route = request.route;

  let pool = available;

  if (policy?.allowModels?.length) {
    const allowed = new Set(policy.allowModels);
    pool = pool.filter((spec) => allowed.has(spec.modelId));
  }
  if (policy?.denyModels?.length) {
    const denied = new Set(policy.denyModels);
    pool = pool.filter((spec) => !denied.has(spec.modelId));
  }

  const pinned = policy?.pinByTaskType?.[route.taskType] ?? route.pin;
  if (pinned) {
    const match = pool.filter((spec) => spec.modelId === pinned);
    if (match.length === 0) {
      throw new OrchestratorError("invalid_request", `Pinned model is unavailable: ${pinned}`, {
        modelId: pinned,
      });
    }
    return match;
  }

  const needsTools = Boolean(request.tools?.length);
  const needsVision = requestHasImages(context);
  // Headroom for the completion plus the imprecision of the prompt estimate itself.
  const requiredContext = Math.ceil(estimatedPromptTokens * 1.2) + (request.maxTokens ?? 1_024);

  const filtered = pool.filter((spec) => {
    if (needsTools && !spec.capabilities.tools) return false;
    if (needsVision && !spec.capabilities.vision) return false;
    if (spec.contextWindow < requiredContext) return false;
    if (
      route.maxCostUsd !== undefined &&
      worstCaseCostUsd(spec, estimatedPromptTokens) > route.maxCostUsd
    ) {
      return false;
    }
    return true;
  });

  if (filtered.length === 0) {
    throw new OrchestratorError(
      "invalid_request",
      describeEmptyPool({ needsTools, needsVision, requiredContext, route, poolSize: pool.length }),
    );
  }

  return filtered;
}

function requestHasImages(context: RoutingContext): boolean {
  return context.request.messages.some(
    (message) =>
      Array.isArray(message.content) && message.content.some((part) => part.type === "image"),
  );
}

function describeEmptyPool(args: {
  needsTools: boolean;
  needsVision: boolean;
  requiredContext: number;
  route: RoutingContext["request"]["route"];
  poolSize: number;
}): string {
  const reasons: string[] = [];
  if (args.needsTools) reasons.push("tool support");
  if (args.needsVision) reasons.push("vision support");
  reasons.push(`>= ${args.requiredContext} context tokens`);
  if (args.route.maxCostUsd !== undefined) {
    reasons.push(`worst-case cost <= $${args.route.maxCostUsd}`);
  }

  return `No model satisfies the request constraints (${reasons.join(", ")}) among ${args.poolSize} available model(s)`;
}

const TIER_RANK: Record<ModelSpec["tier"], number> = {
  economy: 0,
  standard: 1,
  premium: 2,
};

export function tierRank(spec: ModelSpec): number {
  return TIER_RANK[spec.tier];
}

export function byCostAscending(a: ModelSpec, b: ModelSpec): number {
  return blendedCostPerMTok(a) - blendedCostPerMTok(b);
}

export function byTierDescending(a: ModelSpec, b: ModelSpec): number {
  const rank = tierRank(b) - tierRank(a);
  // Among equally capable tiers, spending less is strictly better.
  return rank !== 0 ? rank : byCostAscending(a, b);
}
