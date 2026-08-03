import {
  type Clock,
  type IdGenerator,
  type ModelSpec,
  type RoutingDecision,
  type TaskType,
  systemClock,
  systemIds,
} from "@orchestrator/shared";
import { byCostAscending, byTierDescending, selectCandidates, tierRank } from "./candidates.js";
import { DEFAULT_CHAIN_LENGTH, buildFallbackChain } from "./fallback.js";
import { extractFeatures } from "./features.js";
import type { Router, RoutingContext, TenantPolicies } from "./router.js";

/**
 * Task types where the cheapest model is a false economy: a wrong answer gets retried by a human,
 * which costs more than the model ever saved. In `balanced` mode these floor at `standard`.
 */
const MIN_TIER_BY_TASK: Partial<Record<TaskType, number>> = {
  code: 1,
  reasoning: 1,
};

export interface StaticRouterConfig {
  policies?: TenantPolicies;
  chainLength?: number;
  ids?: IdGenerator;
  clock?: Clock;
}

/**
 * Deterministic rule-based routing.
 *
 * This is the baseline the adaptive router has to beat. Without it there is no way to tell whether
 * the bandit is adding value or just adding variance — so it stays maintained, not deprecated.
 */
export class StaticRouter implements Router {
  private readonly policies: TenantPolicies;
  private readonly chainLength: number;
  private readonly ids: IdGenerator;
  private readonly clock: Clock;

  constructor(config: StaticRouterConfig = {}) {
    this.policies = config.policies ?? {};
    this.chainLength = config.chainLength ?? DEFAULT_CHAIN_LENGTH;
    this.ids = config.ids ?? systemIds;
    this.clock = config.clock ?? systemClock;
  }

  select(context: RoutingContext): RoutingDecision {
    const policy = this.policies[context.request.tenantId];
    const candidates = selectCandidates(context, policy);
    const route = context.request.route;

    const pinned = policy?.pinByTaskType?.[route.taskType] ?? route.pin;
    const { primary, reason } = pinned
      ? { primary: candidates[0] as ModelSpec, reason: `pinned to ${pinned}` }
      : this.pick(candidates, context);

    return {
      decisionId: this.ids.generate("dec"),
      modelId: primary.modelId,
      fallbacks: buildFallbackChain(primary, candidates, this.chainLength),
      strategy: pinned ? "pinned" : "static",
      reason,
      shadowModelId: null,
      features: extractFeatures(context),
      taskType: route.taskType,
      routeMode: route.mode,
      createdAt: this.clock.now(),
    };
  }

  /** Static rules have nothing to learn. */
  observe(): void {}

  private pick(
    candidates: ModelSpec[],
    context: RoutingContext,
  ): { primary: ModelSpec; reason: string } {
    const route = context.request.route;
    const ordered = this.preferFastEnough(candidates, context);

    switch (route.mode) {
      case "cheap": {
        const primary = [...ordered].sort(byCostAscending)[0] as ModelSpec;
        return { primary, reason: "cheap mode: lowest blended cost per token" };
      }

      case "best": {
        const primary = [...ordered].sort(byTierDescending)[0] as ModelSpec;
        return { primary, reason: `best mode: highest tier available (${primary.tier})` };
      }

      case "balanced": {
        const floor = MIN_TIER_BY_TASK[route.taskType] ?? 0;
        const eligible = ordered.filter((spec) => tierRank(spec) >= floor);
        const pool = eligible.length > 0 ? eligible : ordered;
        const primary = [...pool].sort(byCostAscending)[0] as ModelSpec;

        const reason =
          floor > 0
            ? `balanced mode: cheapest model at or above the '${route.taskType}' tier floor`
            : "balanced mode: cheapest model meeting all constraints";
        return { primary, reason };
      }
    }
  }

  /**
   * Applies `maxLatencyMs` as a preference rather than a hard filter: prefer models expected to fit
   * the budget, but never eliminate every option over an estimate.
   */
  private preferFastEnough(candidates: ModelSpec[], context: RoutingContext): ModelSpec[] {
    const limit = context.request.route.maxLatencyMs;
    if (limit === undefined) return candidates;

    const fastEnough = candidates.filter((spec) => spec.typicalLatencyMs <= limit);
    return fastEnough.length > 0 ? fastEnough : candidates;
  }
}
