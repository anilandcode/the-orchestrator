import {
  type Clock,
  type IdGenerator,
  type ModelSpec,
  type RoutingDecision,
  type RoutingOutcome,
  type TaskType,
  systemClock,
  systemIds,
} from "@orchestrator/shared";
import type { Bandit } from "./bandit/bandit.js";
import type { RouterState, StateStore } from "./bandit/state-store.js";
import { selectCandidates } from "./candidates.js";
import { DEFAULT_CHAIN_LENGTH, buildFallbackChain } from "./fallback.js";
import { extractFeatures } from "./features.js";
import type { Router, RoutingContext, TenantPolicies } from "./router.js";
import type { StaticRouter } from "./static-router.js";

/**
 * `static`   — rules only; the bandit is not consulted at all.
 * `shadow`   — the static pick executes, the bandit's pick is recorded beside it. **Default.**
 * `adaptive` — the bandit's pick executes.
 */
export type RouterMode = "static" | "shadow" | "adaptive";

export const DEFAULT_ROUTER_MODE: RouterMode = "shadow";
export const DEFAULT_COLD_START_PULLS = 25;
export const DEFAULT_STATE_KEY = "router:v1";

export interface AdaptiveRouterConfig {
  bandit: Bandit;
  /** Consulted in shadow mode, during cold start, and whenever the bandit cannot decide. */
  baseline: StaticRouter;
  mode?: RouterMode;
  /** Observations of an arm *for a task type* before the bandit is trusted over the baseline. */
  coldStartPulls?: number;
  policies?: TenantPolicies;
  chainLength?: number;
  stateStore?: StateStore;
  stateKey?: string;
  /** Checkpoint cadence. Writing on every observation would put a disk write on the hot path. */
  persistEvery?: number;
  ids?: IdGenerator;
  clock?: Clock;
}

/**
 * The contextual bandit router, wrapped in the safeguards that make it safe to deploy.
 *
 * The bandit itself is only part of the story. Three things around it matter as much:
 *
 *   1. **Cold start.** A bandit with no data is worse than a rule, so until an arm has been observed
 *      enough times *for this task type*, the deterministic baseline decides.
 *   2. **New-model onboarding.** A model added to the registry needs no retraining: LinUCB's
 *      uncertainty term makes an unseen arm attractive on its own, and the cold-start gate keeps it
 *      from taking real traffic before it has earned it.
 *   3. **Mode gate.** Shipping defaults to `shadow`, which logs what the bandit *would* have done
 *      without letting it steer anything. Promoting to `adaptive` is a decision backed by replaying
 *      those shadow decisions against real outcomes.
 */
export class AdaptiveRouter implements Router {
  private readonly bandit: Bandit;
  private readonly baseline: StaticRouter;
  private readonly mode: RouterMode;
  private readonly coldStartPulls: number;
  private readonly policies: TenantPolicies;
  private readonly chainLength: number;
  private readonly stateStore: StateStore | undefined;
  private readonly stateKey: string;
  private readonly persistEvery: number;
  private readonly ids: IdGenerator;
  private readonly clock: Clock;

  /** `${armId}|${taskType}` -> observations. Kept for diagnostics and replay reporting. */
  private contextPulls = new Map<string, number>();
  /** Observations per task type. This is what the cold-start gate actually reads. */
  private taskPulls = new Map<TaskType, number>();
  private observationsSincePersist = 0;

  /** Decision context retained until the outcome arrives, so reward reaches the right arm. */
  private readonly pending = new Map<
    string,
    { modelId: string; features: number[]; taskType: TaskType }
  >();

  constructor(config: AdaptiveRouterConfig) {
    this.bandit = config.bandit;
    this.baseline = config.baseline;
    this.mode = config.mode ?? DEFAULT_ROUTER_MODE;
    this.coldStartPulls = config.coldStartPulls ?? DEFAULT_COLD_START_PULLS;
    this.policies = config.policies ?? {};
    this.chainLength = config.chainLength ?? DEFAULT_CHAIN_LENGTH;
    this.stateStore = config.stateStore;
    this.stateKey = config.stateKey ?? DEFAULT_STATE_KEY;
    this.persistEvery = config.persistEvery ?? 20;
    this.ids = config.ids ?? systemIds;
    this.clock = config.clock ?? systemClock;

    this.restore();
  }

  select(context: RoutingContext): RoutingDecision {
    const baselineDecision = this.baseline.select(context);
    if (this.mode === "static") return baselineDecision;

    const policy = this.policies[context.request.tenantId];
    const candidates = selectCandidates(context, policy);
    const taskType = context.request.route.taskType;

    // A pinned request is not a routing decision. Never let the bandit override an explicit choice.
    if (baselineDecision.strategy === "pinned") return baselineDecision;

    const features = extractFeatures(context);
    const choice = this.bandit.select(
      candidates.map((spec) => spec.modelId),
      features,
    );

    if (this.mode === "shadow") {
      // Record the counterfactual, execute the baseline. This column is what earns the right to
      // promote the bandit later.
      this.remember(baselineDecision.decisionId, baselineDecision.modelId, features, taskType);
      return { ...baselineDecision, shadowModelId: choice.armId };
    }

    // The gate is on total observations for this TASK TYPE, not on the specific arm the bandit
    // wants.
    //
    // Gating per-arm deadlocks: during cold start the baseline's pick is what executes, so the
    // bandit's preferred arm never accumulates observations, so the gate never opens for any model
    // the static rules do not already favour. Per-arm inexperience is LinUCB's own uncertainty term's
    // job — this gate answers the separate question of whether there is enough signal in this task
    // type to trust the bandit at all.
    const observed = this.taskPullsFor(taskType);
    if (observed < this.coldStartPulls) {
      this.remember(baselineDecision.decisionId, baselineDecision.modelId, features, taskType);
      return {
        ...baselineDecision,
        shadowModelId: choice.armId,
        reason: `${baselineDecision.reason} (bandit deferred: ${observed}/${this.coldStartPulls} observations on '${taskType}')`,
      };
    }

    const primary = candidates.find((spec) => spec.modelId === choice.armId) as ModelSpec;
    const decisionId = this.ids.generate("dec");
    this.remember(decisionId, primary.modelId, features, taskType);

    return {
      decisionId,
      modelId: primary.modelId,
      fallbacks: buildFallbackChain(primary, candidates, this.chainLength),
      strategy: "adaptive",
      reason: choice.explored
        ? `exploration floor: sampling least-observed arm ${primary.modelId}`
        : `bandit: score ${(choice.scores[primary.modelId] ?? 0).toFixed(4)} over ${candidates.length} candidates`,
      shadowModelId: baselineDecision.modelId,
      features,
      taskType,
      routeMode: context.request.route.mode,
      createdAt: this.clock.now(),
    };
  }

  observe(outcome: RoutingOutcome): void {
    // Prefer the remembered decision: it carries the exact features the bandit scored, which the
    // caller may not have kept.
    const pending = outcome.decisionId ? this.pending.get(outcome.decisionId) : undefined;
    if (outcome.decisionId) this.pending.delete(outcome.decisionId);

    const modelId = pending?.modelId ?? outcome.modelId;
    const features = pending?.features ?? outcome.features;
    const taskType = pending?.taskType ?? outcome.taskType;

    this.bandit.update(modelId, features, outcome.reward);

    if (taskType) {
      const key = contextKey(modelId, taskType);
      this.contextPulls.set(key, (this.contextPulls.get(key) ?? 0) + 1);
      this.taskPulls.set(taskType, (this.taskPulls.get(taskType) ?? 0) + 1);
    }

    this.observationsSincePersist += 1;
    if (this.observationsSincePersist >= this.persistEvery) this.persist();
  }

  /** Settle by decision id alone, so callers do not have to retain the feature vector. */
  observeDecision(decisionId: string, reward: number): boolean {
    const pending = this.pending.get(decisionId);
    if (!pending) return false;

    this.observe({
      decisionId,
      modelId: pending.modelId,
      features: pending.features,
      taskType: pending.taskType,
      reward,
    });
    return true;
  }

  contextPullsFor(modelId: string, taskType: TaskType): number {
    return this.contextPulls.get(contextKey(modelId, taskType)) ?? 0;
  }

  /** Total observations for a task type — what the cold-start gate compares against. */
  taskPullsFor(taskType: TaskType): number {
    return this.taskPulls.get(taskType) ?? 0;
  }

  /** Flush learning to the store. Call on shutdown so the last window is not lost. */
  persist(): void {
    if (!this.stateStore) return;
    const state: RouterState = {
      bandit: this.bandit.snapshot(),
      contextPulls: Object.fromEntries(this.contextPulls),
      taskPulls: Object.fromEntries(this.taskPulls),
    };
    this.stateStore.save(this.stateKey, state);
    this.observationsSincePersist = 0;
  }

  private restore(): void {
    const stored = this.stateStore?.load<RouterState>(this.stateKey);
    if (!stored) return;

    // A rejected restore leaves the bandit cold rather than corrupt — see LinUcbBandit.restore.
    if (this.bandit.restore(stored.bandit)) {
      this.contextPulls = new Map(Object.entries(stored.contextPulls ?? {}));
      this.taskPulls = new Map(Object.entries(stored.taskPulls ?? {}) as [TaskType, number][]);
    }
  }

  private remember(
    decisionId: string,
    modelId: string,
    features: number[],
    taskType: TaskType,
  ): void {
    this.pending.set(decisionId, { modelId, features, taskType });
    // Bound the map: an outcome that never arrives must not leak memory forever.
    if (this.pending.size > 10_000) {
      const oldest = this.pending.keys().next().value;
      if (oldest) this.pending.delete(oldest);
    }
  }
}

function contextKey(modelId: string, taskType: TaskType): string {
  return `${modelId}|${taskType}`;
}
