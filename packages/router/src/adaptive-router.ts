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
import type { ModelPrior, Router, RoutingContext, TenantPolicies } from "./router.js";
import type { StaticRouter } from "./static-router.js";

/**
 * `static`   — rules only; the bandit is not consulted at all.
 * `shadow`   — the static pick executes, the bandit's pick is recorded beside it. **Default.**
 * `adaptive` — the bandit's pick executes.
 */
export type RouterMode = "static" | "shadow" | "adaptive";

export const DEFAULT_ROUTER_MODE: RouterMode = "shadow";
/**
 * Upper bound on decisions tracked in memory for outcome and revision matching. Late feedback beyond
 * this horizon is dropped rather than retained forever — an unbounded map is a slow leak, and a
 * signal this stale is worth little anyway.
 */
const MAX_TRACKED_DECISIONS = 10_000;
export const DEFAULT_COLD_START_PULLS = 25;
/**
 * Mean quality-signal confidence a task type must show before the bandit is allowed to steer it.
 *
 * 0.5 sits deliberately between the heuristic floor (~0.2) and a real scorer (judge ~0.6,
 * deterministic ~0.9). A task type graded only by "the call did not error" therefore stays on the
 * static rules, because on such traffic the reward's quality term is a constant and the bandit is
 * effectively guessing.
 *
 * Simulation is the reason this exists: the bandit beat the rules on validator-covered tasks
 * (56.4% vs 45.3% optimal picks) and lost on the rest by more than the covered win was worth.
 */
export const DEFAULT_MIN_QUALITY_CONFIDENCE = 0.5;
/** Observations of a task type needed before its mean confidence is trusted enough to gate on. */
export const DEFAULT_MIN_CONFIDENCE_SAMPLES = 10;
export const DEFAULT_STATE_KEY = "router:v1";

export interface AdaptiveRouterConfig {
  bandit: Bandit;
  /** Consulted in shadow mode, during cold start, and whenever the bandit cannot decide. */
  baseline: StaticRouter;
  mode?: RouterMode;
  /** Observations of an arm *for a task type* before the bandit is trusted over the baseline. */
  coldStartPulls?: number;
  /**
   * Minimum mean quality-signal confidence for a task type before the bandit may steer it.
   * Set to 0 to disable gating entirely and let the bandit route everything.
   */
  minQualityConfidence?: number;
  minConfidenceSamples?: number;
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
  private readonly minQualityConfidence: number;
  private readonly minConfidenceSamples: number;
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
  /**
   * Running mean of quality-signal confidence per task type.
   *
   * Deliberately learned rather than configured. A hard-coded list of "task types we can grade"
   * would couple the router to whichever validators happen to exist and go stale the moment one is
   * added or removed. This way, adding a validator makes the router start trusting the bandit on
   * that task on its own.
   */
  private taskConfidence = new Map<TaskType, { sum: number; count: number }>();
  private observationsSincePersist = 0;

  /** Decision context retained until the outcome arrives, so reward reaches the right arm. */
  private readonly pending = new Map<
    string,
    { modelId: string; features: number[]; taskType: TaskType }
  >();

  /**
   * Rewards already applied, keyed by decision. Retained so a later, better quality signal can
   * correct the exact value rather than pile a second observation on top of it.
   */
  private readonly applied = new Map<
    string,
    { modelId: string; features: number[]; taskType?: TaskType; reward: number }
  >();

  constructor(config: AdaptiveRouterConfig) {
    this.bandit = config.bandit;
    this.baseline = config.baseline;
    this.mode = config.mode ?? DEFAULT_ROUTER_MODE;
    this.coldStartPulls = config.coldStartPulls ?? DEFAULT_COLD_START_PULLS;
    this.minQualityConfidence = config.minQualityConfidence ?? DEFAULT_MIN_QUALITY_CONFIDENCE;
    this.minConfidenceSamples = config.minConfidenceSamples ?? DEFAULT_MIN_CONFIDENCE_SAMPLES;
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

    // Quality-observability gate: only steer task types where the reward's quality term actually
    // carries information.
    //
    // On traffic graded solely by "the call did not error", quality is a constant, so the bandit is
    // ranking on cost and latency alone — which the static rules already encode, and it pays a real
    // exploration cost to rediscover. Measurement bore this out: the bandit beat the rules on
    // validator-covered tasks and lost by more than that on the rest.
    const confidence = this.qualityConfidenceFor(taskType);
    if (confidence !== undefined && confidence < this.minQualityConfidence) {
      this.remember(baselineDecision.decisionId, baselineDecision.modelId, features, taskType);
      return {
        ...baselineDecision,
        shadowModelId: choice.armId,
        reason: `${baselineDecision.reason} (bandit gated: mean quality confidence ${confidence.toFixed(2)} on '${taskType}' is below ${this.minQualityConfidence})`,
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

      if (outcome.qualityConfidence !== undefined) {
        const existing = this.taskConfidence.get(taskType) ?? { sum: 0, count: 0 };
        this.taskConfidence.set(taskType, {
          sum: existing.sum + outcome.qualityConfidence,
          count: existing.count + 1,
        });
      }
    }

    // Retained so a late-arriving quality signal can correct exactly this reward. Without the
    // applied value there is no delta to apply, and re-teaching would double-count.
    if (outcome.decisionId) {
      this.applied.set(outcome.decisionId, {
        modelId,
        features,
        reward: outcome.reward,
        ...(taskType ? { taskType } : {}),
      });
      this.evictOldest(this.applied);
    }

    this.observationsSincePersist += 1;
    if (this.observationsSincePersist >= this.persistEvery) this.persist();
  }

  /**
   * Replace the reward previously applied for a decision.
   *
   * Called when a better quality signal lands — a deferred judge, or a human via `/v1/feedback`.
   * Returns false when the decision is no longer tracked, which is expected rather than exceptional:
   * feedback can arrive after a restart or after the bounded map has evicted the entry.
   */
  reviseOutcome(decisionId: string, newReward: number): boolean {
    const previous = this.applied.get(decisionId);
    if (!previous) return false;

    this.bandit.revise(previous.modelId, previous.features, previous.reward, newReward);
    this.applied.set(decisionId, { ...previous, reward: newReward });

    this.observationsSincePersist += 1;
    if (this.observationsSincePersist >= this.persistEvery) this.persist();
    return true;
  }

  /** The reward currently credited to a decision, if it is still tracked. */
  appliedRewardFor(decisionId: string): number | undefined {
    return this.applied.get(decisionId)?.reward;
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

  /**
   * Mean quality-signal confidence seen for a task type.
   *
   * `undefined` means "not enough evidence to judge", which is treated as passing rather than
   * failing: a caller that reports no confidence at all should get the pre-gating behaviour, not
   * silently lose adaptive routing everywhere.
   */
  qualityConfidenceFor(taskType: TaskType): number | undefined {
    const stats = this.taskConfidence.get(taskType);
    if (!stats || stats.count < this.minConfidenceSamples) return undefined;
    return stats.sum / stats.count;
  }

  /** Task types the bandit is currently permitted to steer. Useful for dashboards and debugging. */
  steeredTaskTypes(): TaskType[] {
    return [...this.taskConfidence.keys()].filter((taskType) => {
      const confidence = this.qualityConfidenceFor(taskType);
      return confidence === undefined || confidence >= this.minQualityConfidence;
    });
  }

  /**
   * Apply external priors — benchmark scores, warm-start evals — to arms that have no real data.
   *
   * **Tilt only.** This deliberately routes to `bandit.seed()` rather than through `observe()`.
   * `observe()` increments `taskPulls`, which is exactly what the cold-start gate reads; seeding
   * through it would let a benchmark score open a gate that was designed to require real traffic.
   * The gates are untouched here by construction, not by care.
   *
   * An arm with real observations is skipped outright. Real evidence already outweighs a prior
   * arithmetically, so re-seeding would buy nothing — and un-applying a weighted update later is
   * numerically fragile, so the simplest correct rule is to never need to.
   */
  applyPriors(priors: ModelPrior[]): { seeded: number; skipped: number } {
    let seeded = 0;
    let skipped = 0;

    for (const prior of priors) {
      if (this.bandit.pulls(prior.modelId) > 0) {
        skipped += 1;
        continue;
      }

      this.bandit.seed(prior.modelId, prior.features, prior.reward, prior.weight);
      seeded += 1;
    }

    if (seeded > 0) this.persist();
    return { seeded, skipped };
  }

  /** Seeded weight applied to an arm — reported separately from real observations. */
  syntheticPullsFor(modelId: string): number {
    return this.bandit.syntheticPulls(modelId);
  }

  /** Flush learning to the store. Call on shutdown so the last window is not lost. */
  persist(): void {
    if (!this.stateStore) return;
    const state: RouterState = {
      bandit: this.bandit.snapshot(),
      contextPulls: Object.fromEntries(this.contextPulls),
      taskPulls: Object.fromEntries(this.taskPulls),
      taskConfidence: Object.fromEntries(this.taskConfidence),
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
      this.taskConfidence = new Map(
        Object.entries(stored.taskConfidence ?? {}) as [TaskType, { sum: number; count: number }][],
      );
    }
  }

  private remember(
    decisionId: string,
    modelId: string,
    features: number[],
    taskType: TaskType,
  ): void {
    this.pending.set(decisionId, { modelId, features, taskType });
    this.evictOldest(this.pending);
  }

  /**
   * Bound a tracking map so an outcome (or a revision) that never arrives cannot leak memory.
   * Map iteration is insertion-ordered, so the first key is the oldest.
   */
  private evictOldest(map: Map<string, unknown>): void {
    if (map.size <= MAX_TRACKED_DECISIONS) return;
    const oldest = map.keys().next().value;
    if (oldest !== undefined) map.delete(oldest);
  }
}

function contextKey(modelId: string, taskType: TaskType): string {
  return `${modelId}|${taskType}`;
}
