import type { Bandit, BanditChoice, BanditState, SerializedArm } from "./bandit.js";

/**
 * Beta-Bernoulli Thompson sampling.
 *
 * The non-contextual alternative to LinUCB, kept so the replay harness can compare them on the same
 * traffic rather than asserting that the contextual model is better. It ignores the feature vector
 * entirely — each arm is just a running success rate — which makes it a useful control: if Thompson
 * matches LinUCB on your traffic, the features are not earning their complexity.
 *
 * Rewards are continuous in [0,1], so they update the Beta posterior fractionally rather than as
 * binary wins and losses.
 */

export const THOMPSON_STATE_VERSION = 1;

/** Beta requires strictly positive parameters; corrections are clamped rather than allowed to zero. */
const MIN_BETA_PARAM = 1e-6;

interface Arm {
  alpha: number;
  beta: number;
  pulls: number;
  syntheticPulls: number;
  totalReward: number;
}

export interface ThompsonConfig {
  /** Prior successes. 1/1 is the uniform prior. */
  priorAlpha?: number;
  priorBeta?: number;
  random?: () => number;
}

export class ThompsonBandit implements Bandit {
  readonly kind = "thompson";

  private readonly priorAlpha: number;
  private readonly priorBeta: number;
  private readonly random: () => number;
  private readonly arms = new Map<string, Arm>();

  constructor(config: ThompsonConfig = {}) {
    this.priorAlpha = config.priorAlpha ?? 1;
    this.priorBeta = config.priorBeta ?? 1;
    this.random = config.random ?? Math.random;
  }

  select(armIds: string[]): BanditChoice {
    if (armIds.length === 0) throw new Error("ThompsonBandit.select called with no arms");

    const scores: Record<string, number> = {};
    let bestArm = armIds[0] as string;
    let bestScore = Number.NEGATIVE_INFINITY;

    for (const armId of armIds) {
      const arm = this.arm(armId);
      const sample = sampleBeta(arm.alpha, arm.beta, this.random);
      scores[armId] = sample;
      if (sample > bestScore) {
        bestScore = sample;
        bestArm = armId;
      }
    }

    // Exploration is intrinsic here: it comes from sampling the posterior, not from a bonus term.
    return { armId: bestArm, scores, explored: false };
  }

  update(armId: string, _features: number[], reward: number): void {
    const arm = this.arm(armId);
    const clamped = clamp01(reward);
    arm.alpha += clamped;
    arm.beta += 1 - clamped;
    arm.pulls += 1;
    arm.totalReward += clamped;
  }

  /**
   * Correct an already-applied reward by shifting mass between the Beta parameters.
   *
   * Unlike LinUCB this is only *approximately* reversible: alpha and beta are clamped away from zero
   * to keep the distribution proper, so a correction against a near-degenerate posterior can lose a
   * sliver of the delta. That is acceptable for a control arm and is one more reason LinUCB is the
   * primary — its correction is exact.
   */
  revise(armId: string, _features: number[], oldReward: number, newReward: number): void {
    const arm = this.arms.get(armId);
    if (!arm) return;

    const delta = clamp01(newReward) - clamp01(oldReward);
    if (delta === 0) return;

    arm.alpha = Math.max(MIN_BETA_PARAM, arm.alpha + delta);
    arm.beta = Math.max(MIN_BETA_PARAM, arm.beta - delta);
    arm.totalReward += delta;
  }

  /**
   * Seed a Beta prior worth `weight` pseudo-observations. Same shape as LinUCB's: the posterior is
   * a sum of evidence, so external evidence is just more of it, weighted.
   */
  seed(armId: string, _features: number[], reward: number, weight: number): void {
    if (weight <= 0) return;

    const arm = this.arm(armId);
    const clamped = clamp01(reward);
    arm.alpha += weight * clamped;
    arm.beta += weight * (1 - clamped);
    arm.syntheticPulls += weight;
  }

  syntheticPulls(armId: string): number {
    return this.arms.get(armId)?.syntheticPulls ?? 0;
  }

  pulls(armId: string): number {
    return this.arms.get(armId)?.pulls ?? 0;
  }

  averageReward(armId: string): number {
    const arm = this.arms.get(armId);
    return arm && arm.pulls > 0 ? arm.totalReward / arm.pulls : 0;
  }

  snapshot(): BanditState {
    const arms: Record<string, SerializedArm> = {};
    for (const [armId, arm] of this.arms) {
      arms[armId] = {
        alpha: arm.alpha,
        beta: arm.beta,
        pulls: arm.pulls,
        syntheticPulls: arm.syntheticPulls,
        totalReward: arm.totalReward,
      };
    }
    return { kind: this.kind, version: THOMPSON_STATE_VERSION, dimension: 0, arms };
  }

  restore(state: BanditState): boolean {
    if (state.kind !== this.kind || state.version !== THOMPSON_STATE_VERSION) return false;

    this.arms.clear();
    for (const [armId, serialized] of Object.entries(state.arms)) {
      this.arms.set(armId, {
        alpha: serialized.alpha ?? this.priorAlpha,
        beta: serialized.beta ?? this.priorBeta,
        pulls: serialized.pulls,
        syntheticPulls: serialized.syntheticPulls ?? 0,
        totalReward: serialized.totalReward,
      });
    }
    return true;
  }

  private arm(armId: string): Arm {
    const existing = this.arms.get(armId);
    if (existing) return existing;

    const arm: Arm = {
      alpha: this.priorAlpha,
      beta: this.priorBeta,
      pulls: 0,
      syntheticPulls: 0,
      totalReward: 0,
    };
    this.arms.set(armId, arm);
    return arm;
  }
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/** Beta(a,b) via the ratio of two Gamma draws. */
export function sampleBeta(a: number, b: number, random: () => number): number {
  const x = sampleGamma(a, random);
  const y = sampleGamma(b, random);
  const total = x + y;
  return total > 0 ? x / total : 0.5;
}

/** Marsaglia-Tsang gamma sampler (shape >= 1), with the standard boost for shape < 1. */
function sampleGamma(shape: number, random: () => number): number {
  if (shape < 1) {
    const boost = sampleGamma(shape + 1, random);
    return boost * random() ** (1 / shape);
  }

  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  for (;;) {
    const normal = sampleNormal(random);
    const v = (1 + c * normal) ** 3;
    if (v <= 0) continue;

    const u = random();
    if (u < 1 - 0.0331 * normal ** 4) return d * v;
    if (Math.log(u) < 0.5 * normal ** 2 + d * (1 - v + Math.log(v))) return d * v;
  }
}

/** Box-Muller. */
function sampleNormal(random: () => number): number {
  // Guard against log(0) from a generator that can return exactly 0.
  const u1 = Math.max(random(), Number.EPSILON);
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
