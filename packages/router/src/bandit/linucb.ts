import type { Bandit, BanditChoice, BanditState, SerializedArm } from "./bandit.js";

/**
 * Disjoint LinUCB.
 *
 * Each model is an arm with its own ridge regression over the request's feature vector. The score is
 * a predicted reward plus an uncertainty bonus:
 *
 *     score(arm) = θᵀx + α · sqrt(xᵀ A⁻¹ x)
 *
 * The second term is what makes this self-optimizing rather than merely predictive: an arm nobody has
 * tried yet carries maximum uncertainty, so it gets explored on its own without any hand-written
 * "try the new model sometimes" rule. That is also why a newly registered model needs no retraining —
 * it simply enters with an uninformative prior and high uncertainty.
 *
 * A⁻¹ is maintained incrementally by the Sherman-Morrison identity, so no matrix inversion runs on
 * the request path.
 */

interface Arm {
  /** A⁻¹, row-major, d×d. Starts at I/λ. */
  aInv: Float64Array;
  /** Response vector, d. */
  b: Float64Array;
  pulls: number;
  /** Weight of seeded prior evidence. Kept apart from `pulls` so real data stays countable. */
  syntheticPulls: number;
  totalReward: number;
}

export const LINUCB_STATE_VERSION = 1;

export interface LinUcbConfig {
  dimension: number;
  /** Exploration weight. Higher explores more; 0 is pure exploitation. */
  alpha?: number;
  /** Ridge regularization. */
  lambda?: number;
  /**
   * Probability of ignoring the scores and pulling the least-explored arm.
   *
   * **Defaults to 0, deliberately.** LinUCB's uncertainty term already explores, and does it
   * proportionally to what it does not know. A flat random floor explores unconditionally, at a cost
   * equal to its rate — and simulation showed that a 3% floor against a ~2% headroom over the static
   * baseline spends more than optimal routing could ever return. Size any floor you add against the
   * headroom you are actually chasing.
   */
  explorationFloor?: number;
  random?: () => number;
}

export class LinUcbBandit implements Bandit {
  readonly kind = "linucb";
  readonly dimension: number;

  private readonly alpha: number;
  private readonly lambda: number;
  private readonly explorationFloor: number;
  private readonly random: () => number;
  private readonly arms = new Map<string, Arm>();

  constructor(config: LinUcbConfig) {
    this.dimension = config.dimension;
    this.alpha = config.alpha ?? 0.6;
    this.lambda = config.lambda ?? 1;
    this.explorationFloor = config.explorationFloor ?? 0;
    this.random = config.random ?? Math.random;
  }

  select(armIds: string[], features: number[]): BanditChoice {
    if (armIds.length === 0) throw new Error("LinUcbBandit.select called with no arms");

    const x = this.toVector(features);
    const scores: Record<string, number> = {};

    let bestArm = armIds[0] as string;
    let bestScore = Number.NEGATIVE_INFINITY;

    for (const armId of armIds) {
      const arm = this.arm(armId);
      const aInvX = matVec(arm.aInv, x, this.dimension);

      const theta = matVec(arm.aInv, arm.b, this.dimension);
      const mean = dot(theta, x);
      // Numerically this is non-negative by construction, but floating point can nudge it below 0.
      const variance = Math.max(0, dot(x, aInvX));
      const score = mean + this.alpha * Math.sqrt(variance);

      scores[armId] = score;
      if (score > bestScore) {
        bestScore = score;
        bestArm = armId;
      }
    }

    if (this.random() < this.explorationFloor) {
      const leastExplored = armIds.reduce((least, armId) =>
        this.pulls(armId) < this.pulls(least) ? armId : least,
      );
      return { armId: leastExplored, scores, explored: true };
    }

    return { armId: bestArm, scores, explored: false };
  }

  update(armId: string, features: number[], reward: number): void {
    const arm = this.arm(armId);
    this.applyObservation(arm, this.toVector(features), reward, 1);

    arm.pulls += 1;
    arm.totalReward += reward;
  }

  /**
   * Seed a prior worth `weight` pseudo-observations.
   *
   * In LinUCB a prior *is* observations: `A` and `b` are additive sums, so external evidence — a
   * benchmark score, a warm-start eval — enters through exactly the same arithmetic as a real
   * outcome, just weighted. That also gives decay for free. After 100 real observations a prior
   * worth 10 is 9% of the evidence, with no half-life to tune and no schedule to get wrong.
   *
   * `pulls` is deliberately NOT incremented. It feeds LinUCB's exploration floor and every telemetry
   * report, so counting seeded evidence there would make an untried arm look explored and suppress
   * the exploration it most needs. Synthetic weight is tracked separately.
   */
  seed(armId: string, features: number[], reward: number, weight: number): void {
    if (weight <= 0) return;

    const arm = this.arm(armId);
    this.applyObservation(arm, this.toVector(features), reward, weight);
    arm.syntheticPulls += weight;
  }

  /** Real observations only — what the exploration floor and reporting should see. */
  syntheticPulls(armId: string): number {
    return this.arms.get(armId)?.syntheticPulls ?? 0;
  }

  /**
   * One weighted rank-1 update, shared by `update` and `seed`.
   *
   * Sherman-Morrison for `A ← A + w·x·xᵀ`:
   *   A⁻¹ ← A⁻¹ − w·(A⁻¹x)(A⁻¹x)ᵀ / (1 + w·xᵀA⁻¹x)
   *
   * Routing both callers through here is what makes `seed(x, r, N)` and N calls to `update(x, r)`
   * equivalent by construction rather than by coincidence.
   */
  private applyObservation(arm: Arm, x: Float64Array, reward: number, weight: number): void {
    const d = this.dimension;
    const aInvX = matVec(arm.aInv, x, d);
    const denominator = 1 + weight * dot(x, aInvX);

    for (let i = 0; i < d; i++) {
      const rowOffset = i * d;
      const scaled = (weight * (aInvX[i] ?? 0)) / denominator;
      for (let j = 0; j < d; j++) {
        arm.aInv[rowOffset + j] = (arm.aInv[rowOffset + j] ?? 0) - scaled * (aInvX[j] ?? 0);
      }
    }

    for (let i = 0; i < d; i++) {
      arm.b[i] = (arm.b[i] ?? 0) + weight * reward * (x[i] ?? 0);
    }
  }

  /**
   * Correct a reward that was already applied.
   *
   * Quality signals arrive at different times: a validator refines the provisional reward moments
   * later, a sampled judge minutes later, a human hours later. By then the arm has already learned
   * from the provisional value.
   *
   * The correction is exact rather than approximate, which is a genuine property of the algorithm:
   * `A` is built from `x xᵀ` alone and never depends on the reward, and `b` accumulates `reward · x`
   * linearly. Adding `(new − old) · x` therefore lands the arm in the state it would have reached had
   * the final reward been known up front — no double counting, and no need to stall learning behind
   * a timeout waiting for the signal to settle.
   *
   * `pulls` is deliberately untouched: this is one observation being corrected, not a second one.
   */
  revise(armId: string, features: number[], oldReward: number, newReward: number): void {
    const arm = this.arms.get(armId);
    // Nothing to correct on an arm that was never pulled — silently ignoring is right here, since a
    // late-arriving signal for an evicted or restored-away decision is expected, not exceptional.
    if (!arm) return;

    const delta = newReward - oldReward;
    if (delta === 0) return;

    const x = this.toVector(features);
    for (let i = 0; i < this.dimension; i++) {
      arm.b[i] = (arm.b[i] ?? 0) + delta * (x[i] ?? 0);
    }
    arm.totalReward += delta;
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
        aInv: Array.from(arm.aInv),
        b: Array.from(arm.b),
        pulls: arm.pulls,
        syntheticPulls: arm.syntheticPulls,
        totalReward: arm.totalReward,
      };
    }
    return { kind: this.kind, version: LINUCB_STATE_VERSION, dimension: this.dimension, arms };
  }

  /**
   * Restores persisted state. Returns false — leaving the bandit cold rather than corrupt — when the
   * stored state does not match this build's feature layout. Reinterpreting old coefficients against
   * a new feature vector would produce confident nonsense, which is worse than starting over.
   */
  restore(state: BanditState): boolean {
    if (state.kind !== this.kind) return false;
    if (state.version !== LINUCB_STATE_VERSION) return false;
    if (state.dimension !== this.dimension) return false;

    this.arms.clear();
    const d = this.dimension;

    for (const [armId, serialized] of Object.entries(state.arms)) {
      if (serialized.aInv?.length !== d * d || serialized.b?.length !== d) return false;
      this.arms.set(armId, {
        aInv: Float64Array.from(serialized.aInv),
        b: Float64Array.from(serialized.b),
        pulls: serialized.pulls,
        // Absent in state written before priors existed; treating it as 0 is correct.
        syntheticPulls: serialized.syntheticPulls ?? 0,
        totalReward: serialized.totalReward,
      });
    }

    return true;
  }

  private arm(armId: string): Arm {
    const existing = this.arms.get(armId);
    if (existing) return existing;

    const d = this.dimension;
    const aInv = new Float64Array(d * d);
    // A = λI, so A⁻¹ = I/λ.
    for (let i = 0; i < d; i++) aInv[i * d + i] = 1 / this.lambda;

    const arm: Arm = {
      aInv,
      b: new Float64Array(d),
      pulls: 0,
      syntheticPulls: 0,
      totalReward: 0,
    };
    this.arms.set(armId, arm);
    return arm;
  }

  private toVector(features: number[]): Float64Array {
    if (features.length !== this.dimension) {
      throw new Error(`Feature vector has length ${features.length}, expected ${this.dimension}`);
    }
    return Float64Array.from(features);
  }
}

function matVec(matrix: Float64Array, vector: Float64Array, d: number): Float64Array {
  const out = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    const rowOffset = i * d;
    let total = 0;
    for (let j = 0; j < d; j++) total += (matrix[rowOffset + j] ?? 0) * (vector[j] ?? 0);
    out[i] = total;
  }
  return out;
}

function dot(a: Float64Array, b: Float64Array): number {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += (a[i] ?? 0) * (b[i] ?? 0);
  return total;
}
