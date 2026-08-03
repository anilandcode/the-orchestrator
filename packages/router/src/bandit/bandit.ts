/** Serialized arm state, versioned so a feature-layout change cannot be misread as old learning. */
export interface BanditState {
  kind: string;
  /** Bumped whenever the on-disk shape changes. */
  version: number;
  /** Feature-vector length this state was trained against. */
  dimension: number;
  arms: Record<string, SerializedArm>;
}

export interface SerializedArm {
  /** Inverse of the ridge design matrix, row-major. Empty for non-contextual bandits. */
  aInv?: number[];
  b?: number[];
  /** Beta-Bernoulli parameters, for Thompson sampling. */
  alpha?: number;
  beta?: number;
  pulls: number;
  /** Weight of seeded prior evidence, kept separate from real observations. */
  syntheticPulls?: number;
  totalReward: number;
}

export interface BanditChoice {
  armId: string;
  /** Per-arm scores, kept so a decision can be explained rather than merely asserted. */
  scores: Record<string, number>;
  /** True when the choice came from the exploration floor rather than the model. */
  explored: boolean;
}

export interface Bandit {
  readonly kind: string;
  /** Picks among the arms the caller says are currently valid. */
  select(armIds: string[], features: number[]): BanditChoice;
  update(armId: string, features: number[], reward: number): void;
  /**
   * Correct a reward already applied by `update`, without counting a second observation.
   *
   * Quality signals land at different times — an inline validator, then a sampled judge, then a
   * human. Without revision the bandit would be permanently trained on whichever signal happened to
   * arrive first.
   */
  revise(armId: string, features: number[], oldReward: number, newReward: number): void;
  /**
   * Apply external evidence worth `weight` pseudo-observations.
   *
   * Must not increment `pulls`: seeded evidence is a starting guess, not something the arm has
   * actually been tried on, and conflating the two would suppress exploration and corrupt reporting.
   */
  seed(armId: string, features: number[], reward: number, weight: number): void;
  /** Total seeded weight applied to an arm. */
  syntheticPulls(armId: string): number;
  pulls(armId: string): number;
  averageReward(armId: string): number;
  snapshot(): BanditState;
  restore(state: BanditState): boolean;
}
