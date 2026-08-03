import { describe, expect, it } from "vitest";
import { LinUcbBandit } from "./linucb.js";
import { ThompsonBandit } from "./thompson.js";

/**
 * Prior seeding.
 *
 * The gate on this feature: seeded evidence must enter the model through exactly the same arithmetic
 * as a real observation, and must never masquerade as one. Get the first wrong and priors silently
 * distort learning; get the second wrong and a benchmark score opens a gate that was supposed to
 * require real traffic.
 */

function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1_664_525 + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

const D = 5;
const bandit = () => new LinUcbBandit({ dimension: D, explorationFloor: 0, random: seeded(1) });

const CONTEXT_A = [1, 0.4, 1, 0, 0.25];
const CONTEXT_B = [1, 0.9, 0, 1, 0.75];

describe("LinUcbBandit.seed", () => {
  it("is equivalent to N repeated observations", () => {
    // The core property. `A` and `b` are additive sums, so one weighted rank-1 update must land in
    // the same place as N unit updates. Floating point differs by accumulation order, hence the
    // tolerance rather than exact equality.
    const seededBandit = bandit();
    seededBandit.seed("arm", CONTEXT_A, 0.8, 10);

    const repeated = bandit();
    for (let i = 0; i < 10; i++) repeated.update("arm", CONTEXT_A, 0.8);

    const a = seededBandit.snapshot().arms.arm;
    const b = repeated.snapshot().arms.arm;

    for (const [index, value] of (a?.b ?? []).entries()) {
      expect(value).toBeCloseTo(b?.b?.[index] as number, 10);
    }
    for (const [index, value] of (a?.aInv ?? []).entries()) {
      expect(value).toBeCloseTo(b?.aInv?.[index] as number, 10);
    }
  });

  it("holds for fractional weights", () => {
    // A prior worth half an observation is meaningful when confidence in the source is low.
    const half = bandit();
    half.seed("arm", CONTEXT_A, 0.6, 0.5);

    const scores = half.select(["arm", "untouched"], CONTEXT_A).scores;
    expect(scores.arm).not.toBe(scores.untouched);
  });

  it("produces the same predictions as the equivalent real observations", () => {
    const seededBandit = bandit();
    seededBandit.seed("good", CONTEXT_A, 0.9, 20);
    seededBandit.seed("bad", CONTEXT_A, 0.1, 20);

    const repeated = bandit();
    for (let i = 0; i < 20; i++) {
      repeated.update("good", CONTEXT_A, 0.9);
      repeated.update("bad", CONTEXT_A, 0.1);
    }

    const seededScores = seededBandit.select(["good", "bad"], CONTEXT_A).scores;
    const repeatedScores = repeated.select(["good", "bad"], CONTEXT_A).scores;

    expect(seededScores.good).toBeCloseTo(repeatedScores.good as number, 8);
    expect(seededScores.bad).toBeCloseTo(repeatedScores.bad as number, 8);
  });

  describe("seeded evidence is not real evidence", () => {
    it("does not increment pulls", () => {
      // `pulls` drives the exploration floor and every report. Counting a benchmark score there
      // would make an untried arm look explored.
      const b = bandit();
      b.seed("arm", CONTEXT_A, 0.9, 25);

      expect(b.pulls("arm")).toBe(0);
      expect(b.syntheticPulls("arm")).toBe(25);
    });

    it("leaves averageReward reporting real outcomes only", () => {
      const b = bandit();
      b.seed("arm", CONTEXT_A, 0.9, 50);
      expect(b.averageReward("arm")).toBe(0);

      b.update("arm", CONTEXT_A, 0.2);
      // One real observation of 0.2 — the seeded 0.9 must not pull this toward 0.9.
      expect(b.averageReward("arm")).toBeCloseTo(0.2, 10);
    });

    it("still lets the exploration floor find a seeded-but-untried arm", () => {
      const b = new LinUcbBandit({ dimension: D, explorationFloor: 1, random: seeded(3) });
      for (let i = 0; i < 30; i++) b.update("tried", CONTEXT_A, 0.7);
      b.seed("seeded-only", CONTEXT_A, 0.9, 30);

      expect(b.select(["tried", "seeded-only"], CONTEXT_A).armId).toBe("seeded-only");
    });
  });

  describe("priors decay under real evidence", () => {
    it("is outweighed once enough real observations disagree", () => {
      // The property that makes priors safe: they are evidence, so more evidence overrules them.
      // No decay schedule, no half-life — just arithmetic.
      const b = bandit();
      b.seed("overrated", CONTEXT_A, 0.95, 10);
      b.seed("underrated", CONTEXT_A, 0.2, 10);

      expect(b.select(["overrated", "underrated"], CONTEXT_A).armId).toBe("overrated");

      for (let i = 0; i < 200; i++) {
        b.update("overrated", CONTEXT_A, 0.1);
        b.update("underrated", CONTEXT_A, 0.9);
      }

      expect(b.select(["overrated", "underrated"], CONTEXT_A).armId).toBe("underrated");
    });

    it("pulls the mean estimate closer to the seeded value as weight grows", () => {
      // Measured with alpha = 0, so the score *is* θᵀx.
      //
      // This is deliberately not asserted on the UCB score, because the score is not monotonic in
      // prior weight: a heavier prior also makes the estimate more confident, shrinking the
      // uncertainty bonus. An optimistic heavy prior can therefore score BELOW a light one. The mean
      // is the quantity that behaves the way intuition expects.
      const exploitOnly = () =>
        new LinUcbBandit({ dimension: D, alpha: 0, explorationFloor: 0, random: seeded(1) });

      const light = exploitOnly();
      light.seed("arm", CONTEXT_A, 0.9, 1);

      const heavy = exploitOnly();
      heavy.seed("arm", CONTEXT_A, 0.9, 50);

      const lightMean = light.select(["arm"], CONTEXT_A).scores.arm as number;
      const heavyMean = heavy.select(["arm"], CONTEXT_A).scores.arm as number;

      expect(heavyMean).toBeGreaterThan(lightMean);
      // A heavy prior approaches the seeded value; a weight-1 prior stays well short of it.
      expect(heavyMean).toBeGreaterThan(0.85);
      expect(lightMean).toBeLessThan(0.7);
    });

    it("makes a heavier prior harder to overturn", () => {
      // The practical consequence of prior weight: how much real evidence it takes to overrule.
      const contest = (weight: number) => {
        const b = bandit();
        b.seed("seeded", CONTEXT_A, 0.95, weight);
        for (let i = 0; i < 15; i++) {
          b.update("seeded", CONTEXT_A, 0.1);
          b.update("rival", CONTEXT_A, 0.6);
        }
        return b.select(["seeded", "rival"], CONTEXT_A).armId;
      };

      // 15 real observations overturn a light prior but not yet a very heavy one.
      expect(contest(2)).toBe("rival");
      expect(contest(200)).toBe("seeded");
    });
  });

  describe("context sensitivity", () => {
    it("does not transfer a prior into an orthogonal context", () => {
      // A prior saying "strong at code" must not silently become "strong at everything".
      //
      // Orthogonal vectors are used deliberately. The real feature vector carries a bias term that
      // every context shares, so partial transfer between contexts is expected and arguably correct
      // — a generally strong model is generally strong. What must NOT happen is transfer along a
      // dimension the prior said nothing about, and orthogonality is how that gets isolated.
      const seededDirection = [0, 1, 0, 0, 0];
      const orthogonal = [0, 0, 0, 1, 0];

      const exploitOnly = new LinUcbBandit({
        dimension: D,
        alpha: 0,
        explorationFloor: 0,
        random: seeded(1),
      });
      exploitOnly.seed("specialist", seededDirection, 0.95, 30);

      expect(exploitOnly.select(["specialist"], seededDirection).scores.specialist).toBeGreaterThan(
        0.8,
      );
      // Nothing was claimed about this direction, so nothing is predicted for it.
      expect(exploitOnly.select(["specialist"], orthogonal).scores.specialist).toBeCloseTo(0, 10);
    });
  });

  it("ignores non-positive weights", () => {
    const b = bandit();
    b.seed("arm", CONTEXT_A, 0.9, 0);
    b.seed("arm", CONTEXT_A, 0.9, -5);
    expect(b.syntheticPulls("arm")).toBe(0);
  });

  it("survives a persist/restore round trip", () => {
    const original = bandit();
    original.seed("arm", CONTEXT_A, 0.85, 12);

    const restored = bandit();
    expect(restored.restore(JSON.parse(JSON.stringify(original.snapshot())))).toBe(true);
    expect(restored.syntheticPulls("arm")).toBe(12);
    expect(restored.select(["arm"], CONTEXT_A).scores.arm).toBeCloseTo(
      original.select(["arm"], CONTEXT_A).scores.arm as number,
      10,
    );
  });

  it("treats state written before priors existed as having none", () => {
    const legacy = {
      kind: "linucb",
      version: 1,
      dimension: D,
      arms: {
        arm: {
          aInv: Array.from({ length: D * D }, (_, i) => (i % (D + 1) === 0 ? 1 : 0)),
          b: Array.from({ length: D }, () => 0),
          pulls: 4,
          totalReward: 2,
        },
      },
    };

    const b = bandit();
    expect(b.restore(legacy)).toBe(true);
    expect(b.syntheticPulls("arm")).toBe(0);
    expect(b.pulls("arm")).toBe(4);
  });
});

describe("ThompsonBandit.seed", () => {
  it("is equivalent to N repeated observations", () => {
    const seededBandit = new ThompsonBandit({ random: seeded(9) });
    seededBandit.seed("arm", [], 0.75, 8);

    const repeated = new ThompsonBandit({ random: seeded(9) });
    for (let i = 0; i < 8; i++) repeated.update("arm", [], 0.75);

    const a = seededBandit.snapshot().arms.arm;
    const b = repeated.snapshot().arms.arm;
    expect(a?.alpha).toBeCloseTo(b?.alpha as number, 10);
    expect(a?.beta).toBeCloseTo(b?.beta as number, 10);
  });

  it("does not count seeded weight as pulls", () => {
    const b = new ThompsonBandit({ random: seeded(2) });
    b.seed("arm", [], 0.9, 15);
    expect(b.pulls("arm")).toBe(0);
    expect(b.syntheticPulls("arm")).toBe(15);
  });
});
