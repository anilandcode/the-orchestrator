import { describe, expect, it } from "vitest";
import { LinUcbBandit } from "./linucb.js";
import { ThompsonBandit } from "./thompson.js";

/** Deterministic generator so exploration decisions are reproducible. */
function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1_664_525 + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

const D = 4;
const bandit = (overrides = {}) =>
  new LinUcbBandit({ dimension: D, explorationFloor: 0, random: seeded(1), ...overrides });

const CONTEXT_A = [1, 1, 0, 0];
const CONTEXT_B = [1, 0, 1, 0];

describe("LinUcbBandit", () => {
  it("converges on the arm with the higher reward", () => {
    const b = bandit();
    for (let i = 0; i < 200; i++) {
      b.update("good", CONTEXT_A, 0.9);
      b.update("bad", CONTEXT_A, 0.1);
    }
    expect(b.select(["good", "bad"], CONTEXT_A).armId).toBe("good");
  });

  it("learns different winners for different contexts", () => {
    // This is the whole point of a *contextual* bandit: one global ranking would fail this.
    const b = bandit();
    for (let i = 0; i < 300; i++) {
      b.update("specialist_a", CONTEXT_A, 0.95);
      b.update("specialist_a", CONTEXT_B, 0.05);
      b.update("specialist_b", CONTEXT_A, 0.05);
      b.update("specialist_b", CONTEXT_B, 0.95);
    }

    expect(b.select(["specialist_a", "specialist_b"], CONTEXT_A).armId).toBe("specialist_a");
    expect(b.select(["specialist_a", "specialist_b"], CONTEXT_B).armId).toBe("specialist_b");
  });

  it("prefers an unexplored arm over a known-mediocre one", () => {
    // The uncertainty bonus is what makes a newly registered model get tried without any explicit
    // "try new models" rule.
    const b = bandit({ alpha: 1 });
    for (let i = 0; i < 50; i++) b.update("known", CONTEXT_A, 0.5);

    expect(b.select(["known", "brand_new"], CONTEXT_A).armId).toBe("brand_new");
  });

  it("stops exploring an arm once it is confidently bad", () => {
    const b = bandit({ alpha: 0.6 });
    for (let i = 0; i < 500; i++) {
      b.update("good", CONTEXT_A, 0.9);
      b.update("terrible", CONTEXT_A, 0.0);
    }
    expect(b.select(["good", "terrible"], CONTEXT_A).armId).toBe("good");
  });

  it("onboards a new arm without retraining or disturbing existing arms", () => {
    const b = bandit({ alpha: 1 });
    for (let i = 0; i < 300; i++) {
      b.update("incumbent", CONTEXT_A, 0.9);
      b.update("weak", CONTEXT_A, 0.2);
    }

    // Adding an arm costs nothing: no retraining, and prior learning is untouched.
    const choice = b.select(["incumbent", "weak", "newcomer"], CONTEXT_A);
    expect(choice.armId).toBe("newcomer");
    expect(b.pulls("incumbent")).toBe(300);
    expect(b.averageReward("incumbent")).toBeCloseTo(0.9, 10);

    // And once the newcomer proves poor, the incumbent takes the traffic back.
    for (let i = 0; i < 300; i++) b.update("newcomer", CONTEXT_A, 0.1);
    expect(b.select(["incumbent", "weak", "newcomer"], CONTEXT_A).armId).toBe("incumbent");
  });

  it("bounds the exploration bonus so a strong incumbent is not displaced by optimism alone", () => {
    // With a modest alpha the newcomer's uncertainty bonus (~0.42) cannot outbid a proven 0.9 arm.
    // That is deliberate: alpha is the dial between exploring new models and protecting good traffic.
    const b = bandit({ alpha: 0.3 });
    for (let i = 0; i < 300; i++) b.update("incumbent", CONTEXT_A, 0.9);

    expect(b.select(["incumbent", "newcomer"], CONTEXT_A).armId).toBe("incumbent");

    // Against a weaker incumbent, the same bonus is enough to justify a look.
    const weakPool = bandit({ alpha: 0.3 });
    for (let i = 0; i < 300; i++) weakPool.update("incumbent", CONTEXT_A, 0.3);
    expect(weakPool.select(["incumbent", "newcomer"], CONTEXT_A).armId).toBe("newcomer");
  });

  it("exposes per-arm scores so a decision can be explained", () => {
    const b = bandit();
    const choice = b.select(["a", "b"], CONTEXT_A);
    expect(Object.keys(choice.scores).sort()).toEqual(["a", "b"]);
  });

  it("only considers the arms it was handed", () => {
    const b = bandit();
    for (let i = 0; i < 100; i++) b.update("unavailable", CONTEXT_A, 1);
    expect(b.select(["a", "b"], CONTEXT_A).armId).not.toBe("unavailable");
  });

  it("tracks pulls and average reward per arm", () => {
    const b = bandit();
    b.update("a", CONTEXT_A, 1);
    b.update("a", CONTEXT_A, 0);
    expect(b.pulls("a")).toBe(2);
    expect(b.averageReward("a")).toBeCloseTo(0.5, 10);
    expect(b.pulls("never-seen")).toBe(0);
  });

  it("rejects a feature vector of the wrong length", () => {
    expect(() => bandit().select(["a"], [1, 2])).toThrow(/expected 4/);
  });

  it("throws rather than guessing when handed no arms", () => {
    expect(() => bandit().select([], CONTEXT_A)).toThrow(/no arms/);
  });

  describe("exploration floor", () => {
    it("pulls the least-explored arm when the floor triggers", () => {
      const b = new LinUcbBandit({ dimension: D, explorationFloor: 1, random: seeded(7) });
      for (let i = 0; i < 100; i++) b.update("popular", CONTEXT_A, 0.9);

      const choice = b.select(["popular", "neglected"], CONTEXT_A);
      expect(choice.armId).toBe("neglected");
      expect(choice.explored).toBe(true);
    });

    it("is off when the floor is zero", () => {
      const b = new LinUcbBandit({ dimension: D, explorationFloor: 0, random: () => 0 });
      expect(b.select(["a", "b"], CONTEXT_A).explored).toBe(false);
    });
  });

  describe("state persistence", () => {
    it("round-trips learning through a snapshot", () => {
      const original = bandit();
      for (let i = 0; i < 100; i++) {
        original.update("good", CONTEXT_A, 0.9);
        original.update("bad", CONTEXT_A, 0.1);
      }

      const restored = bandit();
      expect(restored.restore(original.snapshot())).toBe(true);
      expect(restored.pulls("good")).toBe(100);
      expect(restored.select(["good", "bad"], CONTEXT_A).armId).toBe("good");
      expect(restored.select(["good", "bad"], CONTEXT_A).scores).toEqual(
        original.select(["good", "bad"], CONTEXT_A).scores,
      );
    });

    it("refuses state trained on a different feature layout", () => {
      // Reinterpreting old coefficients against a new feature vector produces confident nonsense.
      // Starting cold is the safe failure.
      const original = new LinUcbBandit({ dimension: 8 });
      original.update("a", [1, 0, 0, 0, 0, 0, 0, 0], 1);

      const wider = bandit();
      expect(wider.restore(original.snapshot())).toBe(false);
      expect(wider.pulls("a")).toBe(0);
    });

    it("refuses state from a different bandit kind", () => {
      const thompson = new ThompsonBandit({ random: seeded(3) });
      thompson.update("a", [], 1);
      expect(bandit().restore(thompson.snapshot())).toBe(false);
    });

    it("survives a JSON round trip, which is how it actually gets stored", () => {
      const original = bandit();
      for (let i = 0; i < 50; i++) original.update("a", CONTEXT_A, 0.7);

      const restored = bandit();
      expect(restored.restore(JSON.parse(JSON.stringify(original.snapshot())))).toBe(true);
      expect(restored.averageReward("a")).toBeCloseTo(0.7, 10);
    });
  });
});

describe("ThompsonBandit", () => {
  it("converges on the better arm", () => {
    const b = new ThompsonBandit({ random: seeded(42) });
    for (let i = 0; i < 300; i++) {
      b.update("good", [], 0.9);
      b.update("bad", [], 0.1);
    }

    let goodWins = 0;
    for (let i = 0; i < 100; i++) {
      if (b.select(["good", "bad"]).armId === "good") goodWins++;
    }
    expect(goodWins).toBeGreaterThan(90);
  });

  it("ignores context entirely, which is what makes it a useful control", () => {
    // If Thompson matches LinUCB on real traffic, the feature vector is not earning its complexity.
    const b = new ThompsonBandit({ random: seeded(11) });
    b.update("a", [1, 0, 0, 0], 1);
    b.update("a", [0, 1, 1, 1], 1);
    expect(b.pulls("a")).toBe(2);
  });

  it("clamps out-of-range rewards", () => {
    const b = new ThompsonBandit({ random: seeded(5) });
    b.update("a", [], 5);
    expect(b.averageReward("a")).toBe(1);
  });

  it("round-trips through a snapshot", () => {
    const original = new ThompsonBandit({ random: seeded(9) });
    for (let i = 0; i < 40; i++) original.update("a", [], 0.8);

    const restored = new ThompsonBandit({ random: seeded(9) });
    expect(restored.restore(original.snapshot())).toBe(true);
    expect(restored.pulls("a")).toBe(40);
  });

  it("samples within [0,1]", () => {
    const b = new ThompsonBandit({ random: seeded(21) });
    for (let i = 0; i < 200; i++) {
      const score = b.select(["a"]).scores.a as number;
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });
});
