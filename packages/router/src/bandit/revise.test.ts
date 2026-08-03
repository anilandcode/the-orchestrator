import { describe, expect, it } from "vitest";
import { LinUcbBandit } from "./linucb.js";
import { ThompsonBandit } from "./thompson.js";

/**
 * Reward revision is the machinery that lets a late quality signal correct what the bandit already
 * learned. If it is wrong, it corrupts learning silently — there is no error, just a router that
 * slowly becomes confident about the wrong thing. These tests are the gate on that.
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

describe("LinUcbBandit.revise", () => {
  it("lands in the same state as training on the final reward directly", () => {
    // The core property. LinUCB's A depends only on x xᵀ and its b is linear in reward, so
    // correcting by the delta is exact in real arithmetic — here, to floating-point tolerance,
    // since the accumulation order differs between the two paths.
    const revised = bandit();
    revised.update("arm", CONTEXT_A, 0.2);
    revised.revise("arm", CONTEXT_A, 0.2, 0.9);

    const direct = bandit();
    direct.update("arm", CONTEXT_A, 0.9);

    const revisedState = revised.snapshot().arms.arm;
    const directState = direct.snapshot().arms.arm;

    expect(revisedState?.b).toHaveLength(D);
    for (const [index, value] of (revisedState?.b ?? []).entries()) {
      expect(value).toBeCloseTo(directState?.b?.[index] as number, 12);
    }
    // A is untouched by reward, so it must match exactly, not merely closely.
    expect(revisedState?.aInv).toEqual(directState?.aInv);
  });

  it("holds across a long interleaved sequence of updates and revisions", () => {
    const random = seeded(7);
    const revised = bandit();
    const direct = bandit();

    const applied: { arm: string; features: number[]; reward: number }[] = [];

    for (let i = 0; i < 200; i++) {
      const arm = random() < 0.5 ? "a" : "b";
      const features = random() < 0.5 ? CONTEXT_A : CONTEXT_B;
      const provisional = random();

      revised.update(arm, features, provisional);
      applied.push({ arm, features, reward: provisional });

      // A third of the observations later get corrected, as a judge or a human would.
      if (random() < 0.33) {
        const target = applied[Math.floor(random() * applied.length)];
        if (target) {
          const final = random();
          revised.revise(target.arm, target.features, target.reward, final);
          target.reward = final;
        }
      }
    }

    for (const entry of applied) direct.update(entry.arm, entry.features, entry.reward);

    for (const armId of ["a", "b"]) {
      const revisedArm = revised.snapshot().arms[armId];
      const directArm = direct.snapshot().arms[armId];

      expect(revised.pulls(armId)).toBe(direct.pulls(armId));
      for (const [index, value] of (revisedArm?.b ?? []).entries()) {
        expect(value).toBeCloseTo(directArm?.b?.[index] as number, 10);
      }
    }
  });

  it("does not count a revision as a new observation", () => {
    // Revision corrects one observation; treating it as a second would inflate confidence and
    // shrink the exploration bonus for an arm that has actually been tried once.
    const b = bandit();
    b.update("arm", CONTEXT_A, 0.5);
    b.revise("arm", CONTEXT_A, 0.5, 0.1);
    b.revise("arm", CONTEXT_A, 0.1, 0.9);

    expect(b.pulls("arm")).toBe(1);
    expect(b.averageReward("arm")).toBeCloseTo(0.9, 12);
  });

  it("changes which arm wins when the correction reverses the ranking", () => {
    const b = bandit();
    for (let i = 0; i < 40; i++) {
      b.update("optimistic", CONTEXT_A, 0.9);
      b.update("pessimistic", CONTEXT_A, 0.3);
    }
    expect(b.select(["optimistic", "pessimistic"], CONTEXT_A).armId).toBe("optimistic");

    // A judge reveals the optimistic arm was actually poor all along.
    for (let i = 0; i < 40; i++) b.revise("optimistic", CONTEXT_A, 0.9, 0.05);

    expect(b.select(["optimistic", "pessimistic"], CONTEXT_A).armId).toBe("pessimistic");
  });

  it("ignores a revision for an arm it has never seen", () => {
    // Expected, not exceptional: feedback can outlive a restart or a state reset.
    const b = bandit();
    expect(() => b.revise("never-pulled", CONTEXT_A, 0.5, 0.9)).not.toThrow();
    expect(b.pulls("never-pulled")).toBe(0);
  });

  it("is a no-op when the reward did not actually change", () => {
    const b = bandit();
    b.update("arm", CONTEXT_A, 0.6);
    const before = b.snapshot().arms.arm?.b;
    b.revise("arm", CONTEXT_A, 0.6, 0.6);
    expect(b.snapshot().arms.arm?.b).toEqual(before);
  });

  it("survives a persist/restore round trip mid-revision", () => {
    const original = bandit();
    original.update("arm", CONTEXT_A, 0.3);

    const restored = bandit();
    restored.restore(original.snapshot());
    restored.revise("arm", CONTEXT_A, 0.3, 0.8);

    const direct = bandit();
    direct.update("arm", CONTEXT_A, 0.8);

    for (const [index, value] of (restored.snapshot().arms.arm?.b ?? []).entries()) {
      expect(value).toBeCloseTo(direct.snapshot().arms.arm?.b?.[index] as number, 12);
    }
  });
});

describe("ThompsonBandit.revise", () => {
  it("shifts posterior mass toward the corrected reward", () => {
    const b = new ThompsonBandit({ random: seeded(3) });
    b.update("arm", [], 0.1);
    b.revise("arm", [], 0.1, 0.95);

    expect(b.pulls("arm")).toBe(1);
    expect(b.averageReward("arm")).toBeCloseTo(0.95, 10);
  });

  it("keeps Beta parameters positive under an aggressive downward correction", () => {
    // alpha/beta must stay strictly positive or the distribution is improper and sampling breaks.
    const b = new ThompsonBandit({ priorAlpha: 1, priorBeta: 1, random: seeded(4) });
    b.update("arm", [], 1);
    for (let i = 0; i < 50; i++) b.revise("arm", [], 1, 0);

    const score = b.select(["arm"]).scores.arm as number;
    expect(Number.isNaN(score)).toBe(false);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("ignores a revision for an unknown arm", () => {
    const b = new ThompsonBandit({ random: seeded(5) });
    expect(() => b.revise("nope", [], 0.2, 0.7)).not.toThrow();
  });
});
