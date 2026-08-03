import { createSequentialIds } from "@orchestrator/shared";
import { describe, expect, it } from "vitest";
import { AdaptiveRouter } from "./adaptive-router.js";
import { LinUcbBandit } from "./bandit/linucb.js";
import { InMemoryStateStore } from "./bandit/state-store.js";
import { FEATURE_DIMENSION, extractFeatures } from "./features.js";
import type { ModelPrior } from "./router.js";
import { StaticRouter } from "./static-router.js";
import { CHEAPEST, PREMIUM, context } from "./test-helpers.js";

/**
 * The "tilt only" contract.
 *
 * External priors were granted exactly one power: shift where the bandit starts. They may not open
 * the cold-start gate, may not satisfy the quality-observability gate, and may not make a router
 * with no real traffic start steering. These tests are what keeps that decision true as the code
 * changes — the whole safety argument for ingesting gameable benchmark data rests on it.
 */
function build(overrides: { stateStore?: InMemoryStateStore } = {}) {
  const bandit = new LinUcbBandit({
    dimension: FEATURE_DIMENSION,
    alpha: 0.6,
    explorationFloor: 0,
    random: () => 0.99,
  });

  const router = new AdaptiveRouter({
    bandit,
    baseline: new StaticRouter({ ids: createSequentialIds() }),
    mode: "adaptive",
    coldStartPulls: 25,
    ids: createSequentialIds(),
    persistEvery: 1,
    ...(overrides.stateStore ? { stateStore: overrides.stateStore } : {}),
  });

  return { router, bandit };
}

const cheapCtx = context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } });

const prior = (modelId: string, reward: number, weight = 20): ModelPrior => ({
  modelId,
  features: extractFeatures(cheapCtx),
  reward,
  weight,
  source: "benchmark:test",
});

describe("AdaptiveRouter.applyPriors", () => {
  describe("the tilt-only contract", () => {
    it("does not move the cold-start counter", () => {
      // taskPulls is what the cold-start gate reads. If seeding touched it, a benchmark score would
      // open a gate designed to require real traffic.
      const { router } = build();
      router.applyPriors([prior(PREMIUM, 0.95, 100)]);

      expect(router.taskPullsFor("general")).toBe(0);
    });

    it("does not move the quality-confidence counter", () => {
      const { router } = build();
      router.applyPriors([prior(PREMIUM, 0.95, 100)]);

      expect(router.qualityConfidenceFor("general")).toBeUndefined();
    });

    it("leaves a router with no real traffic still routing statically", () => {
      // The end-to-end statement of the contract.
      const { router } = build();
      router.applyPriors([prior(PREMIUM, 0.99, 500)]);

      const decision = router.select(cheapCtx);
      expect(decision.strategy).toBe("static");
      expect(decision.modelId).toBe(CHEAPEST);
    });

    it("does not count seeded weight as real pulls", () => {
      const { router, bandit } = build();
      router.applyPriors([prior(PREMIUM, 0.9, 30)]);

      expect(bandit.pulls(PREMIUM)).toBe(0);
      expect(router.syntheticPullsFor(PREMIUM)).toBe(30);
    });
  });

  describe("what priors actually do", () => {
    it("changes which arm the bandit prefers once the gates open", () => {
      const { router } = build();
      router.applyPriors([prior(PREMIUM, 0.95, 40), prior(CHEAPEST, 0.2, 40)]);

      // Real traffic clears cold start; the prior decides where the bandit starts from.
      const features = extractFeatures(cheapCtx);
      for (let i = 0; i < 25; i++) {
        router.observe({
          modelId: CHEAPEST,
          features,
          reward: 0.5,
          taskType: "general",
          qualityConfidence: 0.9,
        });
      }

      const decision = router.select(cheapCtx);
      expect(decision.strategy).toBe("adaptive");
      expect(decision.modelId).toBe(PREMIUM);
    });

    it("is overruled by enough real evidence", () => {
      // Priors are evidence, so more evidence wins. This is the safety property that makes ingesting
      // gameable benchmark data defensible at all.
      const { router, bandit } = build();
      router.applyPriors([prior(PREMIUM, 0.95, 10)]);

      const features = extractFeatures(cheapCtx);
      expect(bandit.select([PREMIUM, CHEAPEST], features).armId).toBe(PREMIUM);

      for (let i = 0; i < 200; i++) {
        router.observe({
          modelId: PREMIUM,
          features,
          reward: 0.05,
          taskType: "general",
          qualityConfidence: 0.9,
        });
        router.observe({
          modelId: CHEAPEST,
          features,
          reward: 0.9,
          taskType: "general",
          qualityConfidence: 0.9,
        });
      }

      // Asserted between the two contested arms rather than on the global pick: models that were
      // neither seeded nor observed still carry maximum uncertainty, so LinUCB rightly explores
      // them, and the argmax over the full pool says nothing about whether the prior was overruled.
      expect(bandit.select([PREMIUM, CHEAPEST], features).armId).toBe(CHEAPEST);
      expect(router.select(cheapCtx).modelId).not.toBe(PREMIUM);
    });
  });

  describe("idempotence", () => {
    it("skips an arm that already has real observations", () => {
      // Real evidence already outweighs a prior arithmetically, so re-seeding buys nothing — and
      // never needing to un-apply a weighted update keeps this numerically safe.
      const { router } = build();
      router.observe({
        modelId: PREMIUM,
        features: extractFeatures(cheapCtx),
        reward: 0.4,
        taskType: "general",
        qualityConfidence: 0.9,
      });

      const result = router.applyPriors([prior(PREMIUM, 0.95), prior(CHEAPEST, 0.3)]);
      expect(result).toEqual({ seeded: 1, skipped: 1 });
      expect(router.syntheticPullsFor(PREMIUM)).toBe(0);
      expect(router.syntheticPullsFor(CHEAPEST)).toBe(20);
    });

    it("reports what it did, so a boot log can show it", () => {
      const { router } = build();
      expect(router.applyPriors([prior(PREMIUM, 0.9), prior(CHEAPEST, 0.4)])).toEqual({
        seeded: 2,
        skipped: 0,
      });
    });

    it("handles an empty prior set", () => {
      const { router } = build();
      expect(router.applyPriors([])).toEqual({ seeded: 0, skipped: 0 });
    });
  });

  describe("persistence", () => {
    it("survives a restart without re-seeding", () => {
      const stateStore = new InMemoryStateStore();
      const first = build({ stateStore });
      first.router.applyPriors([prior(PREMIUM, 0.9, 25)]);

      const restarted = build({ stateStore });
      expect(restarted.router.syntheticPullsFor(PREMIUM)).toBe(25);

      // Re-applying after a restart must not double the prior.
      restarted.router.applyPriors([prior(PREMIUM, 0.9, 25)]);
      expect(restarted.router.syntheticPullsFor(PREMIUM)).toBe(50);
    });
  });
});
