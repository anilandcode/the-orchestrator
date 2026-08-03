import { type CallEvent, CallEventSchema } from "@orchestrator/shared";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_NORMALIZATION,
  REWARD_WEIGHTS,
  RewardService,
  RollingNormalizer,
  computeReward,
  heuristicQuality,
  percentileOf,
} from "./reward.js";
import { InMemoryCallEventRepository } from "./store/memory.js";

function event(overrides: Partial<CallEvent> = {}): CallEvent {
  return CallEventSchema.parse({
    id: "evt_1",
    tenantId: "local",
    requestId: "req_1",
    attempt: 1,
    provider: "openai",
    modelId: "openai/gpt-4o-mini",
    taskType: "general",
    routeMode: "balanced",
    promptTokens: 100,
    completionTokens: 50,
    costUsd: 0,
    latencyMs: 0,
    status: "success",
    finishReason: "stop",
    createdAt: 1_700_000_000_000,
    ...overrides,
  });
}

describe("reward weights", () => {
  it("sums to 1 for every mode, keeping rewards comparable across modes", () => {
    for (const weights of Object.values(REWARD_WEIGHTS)) {
      expect(weights.quality + weights.cost + weights.latency).toBeCloseTo(1, 10);
    }
  });

  it("weights cost highest in cheap mode and quality highest in best mode", () => {
    expect(REWARD_WEIGHTS.cheap.cost).toBeGreaterThan(REWARD_WEIGHTS.best.cost);
    expect(REWARD_WEIGHTS.best.quality).toBeGreaterThan(REWARD_WEIGHTS.cheap.quality);
  });
});

describe("computeReward", () => {
  it("scores a free, instant, perfect call at 1", () => {
    expect(computeReward(event({ costUsd: 0, latencyMs: 0, qualityScore: 1 }))).toBeCloseTo(1, 10);
  });

  it("scores a maximally expensive, maximally slow, worthless call at 0", () => {
    const reward = computeReward(event({ costUsd: 999, latencyMs: 999_999, qualityScore: 0 }));
    expect(reward).toBeCloseTo(0, 10);
  });

  describe("failure handling", () => {
    it("scores a failed call at exactly 0", () => {
      expect(computeReward(event({ status: "error", errorClass: "timeout" }))).toBe(0);
    });

    it("never ranks a fast free failure above a slow expensive success", () => {
      // The trap this guards: a failure costs $0 and returns instantly, so a naive weighted sum
      // would score it near the maximum and teach the bandit to prefer models that fail.
      const instantFailure = computeReward(
        event({ status: "error", errorClass: "provider_unavailable", costUsd: 0, latencyMs: 1 }),
      );
      const slowExpensiveSuccess = computeReward(
        event({ status: "success", costUsd: 0.04, latencyMs: 9_000 }),
      );

      expect(instantFailure).toBe(0);
      expect(slowExpensiveSuccess).toBeGreaterThan(instantFailure);
    });

    it("ignores a supplied quality score on a failed call", () => {
      expect(computeReward(event({ status: "error" }), { quality: 1 })).toBe(0);
    });
  });

  describe("cost and latency terms", () => {
    it("penalizes a more expensive call, all else equal", () => {
      const cheap = computeReward(event({ costUsd: 0.001, latencyMs: 1_000 }));
      const pricey = computeReward(event({ costUsd: 0.04, latencyMs: 1_000 }));
      expect(cheap).toBeGreaterThan(pricey);
    });

    it("penalizes a slower call, all else equal", () => {
      const fast = computeReward(event({ costUsd: 0.001, latencyMs: 500 }));
      const slow = computeReward(event({ costUsd: 0.001, latencyMs: 8_000 }));
      expect(fast).toBeGreaterThan(slow);
    });

    it("clamps rather than going negative past the normalization scale", () => {
      const reward = computeReward(
        event({ costUsd: DEFAULT_NORMALIZATION.costScaleUsd * 100, latencyMs: 0, qualityScore: 1 }),
      );
      // The cost term bottoms out at 0 instead of dragging the total below zero.
      const weights = REWARD_WEIGHTS.balanced;
      expect(reward).toBeCloseTo(weights.quality + weights.latency, 10);
    });
  });

  describe("mode sensitivity", () => {
    it("prefers the cheaper model more strongly in cheap mode than in best mode", () => {
      const cheapModeGap =
        computeReward(event({ routeMode: "cheap", costUsd: 0.001, qualityScore: 0.8 })) -
        computeReward(event({ routeMode: "cheap", costUsd: 0.04, qualityScore: 0.8 }));

      const bestModeGap =
        computeReward(event({ routeMode: "best", costUsd: 0.001, qualityScore: 0.8 })) -
        computeReward(event({ routeMode: "best", costUsd: 0.04, qualityScore: 0.8 }));

      expect(cheapModeGap).toBeGreaterThan(bestModeGap);
    });

    it("rewards a quality jump more in best mode than in cheap mode", () => {
      const bestGap =
        computeReward(event({ routeMode: "best", qualityScore: 1 })) -
        computeReward(event({ routeMode: "best", qualityScore: 0.5 }));
      const cheapGap =
        computeReward(event({ routeMode: "cheap", qualityScore: 1 })) -
        computeReward(event({ routeMode: "cheap", qualityScore: 0.5 }));

      expect(bestGap).toBeGreaterThan(cheapGap);
    });
  });

  it("prefers an explicit quality score over the stored one", () => {
    const withStored = event({ qualityScore: 0.2 });
    expect(computeReward(withStored, { quality: 1 })).toBeGreaterThan(computeReward(withStored));
  });
});

describe("heuristicQuality", () => {
  it("scores a clean stop highest", () => {
    expect(heuristicQuality(event({ finishReason: "stop" }))).toBe(0.8);
  });

  it("treats truncation as a partial answer", () => {
    expect(heuristicQuality(event({ finishReason: "length" }))).toBe(0.5);
  });

  it("scores filtered content at 0", () => {
    expect(heuristicQuality(event({ finishReason: "content_filter" }))).toBe(0);
  });

  it("scores any error at 0", () => {
    expect(heuristicQuality(event({ status: "error", finishReason: null }))).toBe(0);
  });
});

describe("RollingNormalizer", () => {
  it("uses the fallback scale until enough samples exist", () => {
    const normalizer = new RollingNormalizer(500, 0.95, 20);
    for (let i = 0; i < 5; i++) normalizer.observe(event({ costUsd: 1, latencyMs: 1 }));
    expect(normalizer.statsFor("general")).toEqual(DEFAULT_NORMALIZATION);
  });

  it("switches to observed percentiles once warmed up", () => {
    const normalizer = new RollingNormalizer(500, 0.95, 20);
    for (let i = 0; i < 100; i++) {
      normalizer.observe(event({ costUsd: 0.002, latencyMs: 1_500 }));
    }
    const stats = normalizer.statsFor("general");
    expect(stats.costScaleUsd).toBeCloseTo(0.002, 6);
    expect(stats.latencyScaleMs).toBeCloseTo(1_500, 6);
  });

  it("keeps task types on separate scales", () => {
    const normalizer = new RollingNormalizer(500, 0.95, 20);
    for (let i = 0; i < 50; i++) {
      normalizer.observe(event({ taskType: "code", costUsd: 0.02, latencyMs: 5_000 }));
      normalizer.observe(event({ taskType: "classification", costUsd: 0.0001, latencyMs: 300 }));
    }
    expect(normalizer.statsFor("code").costScaleUsd).toBeGreaterThan(
      normalizer.statsFor("classification").costScaleUsd,
    );
  });

  it("ignores failures, which have no meaningful cost or latency", () => {
    const normalizer = new RollingNormalizer(500, 0.95, 20);
    for (let i = 0; i < 50; i++) {
      normalizer.observe(event({ status: "error", costUsd: 0, latencyMs: 0 }));
    }
    expect(normalizer.statsFor("general")).toEqual(DEFAULT_NORMALIZATION);
  });

  it("drops observations outside the window so the scale can track a shifting pool", () => {
    const normalizer = new RollingNormalizer(30, 0.95, 20);
    for (let i = 0; i < 30; i++) normalizer.observe(event({ costUsd: 1, latencyMs: 1 }));
    for (let i = 0; i < 30; i++) normalizer.observe(event({ costUsd: 0.01, latencyMs: 1 }));
    expect(normalizer.statsFor("general").costScaleUsd).toBeCloseTo(0.01, 6);
  });
});

describe("RewardService", () => {
  it("persists the score back onto the stored event", () => {
    const repository = new InMemoryCallEventRepository();
    const stored = event({ costUsd: 0.001, latencyMs: 500 });
    repository.record(stored);

    const reward = new RewardService(repository).settle(stored);

    const [reloaded] = repository.query();
    expect(reloaded?.reward).toBeCloseTo(reward, 10);
    expect(reloaded?.qualityScore).toBe(0.8);
  });

  it("records an explicit quality signal when one is supplied", () => {
    const repository = new InMemoryCallEventRepository();
    const stored = event();
    repository.record(stored);

    new RewardService(repository).settle(stored, 0.25);

    expect(repository.query()[0]?.qualityScore).toBe(0.25);
  });

  it("scores a failed call at zero on both axes", () => {
    const repository = new InMemoryCallEventRepository();
    const stored = event({ status: "error", errorClass: "timeout", finishReason: null });
    repository.record(stored);

    expect(new RewardService(repository).settle(stored)).toBe(0);
    expect(repository.query()[0]?.qualityScore).toBe(0);
  });
});

describe("percentileOf", () => {
  it("returns 0 for an empty set", () => {
    expect(percentileOf([], 0.95)).toBe(0);
  });

  it("returns the max at p100 and the min at p0", () => {
    const values = [5, 1, 9, 3];
    expect(percentileOf(values, 1)).toBe(9);
    expect(percentileOf(values, 0)).toBe(1);
  });

  it("does not mutate the input", () => {
    const values = [3, 1, 2];
    percentileOf(values, 0.5);
    expect(values).toEqual([3, 1, 2]);
  });
});
