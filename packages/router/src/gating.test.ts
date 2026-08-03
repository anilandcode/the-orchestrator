import { createSequentialIds } from "@orchestrator/shared";
import { describe, expect, it } from "vitest";
import { AdaptiveRouter } from "./adaptive-router.js";
import { LinUcbBandit } from "./bandit/linucb.js";
import { InMemoryStateStore } from "./bandit/state-store.js";
import { FEATURE_DIMENSION, extractFeatures } from "./features.js";
import { StaticRouter } from "./static-router.js";
import { CHEAPEST, PREMIUM, context } from "./test-helpers.js";

/**
 * The quality-observability gate.
 *
 * Simulation showed the bandit beats the static rules on tasks a validator can grade (56.4% vs 45.3%
 * optimal picks) and loses on the rest by more than that win is worth. This gate is how that finding
 * became behaviour: steer only where the reward's quality term actually carries information.
 */
function build(overrides: { minQualityConfidence?: number; stateStore?: InMemoryStateStore } = {}) {
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
    coldStartPulls: 0,
    minConfidenceSamples: 10,
    ids: createSequentialIds(),
    persistEvery: 1,
    ...(overrides.minQualityConfidence !== undefined
      ? { minQualityConfidence: overrides.minQualityConfidence }
      : {}),
    ...(overrides.stateStore ? { stateStore: overrides.stateStore } : {}),
  });

  return { router, bandit };
}

function train(
  router: AdaptiveRouter,
  modelId: string,
  reward: number,
  times: number,
  confidence: number | undefined,
  ctx = context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } }),
) {
  const features = extractFeatures(ctx);
  for (let i = 0; i < times; i++) {
    router.observe({
      modelId,
      features,
      reward,
      taskType: ctx.request.route.taskType,
      ...(confidence !== undefined ? { qualityConfidence: confidence } : {}),
    });
  }
}

const cheapCtx = context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } });

describe("quality-observability gate", () => {
  it("declines to steer a task type graded only by the heuristic floor", () => {
    const { router } = build();
    // 0.2 is the confidence of "the call did not error" — a constant, carrying no signal about
    // which model to prefer.
    train(router, PREMIUM, 0.99, 40, 0.2);

    const decision = router.select(cheapCtx);
    expect(decision.strategy).toBe("static");
    expect(decision.modelId).toBe(CHEAPEST);
    expect(decision.reason).toMatch(/bandit gated: mean quality confidence 0\.20/);
    // The bandit's preference is still recorded, so shadow analysis keeps working.
    expect(decision.shadowModelId).toBe(PREMIUM);
  });

  it("steers a task type a deterministic validator can grade", () => {
    const { router } = build();
    train(router, PREMIUM, 0.99, 40, 0.9);

    const decision = router.select(cheapCtx);
    expect(decision.strategy).toBe("adaptive");
    expect(decision.modelId).toBe(PREMIUM);
  });

  it("gates per task type, so one ungradeable task does not disable the rest", () => {
    const { router } = build();
    const codeCtx = context({
      messages: [{ role: "user", content: "x" }],
      route: { mode: "cheap", taskType: "code" },
    });

    train(router, PREMIUM, 0.99, 40, 0.9, codeCtx); // validator-graded
    train(router, PREMIUM, 0.99, 40, 0.2, cheapCtx); // heuristic only

    expect(router.select(codeCtx).strategy).toBe("adaptive");
    expect(router.select(cheapCtx).strategy).toBe("static");
    expect(router.steeredTaskTypes()).toEqual(["code"]);
  });

  it("starts steering a task type once validators begin covering it", () => {
    // Self-maintaining by design: adding a validator should not require reconfiguring the router.
    const { router } = build();
    train(router, PREMIUM, 0.99, 30, 0.2);
    expect(router.select(cheapCtx).strategy).toBe("static");

    // A deterministic validator now covers this task; the running mean climbs past the threshold.
    train(router, PREMIUM, 0.99, 90, 0.9);
    expect(router.qualityConfidenceFor("general")).toBeGreaterThan(0.5);
    expect(router.select(cheapCtx).strategy).toBe("adaptive");
  });

  it("uses a running mean, so mixed coverage is judged on balance", () => {
    const { router } = build();
    train(router, PREMIUM, 0.99, 50, 0.9);
    train(router, PREMIUM, 0.99, 50, 0.2);

    // Mean is 0.55 — above the 0.5 threshold, so it still steers.
    expect(router.qualityConfidenceFor("general")).toBeCloseTo(0.55, 6);
    expect(router.select(cheapCtx).strategy).toBe("adaptive");
  });

  it("waits for enough samples before gating on a mean", () => {
    const { router } = build();
    // Below minConfidenceSamples: not enough evidence to judge, so it does not block.
    train(router, PREMIUM, 0.99, 5, 0.2);
    expect(router.qualityConfidenceFor("general")).toBeUndefined();
    expect(router.select(cheapCtx).strategy).toBe("adaptive");
  });

  it("does not gate callers that report no confidence at all", () => {
    // Backwards compatible: omitting confidence yields the pre-gating behaviour rather than
    // silently disabling adaptive routing everywhere.
    const { router } = build();
    train(router, PREMIUM, 0.99, 40, undefined);

    expect(router.qualityConfidenceFor("general")).toBeUndefined();
    expect(router.select(cheapCtx).strategy).toBe("adaptive");
  });

  it("can be disabled entirely", () => {
    const { router } = build({ minQualityConfidence: 0 });
    train(router, PREMIUM, 0.99, 40, 0.2);
    expect(router.select(cheapCtx).strategy).toBe("adaptive");
  });

  it("is on by default", () => {
    const bandit = new LinUcbBandit({ dimension: FEATURE_DIMENSION });
    const router = new AdaptiveRouter({
      bandit,
      baseline: new StaticRouter({ ids: createSequentialIds() }),
      mode: "adaptive",
      coldStartPulls: 0,
    });

    for (let i = 0; i < 40; i++) {
      router.observe({
        modelId: PREMIUM,
        features: extractFeatures(cheapCtx),
        reward: 0.99,
        taskType: "general",
        qualityConfidence: 0.2,
      });
    }
    expect(router.select(cheapCtx).strategy).toBe("static");
  });

  it("survives a restart with its gating state intact", () => {
    // Otherwise the router re-learns which tasks it can grade after every deploy, and steers
    // ungradeable traffic in the meantime.
    const stateStore = new InMemoryStateStore();
    const first = build({ stateStore });
    train(first.router, PREMIUM, 0.99, 40, 0.2);
    first.router.persist();

    const restarted = build({ stateStore });
    expect(restarted.router.qualityConfidenceFor("general")).toBeCloseTo(0.2, 6);
    expect(restarted.router.select(cheapCtx).strategy).toBe("static");
  });

  it("still defers to the cold-start gate first", () => {
    // Confidence is high enough to steer, but there are not enough observations yet. Cold start is
    // the outer gate: no amount of signal quality substitutes for having seen the task at all.
    const gatedRouter = new AdaptiveRouter({
      bandit: new LinUcbBandit({ dimension: FEATURE_DIMENSION, explorationFloor: 0 }),
      baseline: new StaticRouter({ ids: createSequentialIds() }),
      mode: "adaptive",
      coldStartPulls: 50,
      minConfidenceSamples: 10,
      ids: createSequentialIds(),
    });

    train(gatedRouter, PREMIUM, 0.99, 20, 0.9);
    expect(gatedRouter.select(cheapCtx).reason).toMatch(/bandit deferred/);
  });

  it("never gates a pinned request, which was never a routing decision", () => {
    const { router } = build();
    train(router, CHEAPEST, 0.99, 40, 0.2);

    const decision = router.select(
      context({
        messages: [{ role: "user", content: "hi" }],
        route: { mode: "cheap", pin: PREMIUM },
      }),
    );
    expect(decision.modelId).toBe(PREMIUM);
    expect(decision.strategy).toBe("pinned");
  });
});
