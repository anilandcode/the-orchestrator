import { createSequentialIds } from "@orchestrator/shared";
import { describe, expect, it } from "vitest";
import { AdaptiveRouter, type RouterMode } from "./adaptive-router.js";
import { LinUcbBandit } from "./bandit/linucb.js";
import { InMemoryStateStore } from "./bandit/state-store.js";
import { FEATURE_DIMENSION, extractFeatures } from "./features.js";
import { StaticRouter } from "./static-router.js";
import { CHEAPEST, PREMIUM, context } from "./test-helpers.js";

function build(
  overrides: {
    mode?: RouterMode;
    coldStartPulls?: number;
    alpha?: number;
    stateStore?: InMemoryStateStore;
    policies?: Record<string, { pinByTaskType?: Record<string, string> }>;
  } = {},
) {
  const bandit = new LinUcbBandit({
    dimension: FEATURE_DIMENSION,
    alpha: overrides.alpha ?? 0.6,
    explorationFloor: 0,
    random: () => 0.99,
  });

  const router = new AdaptiveRouter({
    bandit,
    baseline: new StaticRouter({ ids: createSequentialIds(), policies: overrides.policies }),
    mode: overrides.mode ?? "adaptive",
    coldStartPulls: overrides.coldStartPulls ?? 0,
    ids: createSequentialIds(),
    persistEvery: 1,
    ...(overrides.stateStore ? { stateStore: overrides.stateStore } : {}),
    ...(overrides.policies ? { policies: overrides.policies } : {}),
  });

  return { router, bandit };
}

/** Feed the bandit enough observations that an arm clears the cold-start gate. */
function train(
  router: AdaptiveRouter,
  modelId: string,
  reward: number,
  times: number,
  ctx = context(),
) {
  const features = extractFeatures(ctx);
  for (let i = 0; i < times; i++) {
    router.observe({ modelId, features, reward, taskType: ctx.request.route.taskType });
  }
}

describe("AdaptiveRouter mode gate", () => {
  it("ignores the bandit entirely in static mode", () => {
    const { router } = build({ mode: "static" });
    train(router, PREMIUM, 1, 100);

    const decision = router.select(
      context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } }),
    );
    expect(decision.modelId).toBe(CHEAPEST);
    expect(decision.strategy).toBe("static");
    expect(decision.shadowModelId).toBeNull();
  });

  it("executes the static pick but records the bandit's in shadow mode", () => {
    const { router } = build({ mode: "shadow" });
    train(router, PREMIUM, 1, 100);
    train(router, CHEAPEST, 0, 100);

    const decision = router.select(
      context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } }),
    );

    // The static choice is what runs.
    expect(decision.modelId).toBe(CHEAPEST);
    expect(decision.strategy).toBe("static");
    // The bandit's counterfactual is recorded beside it — this column is what justifies promotion.
    expect(decision.shadowModelId).toBe(PREMIUM);
  });

  it("lets the bandit steer in adaptive mode once it has data", () => {
    const { router } = build({ mode: "adaptive", coldStartPulls: 10 });
    const ctx = context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } });

    train(router, PREMIUM, 0.95, 50, ctx);
    train(router, CHEAPEST, 0.05, 50, ctx);

    const decision = router.select(ctx);
    expect(decision.modelId).toBe(PREMIUM);
    expect(decision.strategy).toBe("adaptive");
    // The static pick is retained for comparison in the other direction.
    expect(decision.shadowModelId).toBe(CHEAPEST);
  });

  it("defaults to shadow mode", () => {
    const bandit = new LinUcbBandit({ dimension: FEATURE_DIMENSION });
    const router = new AdaptiveRouter({
      bandit,
      baseline: new StaticRouter({ ids: createSequentialIds() }),
    });
    // Shipping with the bandit steering real traffic by default would be the wrong default.
    expect(router.select(context()).strategy).toBe("static");
  });
});

describe("AdaptiveRouter cold start", () => {
  it("defers to the baseline until the task type has enough observations", () => {
    const { router } = build({ mode: "adaptive", coldStartPulls: 25 });
    const ctx = context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } });

    train(router, PREMIUM, 0.99, 5, ctx);

    const decision = router.select(ctx);
    expect(decision.modelId).toBe(CHEAPEST);
    expect(decision.strategy).toBe("static");
    expect(decision.reason).toMatch(/bandit deferred: 5\/25/);
    // The bandit's preference is still recorded while it waits.
    expect(decision.shadowModelId).toBe(PREMIUM);
  });

  it("gates per task type, not globally", () => {
    // Experience on `code` says nothing about `extraction`, so the gate is scoped per task.
    const { router } = build({ mode: "adaptive", coldStartPulls: 20 });
    const codeCtx = context({
      messages: [{ role: "user", content: "x" }],
      route: { mode: "cheap", taskType: "code" },
    });
    const extractionCtx = context({
      messages: [{ role: "user", content: "x" }],
      route: { mode: "cheap", taskType: "extraction" },
    });

    train(router, PREMIUM, 0.99, 40, codeCtx);

    expect(router.taskPullsFor("code")).toBe(40);
    expect(router.taskPullsFor("extraction")).toBe(0);
    expect(router.select(extractionCtx).strategy).toBe("static");
  });

  it("does not deadlock on arms the baseline never picks", () => {
    // The gate counts observations for the task type, not for the arm the bandit wants. Gating
    // per-arm would be self-blocking: during cold start the baseline's pick is what executes, so a
    // model the static rules never favour would never accumulate pulls and the gate would never open.
    const { router } = build({ mode: "adaptive", coldStartPulls: 10 });
    const ctx = context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } });

    // Only the baseline's own pick is ever observed, exactly as it would be in cold start.
    train(router, CHEAPEST, 0.1, 10, ctx);
    // The bandit prefers a model with zero observations of its own — and is now allowed to say so.
    train(router, PREMIUM, 0.99, 1, ctx);

    const decision = router.select(ctx);
    expect(decision.strategy).toBe("adaptive");
    expect(decision.modelId).toBe(PREMIUM);
    expect(router.contextPullsFor(PREMIUM, "general")).toBe(1);
  });

  it("takes over once the gate is cleared", () => {
    const { router } = build({ mode: "adaptive", coldStartPulls: 20 });
    const ctx = context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } });

    train(router, PREMIUM, 0.99, 19, ctx);
    expect(router.select(ctx).strategy).toBe("static");

    train(router, PREMIUM, 0.99, 1, ctx);
    expect(router.select(ctx).strategy).toBe("adaptive");
  });
});

describe("AdaptiveRouter safety", () => {
  it("never overrides an explicit pin", () => {
    const { router } = build({ mode: "adaptive", coldStartPulls: 0 });
    train(router, CHEAPEST, 1, 100);

    const decision = router.select(
      context({
        messages: [{ role: "user", content: "hi" }],
        route: { mode: "cheap", pin: PREMIUM },
      }),
    );
    expect(decision.modelId).toBe(PREMIUM);
    expect(decision.strategy).toBe("pinned");
  });

  it("never overrides a tenant task pin", () => {
    const { router } = build({
      mode: "adaptive",
      coldStartPulls: 0,
      policies: { acme: { pinByTaskType: { code: PREMIUM } } },
    });
    train(router, CHEAPEST, 1, 100);

    const decision = router.select(
      context({
        tenantId: "acme",
        messages: [{ role: "user", content: "hi" }],
        route: { taskType: "code" },
      }),
    );
    expect(decision.modelId).toBe(PREMIUM);
  });

  it("only picks among models the gateway can reach", () => {
    const { router } = build({ mode: "adaptive", coldStartPulls: 0 });
    const openAiOnly = context().available.filter((spec) => spec.provider === "openai");
    train(router, PREMIUM, 1, 100);

    const decision = router.select(
      context({ messages: [{ role: "user", content: "hi" }] }, { available: openAiOnly }),
    );
    expect(decision.modelId).not.toBe(PREMIUM);
  });

  it("respects hard constraints the bandit knows nothing about", () => {
    const { router } = build({ mode: "adaptive", coldStartPulls: 0 });
    train(router, PREMIUM, 1, 200);

    // The bandit loves the premium model, but the budget forbids it.
    const decision = router.select(
      context({
        messages: [{ role: "user", content: "hi" }],
        route: { mode: "best", maxCostUsd: 0.01 },
      }),
    );
    expect(decision.modelId).not.toBe(PREMIUM);
  });

  it("always attaches a fallback chain", () => {
    const { router } = build({ mode: "adaptive", coldStartPulls: 0 });
    train(router, PREMIUM, 1, 50);

    const decision = router.select(context());
    expect(decision.fallbacks.length).toBeGreaterThan(0);
    expect(decision.fallbacks).not.toContain(decision.modelId);
  });

  it("explains an adaptive decision with the winning score", () => {
    const { router } = build({ mode: "adaptive", coldStartPulls: 0 });
    train(router, PREMIUM, 0.9, 50);
    expect(router.select(context()).reason).toMatch(/bandit: score/);
  });
});

describe("AdaptiveRouter learning attribution", () => {
  it("settles an outcome by decision id without the caller retaining features", () => {
    const { router, bandit } = build({ mode: "adaptive", coldStartPulls: 0 });
    const decision = router.select(context());

    expect(router.observeDecision(decision.decisionId, 0.9)).toBe(true);
    expect(bandit.pulls(decision.modelId)).toBe(1);
    expect(bandit.averageReward(decision.modelId)).toBeCloseTo(0.9, 10);
  });

  it("reports an unknown decision id rather than silently crediting the wrong arm", () => {
    const { router } = build();
    expect(router.observeDecision("dec_never_issued", 1)).toBe(false);
  });

  it("credits the model that actually ran, not the one the bandit preferred", () => {
    // In shadow mode the static pick executes, so that is the arm the reward belongs to.
    const { router, bandit } = build({ mode: "shadow", coldStartPulls: 0 });
    train(router, PREMIUM, 1, 40);

    const decision = router.select(
      context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } }),
    );
    expect(decision.modelId).toBe(CHEAPEST);
    expect(decision.shadowModelId).toBe(PREMIUM);

    const premiumPullsBefore = bandit.pulls(PREMIUM);
    router.observeDecision(decision.decisionId, 0.4);

    expect(bandit.pulls(CHEAPEST)).toBe(1);
    expect(bandit.pulls(PREMIUM)).toBe(premiumPullsBefore);
  });
});

describe("AdaptiveRouter persistence", () => {
  it("survives a restart with its learning intact", () => {
    const stateStore = new InMemoryStateStore();
    const ctx = context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } });

    const first = build({ mode: "adaptive", coldStartPulls: 20, stateStore });
    train(first.router, PREMIUM, 0.95, 30, ctx);
    train(first.router, CHEAPEST, 0.05, 30, ctx);
    first.router.persist();

    // A router that forgets on deploy is permanently in cold start.
    const restarted = build({ mode: "adaptive", coldStartPulls: 20, stateStore });
    expect(restarted.router.contextPullsFor(PREMIUM, "general")).toBe(30);
    expect(restarted.router.taskPullsFor("general")).toBe(60);
    expect(restarted.router.select(ctx).modelId).toBe(PREMIUM);
  });

  it("starts cold rather than corrupt when stored state predates a feature change", () => {
    const stateStore = new InMemoryStateStore();
    stateStore.save("router:v1", {
      bandit: { kind: "linucb", version: 1, dimension: 3, arms: {} },
      contextPulls: { [`${PREMIUM}|general`]: 999 },
      taskPulls: { general: 999 },
    });

    const { router } = build({ mode: "adaptive", coldStartPulls: 20, stateStore });
    // The mismatched state is discarded wholesale — including its pull counts, which would
    // otherwise wave an untrained bandit straight past the cold-start gate.
    expect(router.contextPullsFor(PREMIUM, "general")).toBe(0);
    expect(router.taskPullsFor("general")).toBe(0);
    expect(router.select(context()).strategy).toBe("static");
  });

  it("works without a state store at all", () => {
    const { router } = build({ mode: "adaptive", coldStartPulls: 0 });
    expect(() => router.persist()).not.toThrow();
    expect(router.select(context()).modelId).toBeTruthy();
  });
});

describe("extractFeatures", () => {
  it("produces a vector of the declared dimension", () => {
    expect(extractFeatures(context())).toHaveLength(FEATURE_DIMENSION);
  });

  it("separates task types", () => {
    const code = extractFeatures(
      context({ messages: [{ role: "user", content: "x" }], route: { taskType: "code" } }),
    );
    const creative = extractFeatures(
      context({ messages: [{ role: "user", content: "x" }], route: { taskType: "creative" } }),
    );
    expect(code).not.toEqual(creative);
  });

  it("encodes the caller's preference weights, which is what conditions routing on intent", () => {
    const cheap = extractFeatures(
      context({ messages: [{ role: "user", content: "x" }], route: { mode: "cheap" } }),
    );
    const best = extractFeatures(
      context({ messages: [{ role: "user", content: "x" }], route: { mode: "best" } }),
    );
    expect(cheap).not.toEqual(best);
  });

  it("flags tool use and images", () => {
    const withTools = extractFeatures(
      context({
        messages: [{ role: "user", content: "x" }],
        tools: [{ name: "f", parameters: {} }],
      }),
    );
    const plain = extractFeatures(context());
    expect(withTools).not.toEqual(plain);
  });

  it("keeps every component within [0,1] so none dominates by scale", () => {
    const features = extractFeatures(
      context(
        { messages: [{ role: "user", content: "x" }] },
        { estimatedPromptTokens: 900_000, turnIndex: 500 },
      ),
    );
    for (const value of features) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });
});
