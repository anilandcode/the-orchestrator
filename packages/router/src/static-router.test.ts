import { OrchestratorError, createSequentialIds, defaultRegistry } from "@orchestrator/shared";
import { describe, expect, it } from "vitest";
import { buildFallbackChain } from "./fallback.js";
import { StaticRouter } from "./static-router.js";
import { ALL_MODELS, CHEAPEST, HUGE_CONTEXT, PREMIUM, context } from "./test-helpers.js";

const router = (config = {}) => new StaticRouter({ ids: createSequentialIds(), ...config });

describe("StaticRouter mode rules", () => {
  it("picks the cheapest model in cheap mode", () => {
    const decision = router().select(
      context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } }),
    );
    expect(decision.modelId).toBe(CHEAPEST);
    expect(decision.strategy).toBe("static");
  });

  it("picks the highest tier in best mode", () => {
    const decision = router().select(
      context({ messages: [{ role: "user", content: "hi" }], route: { mode: "best" } }),
    );
    expect(defaultRegistry.require(decision.modelId).tier).toBe("premium");
  });

  it("picks the cheapest qualifying model in balanced mode", () => {
    const decision = router().select(
      context({ messages: [{ role: "user", content: "hi" }], route: { mode: "balanced" } }),
    );
    expect(decision.modelId).toBe(CHEAPEST);
  });

  it("floors code tasks above economy in balanced mode", () => {
    // The cheapest model is a false economy here: a wrong answer costs more human time than the
    // model ever saved.
    const decision = router().select(
      context({
        messages: [{ role: "user", content: "refactor this" }],
        route: { mode: "balanced", taskType: "code" },
      }),
    );
    expect(defaultRegistry.require(decision.modelId).tier).not.toBe("economy");
  });

  it("still honours cheap mode for code tasks — the floor is balanced-mode only", () => {
    const decision = router().select(
      context({
        messages: [{ role: "user", content: "x" }],
        route: { mode: "cheap", taskType: "code" },
      }),
    );
    expect(decision.modelId).toBe(CHEAPEST);
  });

  it("explains itself", () => {
    const decision = router().select(
      context({ messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } }),
    );
    expect(decision.reason).toMatch(/cheap mode/);
  });

  it("is deterministic across repeated calls", () => {
    const staticRouter = router();
    const first = staticRouter.select(context());
    const second = staticRouter.select(context());
    expect(first.modelId).toBe(second.modelId);
    expect(first.decisionId).not.toBe(second.decisionId);
  });
});

describe("StaticRouter constraint filtering", () => {
  it("excludes models without tool support when tools are requested", () => {
    const noTools = ALL_MODELS.map((spec) =>
      spec.modelId === CHEAPEST
        ? { ...spec, capabilities: { ...spec.capabilities, tools: false } }
        : spec,
    );

    const decision = router().select(
      context(
        {
          messages: [{ role: "user", content: "hi" }],
          tools: [{ name: "f", parameters: {} }],
          route: { mode: "cheap" },
        },
        { available: noTools },
      ),
    );
    expect(decision.modelId).not.toBe(CHEAPEST);
  });

  it("excludes models whose context window cannot hold the prompt", () => {
    const decision = router().select(
      context(
        { messages: [{ role: "user", content: "hi" }], route: { mode: "cheap" } },
        { estimatedPromptTokens: 300_000 },
      ),
    );
    // Only the million-token model can hold this.
    expect(decision.modelId).toBe(HUGE_CONTEXT);
  });

  it("treats a cost ceiling as a hard limit", () => {
    // Silently blowing past a stated spend limit is worse than refusing the request.
    const decision = router().select(
      context({
        messages: [{ role: "user", content: "hi" }],
        route: { mode: "best", maxCostUsd: 0.01 },
      }),
    );
    expect(decision.modelId).not.toBe(PREMIUM);
  });

  it("throws a classified error when no model satisfies the constraints", () => {
    const attempt = () =>
      router().select(
        context({
          messages: [{ role: "user", content: "hi" }],
          route: { mode: "cheap", maxCostUsd: 0.000_000_1 },
        }),
      );

    expect(attempt).toThrow(OrchestratorError);
    expect(attempt).toThrow(/No model satisfies/);
  });

  it("treats a latency ceiling as a preference, not a hard filter", () => {
    // typicalLatencyMs is a prior, not a measurement — eliminating every candidate over a guess
    // would fail requests that would have succeeded.
    const decision = router().select(
      context({
        messages: [{ role: "user", content: "hi" }],
        route: { mode: "best", maxLatencyMs: 1 },
      }),
    );
    expect(decision.modelId).toBeTruthy();
  });

  it("prefers a model inside the latency budget when one exists", () => {
    const decision = router().select(
      context({
        messages: [{ role: "user", content: "hi" }],
        route: { mode: "best", maxLatencyMs: 1_200 },
      }),
    );
    expect(defaultRegistry.require(decision.modelId).typicalLatencyMs).toBeLessThanOrEqual(1_200);
  });

  it("only routes to models the gateway can actually reach", () => {
    const openAiOnly = ALL_MODELS.filter((spec) => spec.provider === "openai");
    const decision = router().select(
      context(
        { messages: [{ role: "user", content: "hi" }], route: { mode: "best" } },
        { available: openAiOnly },
      ),
    );
    expect(defaultRegistry.require(decision.modelId).provider).toBe("openai");
  });
});

describe("StaticRouter pinning and tenant policy", () => {
  it("honours an explicit pin over every rule", () => {
    const decision = router().select(
      context({
        messages: [{ role: "user", content: "hi" }],
        route: { mode: "cheap", pin: PREMIUM },
      }),
    );
    expect(decision.modelId).toBe(PREMIUM);
    expect(decision.strategy).toBe("pinned");
  });

  it("fails loudly when a pinned model is unavailable rather than silently substituting", () => {
    expect(() =>
      router().select(
        context({ messages: [{ role: "user", content: "hi" }], route: { pin: "vendor/nope" } }),
      ),
    ).toThrow(/Pinned model is unavailable/);
  });

  it("restricts a tenant to its allowlist", () => {
    const decision = router({
      policies: { acme: { allowModels: [PREMIUM] } },
    }).select(
      context({
        tenantId: "acme",
        messages: [{ role: "user", content: "hi" }],
        route: { mode: "cheap" },
      }),
    );

    expect(decision.modelId).toBe(PREMIUM);
  });

  it("respects a tenant denylist", () => {
    const decision = router({
      policies: { acme: { denyModels: [CHEAPEST] } },
    }).select(
      context({
        tenantId: "acme",
        messages: [{ role: "user", content: "hi" }],
        route: { mode: "cheap" },
      }),
    );

    expect(decision.modelId).not.toBe(CHEAPEST);
  });

  it("applies a per-task tenant pin", () => {
    const decision = router({
      policies: { acme: { pinByTaskType: { code: PREMIUM } } },
    }).select(
      context({
        tenantId: "acme",
        messages: [{ role: "user", content: "hi" }],
        route: { mode: "cheap", taskType: "code" },
      }),
    );

    expect(decision.modelId).toBe(PREMIUM);
    expect(decision.strategy).toBe("pinned");
  });

  it("leaves other tenants unaffected by a policy", () => {
    const staticRouter = router({ policies: { acme: { allowModels: [PREMIUM] } } });
    const decision = staticRouter.select(
      context({
        tenantId: "globex",
        messages: [{ role: "user", content: "hi" }],
        route: { mode: "cheap" },
      }),
    );
    expect(decision.modelId).toBe(CHEAPEST);
  });
});

describe("buildFallbackChain", () => {
  it("prefers a different provider first", () => {
    // Outages and rate limits are usually provider-wide, so a same-vendor fallback often dies too.
    const primary = defaultRegistry.require(CHEAPEST);
    const chain = buildFallbackChain(primary, ALL_MODELS, 3);

    expect(chain.length).toBeGreaterThan(0);
    expect(defaultRegistry.require(chain[0] as string).provider).not.toBe(primary.provider);
  });

  it("never includes the primary", () => {
    const primary = defaultRegistry.require(CHEAPEST);
    expect(buildFallbackChain(primary, ALL_MODELS, 5)).not.toContain(CHEAPEST);
  });

  it("respects the chain length cap", () => {
    const primary = defaultRegistry.require(CHEAPEST);
    expect(buildFallbackChain(primary, ALL_MODELS, 2)).toHaveLength(2);
  });

  it("returns an empty chain when the primary is the only candidate", () => {
    const primary = defaultRegistry.require(CHEAPEST);
    expect(buildFallbackChain(primary, [primary], 3)).toEqual([]);
  });

  it("includes a larger-context model so context overflow has somewhere to go", () => {
    const primary = defaultRegistry.require(CHEAPEST);
    const chain = buildFallbackChain(primary, ALL_MODELS, 3);
    const maxChainContext = Math.max(
      ...chain.map((id) => defaultRegistry.require(id).contextWindow),
    );
    expect(maxChainContext).toBeGreaterThan(primary.contextWindow);
  });

  it("attaches a chain to every decision", () => {
    const decision = router().select(context());
    expect(decision.fallbacks.length).toBeGreaterThan(0);
    expect(decision.fallbacks).not.toContain(decision.modelId);
  });
});
