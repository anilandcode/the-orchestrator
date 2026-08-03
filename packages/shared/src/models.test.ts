import { describe, expect, it } from "vitest";
import { ModelRegistry, blendedCostPerMTok, computeCostUsd, worstCaseCostUsd } from "./models.js";
import type { ModelSpec } from "./models.js";
import type { Usage } from "./schemas/chat.js";

const spec: ModelSpec = {
  modelId: "test/model",
  provider: "openai",
  providerModel: "test-model",
  inputCostPerMTok: 10,
  outputCostPerMTok: 30,
  cachedInputCostPerMTok: 1,
  contextWindow: 100_000,
  maxOutputTokens: 1_000,
  tier: "standard",
  capabilities: { tools: true, vision: false, streaming: true, jsonMode: true },
  typicalLatencyMs: 1_000,
};

const usage = (partial: Partial<Usage>): Usage => ({
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  cachedPromptTokens: 0,
  ...partial,
});

describe("computeCostUsd", () => {
  it("prices input and output at their separate rates", () => {
    // 1M prompt @ $10 + 1M completion @ $30
    const cost = computeCostUsd(
      spec,
      usage({ promptTokens: 1_000_000, completionTokens: 1_000_000 }),
    );
    expect(cost).toBeCloseTo(40, 10);
  });

  it("bills cached prompt tokens at the cache rate, not the full input rate", () => {
    const cost = computeCostUsd(
      spec,
      usage({ promptTokens: 1_000_000, cachedPromptTokens: 1_000_000 }),
    );
    // All prompt tokens were cache reads: $1, not $10.
    expect(cost).toBeCloseTo(1, 10);
  });

  it("splits partially cached prompts across both rates", () => {
    const cost = computeCostUsd(
      spec,
      usage({ promptTokens: 1_000_000, cachedPromptTokens: 400_000 }),
    );
    // 600k @ $10/M = $6.00, 400k @ $1/M = $0.40
    expect(cost).toBeCloseTo(6.4, 10);
  });

  it("never returns a negative cost when cached exceeds prompt tokens", () => {
    const cost = computeCostUsd(spec, usage({ promptTokens: 100, cachedPromptTokens: 500 }));
    expect(cost).toBeGreaterThanOrEqual(0);
  });

  it("retains sub-cent precision for small calls", () => {
    const cost = computeCostUsd(spec, usage({ promptTokens: 100, completionTokens: 50 }));
    expect(cost).toBeGreaterThan(0);
    expect(cost).toBeCloseTo((100 / 1e6) * 10 + (50 / 1e6) * 30, 12);
  });
});

describe("worstCaseCostUsd", () => {
  it("assumes the model runs to its full output limit", () => {
    const worst = worstCaseCostUsd(spec, 1_000);
    const expected = (1_000 / 1e6) * 10 + (1_000 / 1e6) * 30;
    expect(worst).toBeCloseTo(expected, 12);
  });

  it("exceeds the cost of a short completion, which is the point of the budget guard", () => {
    expect(worstCaseCostUsd(spec, 1_000)).toBeGreaterThan(
      computeCostUsd(spec, usage({ promptTokens: 1_000, completionTokens: 10 })),
    );
  });
});

describe("ModelRegistry", () => {
  it("orders by blended cost so `cheap` mode has a natural first choice", () => {
    const registry = new ModelRegistry();
    const ordered = registry.listByCost();
    const costs = ordered.map(blendedCostPerMTok);
    expect(costs).toEqual([...costs].sort((a, b) => a - b));
  });

  it("filters out models lacking a required capability", () => {
    const registry = new ModelRegistry([
      spec,
      { ...spec, modelId: "test/no-tools", capabilities: { ...spec.capabilities, tools: false } },
    ]);
    const withTools = registry.list({ requiresTools: true });
    expect(withTools.map((m) => m.modelId)).toEqual(["test/model"]);
  });

  it("filters by context window", () => {
    const registry = new ModelRegistry([spec]);
    expect(registry.list({ minContextWindow: 200_000 })).toHaveLength(0);
    expect(registry.list({ minContextWindow: 50_000 })).toHaveLength(1);
  });

  it("supports price correction without a code release", () => {
    const registry = new ModelRegistry([spec]);
    registry.override("test/model", { inputCostPerMTok: 5 });
    expect(registry.require("test/model").inputCostPerMTok).toBe(5);
    // Untouched fields survive the patch.
    expect(registry.require("test/model").outputCostPerMTok).toBe(30);
  });

  it("accepts models registered at runtime, which is how new arms enter the pool", () => {
    const registry = new ModelRegistry([]);
    expect(registry.has("test/model")).toBe(false);
    registry.register(spec);
    expect(registry.has("test/model")).toBe(true);
  });

  it("throws a named error for unknown models rather than returning undefined silently", () => {
    const registry = new ModelRegistry([]);
    expect(() => registry.require("nope")).toThrow(/Unknown model: nope/);
  });
});
