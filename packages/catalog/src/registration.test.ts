import { ModelRegistry } from "@orchestrator/shared";
import { describe, expect, it } from "vitest";
import { deriveTier, matchesAllowlist, registerFromCatalog, toModelSpec } from "./registration.js";
import type { CatalogEntry } from "./schema.js";

const entry = (overrides: Partial<CatalogEntry> = {}): CatalogEntry => ({
  modelId: "vendor/model",
  provider: null,
  sourceModelId: "vendor/model",
  displayName: "Model",
  inputCostPerMTok: 1,
  outputCostPerMTok: 3,
  contextWindow: 128_000,
  maxOutputTokens: 8_192,
  supportsTools: true,
  supportsVision: false,
  provenance: { source: "openrouter", asOf: 0 },
  ...overrides,
});

/** A catalog the size of the real one, to prove size alone changes nothing. */
const largeCatalog = Array.from({ length: 300 }, (_, i) =>
  entry({ modelId: `vendor/model-${i}`, sourceModelId: `vendor/model-${i}` }),
);

describe("registerFromCatalog", () => {
  describe("the arm-count discipline", () => {
    it("registers only what the allowlist names, however large the catalog", () => {
      // The finding this file exists to enforce: LinUCB regret grows with sqrt(arms × time), so
      // 300 arms against a 2.10% headroom would spend far more on exploration than perfect routing
      // could return. Catalog size must never decide arm count.
      const registry = new ModelRegistry([]);
      const report = registerFromCatalog(registry, largeCatalog, {
        allow: ["vendor/model-1", "vendor/model-2", "vendor/model-3"],
      });

      expect(report.registered).toHaveLength(3);
      expect(registry.list()).toHaveLength(3);
    });

    it("registers nothing when the allowlist is empty", () => {
      const registry = new ModelRegistry([]);
      registerFromCatalog(registry, largeCatalog, { allow: [] });
      expect(registry.list()).toHaveLength(0);
    });

    it("supports a wildcard, which is how someone would opt into the bad outcome", () => {
      // Deliberately possible and deliberately explicit: nobody reaches 300 arms by accident.
      const registry = new ModelRegistry([]);
      registerFromCatalog(registry, largeCatalog, { allow: ["*"] });
      expect(registry.list()).toHaveLength(300);
    });

    it("supports vendor patterns", () => {
      const registry = new ModelRegistry([]);
      registerFromCatalog(
        registry,
        [entry({ modelId: "openai/a" }), entry({ modelId: "meta/b" })],
        { allow: ["openai/*"] },
      );
      expect(registry.list().map((m) => m.modelId)).toEqual(["openai/a"]);
    });
  });

  describe("incomplete data is skipped, never defaulted", () => {
    it("skips an entry with no input price", () => {
      // An assumed price corrupts the reward's cost term, budget filtering, and cheap-mode ordering
      // simultaneously — and a model assumed free wins every budget-constrained route.
      const registry = new ModelRegistry([]);
      const report = registerFromCatalog(registry, [entry({ inputCostPerMTok: null })], {
        allow: ["*"],
      });

      expect(report.registered).toHaveLength(0);
      expect(report.skipped[0]?.reason).toMatch(/incomplete pricing/);
    });

    it("skips an entry with no output price", () => {
      const registry = new ModelRegistry([]);
      registerFromCatalog(registry, [entry({ outputCostPerMTok: null })], { allow: ["*"] });
      expect(registry.list()).toHaveLength(0);
    });

    it("skips an entry with no context window", () => {
      const registry = new ModelRegistry([]);
      const report = registerFromCatalog(registry, [entry({ contextWindow: null })], {
        allow: ["*"],
      });
      expect(report.skipped[0]?.reason).toMatch(/context window/);
    });

    it("accepts a genuinely free model, which is different from a missing price", () => {
      const registry = new ModelRegistry([]);
      registerFromCatalog(
        registry,
        [entry({ modelId: "vendor/free", inputCostPerMTok: 0, outputCostPerMTok: 0 })],
        { allow: ["*"] },
      );
      expect(registry.has("vendor/free")).toBe(true);
    });
  });

  it("never overwrites a hand-maintained spec", () => {
    // Pricing refreshes go through applyToRegistry, which has its own guardrails. This path is for
    // new models only.
    const registry = new ModelRegistry();
    const before = registry.require("openai/gpt-4o-mini").inputCostPerMTok;

    const report = registerFromCatalog(
      registry,
      [entry({ modelId: "openai/gpt-4o-mini", inputCostPerMTok: 999 })],
      { allow: ["*"] },
    );

    expect(report.skipped[0]?.reason).toBe("already registered");
    expect(registry.require("openai/gpt-4o-mini").inputCostPerMTok).toBe(before);
  });

  it("reports what it did, so a boot log can show it", () => {
    const registry = new ModelRegistry([]);
    const report = registerFromCatalog(
      registry,
      [entry({ modelId: "a" }), entry({ modelId: "b", inputCostPerMTok: null })],
      { allow: ["*"] },
    );

    expect(report.registered).toEqual(["a"]);
    expect(report.skipped.map((s) => s.modelId)).toEqual(["b"]);
  });
});

describe("toModelSpec", () => {
  it("dispatches through the configured provider, not the vendor", () => {
    // Which vendor built the model is in its id; `provider` is who we send the HTTP request to.
    const spec = toModelSpec(entry({ modelId: "anthropic/claude-x" }), {
      allow: ["*"],
      provider: "openrouter",
    });

    expect(spec.provider).toBe("openrouter");
    expect(spec.providerModel).toBe("vendor/model");
  });

  it("treats unknown capabilities as present rather than absent", () => {
    // `null` means the source said nothing. Reading that as `false` would quietly exclude a capable
    // model from every tool-using request.
    const spec = toModelSpec(entry({ supportsTools: null }), { allow: ["*"] });
    expect(spec.capabilities.tools).toBe(true);
  });

  it("respects an explicit capability denial", () => {
    const spec = toModelSpec(entry({ supportsTools: false }), { allow: ["*"] });
    expect(spec.capabilities.tools).toBe(false);
  });

  it("assigns a latency placeholder that telemetry will supersede", () => {
    const spec = toModelSpec(entry(), { allow: ["*"] });
    expect(spec.typicalLatencyMs).toBeGreaterThan(0);
  });

  it("falls back on a missing output limit", () => {
    const spec = toModelSpec(entry({ maxOutputTokens: null }), {
      allow: ["*"],
      maxOutputTokensFallback: 2_048,
    });
    expect(spec.maxOutputTokens).toBe(2_048);
  });
});

describe("deriveTier", () => {
  it("is monotonic in price", () => {
    const tiers = [deriveTier(0.1, 0.3), deriveTier(3, 15), deriveTier(15, 75)];
    expect(tiers).toEqual(["economy", "standard", "premium"]);
  });

  it("treats a free model as economy", () => {
    expect(deriveTier(0, 0)).toBe("economy");
  });

  it("weights output price, which dominates real chat spend", () => {
    // Same input price, very different output price: the expensive one must not read as economy.
    expect(deriveTier(0.5, 60)).not.toBe("economy");
  });
});

describe("matchesAllowlist", () => {
  it("matches exact ids and wildcards", () => {
    expect(matchesAllowlist("openai/gpt-4o", ["openai/gpt-4o"])).toBe(true);
    expect(matchesAllowlist("openai/gpt-4o", ["openai/*"])).toBe(true);
    expect(matchesAllowlist("openai/gpt-4o", ["*"])).toBe(true);
    expect(matchesAllowlist("meta/llama", ["openai/*"])).toBe(false);
  });

  it("does not let regex metacharacters widen a pattern", () => {
    expect(matchesAllowlist("vendor/axb", ["vendor/a.b"])).toBe(false);
    expect(matchesAllowlist("vendor/a.b", ["vendor/a.b"])).toBe(true);
  });
});
