import { ModelRegistry, defaultRegistry } from "@orchestrator/shared";
import { openDatabase } from "@orchestrator/telemetry";
import { describe, expect, it } from "vitest";
import { applyToRegistry, checkPricing } from "./guardrail.js";
import { mappedTaskTypes, mappingFor, percentileRank } from "./mapping.js";
import { deriveCapabilities, derivePriors } from "./priors.js";
import { CatalogService } from "./service.js";
import { loadBenchmarkFile, parseBenchmarkFile, unverifiedScores } from "./sources/benchmarks.js";
import { normalizeOpenRouter, perTokenToPerMillion } from "./sources/openrouter.js";
import { SqliteCatalogStore } from "./store/catalog-store.js";

const OPENROUTER_FIXTURE = [
  {
    id: "openai/gpt-4o-mini",
    name: "GPT-4o mini",
    context_length: 128_000,
    pricing: { prompt: "0.00000015", completion: "0.0000006" },
    top_provider: { max_completion_tokens: 16_384 },
    architecture: { input_modalities: ["text", "image"] },
    supported_parameters: ["tools", "temperature"],
  },
  {
    id: "anthropic/claude-opus-5",
    name: "Claude Opus 5",
    context_length: 200_000,
    pricing: { prompt: "0.000015", completion: "0.000075" },
    top_provider: { max_completion_tokens: 64_000 },
    architecture: { input_modalities: ["text", "image"] },
    supported_parameters: ["tools"],
  },
  {
    id: "meta-llama/llama-3.3-70b",
    name: "Llama 3.3 70B",
    context_length: 131_072,
    pricing: { prompt: "0.00000012", completion: "0.0000003" },
    architecture: { input_modalities: ["text"] },
    supported_parameters: ["tools"],
  },
];

describe("OpenRouter normalization", () => {
  it("converts per-token dollars into per-million, matching ModelSpec units", () => {
    const [entry] = normalizeOpenRouter(OPENROUTER_FIXTURE, 1_000);
    expect(entry?.inputCostPerMTok).toBeCloseTo(0.15, 10);
    expect(entry?.outputCostPerMTok).toBeCloseTo(0.6, 10);
  });

  it("stamps provenance on every entry", () => {
    // A price with no source and no date cannot be audited or expired.
    const entries = normalizeOpenRouter(OPENROUTER_FIXTURE, 1_234, "https://example/models");
    for (const entry of entries) {
      expect(entry.provenance.source).toBe("openrouter");
      expect(entry.provenance.asOf).toBe(1_234);
      expect(entry.provenance.sourceUrl).toBe("https://example/models");
    }
  });

  it("carries capabilities and context windows through", () => {
    const [mini] = normalizeOpenRouter(OPENROUTER_FIXTURE, 0);
    expect(mini?.contextWindow).toBe(128_000);
    expect(mini?.maxOutputTokens).toBe(16_384);
    expect(mini?.supportsTools).toBe(true);
    expect(mini?.supportsVision).toBe(true);
  });

  it("keeps models with no local adapter as catalog-only", () => {
    // Knowing about Llama is useful; pretending we can call it is not.
    const llama = normalizeOpenRouter(OPENROUTER_FIXTURE, 0).find((e) =>
      e.modelId.startsWith("meta-llama/"),
    );
    expect(llama).toBeDefined();
    expect(llama?.provider).toBeNull();
  });

  describe("price parsing", () => {
    it("distinguishes absent from free", () => {
      // Both differ from a parse failure, and conflating them is how a model comes to look free.
      expect(perTokenToPerMillion(undefined)).toBeNull();
      expect(perTokenToPerMillion("")).toBeNull();
      expect(perTokenToPerMillion("0")).toBe(0);
    });

    it("returns null for anything unparseable rather than zero", () => {
      expect(perTokenToPerMillion("not-a-number")).toBeNull();
      expect(perTokenToPerMillion("-1")).toBeNull();
    });

    it("skips entries with no id", () => {
      expect(normalizeOpenRouter([{ name: "nameless" }], 0)).toHaveLength(0);
    });
  });
});

describe("benchmark file", () => {
  it("loads and validates the shipped file", () => {
    const file = loadBenchmarkFile();
    expect(file.benchmarks.length).toBeGreaterThan(0);
    expect(file.scores.length).toBeGreaterThan(0);
  });

  it("flags the shipped scores as unverified placeholders", () => {
    // The pipeline is trustworthy; the data shipped with it is not, and the system says so rather
    // than letting made-up numbers quietly steer routing.
    const file = loadBenchmarkFile();
    expect(unverifiedScores(file)).toBe(file.scores.length);
  });

  it("rejects a score referencing an undeclared benchmark", () => {
    expect(() =>
      parseBenchmarkFile({
        version: 1,
        benchmarks: [{ id: "known" }],
        scores: [
          {
            benchmarkId: "typo",
            modelId: "m",
            score: 1,
            provenance: { source: "s", asOf: 0 },
          },
        ],
      }),
    ).toThrow(/undefined benchmark/);
  });

  it("rejects a score with no provenance", () => {
    expect(() =>
      parseBenchmarkFile({
        version: 1,
        benchmarks: [{ id: "b" }],
        scores: [{ benchmarkId: "b", modelId: "m", score: 1, provenance: { source: "", asOf: 0 } }],
      }),
    ).toThrow(/provenance/);
  });
});

describe("benchmark-to-task mapping", () => {
  it("abstains on task types with no defensible benchmark", () => {
    // The central honesty rule, same as packages/quality: a fabricated signal is worse than none,
    // because the bandit cannot tell the difference.
    expect(mappingFor("extraction")).toBeUndefined();
    expect(mappingFor("summarization")).toBeUndefined();
    expect(mappingFor("classification")).toBeUndefined();
  });

  it("maps the task types it can defend", () => {
    expect(mappedTaskTypes().sort()).toEqual(["code", "creative", "general", "reasoning"]);
    expect(mappingFor("code")?.some((m) => m.benchmarkId === "swe-bench-verified")).toBe(true);
  });
});

describe("percentileRank", () => {
  it("ranks within a population", () => {
    expect(percentileRank(10, [1, 5, 10, 20])).toBeCloseTo(0.625, 6);
    expect(percentileRank(100, [1, 5, 10])).toBe(1);
    expect(percentileRank(0, [1, 5, 10])).toBe(0);
  });

  it("gives tied scores identical ranks", () => {
    expect(percentileRank(5, [5, 5, 5, 5])).toBe(0.5);
  });

  it("handles degenerate populations", () => {
    expect(percentileRank(1, [])).toBe(0);
    expect(percentileRank(1, [1])).toBe(0.5);
  });
});

describe("capability derivation", () => {
  const scores = loadBenchmarkFile().scores;

  it("produces capabilities only for mapped task types", () => {
    const taskTypes = new Set(deriveCapabilities(scores).map((c) => c.taskType));
    expect(taskTypes.has("code")).toBe(true);
    expect(taskTypes.has("extraction")).toBe(false);
  });

  it("ranks a strong coding model above a weak one", () => {
    const capabilities = deriveCapabilities(scores);
    const opus = capabilities.find(
      (c) => c.modelId === "anthropic/claude-opus-5" && c.taskType === "code",
    );
    const mini = capabilities.find(
      (c) => c.modelId === "openai/gpt-4o-mini" && c.taskType === "code",
    );

    expect(opus?.capability).toBeGreaterThan(mini?.capability as number);
  });

  it("records which benchmarks contributed, so a surprise can be traced", () => {
    const capability = deriveCapabilities(scores).find((c) => c.taskType === "code");
    expect(capability?.contributingBenchmarks).toContain("swe-bench-verified");
  });

  it("says nothing about a model with no benchmark data", () => {
    const capabilities = deriveCapabilities([
      {
        benchmarkId: "arena-elo",
        modelId: "known/model",
        score: 1300,
        provenance: { source: "s", asOf: 0 },
      },
    ]);
    expect(capabilities.every((c) => c.modelId === "known/model")).toBe(true);
  });

  it("reports reduced coverage when only some mapped benchmarks are present", () => {
    // `code` maps to swe-bench (0.7) and mmlu-pro (0.3); supplying only mmlu-pro is 30% coverage.
    const partial = deriveCapabilities([
      {
        benchmarkId: "mmlu-pro",
        modelId: "m",
        score: 80,
        provenance: { source: "s", asOf: 0 },
      },
    ]).find((c) => c.taskType === "code");

    expect(partial?.coverage).toBeCloseTo(0.3, 6);
  });
});

describe("prior derivation", () => {
  const capabilities = deriveCapabilities(loadBenchmarkFile().scores);

  it("produces a prior per route mode, not one averaged across them", () => {
    // "Excellent at code" is three different claims depending on how cost is weighted.
    const priors = derivePriors(capabilities, defaultRegistry).filter(
      (p) => p.modelId === "anthropic/claude-opus-5" && p.taskType === "code",
    );

    expect(priors.map((p) => p.routeMode).sort()).toEqual(["balanced", "best", "cheap"]);
  });

  it("rates an expensive strong model well in best mode and poorly in cheap mode", () => {
    const priors = derivePriors(capabilities, defaultRegistry);
    const opus = (mode: string) =>
      priors.find(
        (p) =>
          p.modelId === "anthropic/claude-opus-5" && p.taskType === "code" && p.routeMode === mode,
      );

    expect(opus("best")?.reward).toBeGreaterThan(opus("cheap")?.reward as number);
  });

  it("keeps priors on the same 0..1 scale as real rewards", () => {
    // Computed with the real reward function, so a prior is directly comparable to live outcomes.
    for (const prior of derivePriors(capabilities, defaultRegistry)) {
      expect(prior.reward).toBeGreaterThanOrEqual(0);
      expect(prior.reward).toBeLessThanOrEqual(1);
    }
  });

  it("weakens the prior when benchmark coverage is thin", () => {
    const thin = deriveCapabilities([
      {
        benchmarkId: "mmlu-pro",
        modelId: "openai/gpt-4o",
        score: 75,
        provenance: { source: "s", asOf: 0 },
      },
    ]);
    const full = capabilities.filter((c) => c.modelId === "openai/gpt-4o" && c.taskType === "code");

    const thinPrior = derivePriors(thin, defaultRegistry).find((p) => p.taskType === "code");
    const fullPrior = derivePriors(full, defaultRegistry).find((p) => p.taskType === "code");

    expect(thinPrior?.weight).toBeLessThan(fullPrior?.weight as number);
  });

  it("skips models the registry cannot reach", () => {
    // A prior for an uncallable model is dead weight in the bandit's state.
    const priors = derivePriors(
      deriveCapabilities([
        {
          benchmarkId: "arena-elo",
          modelId: "meta-llama/llama-3.3-70b",
          score: 1300,
          provenance: { source: "s", asOf: 0 },
        },
      ]),
      defaultRegistry,
    );
    expect(priors).toHaveLength(0);
  });
});

describe("pricing guardrail", () => {
  const registry = () => new ModelRegistry();

  const entry = (modelId: string, input: number | null, output: number | null) => ({
    modelId,
    provider: null,
    sourceModelId: modelId,
    displayName: modelId,
    inputCostPerMTok: input,
    outputCostPerMTok: output,
    contextWindow: null,
    maxOutputTokens: null,
    supportsTools: null,
    supportsVision: null,
    provenance: { source: "test", asOf: 0 },
  });

  it("rejects a zero price for a model known to cost money", () => {
    // The failure this exists to stop: a misparsed field makes a model look free and win every
    // budget-constrained route.
    const report = checkPricing([entry("openai/gpt-4o", 0, 10)], registry());
    expect(report.rejected[0]?.reason).toMatch(/parse failure/);
  });

  it("flags a tenfold swing without rejecting it", () => {
    const report = checkPricing([entry("openai/gpt-4o", 100, 10)], registry());
    expect(report.suspicious).toHaveLength(1);
    expect(report.rejected).toHaveLength(0);
  });

  it("passes an ordinary price cut through as routine", () => {
    const report = checkPricing([entry("openai/gpt-4o", 2.0, 8.0)], registry());
    expect(report.routine.length).toBeGreaterThan(0);
    expect(report.suspicious).toHaveLength(0);
  });

  it("lists catalog models the registry does not know", () => {
    const report = checkPricing([entry("meta-llama/llama-3.3-70b", 0.12, 0.3)], registry());
    expect(report.unknownModels).toEqual(["meta-llama/llama-3.3-70b"]);
  });

  it("treats an absent price as no claim, leaving the existing value alone", () => {
    const reg = registry();
    const before = reg.require("openai/gpt-4o").inputCostPerMTok;

    const entries = [entry("openai/gpt-4o", null, null)];
    applyToRegistry(entries, reg, checkPricing(entries, reg));

    expect(reg.require("openai/gpt-4o").inputCostPerMTok).toBe(before);
  });

  it("does not apply a suspicious change unless explicitly accepted", () => {
    const reg = registry();
    const before = reg.require("openai/gpt-4o").inputCostPerMTok;
    const entries = [entry("openai/gpt-4o", 100, 10)];

    const blocked = applyToRegistry(entries, reg, checkPricing(entries, reg));
    expect(blocked.skipped).toContain("openai/gpt-4o");
    expect(reg.require("openai/gpt-4o").inputCostPerMTok).toBe(before);

    applyToRegistry(entries, reg, checkPricing(entries, reg), { acceptSuspicious: true });
    expect(reg.require("openai/gpt-4o").inputCostPerMTok).toBe(100);
  });
});

describe("CatalogService", () => {
  const fixtureFetch = (async () =>
    new Response(JSON.stringify({ data: OPENROUTER_FIXTURE }), {
      status: 200,
      headers: { "content-type": "application/json" },
    })) as typeof globalThis.fetch;

  const build = () => {
    const store = new SqliteCatalogStore(openDatabase(":memory:"));
    const registry = new ModelRegistry();
    const service = new CatalogService({
      store,
      registry,
      fetchImpl: fixtureFetch,
      now: () => 5_000,
    });
    return { store, registry, service };
  };

  it("writes an unapplied snapshot and reports what applying would change", async () => {
    // Fetching must never move routing on its own; pricing feeds cost, reward, and budget filters.
    const { registry, service } = build();
    const before = registry.require("openai/gpt-4o-mini").inputCostPerMTok;

    const result = await service.refresh();
    expect(result.snapshot.version).toBe(1);
    expect(registry.require("openai/gpt-4o-mini").inputCostPerMTok).toBe(before);
    expect(service.status().version).toBeNull();
  });

  it("surfaces unverified benchmark data on refresh", async () => {
    const { service } = build();
    expect((await service.refresh()).unverifiedBenchmarkScores).toBeGreaterThan(0);
  });

  it("applies a snapshot on request", async () => {
    const { registry, service } = build();
    const { snapshot } = await service.refresh();

    service.apply(snapshot.version);
    expect(registry.require("anthropic/claude-opus-5").contextWindow).toBe(200_000);
    expect(service.status().version).toBe(1);
  });

  it("reloads the applied snapshot at boot", async () => {
    const { store, service } = build();
    const { snapshot } = await service.refresh();
    service.apply(snapshot.version);

    const freshRegistry = new ModelRegistry();
    const rebooted = new CatalogService({ store, registry: freshRegistry, now: () => 6_000 });
    expect(rebooted.applyStored()?.updated.length).toBeGreaterThan(0);
  });

  it("produces router priors with a caller-supplied feature vector", async () => {
    const { service } = build();
    const { snapshot } = await service.refresh();
    service.apply(snapshot.version);

    const priors = service.routerPriors(() => [1, 0, 0]);
    expect(priors.length).toBeGreaterThan(0);
    expect(priors[0]?.features).toEqual([1, 0, 0]);
    expect(priors[0]?.source).toMatch(/^benchmarks:/);
  });

  it("reports nothing believed before anything is applied", () => {
    const { service } = build();
    expect(service.status()).toMatchObject({ version: null, models: 0, stale: false });
  });

  it("flags a stale snapshot", async () => {
    const store = new SqliteCatalogStore(openDatabase(":memory:"));
    const registry = new ModelRegistry();
    const service = new CatalogService({ store, registry, fetchImpl: fixtureFetch, now: () => 0 });

    const { snapshot } = await service.refresh();
    service.apply(snapshot.version);

    const later = new CatalogService({
      store,
      registry,
      now: () => 90 * 24 * 60 * 60 * 1_000,
    });
    expect(later.status().stale).toBe(true);
  });

  it("rejects applying a version that does not exist", () => {
    const { service } = build();
    expect(() => service.apply(99)).toThrow(/Unknown catalog version/);
  });
});
