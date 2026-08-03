import type { ProviderId, Usage } from "./schemas/chat.js";

/**
 * The model registry.
 *
 * This table is the single source of truth for cost. A provider response's own cost field (where one
 * exists) is never trusted — usage tokens are reported, dollars are computed here. That keeps billing
 * consistent across providers and keeps the router's cost term comparable between arms.
 *
 * PRICING IS CONFIGURATION, NOT TRUTH. Verify these numbers against the provider pricing pages before
 * charging anyone. `ModelRegistry.override()` exists so a deployment can correct a price without
 * shipping a release.
 */

export type ModelTier = "economy" | "standard" | "premium";

export interface ModelCapabilities {
  tools: boolean;
  vision: boolean;
  streaming: boolean;
  jsonMode: boolean;
}

export interface ModelSpec {
  /** Stable internal id. Used in routing decisions and stored events — do not rename casually. */
  modelId: string;
  provider: ProviderId;
  /** The name actually sent on the wire, which providers version independently of our id. */
  providerModel: string;

  inputCostPerMTok: number;
  outputCostPerMTok: number;
  /** Cache-read price. Falls back to the full input price when a provider has no cache discount. */
  cachedInputCostPerMTok?: number;

  contextWindow: number;
  maxOutputTokens: number;
  tier: ModelTier;
  capabilities: ModelCapabilities;

  /**
   * Rough expected end-to-end latency for a short completion, in ms. Used only to seed cold-start
   * priors and to enforce `maxLatencyMs` filters — measured latency from telemetry supersedes it as
   * soon as any real data exists.
   */
  typicalLatencyMs: number;
}

const ALL_CAPABILITIES: ModelCapabilities = {
  tools: true,
  vision: true,
  streaming: true,
  jsonMode: true,
};

export const DEFAULT_MODELS: readonly ModelSpec[] = Object.freeze([
  // --- OpenAI -------------------------------------------------------------
  {
    modelId: "openai/gpt-4o-mini",
    provider: "openai",
    providerModel: "gpt-4o-mini",
    inputCostPerMTok: 0.15,
    outputCostPerMTok: 0.6,
    cachedInputCostPerMTok: 0.075,
    contextWindow: 128_000,
    maxOutputTokens: 16_384,
    tier: "economy",
    capabilities: ALL_CAPABILITIES,
    typicalLatencyMs: 900,
  },
  {
    modelId: "openai/gpt-4.1",
    provider: "openai",
    providerModel: "gpt-4.1",
    inputCostPerMTok: 2.0,
    outputCostPerMTok: 8.0,
    cachedInputCostPerMTok: 0.5,
    contextWindow: 1_047_576,
    maxOutputTokens: 32_768,
    tier: "standard",
    capabilities: ALL_CAPABILITIES,
    typicalLatencyMs: 1_800,
  },
  {
    modelId: "openai/gpt-4o",
    provider: "openai",
    providerModel: "gpt-4o",
    inputCostPerMTok: 2.5,
    outputCostPerMTok: 10.0,
    cachedInputCostPerMTok: 1.25,
    contextWindow: 128_000,
    maxOutputTokens: 16_384,
    tier: "standard",
    capabilities: ALL_CAPABILITIES,
    typicalLatencyMs: 2_000,
  },

  // --- Anthropic ----------------------------------------------------------
  {
    modelId: "anthropic/claude-haiku-4-5",
    provider: "anthropic",
    providerModel: "claude-haiku-4-5-20251001",
    inputCostPerMTok: 1.0,
    outputCostPerMTok: 5.0,
    cachedInputCostPerMTok: 0.1,
    contextWindow: 200_000,
    maxOutputTokens: 64_000,
    tier: "economy",
    capabilities: ALL_CAPABILITIES,
    typicalLatencyMs: 1_100,
  },
  {
    modelId: "anthropic/claude-sonnet-5",
    provider: "anthropic",
    providerModel: "claude-sonnet-5",
    inputCostPerMTok: 3.0,
    outputCostPerMTok: 15.0,
    cachedInputCostPerMTok: 0.3,
    contextWindow: 200_000,
    maxOutputTokens: 64_000,
    tier: "standard",
    capabilities: ALL_CAPABILITIES,
    typicalLatencyMs: 2_200,
  },
  {
    modelId: "anthropic/claude-opus-5",
    provider: "anthropic",
    providerModel: "claude-opus-5",
    inputCostPerMTok: 15.0,
    outputCostPerMTok: 75.0,
    cachedInputCostPerMTok: 1.5,
    contextWindow: 200_000,
    maxOutputTokens: 64_000,
    tier: "premium",
    capabilities: ALL_CAPABILITIES,
    typicalLatencyMs: 4_000,
  },
]);

export interface ModelFilter {
  provider?: ProviderId;
  tier?: ModelTier;
  requiresTools?: boolean;
  requiresVision?: boolean;
  minContextWindow?: number;
}

export class ModelRegistry {
  private readonly models = new Map<string, ModelSpec>();

  constructor(specs: readonly ModelSpec[] = DEFAULT_MODELS) {
    for (const spec of specs) this.models.set(spec.modelId, spec);
  }

  /** Add a model at runtime. The adaptive router treats unseen arms as cold-start, not as errors. */
  register(spec: ModelSpec): void {
    this.models.set(spec.modelId, spec);
  }

  /** Patch fields on an existing model — typically pricing that drifted since release. */
  override(modelId: string, patch: Partial<Omit<ModelSpec, "modelId">>): void {
    const existing = this.require(modelId);
    this.models.set(modelId, { ...existing, ...patch });
  }

  get(modelId: string): ModelSpec | undefined {
    return this.models.get(modelId);
  }

  require(modelId: string): ModelSpec {
    const spec = this.models.get(modelId);
    if (!spec) throw new Error(`Unknown model: ${modelId}`);
    return spec;
  }

  has(modelId: string): boolean {
    return this.models.has(modelId);
  }

  list(filter: ModelFilter = {}): ModelSpec[] {
    return [...this.models.values()].filter((m) => {
      if (filter.provider && m.provider !== filter.provider) return false;
      if (filter.tier && m.tier !== filter.tier) return false;
      if (filter.requiresTools && !m.capabilities.tools) return false;
      if (filter.requiresVision && !m.capabilities.vision) return false;
      if (filter.minContextWindow && m.contextWindow < filter.minContextWindow) return false;
      return true;
    });
  }

  /** Model ids sorted cheapest-first by blended cost. The natural order for `cheap` mode. */
  listByCost(filter: ModelFilter = {}): ModelSpec[] {
    return this.list(filter).sort((a, b) => blendedCostPerMTok(a) - blendedCostPerMTok(b));
  }
}

/**
 * A single comparable price per model, assuming a 3:1 input:output ratio — roughly what chat traffic
 * looks like. Used for ordering and for budget filtering, not for billing.
 */
export function blendedCostPerMTok(spec: ModelSpec): number {
  return spec.inputCostPerMTok * 0.75 + spec.outputCostPerMTok * 0.25;
}

/** Dollars for one call. The only place token counts turn into money. */
export function computeCostUsd(spec: ModelSpec, usage: Usage): number {
  const cachedRate = spec.cachedInputCostPerMTok ?? spec.inputCostPerMTok;
  const uncachedPromptTokens = Math.max(0, usage.promptTokens - usage.cachedPromptTokens);

  const cost =
    (uncachedPromptTokens / 1_000_000) * spec.inputCostPerMTok +
    (usage.cachedPromptTokens / 1_000_000) * cachedRate +
    (usage.completionTokens / 1_000_000) * spec.outputCostPerMTok;

  // Sub-cent precision matters: a million cheap calls is where the rounding error lives.
  return Number(cost.toFixed(10));
}

/**
 * Worst-case cost if the model ran to its full output limit. Used to enforce `maxCostUsd` *before*
 * committing to a call, since the actual completion length is unknown at routing time.
 */
export function worstCaseCostUsd(spec: ModelSpec, promptTokens: number): number {
  return computeCostUsd(spec, {
    promptTokens,
    completionTokens: spec.maxOutputTokens,
    totalTokens: promptTokens + spec.maxOutputTokens,
    cachedPromptTokens: 0,
  });
}

export const defaultRegistry = new ModelRegistry();
