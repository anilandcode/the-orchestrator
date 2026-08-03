import type { ProviderId } from "@orchestrator/shared";
import { type CatalogEntry, CatalogEntrySchema } from "../schema.js";

/**
 * The OpenRouter public model catalog.
 *
 * Chosen because it needs no authentication, covers several hundred models across providers, and
 * publishes live pricing — which is what lets this system stop carrying a hand-written price table
 * that nobody verified.
 *
 * It is still a third party. Prices arrive as strings in dollars *per token*, occasionally as "0"
 * for free tiers, and occasionally absent. Everything below is defensive for that reason.
 */

const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";

interface OpenRouterModel {
  id?: string;
  name?: string;
  context_length?: number;
  pricing?: { prompt?: string; completion?: string };
  top_provider?: { max_completion_tokens?: number | null };
  architecture?: { input_modalities?: string[] };
  supported_parameters?: string[];
}

export interface OpenRouterSourceConfig {
  fetchImpl?: typeof globalThis.fetch;
  url?: string;
  now?: () => number;
}

export async function fetchOpenRouterCatalog(
  config: OpenRouterSourceConfig = {},
): Promise<CatalogEntry[]> {
  const fetchImpl = config.fetchImpl ?? globalThis.fetch;
  const url = config.url ?? OPENROUTER_MODELS_URL;

  const response = await fetchImpl(url, { headers: { accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`OpenRouter catalog fetch failed (${response.status})`);
  }

  const body = (await response.json()) as { data?: OpenRouterModel[] };
  return normalizeOpenRouter(body.data ?? [], config.now?.() ?? Date.now(), url);
}

export function normalizeOpenRouter(
  models: OpenRouterModel[],
  asOf: number,
  sourceUrl = OPENROUTER_MODELS_URL,
): CatalogEntry[] {
  const entries: CatalogEntry[] = [];

  for (const model of models) {
    if (!model.id) continue;

    const modalities = model.architecture?.input_modalities ?? [];
    const parameters = model.supported_parameters ?? [];

    entries.push(
      CatalogEntrySchema.parse({
        // OpenRouter already uses `vendor/model`, which is the shape our registry uses.
        modelId: model.id,
        provider: toProviderId(model.id),
        sourceModelId: model.id,
        displayName: model.name ?? model.id,
        inputCostPerMTok: perTokenToPerMillion(model.pricing?.prompt),
        outputCostPerMTok: perTokenToPerMillion(model.pricing?.completion),
        contextWindow:
          model.context_length && model.context_length > 0 ? model.context_length : null,
        maxOutputTokens: model.top_provider?.max_completion_tokens ?? null,
        supportsTools: parameters.length > 0 ? parameters.includes("tools") : null,
        supportsVision: modalities.length > 0 ? modalities.includes("image") : null,
        provenance: { source: "openrouter", sourceUrl, asOf },
      }),
    );
  }

  return entries;
}

/**
 * OpenRouter quotes dollars per token as a string. Converting to per-million keeps the units
 * consistent with `ModelSpec`, where every other price already lives.
 *
 * A missing price and a genuinely free model are different things, and both differ from a parse
 * failure. Anything unparseable becomes `null` — "we do not know" — rather than `0`, which would
 * make a model look free and win every budget-constrained route.
 */
export function perTokenToPerMillion(raw: string | undefined): number | null {
  if (raw === undefined || raw === null || raw === "") return null;

  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) return null;

  return value * 1_000_000;
}

/** Only providers this system has an adapter for get a provider id; the rest stay catalog-only. */
function toProviderId(modelId: string): ProviderId | null {
  const vendor = modelId.split("/")[0];
  if (vendor === "openai") return "openai";
  if (vendor === "anthropic") return "anthropic";
  return null;
}
