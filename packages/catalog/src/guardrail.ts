import type { ModelRegistry } from "@orchestrator/shared";
import type { CatalogEntry } from "./schema.js";

/**
 * Guardrails on ingested pricing.
 *
 * Ingested prices feed `computeCostUsd`, the reward's cost term, `maxCostUsd` budget filtering, and
 * `cheap`-mode ordering. A single misparsed field that lands as `0` would make a model look free and
 * win every budget-constrained route — silently, and in a way that looks like the router working.
 *
 * So an ingested price is not trusted merely because it parsed.
 */
export interface PriceChange {
  modelId: string;
  field: "inputCostPerMTok" | "outputCostPerMTok";
  from: number;
  to: number;
  ratio: number;
}

export interface GuardrailReport {
  /** Changes safe to apply without comment. */
  routine: PriceChange[];
  /** Large swings — plausible but worth a human glance. */
  suspicious: PriceChange[];
  /** Rejected outright; never applied. */
  rejected: { modelId: string; reason: string }[];
  /** Models present in the catalog that the registry does not know. */
  unknownModels: string[];
}

/** A change beyond this multiple in either direction needs confirmation. */
export const DEFAULT_MAX_PRICE_RATIO = 10;

export function checkPricing(
  entries: CatalogEntry[],
  registry: ModelRegistry,
  options: { maxRatio?: number } = {},
): GuardrailReport {
  const maxRatio = options.maxRatio ?? DEFAULT_MAX_PRICE_RATIO;
  const report: GuardrailReport = {
    routine: [],
    suspicious: [],
    rejected: [],
    unknownModels: [],
  };

  for (const entry of entries) {
    const spec = registry.get(entry.modelId);
    if (!spec) {
      report.unknownModels.push(entry.modelId);
      continue;
    }

    for (const field of ["inputCostPerMTok", "outputCostPerMTok"] as const) {
      const incoming = entry[field];
      // Absent is fine — it means the source said nothing, and the existing value stands.
      if (incoming === null) continue;

      // A free model is a real thing, but a price of exactly zero arriving for a model we know
      // charges money is far more likely to be a parse failure than a giveaway.
      if (incoming === 0 && spec[field] > 0) {
        report.rejected.push({
          modelId: entry.modelId,
          reason: `${field} arrived as 0 for a model previously priced at ${spec[field]}; treating as a parse failure rather than a free tier`,
        });
        continue;
      }

      const existing = spec[field];
      if (existing === incoming) continue;

      const ratio = existing === 0 ? Number.POSITIVE_INFINITY : incoming / existing;
      const change: PriceChange = {
        modelId: entry.modelId,
        field,
        from: existing,
        to: incoming,
        ratio,
      };

      // Providers do cut prices. They rarely cut them tenfold, and a tenfold *rise* is almost
      // certainly a unit error.
      if (ratio > maxRatio || ratio < 1 / maxRatio) report.suspicious.push(change);
      else report.routine.push(change);
    }
  }

  return report;
}

/**
 * Apply catalog facts to the live registry.
 *
 * Only fields the source actually reported are written, so a partial catalog never blanks out known
 * good values. Rejected entries are skipped entirely; suspicious ones apply only when the caller has
 * explicitly accepted them.
 */
export function applyToRegistry(
  entries: CatalogEntry[],
  registry: ModelRegistry,
  report: GuardrailReport,
  options: { acceptSuspicious?: boolean } = {},
): { updated: string[]; skipped: string[] } {
  const rejected = new Set(report.rejected.map((entry) => entry.modelId));
  const suspicious = new Set(report.suspicious.map((change) => change.modelId));

  const updated: string[] = [];
  const skipped: string[] = [];

  for (const entry of entries) {
    if (!registry.has(entry.modelId)) continue;

    if (rejected.has(entry.modelId)) {
      skipped.push(entry.modelId);
      continue;
    }
    if (suspicious.has(entry.modelId) && !options.acceptSuspicious) {
      skipped.push(entry.modelId);
      continue;
    }

    const patch: Record<string, unknown> = {};
    if (entry.inputCostPerMTok !== null) patch.inputCostPerMTok = entry.inputCostPerMTok;
    if (entry.outputCostPerMTok !== null) patch.outputCostPerMTok = entry.outputCostPerMTok;
    if (entry.contextWindow !== null) patch.contextWindow = entry.contextWindow;
    if (entry.maxOutputTokens !== null) patch.maxOutputTokens = entry.maxOutputTokens;

    if (Object.keys(patch).length === 0) continue;

    registry.override(entry.modelId, patch);
    updated.push(entry.modelId);
  }

  return { updated, skipped };
}
