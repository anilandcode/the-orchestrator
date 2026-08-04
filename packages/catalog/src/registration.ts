import {
  type ModelRegistry,
  type ModelSpec,
  type ModelTier,
  type ProviderId,
  blendedCostPerMTok,
} from "@orchestrator/shared";
import type { CatalogEntry } from "./schema.js";

/**
 * Turning catalog knowledge into callable models.
 *
 * **Registration is opt-in, and that is the whole point of this file.**
 *
 * The catalog knows about several hundred models. Registering all of them would make every one a
 * bandit arm, and more arms is emphatically not better: LinUCB's regret grows roughly with
 * `sqrt(arms × time)`, so going from 6 arms to 300 multiplies exploration cost by about 7× — against
 * a headroom over static rules measured at 2.10%. Exploration that costs more than perfect routing
 * could ever return is not a trade worth making. It is the same arithmetic that put
 * `explorationFloor` at 0.
 *
 * So the three layers stay distinct:
 *
 *   catalog (~300, knowledge) → registry (curated, callable) → candidates (small, per request)
 *
 * `registerFromCatalog` is the boundary between the first two, and it requires an explicit allowlist.
 */

/** Blended-cost thresholds, in USD per million tokens. */
const ECONOMY_MAX = 1.5;
const PREMIUM_MIN = 12;

/**
 * Latency defaults by tier, in milliseconds.
 *
 * A catalog cannot tell you how fast a model is. These are placeholders that let a model be routable
 * at all, and telemetry supersedes them the moment any real call lands — `typicalLatencyMs` only ever
 * seeds cold-start priors and `maxLatencyMs` filtering, never the reward, which uses measured latency.
 */
const DEFAULT_LATENCY_MS: Record<ModelTier, number> = {
  economy: 1_200,
  standard: 2_500,
  premium: 5_000,
};

export interface RegistrationOptions {
  /**
   * Model ids permitted to become callable. `*` is accepted but should be understood as "make every
   * catalog model a bandit arm", which the note above argues against.
   */
  allow: string[];
  /** The adapter that will serve these models. */
  provider?: ProviderId;
  maxOutputTokensFallback?: number;
}

export interface RegistrationReport {
  registered: string[];
  /** Matched the allowlist but could not be registered, with the reason. */
  skipped: { modelId: string; reason: string }[];
}

/**
 * Register allowlisted catalog entries as callable models.
 *
 * An entry without complete pricing is skipped rather than defaulted. `computeCostUsd` feeds the
 * reward's cost term, `maxCostUsd` budget filtering, and `cheap`-mode ordering — a model with an
 * assumed price would corrupt all three, and a model assumed free would win every budget-constrained
 * route.
 */
export function registerFromCatalog(
  registry: ModelRegistry,
  entries: CatalogEntry[],
  options: RegistrationOptions,
): RegistrationReport {
  const report: RegistrationReport = { registered: [], skipped: [] };

  for (const entry of entries) {
    if (!matchesAllowlist(entry.modelId, options.allow)) continue;

    // Never silently replace a hand-maintained spec with catalog guesses. Pricing refreshes go
    // through `applyToRegistry`, which has its own guardrails; this path is only for new models.
    if (registry.has(entry.modelId)) {
      report.skipped.push({ modelId: entry.modelId, reason: "already registered" });
      continue;
    }

    if (entry.inputCostPerMTok === null || entry.outputCostPerMTok === null) {
      report.skipped.push({
        modelId: entry.modelId,
        reason: "incomplete pricing; cost, reward, and budget filtering would all be wrong",
      });
      continue;
    }

    if (entry.contextWindow === null) {
      report.skipped.push({
        modelId: entry.modelId,
        reason: "no context window; candidate filtering could not tell if a prompt fits",
      });
      continue;
    }

    registry.register(toModelSpec(entry, options));
    report.registered.push(entry.modelId);
  }

  return report;
}

export function toModelSpec(entry: CatalogEntry, options: RegistrationOptions): ModelSpec {
  const inputCostPerMTok = entry.inputCostPerMTok ?? 0;
  const outputCostPerMTok = entry.outputCostPerMTok ?? 0;
  const tier = deriveTier(inputCostPerMTok, outputCostPerMTok);

  return {
    modelId: entry.modelId,
    // The dispatching provider, not the vendor. Which vendor built the model is in its id.
    provider: options.provider ?? entry.provider ?? "openrouter",
    // OpenRouter takes the same `vendor/model` id we key on.
    providerModel: entry.sourceModelId,
    inputCostPerMTok,
    outputCostPerMTok,
    contextWindow: entry.contextWindow ?? 0,
    maxOutputTokens: entry.maxOutputTokens ?? options.maxOutputTokensFallback ?? 4_096,
    tier,
    capabilities: {
      // `null` means the source did not say. Reading unknown as `false` would quietly exclude a
      // capable model from every tool-using request.
      tools: entry.supportsTools ?? true,
      vision: entry.supportsVision ?? false,
      streaming: true,
      jsonMode: true,
    },
    typicalLatencyMs: DEFAULT_LATENCY_MS[tier],
  };
}

/**
 * Tier from price.
 *
 * Price is a *proxy* for capability tier, not a definition of it — a badly-priced model lands in the
 * wrong band, and an overpriced weak model would be treated as premium by `best` mode. That is
 * acceptable because tier only shapes the static baseline's ordering; the bandit learns the real
 * ranking from outcomes and will overrule it.
 */
export function deriveTier(inputCostPerMTok: number, outputCostPerMTok: number): ModelTier {
  const blended = blendedCostPerMTok({
    inputCostPerMTok,
    outputCostPerMTok,
  } as ModelSpec);

  if (blended <= ECONOMY_MAX) return "economy";
  if (blended >= PREMIUM_MIN) return "premium";
  return "standard";
}

/** `*` matches any run of characters, so `openai/*` or a bare `*` both work. */
export function matchesAllowlist(modelId: string, allow: string[]): boolean {
  return allow.some((pattern) => {
    if (pattern === "*") return true;
    const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*");
    return new RegExp(`^${escaped}$`).test(modelId);
  });
}
