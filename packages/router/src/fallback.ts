import type { ModelSpec } from "@orchestrator/shared";
import { byCostAscending } from "./candidates.js";

export const DEFAULT_CHAIN_LENGTH = 2;

/**
 * Orders the fallback chain that follows the primary choice.
 *
 * Priority, in order:
 *   1. **A different provider.** The failures worth failing over from — outages, rate limits — are
 *      usually provider-wide, so a second model at the same vendor often fails the same way.
 *   2. **A larger context window.** `context_length_exceeded` is fallback-eligible precisely because
 *      a bigger model can serve it, so the chain should contain one.
 *   3. **Cheaper.** All else equal, do not make a failure expensive.
 */
export function buildFallbackChain(
  primary: ModelSpec,
  candidates: ModelSpec[],
  maxLength: number = DEFAULT_CHAIN_LENGTH,
): string[] {
  return candidates
    .filter((spec) => spec.modelId !== primary.modelId)
    .sort((a, b) => {
      const providerDiversity =
        Number(b.provider !== primary.provider) - Number(a.provider !== primary.provider);
      if (providerDiversity !== 0) return providerDiversity;

      const contextHeadroom = b.contextWindow - a.contextWindow;
      if (contextHeadroom !== 0) return contextHeadroom;

      return byCostAscending(a, b);
    })
    .slice(0, maxLength)
    .map((spec) => spec.modelId);
}
