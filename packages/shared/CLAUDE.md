# packages/shared

The contract layer. **Imports nothing else from this repo** — if this package needs something from
another package, the dependency is pointing the wrong way.

## Owns

- `src/schemas/` — the wire contracts (`UnifiedChatRequest/Response/Chunk`, `CallEvent`, `RoutingDecision`)
- `src/errors.ts` — the normalized error taxonomy and its retry/fallback policy
- `src/models.ts` — the model registry: pricing, context windows, capabilities
- `src/clock.ts`, `src/ids.ts` — injectable time and id sources so tests stay deterministic

## Rules

- **Pricing in `models.ts` is configuration, not truth.** Verify against provider pricing pages before
  billing anyone. `ModelRegistry.override()` exists so deployments can correct it without a release.
- Changing a schema field is a cross-layer breaking change. The gateway writes `CallEvent`s, the router
  reads them, and the replay harness reads historical ones — a rename silently invalidates stored data.
- Error policy (`retryable` / `fallbackEligible`) lives here and only here. Both the gateway's retry loop
  and the router's fallback chain read it.
