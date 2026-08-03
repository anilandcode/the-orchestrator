# packages/router

Model selection. Decides *what* to call; never performs a call itself.

## Owns

- `src/router.ts` — the `Router` interface (`select` / `observe`)
- `src/candidates.ts` — capability, context, and budget filtering shared by both routers
- `src/static-router.ts` + `src/fallback.ts` — the deterministic baseline
- `src/features.ts` — the context vector the bandit conditions on
- `src/bandit/` — LinUCB, Thompson sampling, and persisted arm state
- `src/adaptive-router.ts` — the bandit wrapped in its safeguards

## Rules

- **Never import `@orchestrator/gateway`.** The router returns a decision; something else executes it.
- **`select` must be cheap and synchronous.** It runs ahead of every model call, on the request path.
- **The static router is not deprecated.** It is the baseline the bandit must beat, the cold-start
  fallback, and what actually executes in `shadow` mode. Keep it working.
- **`ROUTER_MODE` defaults to `shadow`.** Changing that default is a product decision backed by replay
  evidence, not a code cleanup.
- **A budget ceiling (`maxCostUsd`) is a hard filter.** Silently exceeding a stated spend limit is
  worse than refusing the request. Latency is best-effort by contrast — we cannot know it in advance.
- Arm state is versioned. A dimension change must reset the state rather than reinterpret old numbers
  against a new feature layout.
