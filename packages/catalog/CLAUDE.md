# packages/catalog

What the system knows about models before it has tried them: pricing, context windows, and public
benchmark results, turned into router priors.

## Owns

- `src/sources/` — ingestion (OpenRouter catalog, curated benchmark file)
- `src/data/benchmarks.json` — the curated scores, reviewed in diffs
- `src/mapping.ts` — benchmark → task-type weights, the judgment layer
- `src/priors.ts` — benchmark scores → `ModelPrior`, via the real reward function
- `src/store/` — versioned persistence with provenance

## Rules

- **Every claim carries a source and a date.** A benchmark number with no provenance cannot be
  audited, and "why did the router start there?" is a question that will get asked.
- **Priors tilt; they never open a gate.** Seeding goes through `Bandit.seed()`, never
  `AdaptiveRouter.observe()`. `packages/router/src/priors.test.ts` enforces this — if you find
  yourself wanting to relax it, that is a product decision, not a refactor.
- **Abstain rather than invent.** A task type with no defensible benchmark mapping gets no prior.
  Same principle as `packages/quality`: a fabricated signal is worse than none.
- **Never import gateway, router, or orchestrator.** This package produces data those layers consume.
- **Pricing is guarded on ingest.** A misparsed zero would make a model look free and win every
  budget-constrained route. Zero is rejected; large swings need explicit confirmation.
- Benchmarks are saturated, gamed, and often contaminated. Treat them as a weak prior about general
  capability, never as evidence about a specific customer's traffic.
