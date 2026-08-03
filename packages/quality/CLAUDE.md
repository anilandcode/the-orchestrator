# packages/quality

Produces the quality term of the reward. This is the input the bandit weights most heavily
(0.6 balanced, 0.85 best), so a mistake here does not cause a bug — it causes a router that
confidently optimizes for the wrong thing.

## Owns

- `src/scorer.ts` — the `QualityScorer` contract
- `src/pipeline.ts` — precedence, provenance, and which scorers run where
- `src/validators/` — deterministic, free, inline
- `src/judge/` — sampled LLM judge, deferred, costs money

## Rules

- **May import `@orchestrator/gateway`** (the judge makes model calls).
  **Must never import `@orchestrator/router`** — a judge that routed through the bandit would let the
  bandit grade its own homework.
- **`undefined` means "no opinion", `0` means "this was bad".** Never conflate them. A validator that
  does not apply to a task must abstain; returning 0 would teach the bandit that every model fails at
  every task the validators happen not to cover, which is worse than the constant heuristic.
- **Precedence, not blending.** The highest-authority available signal wins outright. Averaging a
  confident human score with a vague heuristic produces a number that means nothing.
- **Nothing in `validators/` may do I/O.** They run inline on the request path.
- **The judge never runs on the hot path** and always respects its spend cap. A judge billing more
  than the traffic it grades is a real failure mode, not a hypothetical one.
