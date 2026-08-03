# packages/memory

Tiered memory: a recent-turn buffer, and semantic recall across sessions.

## Owns

- `src/embedder.ts` — the embedding contract, plus an offline implementation
- `src/store/` — persistence and similarity search
- `src/service.ts` — write and retrieval policy
- `src/summarize.ts` — how a long session gets compressed

## Rules

- **Tenant isolation is enforced in the store, not the caller.** Every query is scoped by
  `tenantId`. A retrieval bug that leaks one customer's context into another's prompt is the worst
  failure this system can have, so the scoping lives where it cannot be forgotten.
- **Never import the gateway or the router.** The embedder and the summarizer arrive injected, for
  the same reason the quality judge is pinned: memory must not be able to influence routing.
- **Retrieval quality must be measurable before it is trusted.** The research is explicit that recall
  degrades as the store grows; `retrieval-eval.ts` exists so that is observed rather than assumed.
- **Write policy is not "store everything".** An assistant that remembers every utterance retrieves
  noise. What gets stored is a product decision, made explicitly in `service.ts`.
- Brute-force cosine is honest at this scale and stated as such. Past roughly 10^5 rows per tenant it
  needs a real index (pgvector, sqlite-vec) — that is a driver swap behind `MemoryStore`.
