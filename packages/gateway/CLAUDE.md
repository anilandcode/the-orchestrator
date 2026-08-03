# packages/gateway

Normalizes provider APIs and executes calls. The narrow waist of the system.

## Owns

- `src/provider-adapter.ts` — the adapter contract
- `src/adapters/` — one file per provider, and the **only** place provider quirks may live
- `src/gateway.ts` — execution: timeout, retry, fallback chain, cost accounting, event emission
- `src/sse.ts` — shared SSE frame parser

## Rules

- **Never import `@orchestrator/router`.** The gateway receives an `ExecutionPlan` and executes it. If
  it ever needed to ask "which model should I use?", routing would stop being swappable.
- **Never import `@orchestrator/telemetry`.** Events go out through the injected `CallEventSink`
  interface from `shared`.
- **One `CallEvent` per attempt.** Three provider calls write three events, including the failures.
  The bandit's reward attribution depends on this being literally true.
- **Adapters do not compute cost or latency.** They return `{ message, finishReason, usage }`; the
  gateway owns all accounting so it is identical across providers.
- **Adapters classify their own errors** into the `shared` taxonomy using the provider's error body,
  falling back to `classifyHttpStatus` only when the body is unhelpful.
- All network access goes through the injected `fetchImpl`, so every test runs offline.
