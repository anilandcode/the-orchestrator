# packages/orchestrator

Multi-step workflow execution. Owns *when* things run and what happens on failure; owns nothing about
*how* a step is performed.

## Owns

- `src/schema.ts` — the serializable workflow definition
- `src/guards.ts` — declarative edge conditions
- `src/state.ts` — run state and the event log it is derived from
- `src/runner.ts` — the step loop, retries, pause/resume
- `src/store/` — durable run persistence

## Rules

- **Never import `@orchestrator/gateway` or `@orchestrator/router`.** Steps run through injected
  `NodeExecutor`s. The orchestrator that knew how to call a provider would be a second gateway, and
  the one that knew how to pick a model would make routing unswappable.
- **Workflow definitions must stay JSON-serializable.** They are stored, versioned, and sent over the
  wire. That rules out functions as guards — hence the declarative condition schema, which is
  deliberately small rather than a general expression language.
- **A run's history is the source of truth.** State is derived from the event log, never patched in
  place, so a crashed run resumes from what actually happened rather than from a summary of it.
- **Every loop needs a bound.** `maxSteps` is not optional paranoia: a guard that never goes false
  would otherwise bill a provider forever.
