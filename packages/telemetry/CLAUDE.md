# packages/telemetry

Event persistence, reward computation, and aggregation. This is the feedback loop the adaptive router
learns from — if this package is wrong, the bandit optimizes for the wrong thing.

## Owns

- `src/store/database.ts` — connection + migration runner (shared with the router's arm-state table)
- `src/store/repository.ts` — `CallEventRepository` interface
- `src/store/sqlite.ts`, `src/store/memory.ts` — the two implementations
- `src/reward.ts` — the reward function and its weights
- `src/aggregate.ts` — per-model win rate, latency percentiles, cost per success

## Rules

- **No SQL outside `src/store/`.** That boundary is what makes the Postgres migration a driver swap.
- **A failed call scores exactly 0.** It cost nothing and returned fast, so a naive weighted sum would
  rank failure *above* a slow successful answer. `computeReward` short-circuits on error for this
  reason — do not "simplify" it away.
- **Cost and latency normalize against rolling per-task-type percentiles**, not fixed constants.
  Absolute dollar and millisecond figures drift every time the model pool changes.
- Reward weights must sum to 1 per route mode, so rewards stay comparable across modes.
