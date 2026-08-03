# Phase 4.5 — Make the Quality Signal Real

## Why this comes before Phase 5

The reward function weights quality at **0.6 in `balanced` mode and 0.85 in `best` mode**. The current
quality signal is a three-value heuristic in `packages/telemetry/src/reward.ts`:

```
success   -> 0.8
truncated -> 0.5
failure   -> 0
```

For the overwhelming majority of calls this is a **constant**. In `best` mode the bandit is therefore
learning almost nothing about the dimension it weights most heavily — it is effectively ranking on
cost and latency, which the static rules already do well. This is a substantial part of why LinUCB
closed only 31.4% of the static→oracle gap in simulation.

Every layer above the router (orchestration, memory, MCP) assumes routing works. Building them on an
unvalidated wedge means three more layers depending on an unproven claim. The gateway, telemetry
store, bandit, persistence, and replay harness are all built and tested. **The loop is closed except
for the one input that carries the signal.**

## Scope

Four quality sources, layered by authority, plus the machinery to revise a reward after the bandit
has already learned from it.

| Source | When it lands | Cost | Authority |
|---|---|---|---|
| Client feedback (`/v1/feedback`) | Arbitrarily later | Free | Highest — a human judged it |
| Deterministic validators | Inline, sub-millisecond | Free | High where they apply |
| LLM judge (sampled) | Off hot path, seconds | Money per judged call | Medium, generalizes |
| Heuristic | Inline | Free | Floor — "we do not know" |

---

## The hard part: rewards must be revisable

The bandit learns from a provisional reward at request time. A validator may refine it moments later;
a judge minutes later; a human hours later. Re-teaching naively would double-count.

**LinUCB makes this exact, not approximate.** The update is:

```
A += x xᵀ          <- does not depend on reward
b += reward · x    <- linear in reward
```

So a correction is simply `b += (r_new − r_old) · x`, with `A` and the pull count untouched. The arm
ends up in precisely the state it would have reached had the correct reward been known up front. This
is a real property of the algorithm, not a convenient approximation, and it is why revision is worth
building rather than deferring the update behind a timeout.

Thompson needs `alpha += Δ`, `beta -= Δ`, clamped to stay positive.

### New bandit method

```ts
interface Bandit {
  // ... existing
  /** Correct a previously-applied reward. Exact for LinUCB. */
  revise(armId: string, features: number[], oldReward: number, newReward: number): void;
}
```

---

## New package: `packages/quality`

Layer position: `shared <- gateway <- quality`. Nothing depends on it except `apps/api`.

It may import the gateway because the LLM judge must make a model call. **It must never import the
router** — a judge that routed through the bandit would let the bandit grade its own homework.

```
packages/quality/
├── src/
│   ├── scorer.ts              # QualityScorer interface + QualityInput
│   ├── pipeline.ts            # precedence, provenance, confidence
│   ├── validators/
│   │   ├── finish-reason.ts   # current heuristic, as the floor
│   │   ├── json-schema.ts     # extraction: does output match declared schema
│   │   ├── tool-call.ts       # do tool arguments validate against the tool's schema
│   │   └── code.ts            # code: does it parse; optional tsc --noEmit
│   └── judge/
│       ├── llm-judge.ts       # sampled, rubric-based, cost-capped
│       └── rubric.ts          # per-task-type rubrics
```

### Interface

```ts
export interface QualityInput {
  request: UnifiedChatRequest;
  response: UnifiedChatResponse;
  event: CallEvent;
}

export interface QualityAssessment {
  score: number;        // 0..1
  confidence: number;   // 0..1 — how much weight the pipeline should give this
  source: string;       // provenance, persisted for audit
}

export interface QualityScorer {
  readonly name: string;
  readonly stage: "inline" | "deferred";
  /** undefined = "no opinion on this call", which is different from a score of 0. */
  score(input: QualityInput): Promise<QualityAssessment | undefined> | QualityAssessment | undefined;
}
```

`undefined` vs `0` is the distinction that matters most here: a validator that does not apply to a
task must abstain, not condemn. Conflating them would teach the bandit that every model fails at
every task the validators do not cover.

### Pipeline precedence

Highest-authority *available* signal wins outright rather than blending. Blending a confident human
score with a vague heuristic produces a number that means nothing. Confidence is used to decide
whether a deferred scorer is worth running at all, not to average.

---

## Deterministic validators

- **`finish-reason`** — today's heuristic, kept as the abstain-floor.
- **`tool-call`** — validate emitted tool arguments against the tool's own JSON Schema. Applies to
  every tool-calling request, needs no configuration, and catches a real failure mode (models emitting
  plausible-looking but invalid arguments).
- **`json-schema`** — for `extraction` tasks where the caller declared an output schema. Requires a
  new optional `route.outputSchema` field on `UnifiedChatRequest`.
- **`code`** — extract fenced code blocks and parse them. Start with parse-only (fast, no sandbox).
  Executing untrusted model-generated code needs an isolated sandbox and is explicitly **out of scope**
  for this phase.

## LLM judge

- Samples a configurable fraction (`QUALITY_JUDGE_SAMPLE_RATE`, default `0.05`).
- Runs **after** the response is returned — never on the hot path.
- Pins its own model (`QUALITY_JUDGE_MODEL`); must not consult the router.
- Hard spend cap (`QUALITY_JUDGE_MAX_USD_PER_HOUR`) with a circuit breaker, because a runaway judge
  billing more than the traffic it grades is a real failure mode.
- Emits its own `CallEvent` tagged as judge traffic, excluded from routing statistics — otherwise the
  judge's own calls pollute the data it exists to grade.

## Held-out eval set

`tools/eval/` — fixtures per task type, run across every reachable model on demand.

```
tools/eval/fixtures/<task-type>/*.json   # prompt + expectation
pnpm eval                                # runs all models, writes report + priors
```

Output is a **warm-start prior file**: per-model, per-task expected quality. Loading it at boot
addresses cold start far better than deferring to static rules, because the bandit starts with
measured priors instead of no information. This directly targets the cold-start risk the research doc
flags.

Costs real money per run. Opt-in, never in CI.

---

## Schema changes

`call_events` gains, via a new migration (`003_quality_provenance`):

```sql
quality_source     TEXT     -- which scorer produced the score
quality_confidence REAL     -- 0..1
quality_revisions  INTEGER  -- how many times the reward was corrected
```

Provenance is not optional bookkeeping: without it you cannot tell whether a model looks good because
it is good or because only the lenient scorer ever rated it.

---

## Files touched

| File | Change |
|---|---|
| `packages/quality/**` | New package |
| `packages/router/src/bandit/bandit.ts` | Add `revise()` to the interface |
| `packages/router/src/bandit/linucb.ts` | Exact `b` correction |
| `packages/router/src/bandit/thompson.ts` | Beta parameter correction |
| `packages/router/src/adaptive-router.ts` | `reviseOutcome()`; track applied rewards per decision |
| `packages/telemetry/src/reward.ts` | `RewardService.rescore()` |
| `packages/telemetry/src/migrations/` | `003_quality_provenance` |
| `packages/shared/src/schemas/chat.ts` | Optional `route.outputSchema` |
| `apps/api/src/server.ts` | Run inline scorers; queue deferred; `/v1/feedback` revises |
| `tools/eval/**` | New |

## Verification

1. **Unit** — each validator abstains correctly (`undefined`, never `0`); pipeline precedence; cost cap trips.
2. **Revision correctness** — the property test that matters: for a random sequence of rewards and
   corrections, a LinUCB arm updated-then-revised must be **bit-identical** to one trained on the final
   values directly. If that fails, revision is silently corrupting learning.
3. **Simulation** — extend `tools/replay/src/world.ts` so simulated quality is *observable* through a
   validator rather than known outright. Re-run `pnpm replay:simulate`; expect the gap-closed figure
   to move materially above 31.4%. If it does not, the quality signal is not carrying information and
   that is the finding.
4. **Eval** — `pnpm eval` against real keys produces a prior file; confirm warm-started cold start
   beats cold cold-start in simulation.

## Out of scope

Executing model-generated code in a sandbox; multi-judge consensus; learned quality models; anything
in Phases 5–8.

## Success criterion

Stated plainly so it can fail: **the quality term must measurably separate models on the same task.**
If, after this phase, per-model quality distributions on a given task type still overlap almost
entirely, then quality is not learnable from these signals — and the honest response is to reweight
the reward toward cost and latency and say so, rather than keep a 0.85 weight on a number that does
not discriminate.
