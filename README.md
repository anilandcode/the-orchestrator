# The Orchestrator

A provider-agnostic AI orchestrator whose differentiator is a **self-optimizing contextual-bandit
router** — not the proxy around it. Everything else exists to make that router measurable.

```bash
pnpm install
pnpm verify        # typecheck, typecheck tests, lint, 585 tests, simulation
```

No API keys are needed for any of the above. The entire test suite runs offline against mocked
`fetch`, and the routing evaluation runs against a simulated world.

---

## What is actually true about this system

This section is first because it is the part most likely to be assumed wrong.

**The router beats static rules only where outcomes are observable.** Measured, in simulation:

| Strategy | True value | Optimal picks | Regret |
|---|---:|---:|---:|
| static rules (baseline) | 6747.0 | 37.1% | 144.6 |
| bandit, ungated | 6615.8 | 37.3% | 275.8 |
| **bandit, gated to observable tasks** | **6778.6** | **41.5%** | **113.0** |
| oracle (upper bound) | 6891.6 | 100% | 0 |

The ungated bandit **loses** to plain rules. Gated to task types where a validator can actually grade
the answer, it closes 21.9% of the gap to optimal and cuts regret 22%. That narrower claim is the one
this repo supports.

**Two things have never happened here:**

1. **No adapter has made a real API call.** Every test mocks `fetch`. The wire formats are implemented
   from the provider specs and verified against fixtures, not against live endpoints.
2. **No real traffic has ever been routed.** Every number above comes from a simulated world whose
   quality table this repo defines. Whether your traffic has more headroom than the ~2% the static
   rules leave is unknown, and is the single most important open question.

**Benchmark priors made routing worse.** Seeding the bandit from public benchmark scores was measured
and rejected — see [Model catalog](#model-catalog) below. The mechanism works; the data did not.

---

## Architecture

Dependencies point one way, and the direction is enforced:

```
shared ← gateway ← telemetry ← router ← api
                        ↑        ↑
              memory, mcp, catalog, orchestrator
```

| Package | Responsibility |
|---|---|
| [`shared`](packages/shared) | Zod contracts, error taxonomy, model registry. Imports nothing from this repo. |
| [`gateway`](packages/gateway) | Provider adapters, retry, fallback, streaming, cost accounting. **Never imports the router.** |
| [`telemetry`](packages/telemetry) | Event store, reward function, aggregation. **All SQL lives here.** |
| [`router`](packages/router) | Selection only. Decides *what* to call; never calls anything. |
| [`quality`](packages/quality) | Produces the reward's quality term. Validators, sampled LLM judge. |
| [`orchestrator`](packages/orchestrator) | Durable workflow graphs. Imports neither gateway nor router. |
| [`memory`](packages/memory) | Session buffer plus semantic recall, tenant-isolated. |
| [`mcp`](packages/mcp) | MCP client, tool registry, deny-by-default policy, audit. |
| [`catalog`](packages/catalog) | External model knowledge: pricing, benchmarks, priors. |
| [`apps/api`](apps/api) | The only place the layers are wired together. |

The inversions are deliberate. A gateway that could ask "which model should I use?" would make routing
unswappable; an orchestrator that knew how to call a provider would be a second gateway.

---

## Design rules worth knowing before changing anything

**One `CallEvent` per attempt, never per request.** A request that fails over twice writes three
events, including the failures. Collapsing them would credit a fallback's success to the model that
broke — which is precisely the signal the bandit learns from.

**A failed call scores exactly 0 reward.** Not "cheap and fast". A failure costs nothing and returns
instantly, so a naive weighted sum would rank it *above* a slow correct answer and teach the router
to prefer models that fail.

**Abstention is not the same as zero.** A quality scorer with no opinion returns `undefined`. A
validator that returned `0` for tasks it does not cover would teach the bandit that every model fails
at everything it cannot see.

**Cost is always computed from the model registry**, never read from a provider response.

**Priors tilt; they never open a gate.** External evidence seeds the bandit's starting point and
cannot satisfy the cold-start or observability gates. Enforced by
[`priors.test.ts`](packages/router/src/priors.test.ts).

---

## Running it

```bash
cp .env.example .env     # add keys only if you want live calls
pnpm dev:api
```

**One key reaches everything.** `OPENROUTER_API_KEY` alone — free tier included — makes the whole
catalog callable and unblocks `pnpm smoke`, `pnpm eval`, and real pricing. Separate OpenAI and
Anthropic keys still work if you have them.

`OPENROUTER_MODELS` controls which catalog models become *callable*, and it is empty by default. That
is arithmetic rather than caution: LinUCB regret grows with `sqrt(arms x time)`, so opening all ~300
models multiplies exploration cost about 7x against 2.10% of headroom. The catalog stays full of
knowledge; the routing pool stays small on purpose.

`ROUTER_MODE` defaults to `shadow`: the bandit computes and logs its choice on every request but the
static rules execute. Promoting it to `adaptive` is a decision backed by `pnpm replay` against real
traffic, not a config flip.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat` | Unified chat. `stream: true` returns SSE. |
| `POST /v1/feedback` | Human quality signal; revises the reward and re-teaches the router. |
| `POST /v1/runs` | Start a workflow. `/:id/resume` continues a paused one. |
| `POST /v1/memory/recall` | Inspect what memory *would* inject before trusting it. |
| `GET /v1/catalog` | What the system believes about models, sourced and dated. |
| `GET /v1/stats` | Traffic, per-model performance, routing disagreement. |

## Tooling

| Command | What it does | Costs money |
|---|---|---|
| `pnpm verify` | Everything CI runs | No |
| `pnpm replay:simulate` | Routing evaluation against a known-truth world | No |
| `pnpm replay` | Off-policy analysis of **real** logged traffic | No |
| `pnpm catalog:refresh` | Ingest live model pricing (`--apply` to promote) | No |
| `pnpm smoke` | One real call per provider, asserts identical normalization | **Yes** |
| `pnpm eval` | Measure every model against `tools/eval/fixtures` | **Yes** |

## Model catalog

`pnpm catalog:refresh` ingests the public OpenRouter catalog — no auth required — replacing
hand-written prices with dated, sourced, refreshable data. Nothing is applied until you re-run with
`--apply`, because ingested pricing feeds cost accounting, the reward's cost term, and budget
filtering. A price of zero for a model known to cost money is rejected as a parse failure rather than
accepted as a free tier.

**Benchmark priors are disabled by default, on evidence.** A diagnostic arm in the simulation
separates mechanism from data:

```
no priors           early regret  9.7
benchmark priors    early regret 10.0   ← worse
oracle priors       early regret  7.1   ← 27% better
```

Seeding works when the priors are true. The shipped benchmark numbers are **unverified placeholders**
(labelled as such, and a test asserts they still are) and they describe a different world than the one
being routed in. `pnpm eval` is the answer: measure the models on fixtures shaped like your traffic,
which is what public benchmarks were not.

---

## Status

Roadmap phases 0–7 are implemented, plus quality scoring, task-type gating, and the model catalog.
Not built: multi-tenancy, billing, quotas, and a dashboard.

**To move this from "measured in simulation" to "measured in production" you need:** API keys, ten
minutes verifying the pricing table (or one `catalog:refresh --apply`), and a real workload. Until
then every number in this README describes a world this repository invented.
