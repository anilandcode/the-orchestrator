# The Orchestrator — architecture rules

Provider-agnostic AI orchestrator. The differentiator is the **self-optimizing contextual-bandit
router**, not the proxy. Everything else exists to make that router measurable.

## Layer boundaries (enforced, one-directional)

```
shared  <-  gateway  <-  telemetry  <-  router  <-  api
```

- **`packages/shared`** — imports nothing from this repo. Zod schemas, model registry, error taxonomy.
- **`packages/gateway`** — provider adapters and execution. **Must never import the router.** That
  inversion is what keeps routing swappable. Telemetry arrives as an injected `CallEventSink`.
- **`packages/telemetry`** — event persistence, reward computation, aggregation. No provider logic.
- **`packages/router`** — selection only. Decides *what* to call; never performs a call itself.
- **`apps/api`** — the only place the layers are wired together.

## Non-negotiables

1. **One `CallEvent` per attempt, not per request.** A request that fails over twice writes three
   events. Collapsing them teaches the bandit a falsehood about which model actually failed.
2. **Cost is always computed from `packages/shared/src/models.ts`** against reported token usage.
   Never trust a cost field from a provider response.
3. **Provider quirks stay inside their adapter file.** No `if (provider === 'anthropic')` outside
   `packages/gateway/src/adapters/`.
4. **Errors are normalized before they leave the gateway.** Retry and fallback both key off
   `retryable` / `fallbackEligible`, never off HTTP status codes.
5. **No SQL outside `packages/telemetry/src/store/`.** The repository interface is the seam that makes
   the eventual Postgres migration a driver swap.
6. **`ROUTER_MODE` defaults to `shadow`.** The bandit does not steer real traffic until replay against
   real `CallEvent` data justifies it.

## Session guidance

One package per session where possible. Tests before scope expansion. Package-specific rules live in
each package's own `CLAUDE.md`.

---

# context-mode — MANDATORY routing rules

You have context-mode MCP tools available. These rules are NOT optional — they protect your context window from flooding. A single unrouted command can dump 56 KB into context and waste the entire session.

## BLOCKED commands — do NOT attempt these

### curl / wget — BLOCKED
Any Bash command containing `curl` or `wget` is intercepted and replaced with an error message. Do NOT retry.
Instead use:
- `ctx_fetch_and_index(url, source)` to fetch and index web pages
- `ctx_execute(language: "javascript", code: "const r = await fetch(...)")` to run HTTP calls in sandbox

### Inline HTTP — BLOCKED
Any Bash command containing `fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, or `http.request(` is intercepted and replaced with an error message. Do NOT retry with Bash.
Instead use:
- `ctx_execute(language, code)` to run HTTP calls in sandbox — only stdout enters context

### WebFetch — BLOCKED
WebFetch calls are denied entirely. The URL is extracted and you are told to use `ctx_fetch_and_index` instead.
Instead use:
- `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` to query the indexed content

## REDIRECTED tools — use sandbox equivalents

### Bash (>20 lines output)
Bash is ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`, and other short-output commands.
For everything else, use:
- `ctx_batch_execute(commands, queries)` — run multiple commands + search in ONE call
- `ctx_execute(language: "shell", code: "...")` — run in sandbox, only stdout enters context

### Read (for analysis)
If you are reading a file to **Edit** it → Read is correct (Edit needs content in context).
If you are reading to **analyze, explore, or summarize** → use `ctx_execute_file(path, language, code)` instead. Only your printed summary enters context. The raw file content stays in the sandbox.

### Grep (large results)
Grep results can flood context. Use `ctx_execute(language: "shell", code: "grep ...")` to run searches in sandbox. Only your printed summary enters context.

## Tool selection hierarchy

1. **GATHER**: `ctx_batch_execute(commands, queries)` — Primary tool. Runs all commands, auto-indexes output, returns search results. ONE call replaces 30+ individual calls.
2. **FOLLOW-UP**: `ctx_search(queries: ["q1", "q2", ...])` — Query indexed content. Pass ALL questions as array in ONE call.
3. **PROCESSING**: `ctx_execute(language, code)` | `ctx_execute_file(path, language, code)` — Sandbox execution. Only stdout enters context.
4. **WEB**: `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` — Fetch, chunk, index, query. Raw HTML never enters context.
5. **INDEX**: `ctx_index(content, source)` — Store content in FTS5 knowledge base for later search.

## Subagent routing

When spawning subagents (Agent/Task tool), the routing block is automatically injected into their prompt. Bash-type subagents are upgraded to general-purpose so they have access to MCP tools. You do NOT need to manually instruct subagents about context-mode.

## Output constraints

- Keep responses under 500 words.
- Write artifacts (code, configs, PRDs) to FILES — never return them as inline text. Return only: file path + 1-line description.
- When indexing content, use descriptive source labels so others can `ctx_search(source: "label")` later.

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call the `ctx_stats` MCP tool and display the full output verbatim |
| `ctx doctor` | Call the `ctx_doctor` MCP tool, run the returned shell command, display as checklist |
| `ctx upgrade` | Call the `ctx_upgrade` MCP tool, run the returned shell command, display as checklist |
