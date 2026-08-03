# AI Orchestrator End-to-End Build Roadmap

This document is a practical build plan for creating a provider-agnostic, self-optimizing AI orchestrator as a SaaS product. The target system combines a unified model gateway, adaptive routing, graph-based orchestration, hybrid memory, and MCP-native tool interoperability.[web:14][web:15][web:20][web:22][web:31][web:37]

## Product Goal

Build an AI orchestrator that can:

- Connect to any major LLM provider through one normalized interface.[web:14][web:15][web:44]
- Route each request to the best model or agent using learned/adaptive routing based on cost, latency, and success rate.[web:31][web:35][web:37]
- Support multi-step workflows through a graph-based orchestration engine.[web:1][web:4][web:8]
- Persist useful context across sessions with a hybrid memory system.[web:36][web:41]
- Connect to external tools and data sources through MCP so the system can adapt to new environments without custom glue code for every integration.[web:18][web:20][web:27]
- Operate as a multi-tenant SaaS with usage tracking, controls, and extensibility.[web:42][web:44]

## Core Principles

The system should be built as separate layers with strict boundaries. The gateway handles providers, the router handles selection, the orchestrator handles execution flow, the memory service handles persistence and retrieval, and the MCP layer handles tool interoperability.[web:15][web:18][web:31][web:41]

Build from scratch where differentiation matters most: routing intelligence, orchestration semantics, and product UX. Avoid wasting early time rebuilding commodity pieces before the interfaces are clear.[web:44][web:42]

## End-State Architecture

| Layer | Purpose | First Deliverable | Mature Deliverable |
|---|---|---|---|
| Gateway | Normalize provider APIs | Unified chat API for 2 providers | Multi-provider gateway with retries, streaming, usage accounting |
| Router | Choose best model/agent | Static rules | Contextual multi-armed bandit with online learning |
| Orchestrator | Manage multi-step execution | Simple directed graph runner | Durable state machine with resumes, retries, branch control |
| Memory | Persist and retrieve context | Session buffer + summaries | Hybrid short-term + vector + optional graph memory |
| MCP Layer | Standardize tool access | MCP client calling one tool | Full MCP registry, policy layer, auth, tenant-scoped tools |
| SaaS Layer | Package as a product | Single-tenant API | Multi-tenant platform, billing, quotas, dashboards |

## Recommended Monorepo Shape

```text
ai-orchestrator/
├── CLAUDE.md
├── .claude/
│   ├── settings.json
│   ├── rules/
│   └── skills/
├── apps/
│   ├── api/
│   ├── dashboard/
│   └── docs/
├── packages/
│   ├── shared/
│   ├── gateway/
│   ├── router/
│   ├── orchestrator/
│   ├── memory/
│   ├── mcp/
│   ├── telemetry/
│   └── sdk/
├── infra/
│   ├── docker/
│   ├── terraform/
│   └── k8s/
└── research/
```

A hierarchical Claude Code setup is useful in monorepos because root and per-package `CLAUDE.md` files help keep the agent aligned with architectural boundaries and package-specific rules.[web:47][web:49][web:50][web:60]

## Phase Plan

## Phase 0 — Foundation

### Goal

Create the codebase, development harness, package boundaries, testing standards, and deployment conventions.

### Outcome

By the end of this phase, the project has a stable monorepo, shared schemas, package-level ownership, and a Claude Code workflow that does not leak responsibilities across packages.[web:49][web:50][web:52]

### Tasks

- Set up a pnpm TypeScript monorepo with strict mode.
- Add root `CLAUDE.md` and package-specific `CLAUDE.md` files.
- Create packages for `shared`, `gateway`, `router`, `orchestrator`, `memory`, `mcp`, `telemetry`, and `sdk`.
- Add apps for `api` and `dashboard`.
- Standardize linting, testing, formatting, and release scripts.
- Add `.env.example`, secrets handling, and local Docker Compose.

### Deliverables

- Monorepo scaffold.
- Shared Zod schemas.
- Basic CI pipeline.
- Claude Code operating rules.

### Exit Criteria

- Each package builds independently.
- Shared types can be imported cleanly.
- Claude Code can work package-by-package without crossing layers accidentally.[web:49][web:53][web:60]

## Phase 1 — Gateway MVP

### Goal

Build a provider-agnostic gateway that normalizes requests and responses across at least two LLM providers.

### Why First

The gateway is the narrow waist of the system. Adaptive routing, memory, orchestration, and SaaS metering all depend on a unified request/response contract.[web:14][web:15][web:44]

### Tasks

- Define `UnifiedChatRequest` and `UnifiedChatResponse`.
- Create provider adapters for OpenAI and Anthropic first.
- Normalize message shape, tool-call shape, token usage, latency, and cost fields.
- Support retries, timeouts, and error classification.
- Add mocked tests and one smoke test path.
- Capture cost and token accounting on every request.

### Deliverables

- `Gateway.chat()`.
- `ProviderAdapter` interface.
- OpenAI and Anthropic adapters.
- Usage and latency metrics.

### Exit Criteria

- One request shape works across both providers.
- Tests pass with mocked fetch.
- Gateway logs every success and failure consistently.[web:14][web:15][web:44]

## Phase 2 — Telemetry and Feedback Loop

### Goal

Instrument the system so routing decisions can later learn from real outcomes.

### Why This Matters

Adaptive routing only works if the system records the right signals: latency, cost, success, and downstream quality indicators. Recent routing research treats model selection as an online learning problem and depends on measurable rewards.[web:31][web:35][web:37][web:43]

### Tasks

- Create a `CallEvent` schema.
- Log request metadata, model, provider, latency, tokens, cost, status, and error classes.
- Add feature extraction fields such as estimated prompt length, task type, and structured task tags.
- Build replay datasets from historical traffic.
- Add dashboards for cost, latency, and model win rates.

### Deliverables

- Telemetry package.
- Event store in Postgres, ClickHouse, or a simpler first-step store.
- Reward computation utility.

### Exit Criteria

- Every gateway request creates a structured event.
- Historical logs can be replayed into an offline evaluator.
- Basic reporting exists for routing experiments.[web:31][web:35][web:37]

## Phase 3 — Static Router v1

### Goal

Introduce a deterministic router before moving to online learning.

### Why This Phase Exists

A rule-based router provides a baseline for comparison. Without a baseline, adaptive routing cannot be proven useful.[web:31][web:35]

### Tasks

- Add simple route rules based on task type, tenant preference, max budget, or latency mode.
- Define fallback chains, for example: cheap fast model first, premium model on failure.
- Add policy overrides for specific customers.
- Add evaluation scripts comparing rule-based routing to single-model baselines.

### Deliverables

- Router interface.
- Rule engine.
- Fallback policies.
- Baseline reports.

### Exit Criteria

- Static router can select between at least 3 targets.
- Fallbacks work predictably.
- The team has a measured baseline before adaptive routing starts.[web:31][web:44]

## Phase 4 — Adaptive Router v2

### Goal

Replace fixed rules with a contextual multi-armed bandit router that self-optimizes over time.

### Why This Is the Product Wedge

Dynamic routing research shows contextual bandit approaches can outperform static or random baselines on the tradeoff between quality and cost, and can adapt as new models enter the pool.[web:31][web:35][web:37][web:38]

### Tasks

- Start with Thompson Sampling or UCB.
- Define feature vectors from request context.
- Define reward functions that combine success, cost, and latency.
- Add customer preference weighting for “cheap”, “balanced”, and “best quality” modes.[web:31]
- Add exploration safeguards and cold-start defaults.
- Add offline replay evaluation, then limited online rollout.

### Deliverables

- Adaptive router service.
- Reward function module.
- Evaluation harness.
- Shadow mode deployment path.

### Exit Criteria

- Router performs at least as well as static routing on internal benchmarks.
- Router can onboard a newly added model without full retraining.[web:31][web:35][web:37]

## Phase 5 — Orchestration Engine

### Goal

Build the execution layer for multi-step agent workflows.

### Why Graph-Based

Graph-based orchestration provides more explicit control and production reliability than role-only delegation or chat-style agent loops when workflows become complex.[web:1][web:4][web:8][web:11]

### Tasks

- Define workflow nodes, edges, guards, and retry behavior.
- Implement a graph runner with execution state.
- Add human-in-the-loop checkpoints.
- Add resumable runs and event-sourced transitions.
- Allow each node to specify routing strategy and tool permissions.

### Deliverables

- Graph engine.
- Workflow definition schema.
- Durable execution log.
- Debug trace viewer.

### Exit Criteria

- A multi-node workflow can pause, resume, branch, and retry safely.
- The router can be consulted at each node rather than only once per workflow.[web:1][web:4][web:8]

## Phase 6 — Memory System

### Goal

Add a tiered memory architecture that improves continuity and recall without overloading context windows.

### Recommended Default

Start with short-term buffer memory and vector-based long-term memory. Add graph memory only if your product truly needs structured relationship reasoning.[web:33][web:36][web:41]

### Tasks

- Implement working memory for active execution state.
- Add summarized session memory.
- Add long-term semantic memory using a vector store.
- Define memory write policies so the system stores only useful facts.
- Define retrieval policies before each model call.
- Add tenant-level isolation and retention settings.

### Deliverables

- Memory API.
- Buffer + vector memory store.
- Memory policies and tests.

### Exit Criteria

- The system can recall useful prior context across sessions.
- Retrieval quality is measurable and does not degrade uncontrollably at moderate scale.[web:36][web:39][web:41]

## Phase 7 — MCP Tool Layer

### Goal

Make the orchestrator tool-native through Model Context Protocol.

### Why MCP Matters

MCP was designed to reduce the custom integration burden between AI systems and external tools by providing a shared protocol for discovery and invocation.[web:18][web:20][web:22][web:27]

### Tasks

- Build an MCP client package.
- Integrate one internal MCP server first.
- Add tool registry, auth, permission checks, and audit logging.
- Add tenant-scoped tool catalogs.
- Allow workflows to declare allowed MCP tool scopes.

### Deliverables

- MCP client.
- Internal tool server examples.
- Tool policy layer.

### Exit Criteria

- Any workflow node can call an MCP tool through a normalized interface.
- Tool permissions are auditable and tenant-aware.[web:18][web:22][web:27]

## Phase 8 — SaaS Platform

### Goal

Turn the orchestration engine into a usable product.

### Tasks

- Build API keys, auth, and tenant isolation.
- Add usage metering and billing.
- Add model policies and customer routing preferences.
- Add workflow templates.
- Add dashboard views for runs, cost, latency, success rates, and memory usage.
- Add rate limits, quotas, and alerting.

### Deliverables

- Public API.
- Admin dashboard.
- Usage reports.
- Tenant management.

### Exit Criteria

- External users can create tenants, send requests, inspect runs, and control their routing/cost preferences.[web:42][web:44]

## Phase 9 — Reliability and Production Hardening

### Goal

Make the system resilient enough for real customers.

### Tasks

- Add circuit breakers and provider failover.
- Add idempotency keys and replay-safe execution.
- Add prompt redaction and secrets filtering.
- Add policy enforcement for tool access and data boundaries.
- Add audit logging, tracing, and incident response playbooks.
- Add synthetic and live canary tests.

### Exit Criteria

- Provider outages degrade gracefully.
- Important workflows can recover from partial failure without corruption.
- Customers can trust the platform for production use.

## Phase 10 — Differentiation and Growth

### Goal

Move beyond parity and build the features that make the orchestrator truly hard to replace.

### Ideas

- Bandit router marketplace presets by use case.
- Bring-your-own-model and bring-your-own-MCP servers.
- Automatic workflow optimization based on outcome traces.
- Memory quality scoring and pruning.
- Team collaboration features around traces and run debugging.
- Vertical templates for sales ops, support, coding agents, and design-to-code systems.

## Execution Order Summary

| Order | Phase | Must Be Done Before |
|---|---|---|
| 0 | Foundation | Everything |
| 1 | Gateway MVP | Router, telemetry, SaaS API |
| 2 | Telemetry | Adaptive router |
| 3 | Static router | Adaptive router validation |
| 4 | Adaptive router | Advanced product differentiation |
| 5 | Orchestration engine | Serious multi-step agent product |
| 6 | Memory | Production-quality assistant behavior |
| 7 | MCP layer | Tool ecosystem and extensibility |
| 8 | SaaS platform | Customer onboarding |
| 9 | Hardening | Broad launch |
| 10 | Differentiation | Scale and market wedge |

## 12-Week Suggested Roadmap

| Week | Focus | Main Output |
|---|---|---|
| 1 | Foundation | Monorepo, CLAUDE.md, package boundaries |
| 2 | Gateway | OpenAI + Anthropic unified gateway |
| 3 | Telemetry | Structured call events and dashboards |
| 4 | Static router | Deterministic rules + fallback chains |
| 5 | Adaptive router offline | Replay harness + reward functions |
| 6 | Adaptive router online | Shadow mode and limited rollout |
| 7 | Orchestrator | Graph runner MVP |
| 8 | Memory | Buffer + vector memory |
| 9 | MCP | One internal tool through MCP |
| 10 | SaaS API | Auth, tenant model, usage metering |
| 11 | Dashboard | Trace view, cost view, routing view |
| 12 | Hardening | Failover, retries, audit, launch prep |

## Definition of Done by Layer

### Gateway Done

- Unified request/response contract works across providers.
- Retries, errors, streaming, and usage accounting are consistent.

### Router Done

- Routing decisions are explainable.
- Performance beats or matches static baselines on target workloads.[web:31][web:35][web:37]

### Orchestrator Done

- Multi-step workflows run durably with retries and resumes.

### Memory Done

- Retrieval meaningfully improves continuity without runaway storage noise.[web:36][web:41]

### MCP Done

- Tools are discoverable, policy-controlled, and reusable across tenants.[web:18][web:22]

### SaaS Done

- A customer can sign up, send traffic, monitor usage, and control spend.

## Claude Code Operating Model

Claude Code works best when each session has a single bounded goal, especially in a monorepo. Root and package-specific `CLAUDE.md` files help enforce layer boundaries and package ownership.[web:47][web:49][web:50][web:60]

### Session Rules

- One package per session when possible.
- One goal per prompt.
- Require tests before expanding scope.
- Keep architecture rules in root `CLAUDE.md`.
- Keep package-specific implementation rules in package `CLAUDE.md`.
- Use separate sessions for gateway, router, memory, and dashboard work.

## Build Priorities

If time is limited, build in this order:

1. Gateway
2. Telemetry
3. Static router
4. Adaptive router
5. Orchestrator
6. Memory
7. MCP
8. SaaS dashboard

This order preserves the learning loop. Without unified calls and feedback logging, the adaptive router will be guesswork rather than a real product.[web:31][web:35][web:37]

## Final Guidance

The best way to win here is not to build every layer at once. Build the narrowest working path first, make it measurable, then add intelligence on top of real traffic. The real moat is the combination of unified execution, adaptive routing, workflow control, and tool interoperability in one product, not just another model proxy.[web:42][web:44][web:31][web:20]
