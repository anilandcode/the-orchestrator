## Overview

This report defines an architecture for a from-scratch, provider-agnostic AI orchestrator that (a) works across any LLM, agent framework, or external tool, (b) uses learned/adaptive routing that self-optimizes on cost, latency, and success rate, and (c) is designed to be sold as a SaaS product. It synthesizes current 2026 industry patterns from LLM gateways, multi-agent frameworks, adaptive routing research, and agent memory systems, and gives a concrete build sequence.

## System Boundaries: What "Any AI System" Means

To be truly adaptable, the orchestrator needs three separate points of abstraction rather than one:

- **Model abstraction**: any LLM provider (OpenAI, Anthropic, Gemini, local models via Ollama) behind one call signature, the same problem solved by LiteLLM (self-hosted proxy) and OpenRouter (hosted, 600+ models).[^1][^2][^3]
- **Agent abstraction**: sub-orchestrated units built in different frameworks (LangGraph state machines, CrewAI role-based crews, custom agents) treated as callable nodes rather than requiring a single framework choice.[^4][^5][^6][^7]
- **Tool/context abstraction**: external tools, databases, and APIs exposed through the Model Context Protocol (MCP), an open standard from Anthropic that solves the "NxM integration problem" — instead of custom glue code between every model and every tool, both sides speak one protocol.[^8][^9][^10][^11]

Building your own orchestrator from scratch means implementing thin, standard interfaces at each of these three boundaries, then building your own intelligence layer on top — rather than adopting someone else's opinionated framework wholesale.

## Core Architecture: Five Layers

| Layer | Responsibility | Design Reference |
|---|---|---|
| Gateway/Adapter | Normalize every model call into one request/response schema | LiteLLM proxy pattern, OpenAI-compatible schema [^3][^1] |
| Routing Engine | Decide which model/agent handles each request | Contextual multi-armed bandit (adaptive) [^12][^13][^14] |
| Orchestration/State | Sequence multi-step workflows, manage retries and branching | Graph-based state machine (LangGraph pattern) [^4][^5][^15] |
| Memory Service | Persist and retrieve context across turns and sessions | Hybrid vector + graph + buffer memory [^16][^17] |
| Tool/MCP Layer | Let any agent call any external tool uniformly | MCP client-server architecture [^18][^9] |

Each layer should be a separate service or module with a clean API boundary, so you can swap implementations (e.g., change the routing algorithm) without touching the gateway or memory layers. This modularity is what lets the system claim to be "adapted to any AI system" — new models, agents, or tools plug into existing interfaces instead of requiring core rewrites.

### Gateway Layer

The gateway is the thinnest, most mechanical layer: it takes a normalized request and dispatches it to whichever underlying provider API the router selects, then normalizes the response back. Building this yourself (rather than depending on LiteLLM) gives full control but means owning retry logic, streaming, rate-limit handling, and provider-specific quirks (function calling formats differ across OpenAI, Anthropic, and Gemini). Given the March 2026 supply-chain security incident reported against LiteLLM, a from-scratch gateway also avoids that specific third-party dependency risk, though it means re-solving problems LiteLLM (47.7k GitHub stars) has already hardened over years.[^19][^1]

### Routing Engine: Learned/Adaptive Design

Since the requirement is self-optimizing routing, the routing engine should be framed as a **contextual multi-armed bandit problem**: each available model/agent is an "arm," the router observes lightweight features of the incoming query (task type, complexity signals, token length), and it learns online which arm to pull based on a reward signal combining accuracy, cost, and latency.[^12][^13][^14]

Key implementation points validated by recent research:

- **No offline calibration needed**: bandit-based routers like MetaLLM optimize routing decisions purely from online accuracy-cost feedback, so new models can be added at runtime without retraining a full model.[^13]
- **Preference-conditioned routing**: let the SaaS customer (or your own system) set a lambda parameter trading off cost vs. quality vs. speed at inference time, rather than hardcoding one objective.[^12]
- **Measured gains**: contextual bandit routing has demonstrated 22% accuracy improvements and 31% reduction in resource cost versus random routing baselines in benchmark studies, and adapts to new models added to the pool without policy retraining.[^14][^20]
- **Feedback loop requirement**: the router needs a reward signal after each call — this typically means logging success/failure (task completion, user correction, downstream validation), latency, and token cost per request, then updating arm statistics (e.g., Thompson sampling or UCB).[^21][^13]

This is meaningfully more sophisticated than the static rule-based routing used by most current commercial routers (Not Diamond, Martian), which mostly do single-shot predictive classification rather than continuously self-optimizing online learning — building your own bandit-based router is a genuine differentiator versus buying.[^22][^1]

### Orchestration/State Layer

For multi-step workflows (the "coordinate multi-agent workflows" requirement from your primary goal), a graph-based state machine is the most defensible pattern for a SaaS product because it gives explicit, debuggable control over execution paths, unlike conversational message-passing (AutoGen-style, which several 2026 comparisons now describe as trending toward maintenance-mode) or purely role-based delegation (CrewAI-style, faster to prototype but less precise control). Building this from scratch means: defining a workflow as nodes (tasks/agents) and edges (transitions, conditionals), executing state transitions durably (so a crashed process can resume), and exposing hooks so the routing engine can be called at each node to pick which model/agent executes it.[^23][^24][^25][^7][^4]

### Memory Layer

Since you weren't sure what's best here, the current industry consensus for 2026 production agents is a **hybrid, tiered memory architecture** rather than picking one storage type:[^16][^17]

| Tier | Purpose | Storage | Latency Target |
|---|---|---|---|
| In-context/working | Active conversation, immediate reasoning | Prompt buffer, no external DB | Instant |
| Buffer/short-term | Recent session history, summarized when long | In-memory or Redis | <10ms |
| Vector/long-term | Semantic recall across sessions, facts, preferences | Pinecone, Qdrant, Weaviate, or pgvector | 50–200ms |
| Graph/structured | Entity relationships, causal chains | Neo4j or similar graph DB | Varies |

The recommended default for a from-scratch build: start with buffer + vector store (covers 90% of use cases — session continuity plus cross-session recall), and add graph memory only if the product needs structured relationship reasoning (e.g., tracking org charts or workflow dependencies). Expose memory as an internal microservice with a single retrieval API the orchestrator calls before every LLM invocation and writes to after every response, rather than letting each agent query storage directly — this keeps memory swappable and independently scalable.[^26][^17][^16]

### Tool/MCP Layer

Rather than building custom integration code for every external tool, implement an MCP client in your orchestrator and expose your own tools (and eventually customer tools) as MCP servers. This is the piece that most directly satisfies "adapted to any AI system" for the tool-use dimension: any MCP-compliant tool or data source becomes usable by any agent in your system without bespoke glue code, and it lets third parties extend your SaaS platform by registering their own MCP servers.[^18][^9][^10][^11][^8]

## Build Sequence for a From-Scratch SaaS

1. **Gateway MVP**: Implement normalized request/response schema and adapters for 2–3 providers (e.g., OpenAI, Anthropic, one open-weight model via a local/hosted endpoint). Get this working with basic static routing first.
2. **Logging and feedback infrastructure**: Before building the adaptive router, instrument every call with cost, latency, and outcome logging — the bandit algorithm is worthless without a reliable reward signal.
3. **Bandit router v1**: Implement a contextual multi-armed bandit (start with a simple algorithm like Thompson sampling or UCB1) using the logged features and rewards; validate it beats static routing on your own test traffic.[^13][^14]
4. **State machine orchestrator**: Build the graph/node execution engine for multi-step workflows, with the router pluggable at each node.
5. **Memory microservice**: Stand up buffer memory + a vector store, exposed via a single retrieval/write API.
6. **MCP integration**: Add an MCP client so agents can call external tools uniformly; expose your own core capabilities as MCP servers too.
7. **SaaS productization**: Add multi-tenancy, per-customer API keys and cost attribution, usage dashboards, and a preference-parameter UI (cost vs. quality slider) so customers can tune the bandit's reward function themselves — this mirrors how commercial routers like Not Diamond and Martian differentiate, but with the added self-optimization edge.[^22][^1]

## Competitive Positioning

| Product | Model | Adaptive Routing | Self-Hostable | Relevant Gap Your Build Fills |
|---|---|---|---|---|
| OpenRouter | Hosted SaaS, 600+ models | Basic auto-mode (uses Not Diamond) | No | No true online learning; ~5.5% fee [^19][^1] |
| LiteLLM | Open-source proxy | No native adaptive routing | Yes | Static config-based routing only [^1] |
| Portkey | Gateway, open core | Rule-based, not bandit-learned | Partial | Strong on compliance/caching, weak on self-optimization [^22][^1] |
| Not Diamond | SaaS routing layer only | Predictive, not continuously online-learned | No | No proxy/gateway capability, single-shot prediction [^22][^1] |
| Martian | Enterprise SaaS | Real-time per-prompt routing | No | Opaque pricing, enterprise-only, ~$1.3B valuation signals big-player focus, leaving room at smaller scale [^22][^1] |

None of the current commercial offerings combine a true self-optimizing contextual bandit router with an integrated state-machine orchestrator, hybrid memory, and MCP-native tool layer in one open, self-hostable SaaS product — this combination is the differentiated wedge for a from-scratch build.[^1][^22]

## Key Risks and Open Questions

Cold-start behavior of the bandit router needs a fallback strategy (e.g., default to a known-good model until enough data accumulates for a given task category) since bandits perform poorly with zero historical data. Memory retrieval quality degrades as the store grows, so retrieval evaluation sets should be built early rather than after scale hits. Finally, MCP is a young standard (introduced November 2024) and its security model (OAuth-based auth, tool-call trust boundaries) is still maturing, so tool-layer security review should be built into the roadmap rather than treated as an afterthought.[^27][^9][^28][^14][^8][^13]

---

## References

1. [Best LLM Router in 2026: OpenRouter, LiteLLM, and Portkey](https://toolchew.com/en/best-llm-router/) - LiteLLM is the best self-hosted router at 8ms P95 and zero per-request cost. OpenRouter wins for ins...

2. [LiteLLM vs OpenRouter: Which Wins for Production AI Agents ...](https://mpiv.ai/blog/litellm-vs-openrouter-which-wins-for-production-ai-agents-2026) - I ran both LiteLLM and OpenRouter in production agent stacks. Here's the honest comparison — pricing...

3. [LiteLLM vs OpenRouter | VIPS Learn](https://learn.engineering.vips.edu/compare/litellm-vs-openrouter) - LiteLLM is an open-source Python library / proxy that unifies LLM APIs. OpenRouter is a hosted servi...

4. [CrewAI vs AutoGen vs LangGraph: Which Multi-Agent Framework in ...](https://dev.to/agdex_ai/crewai-vs-autogen-vs-langgraph-which-multi-agent-framework-in-2026-51m6) - CrewAI vs AutoGen vs LangGraph: Which Multi-Agent Framework in 2026? Three frameworks...

5. [LangGraph vs CrewAI vs AutoGen: The Complete Multi ...](https://dev.to/pockit_tools/langgraph-vs-crewai-vs-autogen-the-complete-multi-agent-ai-orchestration-guide-for-2026-2d63) - A deep dive into the three dominant multi-agent AI frameworks. Learn when to use LangGraph's graph-b...

6. [Agentic AI Frameworks 2026 — LangGraph, CrewAI & AutoGen](https://myengineeringpath.dev/tools/agentic-frameworks/) - LangGraph if you need production control. CrewAI for fast prototyping. Compare state machines vs rol...

7. [CrewAI vs LangGraph vs AutoGen vs OpenAgents — Best ...](https://openagents.org/blog/posts/2026-02-23-open-source-ai-agent-frameworks-compared) - CrewAI is best for role-based team workflows with fast setup. LangGraph is ideal for stateful produc...

8. [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) - The Model Context Protocol (MCP) is an open standard for connecting AI assistants to the systems whe...

9. [Model Context Protocol (MCP) explained: A practical ...](https://codilime.com/blog/model-context-protocol-explained/) - Learn how MCP solves the NxM integration problem and standardizes tool discovery/calls for LLM apps,...

10. [What is Model Context Protocol (MCP)? - IBM](https://www.ibm.com/think/topics/model-context-protocol) - Model context protocol (MCP) serves as a standardization layer for AI applications to communicate ef...

11. [What is the Model Context Protocol (MCP)?](https://www.databricks.com/blog/what-is-model-context-protocol) - Learn what the Model Context Protocol (MCP) is and how it standardizes AI model interaction. Discove...

12. [Cost-Efficient LLM Generation via Preference-Conditioned Dynamic ...](https://arxiv.org/abs/2502.02743) - The rapid advancement in large language models (LLMs) has brought forth a diverse range of models wi...

13. [Dynamic Model Routing and Cascading for Efficient LLM ...](https://arxiv.org/html/2603.04445v1)

14. [A dynamic routing approach for sustainable language model inference](https://repositum.tuwien.at/handle/20.500.12708/216263) - Große Sprachmodelle (LLMs) bieten beispiellose Fähigkeiten, doch ihre breite Anwendung wird durch de...

15. [LangGraph vs CrewAI vs AutoGen: Top 10 AI Agent Frameworks](https://o-mega.ai/articles/langgraph-vs-crewai-vs-autogen-top-10-agent-frameworks-2026) - 2026 agent framework rankings with hard data. LangGraph leads, AutoGen goes legacy, and MCP reshapes...

16. [Architectures, Vector Stores, and GraphRAG](https://mem0.ai/blog/what-is-ai-agent-memory) - AI agent memory allows LLMs to retain and retrieve context across sessions. Learn how agent memory a...

17. [Agent Memory — How AI Agents Remember (2026)](https://myengineeringpath.dev/genai-engineer/agent-memory/) - In-context, buffer, vector store, knowledge graph — 4 memory tiers that let agents remember across s...

18. [Architecture overview - Model Context Protocol](https://modelcontextprotocol.io/docs/learn/architecture)

19. [OpenRouter vs LiteLLM: Which One Actually Wins? (2026)](https://www.youtube.com/watch?v=Jiyjukqx8SQ) - 🏷️ Check Current Price on Amazon: https://amzn.to/3I8udfq
🔖 Bookmark & Use for ANY Amazon Purchase (...

20. [Energy-Efficient Context-Aware Dynamic Routing for Multi- ...](https://shashikantilager.com/assets/pdf/publications/greenserv_2026.pdf)

21. [Multi-Armed Bandits Meet Large Language Models](https://arxiv.org/html/2505.13355v1) - We first examine the role of bandit algorithms in optimizing LLM fine-tuning, prompt engineering, an...

22. [Routers & Gateways](https://www.llmreference.com/routers) - A comprehensive reference guide for technology leaders and engineers to navigate AI language models,...

23. [CrewAI vs LangGraph vs AutoGen 2026 - futureagi.com](https://futureagi.com/blog/crewai-vs-langgraph-vs-autogen-2026/) - CrewAI, LangGraph, and AutoGen compared head to head in 2026: architecture, primitives, debug, eval,...

24. [Faq](https://dev.to/emperorakashi20/crewai-vs-langgraph-vs-autogen-which-multi-agent-framework-should-you-use-in-2026-5h2f) - We've built production workflows on all three. An honest breakdown of where each multi-agent framewo...

25. [AutoGen vs LangGraph vs CrewAI: Which Agent ...](https://dev.to/synsun/autogen-vs-langgraph-vs-crewai-which-agent-framework-actually-holds-up-in-2026-3fl8) - Three weeks ago I was staring at a half-broken agent pipeline that was supposed to autonomously...

26. [AI Agent Memory: Types, Architecture & Code [2026] | TECHSY](https://techsy.io/en/blog/ai-agent-memory-guide) - Master AI agent memory in 2026: 5 memory types, 6 framework comparisons (Mem0, Zep, Letta), Python c...

27. [AI Agent Memory in 2026: How Autonomous Systems Remember and ...](https://skycrumbs.com/blog/ai-agent-memory-2026)

28. [Model Context Protocol (MCP) - Stytch](https://stytch.com/blog/model-context-protocol-introduction/) - Model Context Protocol (MCP) is an open standard that bridges AI models with external data and servi...

