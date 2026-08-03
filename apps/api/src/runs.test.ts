import type { AdapterResult, ProviderAdapter } from "@orchestrator/gateway";
import type { ProviderId, UnifiedChatChunk } from "@orchestrator/shared";
import type { FastifyInstance } from "fastify";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { loadConfig } from "./config.js";
import { type Container, buildContainer } from "./container.js";
import { buildServer } from "./server.js";

const API_KEY = "test-key";

/** Returns a canned completion; counts calls so per-node routing can be asserted. */
class StubAdapter implements ProviderAdapter {
  calls = 0;
  constructor(
    readonly provider: ProviderId,
    private readonly reply = "ok",
  ) {}

  async chat(): Promise<AdapterResult> {
    this.calls += 1;
    return {
      message: { role: "assistant", content: this.reply },
      finishReason: "stop",
      usage: { promptTokens: 50, completionTokens: 10, totalTokens: 60, cachedPromptTokens: 0 },
    };
  }

  async *stream(): AsyncGenerator<UnifiedChatChunk, void, undefined> {
    yield { type: "finish", finishReason: "stop" };
  }
}

/** Classify -> checkpoint -> publish. Exercises model nodes, guards, and human-in-the-loop. */
const approvalWorkflow = {
  id: "wf_review",
  entry: "classify",
  nodes: [
    {
      id: "classify",
      type: "model",
      config: {
        prompt: "Classify this ticket: {{ticket}}",
        outputVar: "category",
        route: { mode: "cheap", taskType: "classification" },
      },
    },
    {
      id: "draft",
      type: "model",
      config: {
        prompt: "Draft a reply about {{category}} for: {{ticket}}",
        outputVar: "draft",
        route: { mode: "balanced", taskType: "general" },
      },
    },
    {
      id: "approve",
      type: "checkpoint",
      config: { prompt: "Send this reply? {{draft}}" },
    },
    {
      id: "send",
      type: "transform",
      config: { set: { sent: "yes", final: "{{draft}}" } },
    },
  ],
  edges: [
    { from: "classify", to: "draft" },
    { from: "draft", to: "approve" },
    { from: "approve", to: "send", when: { path: "decision", op: "eq", value: "approve" } },
  ],
};

function makeApp(adapters: ProviderAdapter[]) {
  const config = loadConfig({
    ORCHESTRATOR_API_KEY: API_KEY,
    ORCHESTRATOR_DB_PATH: ":memory:",
    ROUTER_MODE: "shadow",
  } as NodeJS.ProcessEnv);

  const container = buildContainer(config, { adapters });
  return { app: buildServer(container), container };
}

describe("POST /v1/runs", () => {
  let app: FastifyInstance | undefined;
  let container: Container | undefined;
  let adapter: StubAdapter;

  const auth = { authorization: `Bearer ${API_KEY}` };

  beforeEach(() => {
    adapter = new StubAdapter("openai", "billing");
    ({ app, container } = makeApp([adapter]));
  });

  afterEach(async () => {
    await app?.close();
    container?.close();
    app = undefined;
    container = undefined;
  });

  const start = (workflow: unknown, input: Record<string, unknown> = {}) =>
    app?.inject({ method: "POST", url: "/v1/runs", headers: auth, payload: { workflow, input } });

  it("runs a multi-step workflow up to its checkpoint", async () => {
    const response = await start(approvalWorkflow, { ticket: "I was charged twice" });
    expect(response?.statusCode).toBe(200);

    const state = response?.json();
    expect(state.status).toBe("paused");
    // Two model nodes ran; the checkpoint stopped it before `send`.
    expect(adapter.calls).toBe(2);
    expect(state.variables.category).toBe("billing");
    expect(state.pendingPrompt).toBe("Send this reply? billing");
    expect(state.variables.sent).toBeUndefined();
  });

  it("consults the router once per model node, not once per run", async () => {
    // This is the property that makes orchestration worth pairing with adaptive routing: each step
    // is routed on its own intent.
    await start(approvalWorkflow, { ticket: "x" });

    const decisions = container?.decisions.query() ?? [];
    expect(decisions).toHaveLength(2);
    expect(decisions.map((d) => d.taskType)).toEqual(["classification", "general"]);
    expect(decisions.map((d) => d.routeMode)).toEqual(["cheap", "balanced"]);
  });

  it("feeds workflow calls into the same telemetry and reward loop as direct chat", async () => {
    // Workflow traffic is ordinary traffic; excluding it would teach the router from half the system.
    await start(approvalWorkflow, { ticket: "x" });

    const events = container?.events.query() ?? [];
    expect(events).toHaveLength(2);
    for (const event of events) {
      expect(event.status).toBe("success");
      expect(event.reward).not.toBeNull();
      expect(event.routingDecisionId).toBeTruthy();
    }
  });

  it("resumes from the checkpoint and completes", async () => {
    const started = (await start(approvalWorkflow, { ticket: "x" }))?.json();

    const resumed = await app?.inject({
      method: "POST",
      url: `/v1/runs/${started.runId}/resume`,
      headers: auth,
      payload: { workflow: approvalWorkflow, input: { decision: "approve" } },
    });

    const state = resumed?.json();
    expect(state.status).toBe("completed");
    expect(state.variables.sent).toBe("yes");
    expect(state.variables.final).toBe("billing");
  });

  it("takes no branch when the guard rejects, and still terminates", async () => {
    const started = (await start(approvalWorkflow, { ticket: "x" }))?.json();

    const resumed = await app?.inject({
      method: "POST",
      url: `/v1/runs/${started.runId}/resume`,
      headers: auth,
      payload: { workflow: approvalWorkflow, input: { decision: "reject" } },
    });

    const state = resumed?.json();
    expect(state.status).toBe("completed");
    expect(state.variables.sent).toBeUndefined();
  });

  it("accumulates cost across the whole run", async () => {
    const state = (await start(approvalWorkflow, { ticket: "x" }))?.json();
    expect(state.totalCostUsd).toBeGreaterThan(0);
  });

  it("rejects a workflow with a dangling edge before spending anything", async () => {
    const response = await start({
      id: "bad",
      entry: "a",
      nodes: [{ id: "a", type: "transform", config: { set: {} } }],
      edges: [{ from: "a", to: "nowhere" }],
    });

    expect(response?.statusCode).toBe(400);
    expect(response?.json().error.message).toMatch(/Edge to unknown node/);
    expect(adapter.calls).toBe(0);
  });

  it("rejects an entry node that does not exist", async () => {
    const response = await start({
      id: "bad",
      entry: "missing",
      nodes: [{ id: "a", type: "transform", config: { set: {} } }],
    });
    expect(response?.statusCode).toBe(400);
    expect(response?.json().error.message).toMatch(/Entry node does not exist/);
  });

  it("returns derived state plus the raw event log for debugging", async () => {
    const started = (await start(approvalWorkflow, { ticket: "x" }))?.json();

    const response = await app?.inject({
      method: "GET",
      url: `/v1/runs/${started.runId}`,
      headers: auth,
    });

    const body = response?.json();
    expect(body.state.runId).toBe(started.runId);
    const types = body.events.map((e: { type: string }) => e.type);
    expect(types[0]).toBe("run_started");
    expect(types).toContain("node_succeeded");
    expect(types).toContain("run_paused");
  });

  it("404s an unknown run", async () => {
    const response = await app?.inject({
      method: "GET",
      url: "/v1/runs/run_nope",
      headers: auth,
    });
    expect(response?.statusCode).toBe(404);
  });

  it("lists runs for the tenant", async () => {
    await start(approvalWorkflow, { ticket: "a" });
    await start(approvalWorkflow, { ticket: "b" });

    const response = await app?.inject({ method: "GET", url: "/v1/runs", headers: auth });
    expect(response?.json().data).toHaveLength(2);
  });

  it("requires auth", async () => {
    const response = await app?.inject({
      method: "POST",
      url: "/v1/runs",
      payload: { workflow: approvalWorkflow },
    });
    expect(response?.statusCode).toBe(401);
  });

  it("fails the run rather than hanging when a provider is unreachable", async () => {
    // Only an OpenAI adapter is configured; pin a model whose provider is absent.
    const response = await start({
      id: "wf_pin",
      entry: "a",
      nodes: [
        {
          id: "a",
          type: "model",
          config: { prompt: "hi", outputVar: "out", route: { pin: "anthropic/claude-opus-5" } },
        },
      ],
    });

    const state = response?.json();
    expect(state.status).toBe("failed");
    expect(state.error).toBeTruthy();
  });
});
