import type { AdapterResult, ProviderAdapter } from "@orchestrator/gateway";
import {
  OrchestratorError,
  type ProviderId,
  type UnifiedChatChunk,
  type UnifiedChatResponse,
} from "@orchestrator/shared";
import type { FastifyInstance } from "fastify";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { loadConfig } from "./config.js";
import { type Container, buildContainer } from "./container.js";
import { buildServer } from "./server.js";

const API_KEY = "test-key";

class StubAdapter implements ProviderAdapter {
  calls = 0;
  constructor(
    readonly provider: ProviderId,
    private readonly failWith?: OrchestratorError,
  ) {}

  async chat(): Promise<AdapterResult> {
    this.calls += 1;
    if (this.failWith) throw this.failWith;
    return {
      message: { role: "assistant", content: "pong" },
      finishReason: "stop",
      usage: { promptTokens: 120, completionTokens: 8, totalTokens: 128, cachedPromptTokens: 0 },
    };
  }

  async *stream(): AsyncGenerator<UnifiedChatChunk, void, undefined> {
    yield { type: "finish", finishReason: "stop" };
  }
}

function makeApp(adapters: ProviderAdapter[], env: Record<string, string> = {}) {
  const config = loadConfig({
    ORCHESTRATOR_API_KEY: API_KEY,
    ORCHESTRATOR_DB_PATH: ":memory:",
    ROUTER_MODE: "shadow",
    ...env,
  } as NodeJS.ProcessEnv);

  const container = buildContainer(config, { adapters });
  return { app: buildServer(container), container };
}

const chatBody = (overrides: Record<string, unknown> = {}) => ({
  messages: [{ role: "user", content: "ping" }],
  route: { mode: "cheap", taskType: "general" },
  ...overrides,
});

describe("API", () => {
  let app: FastifyInstance | undefined;
  let container: Container | undefined;

  afterEach(async () => {
    await app?.close();
    container?.close();
    // Cleared so a test that builds no app does not tear down the previous test's container.
    app = undefined;
    container = undefined;
  });

  describe("auth", () => {
    beforeEach(() => {
      ({ app, container } = makeApp([new StubAdapter("openai")]));
    });

    it("rejects a request with no key", async () => {
      const response = await app.inject({ method: "POST", url: "/v1/chat", payload: chatBody() });
      expect(response.statusCode).toBe(401);
    });

    it("rejects a wrong key", async () => {
      const response = await app.inject({
        method: "POST",
        url: "/v1/chat",
        headers: { authorization: "Bearer nope" },
        payload: chatBody(),
      });
      expect(response.statusCode).toBe(401);
    });

    it("leaves the health check open", async () => {
      const response = await app.inject({ method: "GET", url: "/healthz" });
      expect(response.statusCode).toBe(200);
      expect(response.json()).toMatchObject({ status: "ok", routerMode: "shadow" });
    });
  });

  describe("POST /v1/chat", () => {
    beforeEach(() => {
      ({ app, container } = makeApp([new StubAdapter("openai"), new StubAdapter("anthropic")]));
    });

    const post = (payload: unknown) =>
      app.inject({
        method: "POST",
        url: "/v1/chat",
        headers: { authorization: `Bearer ${API_KEY}` },
        payload,
      });

    it("routes, executes, and returns a normalized response", async () => {
      const response = await post(chatBody());
      expect(response.statusCode).toBe(200);

      const body = response.json<UnifiedChatResponse>();
      expect(body.message.content).toBe("pong");
      expect(body.attempts).toBe(1);
      expect(body.costUsd).toBeGreaterThan(0);
      expect(body.routingDecisionId).toBeTruthy();
    });

    it("records a CallEvent with cost, latency, and the routing decision", async () => {
      const body = (await post(chatBody())).json<UnifiedChatResponse>();
      const events = container.events.query({ requestId: body.requestId });

      expect(events).toHaveLength(1);
      expect(events[0]).toMatchObject({
        status: "success",
        modelId: body.modelId,
        routingDecisionId: body.routingDecisionId,
        taskType: "general",
        routeMode: "cheap",
      });
      expect(events[0]?.costUsd).toBeGreaterThan(0);
    });

    it("persists the routing decision so shadow mode can be analysed later", async () => {
      await post(chatBody());
      const decisions = container.decisions.query();

      expect(decisions).toHaveLength(1);
      // In shadow mode the bandit's counterfactual is recorded beside what ran.
      expect(decisions[0]?.strategy).toBe("static");
      expect(decisions[0]?.reason).toBeTruthy();
    });

    it("scores the event, which is what closes the feedback loop", async () => {
      const body = (await post(chatBody())).json<UnifiedChatResponse>();
      const [event] = container.events.query({ requestId: body.requestId });

      expect(event?.reward).not.toBeNull();
      expect(event?.reward).toBeGreaterThan(0);
      expect(event?.qualityScore).not.toBeNull();
    });

    it("rejects a malformed body with 400", async () => {
      const response = await post({ messages: [] });
      expect(response.statusCode).toBe(400);
      expect(response.json().error.type).toBe("invalid_request");
    });

    it("maps a provider failure onto a sensible status and still records it", async () => {
      const rateLimited = new OrchestratorError("rate_limit", "429 slow down");
      ({ app, container } = makeApp([new StubAdapter("openai", rateLimited)]));

      const response = await app.inject({
        method: "POST",
        url: "/v1/chat",
        headers: { authorization: `Bearer ${API_KEY}` },
        payload: chatBody(),
      });

      expect(response.statusCode).toBe(429);
      expect(response.json().error).toMatchObject({ type: "rate_limit", retryable: true });
      // A failed request is still evidence — the attempts must have been logged.
      expect(container.events.query().length).toBeGreaterThan(0);
      expect(container.events.query()[0]?.status).toBe("error");
    });

    it("honours an explicit pin", async () => {
      const response = await post(chatBody({ route: { pin: "anthropic/claude-opus-5" } }));
      expect(response.json<UnifiedChatResponse>().modelId).toBe("anthropic/claude-opus-5");
      expect(container.decisions.query()[0]?.strategy).toBe("pinned");
    });

    it("refuses a request no reachable model can satisfy", async () => {
      const response = await post(chatBody({ route: { mode: "cheap", maxCostUsd: 0.000_000_1 } }));
      expect(response.statusCode).toBe(400);
      expect(response.json().error.message).toMatch(/No model satisfies/);
    });
  });

  describe("GET /v1/models", () => {
    it("lists only models the configured keys can actually reach", async () => {
      ({ app, container } = makeApp([new StubAdapter("openai")]));

      const response = await app.inject({
        method: "GET",
        url: "/v1/models",
        headers: { authorization: `Bearer ${API_KEY}` },
      });

      const providers = new Set(
        response.json<{ data: { provider: string }[] }>().data.map((model) => model.provider),
      );
      expect(providers).toEqual(new Set(["openai"]));
    });
  });

  describe("POST /v1/feedback", () => {
    beforeEach(() => {
      ({ app, container } = makeApp([new StubAdapter("openai")]));
    });

    it("re-scores an event with an explicit quality signal", async () => {
      const chat = await app.inject({
        method: "POST",
        url: "/v1/chat",
        headers: { authorization: `Bearer ${API_KEY}` },
        payload: chatBody(),
      });
      const { requestId } = chat.json<UnifiedChatResponse>();
      const rewardBefore = container.events.query({ requestId })[0]?.reward as number;

      const feedback = await app.inject({
        method: "POST",
        url: "/v1/feedback",
        headers: { authorization: `Bearer ${API_KEY}` },
        payload: { requestId, quality: 0.1 },
      });

      expect(feedback.statusCode).toBe(200);
      const rewardAfter = container.events.query({ requestId })[0]?.reward as number;
      // A poor explicit rating must pull the reward below the optimistic heuristic default.
      expect(rewardAfter).toBeLessThan(rewardBefore);
      expect(container.events.query({ requestId })[0]?.qualityScore).toBe(0.1);
    });

    it("404s on an unknown requestId rather than silently scoring nothing", async () => {
      const response = await app.inject({
        method: "POST",
        url: "/v1/feedback",
        headers: { authorization: `Bearer ${API_KEY}` },
        payload: { requestId: "req_never_existed", quality: 0.5 },
      });
      expect(response.statusCode).toBe(404);
    });
  });

  describe("GET /v1/stats", () => {
    it("summarizes traffic and router state", async () => {
      ({ app, container } = makeApp([new StubAdapter("openai")]));

      await app.inject({
        method: "POST",
        url: "/v1/chat",
        headers: { authorization: `Bearer ${API_KEY}` },
        payload: chatBody(),
      });

      const response = await app.inject({
        method: "GET",
        url: "/v1/stats",
        headers: { authorization: `Bearer ${API_KEY}` },
      });

      const body = response.json();
      expect(body.summary.requests).toBe(1);
      expect(body.models).toHaveLength(1);
      expect(body.router).toMatchObject({ mode: "shadow", decisions: 1 });
    });
  });

  describe("config", () => {
    it("defaults ROUTER_MODE to shadow", () => {
      expect(loadConfig({} as NodeJS.ProcessEnv).routerMode).toBe("shadow");
    });

    it("rejects an invalid ROUTER_MODE instead of silently falling back", () => {
      expect(() => loadConfig({ ROUTER_MODE: "yolo" } as NodeJS.ProcessEnv)).toThrow(
        /Invalid ROUTER_MODE/,
      );
    });
  });
});
