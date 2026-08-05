import type { AdapterResult, ProviderAdapter } from "@orchestrator/gateway";
import type { Message, ProviderId, UnifiedChatChunk } from "@orchestrator/shared";
import type { FastifyInstance } from "fastify";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { loadConfig } from "./config.js";
import { type Container, buildContainer } from "./container.js";
import { buildServer } from "./server.js";

const API_KEY = "test-key";

class CapturingAdapter implements ProviderAdapter {
  seen: Message[][] = [];
  constructor(readonly provider: ProviderId) {}

  async chat(request: { messages: Message[] }): Promise<AdapterResult> {
    this.seen.push(request.messages);
    return {
      message: { role: "assistant", content: "ok" },
      finishReason: "stop",
      usage: { promptTokens: 30, completionTokens: 5, totalTokens: 35, cachedPromptTokens: 0 },
    };
  }

  async *stream(): AsyncGenerator<UnifiedChatChunk, void, undefined> {
    yield { type: "finish", finishReason: "stop" };
  }
}

/**
 * Tenant isolation must not be reachable from request input.
 *
 * The stores enforce scoping correctly, but enforcement is only ever as good as the `tenantId` handed
 * to them. If a caller can name the tenant, every store-level guarantee in the system evaporates at
 * once — memory, tool policy, run history, and usage attribution all key off that single value.
 */
describe("tenant isolation", () => {
  let app!: FastifyInstance;
  let container!: Container;
  let adapter: CapturingAdapter;
  let live = false;

  const auth = { authorization: `Bearer ${API_KEY}` };

  beforeEach(() => {
    adapter = new CapturingAdapter("openai");
    const config = loadConfig({
      ORCHESTRATOR_API_KEY: API_KEY,
      ORCHESTRATOR_DB_PATH: ":memory:",
      ROUTER_MODE: "shadow",
      DEFAULT_TENANT_ID: "acme",
    } as NodeJS.ProcessEnv);

    container = buildContainer(config, { adapters: [adapter] });
    app = buildServer(container);
    live = true;
  });

  afterEach(async () => {
    if (!live) return;
    live = false;
    await app.close();
    container.close();
  });

  const chat = (payload: Record<string, unknown>) =>
    app.inject({ method: "POST", url: "/v1/chat", headers: auth, payload });

  it("ignores a tenantId supplied in the request body", async () => {
    // The attack: name someone else's tenant and write into their memory.
    // The message has to clear the write policy's length floor, or nothing is stored under any
    // tenant and the test would pass for the wrong reason.
    await chat({
      messages: [{ role: "user", content: "My account number is 4471 and I was charged twice" }],
      tenantId: "victim",
      memory: { sessionId: "s1" },
    });

    // The turn must land under the authenticated tenant, never the one the caller asked for.
    expect(container.memory.forget({ tenantId: "victim" })).toBe(0);
    expect(container.memory.forget({ tenantId: "acme" })).toBeGreaterThan(0);
  });

  it("does not recall another tenant's memory when the body names them", async () => {
    // Seed a victim tenant directly, as a second customer's traffic would.
    await container.memory.rememberFact({
      tenantId: "victim",
      text: "The victim deployment key is SWORDFISH-9931 and rotates every Friday",
    });

    await chat({
      // The question deliberately shares no distinctive wording with the secret. Asserting against
      // the whole prompt would otherwise match the attacker's own question rather than any leak —
      // a test that passes whether or not the isolation works.
      messages: [{ role: "user", content: "Remind me of the deployment credential please" }],
      tenantId: "victim",
      memory: { sessionId: "attacker-session" },
    });

    const prompt = JSON.stringify(adapter.seen[0]);
    expect(prompt).not.toMatch(/SWORDFISH-9931/);
    // Nothing was recallable for the authenticated tenant, so no context was injected at all.
    expect(adapter.seen[0]?.some((message) => message.role === "system")).toBe(false);
  });

  it("does not let one tenant rescore another tenant's call", async () => {
    // Feedback is the highest-authority quality signal there is. An unscoped lookup would let anyone
    // who guessed a requestId poison the reward another tenant's router learns from.
    const victimEvents = container.events.query({ tenantId: "acme" });
    expect(victimEvents).toHaveLength(0);

    const response = await chat({ messages: [{ role: "user", content: "hello there" }] });
    const { requestId } = response.json() as { requestId: string };

    // Move the event to another tenant, simulating a second customer's traffic.
    const stored = container.events.query({ requestId })[0];
    expect(stored).toBeDefined();
    container.events.record({ ...(stored as NonNullable<typeof stored>), tenantId: "victim" });

    const feedback = await app.inject({
      method: "POST",
      url: "/v1/feedback",
      headers: auth,
      payload: { requestId, quality: 0 },
    });

    expect(feedback.statusCode).toBe(404);
  });

  it("attributes telemetry to the authenticated tenant", async () => {
    // Usage attribution is billing. A caller who can name the tenant can bill someone else.
    await chat({ messages: [{ role: "user", content: "hi" }], tenantId: "victim" });

    expect(container.events.query({ tenantId: "victim" })).toHaveLength(0);
    expect(container.events.query({ tenantId: "acme" })).toHaveLength(1);
  });

  it("ignores a spoofed tenantId on a streamed request too", async () => {
    await chat({
      messages: [{ role: "user", content: "hi" }],
      tenantId: "victim",
      stream: true,
    });

    expect(container.events.query({ tenantId: "victim" })).toHaveLength(0);
  });

  it("ignores a spoofed tenantId on workflow runs", async () => {
    await app.inject({
      method: "POST",
      url: "/v1/runs",
      headers: auth,
      payload: {
        tenantId: "victim",
        workflow: {
          id: "wf",
          entry: "a",
          nodes: [{ id: "a", type: "transform", config: { set: { x: "1" } } }],
        },
      },
    });

    expect(container.runs.list({ tenantId: "victim" })).toHaveLength(0);
    expect(container.runs.list({ tenantId: "acme" })).toHaveLength(1);
  });
});
