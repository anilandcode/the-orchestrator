import type { AdapterResult, ProviderAdapter } from "@orchestrator/gateway";
import type { Message, ProviderId, UnifiedChatChunk } from "@orchestrator/shared";
import type { FastifyInstance } from "fastify";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { loadConfig } from "./config.js";
import { type Container, buildContainer } from "./container.js";
import { buildServer } from "./server.js";

const API_KEY = "test-key";

/** Captures the messages it was sent, so injected memory context can be asserted on. */
class CapturingAdapter implements ProviderAdapter {
  seen: Message[][] = [];

  constructor(
    readonly provider: ProviderId,
    private readonly reply = "acknowledged",
  ) {}

  async chat(request: { messages: Message[] }): Promise<AdapterResult> {
    this.seen.push(request.messages);
    return {
      message: { role: "assistant", content: this.reply },
      finishReason: "stop",
      usage: { promptTokens: 40, completionTokens: 8, totalTokens: 48, cachedPromptTokens: 0 },
    };
  }

  async *stream(): AsyncGenerator<UnifiedChatChunk, void, undefined> {
    yield { type: "finish", finishReason: "stop" };
  }
}

function makeApp(adapters: ProviderAdapter[]) {
  const config = loadConfig({
    ORCHESTRATOR_API_KEY: API_KEY,
    ORCHESTRATOR_DB_PATH: ":memory:",
    ROUTER_MODE: "shadow",
  } as NodeJS.ProcessEnv);

  const container = buildContainer(config, { adapters });
  return { app: buildServer(container), container };
}

describe("memory in /v1/chat", () => {
  let app: FastifyInstance | undefined;
  let container: Container | undefined;
  let adapter: CapturingAdapter;

  const auth = { authorization: `Bearer ${API_KEY}` };

  beforeEach(() => {
    adapter = new CapturingAdapter("openai");
    ({ app, container } = makeApp([adapter]));
  });

  afterEach(async () => {
    await app?.close();
    container?.close();
    app = undefined;
    container = undefined;
  });

  const chat = (content: string, memory?: Record<string, unknown>) =>
    app?.inject({
      method: "POST",
      url: "/v1/chat",
      headers: auth,
      payload: { messages: [{ role: "user", content }], ...(memory ? { memory } : {}) },
    });

  it("does nothing unless memory is requested", async () => {
    // Silently prepending context would change what the model sees and what it costs, for a caller
    // who never asked for it.
    await chat("My account number is 4471 and I was double charged in March");
    await chat("What did I tell you about my account?");

    expect(container?.memory.forget({ tenantId: "local" })).toBe(0);
    expect(adapter.seen[1]?.every((m) => m.role === "user")).toBe(true);
  });

  it("stores turns and recalls them on a later request in the same session", async () => {
    await chat("My account number is 4471 and I was double charged in March", {
      sessionId: "s1",
    });
    await chat("Remind me about the account number problem I mentioned", { sessionId: "s1" });

    const secondCall = adapter.seen[1] as Message[];
    const systemContext = secondCall.find((m) => m.role === "system");
    expect(systemContext).toBeDefined();
    expect(String(systemContext?.content)).toMatch(/4471/);
  });

  it("keeps sessions separate", async () => {
    await chat("My secret project codename is Bluebird and it ships in autumn", {
      sessionId: "s1",
    });
    await chat("What is the project codename that I mentioned earlier?", { sessionId: "s2" });

    const secondCall = adapter.seen[1] as Message[];
    expect(secondCall.some((m) => String(m.content).includes("Bluebird"))).toBe(false);
  });

  it("does not compound recalled context back into memory", async () => {
    // Writing the injected context back would make each turn remember its own memory, growing
    // without bound.
    await chat("My account number is 4471 and I was double charged in March", { sessionId: "s1" });
    await chat("Remind me about the account number problem I mentioned", { sessionId: "s1" });

    const recall = await app?.inject({
      method: "POST",
      url: "/v1/memory/recall",
      headers: auth,
      payload: { sessionId: "s1", query: "account" },
    });

    const stored = recall?.json().buffer as { text: string }[];
    expect(stored.some((item) => item.text.startsWith("Relevant earlier context"))).toBe(false);
    expect(stored.some((item) => item.text.startsWith("Recent conversation"))).toBe(false);
  });

  it("honours recall:false — write only", async () => {
    await chat("My account number is 4471 and I was double charged in March", { sessionId: "s1" });
    await chat("Anything about my account details from before?", {
      sessionId: "s1",
      recall: false,
    });

    expect((adapter.seen[1] as Message[]).some((m) => m.role === "system")).toBe(false);
  });

  it("honours write:false — recall only", async () => {
    await chat("My account number is 4471 and I was double charged in March", { sessionId: "s1" });
    await chat("A message that should not be stored anywhere in the session", {
      sessionId: "s1",
      write: false,
    });

    const recall = await app?.inject({
      method: "POST",
      url: "/v1/memory/recall",
      headers: auth,
      payload: { sessionId: "s1", query: "stored" },
    });

    const stored = recall?.json().buffer as { text: string }[];
    expect(stored.some((item) => item.text.includes("should not be stored"))).toBe(false);
  });
});

describe("memory endpoints", () => {
  let app: FastifyInstance | undefined;
  let container: Container | undefined;
  const auth = { authorization: `Bearer ${API_KEY}` };

  beforeEach(() => {
    ({ app, container } = makeApp([new CapturingAdapter("openai")]));
  });

  afterEach(async () => {
    await app?.close();
    container?.close();
    app = undefined;
    container = undefined;
  });

  it("stores a durable fact that surfaces in a brand-new session", async () => {
    await app?.inject({
      method: "POST",
      url: "/v1/memory/facts",
      headers: auth,
      payload: { text: "The customer always wants invoices sent to accounts@example.com" },
    });

    const recall = await app?.inject({
      method: "POST",
      url: "/v1/memory/recall",
      headers: auth,
      payload: { sessionId: "never-seen-before", query: "where should invoices be sent" },
    });

    const body = recall?.json();
    expect(body.recalled.length).toBeGreaterThan(0);
    expect(body.context).toMatch(/accounts@example.com/);
  });

  it("forgets a session on request", async () => {
    await app?.inject({
      method: "POST",
      url: "/v1/chat",
      headers: auth,
      payload: {
        messages: [{ role: "user", content: "Something substantive to store in this session" }],
        memory: { sessionId: "doomed" },
      },
    });

    const response = await app?.inject({
      method: "DELETE",
      url: "/v1/memory/sessions/doomed",
      headers: auth,
    });

    expect(response?.json().forgotten).toBeGreaterThan(0);
    expect(container?.memory.forget({ tenantId: "local", sessionId: "doomed" })).toBe(0);
  });

  it("rejects a malformed recall request", async () => {
    const response = await app?.inject({
      method: "POST",
      url: "/v1/memory/recall",
      headers: auth,
      payload: { sessionId: "s" },
    });
    expect(response?.statusCode).toBe(400);
  });

  it("requires auth", async () => {
    const response = await app?.inject({
      method: "POST",
      url: "/v1/memory/facts",
      payload: { text: "x" },
    });
    expect(response?.statusCode).toBe(401);
  });
});
