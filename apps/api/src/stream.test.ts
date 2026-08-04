import type { AdapterResult, ProviderAdapter } from "@orchestrator/gateway";
import { OrchestratorError, type ProviderId, type UnifiedChatChunk } from "@orchestrator/shared";
import type { FastifyInstance } from "fastify";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { loadConfig } from "./config.js";
import { type Container, buildContainer } from "./container.js";
import { buildServer } from "./server.js";

const API_KEY = "test-key";

/** Streams a fixed script, optionally dying partway through. */
class StreamingAdapter implements ProviderAdapter {
  constructor(
    readonly provider: ProviderId,
    private readonly pieces: string[] = ["Hel", "lo"],
    private readonly failAfter?: number,
  ) {}

  async chat(): Promise<AdapterResult> {
    return {
      message: { role: "assistant", content: this.pieces.join("") },
      finishReason: "stop",
      usage: { promptTokens: 20, completionTokens: 5, totalTokens: 25, cachedPromptTokens: 0 },
    };
  }

  async *stream(): AsyncGenerator<UnifiedChatChunk, void, undefined> {
    for (const [index, delta] of this.pieces.entries()) {
      if (this.failAfter !== undefined && index === this.failAfter) {
        throw new OrchestratorError("provider_unavailable", "stream died");
      }
      yield { type: "text", delta };
    }
    yield {
      type: "finish",
      finishReason: "stop",
      usage: { promptTokens: 20, completionTokens: 5, totalTokens: 25, cachedPromptTokens: 0 },
    };
  }
}

function parseSse(body: string): { event: string; data: unknown }[] {
  return body
    .split("\n\n")
    .filter((block) => block.trim())
    .map((block) => {
      const event = block.match(/^event: (.*)$/m)?.[1] ?? "";
      const data = block.match(/^data: (.*)$/m)?.[1] ?? "{}";
      return { event, data: JSON.parse(data) as unknown };
    });
}

describe("streaming /v1/chat", () => {
  let app!: FastifyInstance;
  let container!: Container;
  // Guards teardown for describe blocks that never build a server.
  let live = false;

  const auth = { authorization: `Bearer ${API_KEY}` };

  const make = (adapter: ProviderAdapter) => {
    const config = loadConfig({
      ORCHESTRATOR_API_KEY: API_KEY,
      ORCHESTRATOR_DB_PATH: ":memory:",
      ROUTER_MODE: "shadow",
    } as NodeJS.ProcessEnv);

    const built = buildContainer(config, { adapters: [adapter] });
    return { app: buildServer(built), container: built };
  };

  afterEach(async () => {
    if (!live) return;
    live = false;
    await app.close();
    container.close();
  });

  beforeEach(() => {
    ({ app, container } = make(new StreamingAdapter("openai")));
    live = true;
  });

  const post = (payload: Record<string, unknown>) =>
    app.inject({ method: "POST", url: "/v1/chat", headers: auth, payload });

  it("streams text deltas and a done frame", async () => {
    const response = await post({
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });

    expect(response.headers["content-type"]).toMatch(/text\/event-stream/);
    const frames = parseSse(response.body as string);

    expect(frames.map((f) => f.event)).toContain("text");
    expect(frames.at(-1)?.event).toBe("done");
  });

  it("sends correlation metadata before the body", async () => {
    // A client needs the requestId to submit feedback later, and should not have to wait for the
    // answer to learn it.
    const frames = parseSse(
      (
        await post({
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        })
      )?.body as string,
    );

    expect(frames[0]?.event).toBe("meta");
    expect(frames[0]?.data).toMatchObject({ modelId: expect.any(String) });
  });

  it("still records telemetry for a streamed call", async () => {
    // A streamed request that skipped scoring would be traffic the router never learns from.
    await post({ messages: [{ role: "user", content: "hi" }], stream: true });

    const events = container.events.query() ?? [];
    expect(events).toHaveLength(1);
    expect(events[0]?.status).toBe("success");
    expect(events[0]?.reward).not.toBeNull();
  });

  it("records a routing decision for a streamed call", async () => {
    await post({ messages: [{ role: "user", content: "hi" }], stream: true });
    expect(container.decisions.query()).toHaveLength(1);
  });

  it("delivers a mid-stream failure as an error frame, not a truncated success", async () => {
    // Headers are long gone by then, so a status code is not available. Silently truncating would
    // look to a client like a short but successful answer.
    ({ app, container } = make(new StreamingAdapter("openai", ["one", "two", "three"], 1)));
    live = true;

    const response = await post({
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });

    const frames = parseSse(response.body as string);
    expect(frames.some((f) => f.event === "text")).toBe(true);
    expect(frames.at(-1)?.event).toBe("error");
    expect(frames.at(-1)?.data).toMatchObject({ type: "provider_unavailable" });
  });

  it("scores a failed stream too", async () => {
    ({ app, container } = make(new StreamingAdapter("openai", ["one", "two"], 1)));
    live = true;
    await post({ messages: [{ role: "user", content: "hi" }], stream: true });

    // A failed attempt is still evidence about the model that failed.
    const events = container.events.query() ?? [];
    expect(events.length).toBeGreaterThan(0);
  });

  it("leaves non-streaming requests returning ordinary JSON", async () => {
    const response = await post({ messages: [{ role: "user", content: "hi" }] });
    expect(response.headers["content-type"]).toMatch(/application\/json/);
    expect(response.json().message.content).toBe("Hello");
  });
});
