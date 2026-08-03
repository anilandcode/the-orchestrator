import { describe, expect, it } from "vitest";
import { CallEventSchema } from "./call-event.js";
import { UnifiedChatRequestSchema, UnifiedChatResponseSchema } from "./chat.js";

describe("UnifiedChatRequestSchema", () => {
  it("fills routing defaults so a bare request is still routable", () => {
    const parsed = UnifiedChatRequestSchema.parse({
      messages: [{ role: "user", content: "hi" }],
    });
    expect(parsed.route.mode).toBe("balanced");
    expect(parsed.route.taskType).toBe("general");
    expect(parsed.tenantId).toBe("local");
    expect(parsed.stream).toBe(false);
  });

  it("keeps an explicit pin, which is the escape hatch around routing", () => {
    const parsed = UnifiedChatRequestSchema.parse({
      messages: [{ role: "user", content: "hi" }],
      route: { pin: "openai/gpt-4o" },
    });
    expect(parsed.route.pin).toBe("openai/gpt-4o");
  });

  it("rejects an empty message list", () => {
    expect(() => UnifiedChatRequestSchema.parse({ messages: [] })).toThrow();
  });

  it("accepts multimodal content parts alongside plain strings", () => {
    const parsed = UnifiedChatRequestSchema.parse({
      messages: [
        { role: "user", content: [{ type: "text", text: "what is this" }] },
        { role: "user", content: "and this" },
      ],
    });
    expect(parsed.messages).toHaveLength(2);
  });

  it("round-trips a tool-calling exchange", () => {
    const parsed = UnifiedChatRequestSchema.parse({
      messages: [
        { role: "user", content: "weather in Karachi" },
        {
          role: "assistant",
          content: "",
          toolCalls: [{ id: "call_1", name: "get_weather", arguments: { city: "Karachi" } }],
        },
        { role: "tool", content: "31C", toolCallId: "call_1" },
      ],
      tools: [{ name: "get_weather", parameters: { type: "object" } }],
    });
    expect(parsed.messages[1]?.toolCalls?.[0]?.arguments).toEqual({ city: "Karachi" });
    expect(parsed.messages[2]?.toolCallId).toBe("call_1");
  });

  it("rejects a temperature outside the normalized range", () => {
    expect(() =>
      UnifiedChatRequestSchema.parse({
        messages: [{ role: "user", content: "x" }],
        temperature: 5,
      }),
    ).toThrow();
  });
});

describe("UnifiedChatResponseSchema", () => {
  it("requires an attempt count so retries are always visible", () => {
    const base = {
      id: "resp_1",
      requestId: "req_1",
      provider: "openai" as const,
      modelId: "openai/gpt-4o-mini",
      message: { role: "assistant" as const, content: "hello" },
      finishReason: "stop" as const,
      usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15, cachedPromptTokens: 0 },
      costUsd: 0.0001,
      latencyMs: 400,
    };
    expect(() => UnifiedChatResponseSchema.parse(base)).toThrow();
    expect(UnifiedChatResponseSchema.parse({ ...base, attempts: 1 }).attempts).toBe(1);
  });
});

describe("CallEventSchema", () => {
  const base = {
    id: "evt_1",
    tenantId: "local",
    requestId: "req_1",
    attempt: 1,
    provider: "anthropic" as const,
    modelId: "anthropic/claude-haiku-4-5",
    taskType: "code" as const,
    routeMode: "cheap" as const,
    promptTokens: 100,
    completionTokens: 20,
    costUsd: 0.0002,
    latencyMs: 850,
    status: "success" as const,
    createdAt: 1_700_000_000_000,
  };

  it("defaults the unknown-at-write-time fields to null rather than dropping them", () => {
    const parsed = CallEventSchema.parse(base);
    // Reward and quality are computed later, once the outcome is scored.
    expect(parsed.reward).toBeNull();
    expect(parsed.qualityScore).toBeNull();
    expect(parsed.routingDecisionId).toBeNull();
    expect(parsed.features).toEqual([]);
  });

  it("carries the routing decision id, which is how reward gets attributed back to an arm", () => {
    const parsed = CallEventSchema.parse({ ...base, routingDecisionId: "dec_1", attempt: 3 });
    expect(parsed.routingDecisionId).toBe("dec_1");
    expect(parsed.attempt).toBe(3);
  });

  it("rejects a reward outside 0..1", () => {
    expect(() => CallEventSchema.parse({ ...base, reward: 1.5 })).toThrow();
  });

  it("records an error class on failed attempts", () => {
    const parsed = CallEventSchema.parse({
      ...base,
      status: "error",
      errorClass: "rate_limit",
      completionTokens: 0,
      costUsd: 0,
    });
    expect(parsed.errorClass).toBe("rate_limit");
  });
});
