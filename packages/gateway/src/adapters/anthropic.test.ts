import type { Message, OrchestratorError } from "@orchestrator/shared";
import { describe, expect, it } from "vitest";
import { ANTHROPIC_SPEC, jsonResponse, request, sseResponse, stubFetch } from "../test-helpers.js";
import { AnthropicAdapter, splitSystem } from "./anthropic.js";

const signal = () => new AbortController().signal;

const message = (overrides: Record<string, unknown> = {}) =>
  jsonResponse({
    content: [{ type: "text", text: "hi there" }],
    stop_reason: "end_turn",
    usage: { input_tokens: 12, output_tokens: 4 },
    ...overrides,
  });

describe("AnthropicAdapter.chat", () => {
  it("normalizes a plain completion to the same shape OpenAI produces", async () => {
    const { fetchImpl } = stubFetch([message()]);
    const result = await new AnthropicAdapter({ apiKey: "k", fetchImpl }).chat(
      request(),
      ANTHROPIC_SPEC,
      signal(),
    );

    expect(result.message).toEqual({ role: "assistant", content: "hi there" });
    expect(result.finishReason).toBe("stop");
    expect(result.usage.promptTokens).toBe(12);
    expect(result.usage.completionTokens).toBe(4);
  });

  it("sends the versioned header and api-key auth", async () => {
    const { fetchImpl, calls } = stubFetch([message()]);
    await new AnthropicAdapter({ apiKey: "k", fetchImpl }).chat(
      request(),
      ANTHROPIC_SPEC,
      signal(),
    );

    expect(calls[0]?.headers["x-api-key"]).toBe("k");
    expect(calls[0]?.headers["anthropic-version"]).toBe("2023-06-01");
  });

  it("always sends max_tokens, which the API requires, capped at the model limit", async () => {
    const { fetchImpl, calls } = stubFetch([message(), message()]);
    const adapter = new AnthropicAdapter({ apiKey: "k", fetchImpl });

    await adapter.chat(request(), ANTHROPIC_SPEC, signal());
    expect(calls[0]?.body.max_tokens).toBe(4_096);

    await adapter.chat(request({ maxTokens: 999_999 }), ANTHROPIC_SPEC, signal());
    expect(calls[1]?.body.max_tokens).toBe(ANTHROPIC_SPEC.maxOutputTokens);
  });

  it("clamps temperature into Anthropic's 0..1 range", async () => {
    const { fetchImpl, calls } = stubFetch([message()]);
    await new AnthropicAdapter({ apiKey: "k", fetchImpl }).chat(
      request({ temperature: 1.8 }),
      ANTHROPIC_SPEC,
      signal(),
    );
    expect(calls[0]?.body.temperature).toBe(1);
  });

  it("counts cache reads inside promptTokens so cost stays comparable across providers", async () => {
    const { fetchImpl } = stubFetch([
      message({ usage: { input_tokens: 200, output_tokens: 10, cache_read_input_tokens: 800 } }),
    ]);

    const result = await new AnthropicAdapter({ apiKey: "k", fetchImpl }).chat(
      request(),
      ANTHROPIC_SPEC,
      signal(),
    );

    // Anthropic reports input_tokens EXCLUDING cache reads; the unified contract reports the total
    // the model actually saw, or cached traffic would look free.
    expect(result.usage.promptTokens).toBe(1_000);
    expect(result.usage.cachedPromptTokens).toBe(800);
  });

  it("extracts tool_use blocks into unified tool calls", async () => {
    const { fetchImpl } = stubFetch([
      message({
        content: [
          { type: "text", text: "let me check" },
          { type: "tool_use", id: "tu_1", name: "get_weather", input: { city: "Karachi" } },
        ],
        stop_reason: "tool_use",
      }),
    ]);

    const result = await new AnthropicAdapter({ apiKey: "k", fetchImpl }).chat(
      request(),
      ANTHROPIC_SPEC,
      signal(),
    );

    expect(result.finishReason).toBe("tool_calls");
    expect(result.message.content).toBe("let me check");
    expect(result.message.toolCalls).toEqual([
      { id: "tu_1", name: "get_weather", arguments: { city: "Karachi" } },
    ]);
  });
});

describe("splitSystem", () => {
  it("hoists system messages out of the array into a top-level field", () => {
    const { system, messages } = splitSystem([
      { role: "system", content: "be terse" },
      { role: "user", content: "hi" },
    ] as Message[]);

    expect(system).toBe("be terse");
    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({ role: "user" });
  });

  it("joins multiple system messages rather than dropping any", () => {
    const { system } = splitSystem([
      { role: "system", content: "rule one" },
      { role: "system", content: "rule two" },
      { role: "user", content: "hi" },
    ] as Message[]);
    expect(system).toBe("rule one\n\nrule two");
  });

  it("merges consecutive tool results into one user turn", () => {
    // Two adjacent user messages are rejected by the API, so parallel tool results must combine.
    const { messages } = splitSystem([
      { role: "user", content: "weather in both cities?" },
      {
        role: "assistant",
        content: "",
        toolCalls: [
          { id: "t1", name: "w", arguments: { city: "Karachi" } },
          { id: "t2", name: "w", arguments: { city: "Lahore" } },
        ],
      },
      { role: "tool", content: "31C", toolCallId: "t1" },
      { role: "tool", content: "35C", toolCallId: "t2" },
    ] as Message[]);

    expect(messages).toHaveLength(3);
    const toolTurn = messages[2] as { role: string; content: unknown[] };
    expect(toolTurn.role).toBe("user");
    expect(toolTurn.content).toHaveLength(2);
    expect(toolTurn.content[0]).toMatchObject({ type: "tool_result", tool_use_id: "t1" });
    expect(toolTurn.content[1]).toMatchObject({ type: "tool_result", tool_use_id: "t2" });
  });

  it("emits tool_use blocks for an assistant turn that only called tools", () => {
    const { messages } = splitSystem([
      {
        role: "assistant",
        content: "",
        toolCalls: [{ id: "t1", name: "w", arguments: { city: "Karachi" } }],
      },
    ] as Message[]);

    const content = (messages[0] as { content: unknown[] }).content;
    expect(content).toHaveLength(1);
    expect(content[0]).toMatchObject({ type: "tool_use", id: "t1", name: "w" });
  });
});

describe("AnthropicAdapter error classification", () => {
  it("maps overloaded_error to provider_unavailable, which is retryable", async () => {
    const { fetchImpl } = stubFetch([
      jsonResponse({ error: { type: "overloaded_error", message: "Overloaded" } }, { status: 529 }),
    ]);

    const err = (await new AnthropicAdapter({ apiKey: "k", fetchImpl })
      .chat(request(), ANTHROPIC_SPEC, signal())
      .catch((e: unknown) => e)) as OrchestratorError;

    expect(err.errorClass).toBe("provider_unavailable");
    expect(err.retryable).toBe(true);
  });

  it("recognizes an over-long prompt inside a generic invalid_request_error", async () => {
    const { fetchImpl } = stubFetch([
      jsonResponse(
        { error: { type: "invalid_request_error", message: "prompt is too long: 250000 tokens" } },
        { status: 400 },
      ),
    ]);

    const err = (await new AnthropicAdapter({ apiKey: "k", fetchImpl })
      .chat(request(), ANTHROPIC_SPEC, signal())
      .catch((e: unknown) => e)) as OrchestratorError;

    expect(err.errorClass).toBe("context_length_exceeded");
    expect(err.fallbackEligible).toBe(true);
  });

  it("leaves a genuinely malformed request terminal", async () => {
    const { fetchImpl } = stubFetch([
      jsonResponse(
        { error: { type: "invalid_request_error", message: "messages: field required" } },
        { status: 400 },
      ),
    ]);

    const err = (await new AnthropicAdapter({ apiKey: "k", fetchImpl })
      .chat(request(), ANTHROPIC_SPEC, signal())
      .catch((e: unknown) => e)) as OrchestratorError;

    expect(err.errorClass).toBe("invalid_request");
    expect(err.fallbackEligible).toBe(false);
  });
});

describe("AnthropicAdapter.stream", () => {
  it("assembles usage from message_start and message_delta", async () => {
    const frames = [
      'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":20,"cache_read_input_tokens":5}}}\n\n',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n',
      'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n',
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}\n\n',
      'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ];
    const { fetchImpl } = stubFetch([sseResponse(frames)]);

    const chunks = [];
    for await (const chunk of new AnthropicAdapter({ apiKey: "k", fetchImpl }).stream(
      request(),
      ANTHROPIC_SPEC,
      signal(),
    )) {
      chunks.push(chunk);
    }

    expect(chunks.filter((c) => c.type === "text").map((c) => c.delta)).toEqual(["Hel", "lo"]);

    const finish = chunks.at(-1);
    expect(finish).toMatchObject({ type: "finish", finishReason: "stop" });
    // Input tokens are only ever reported once, at message_start.
    expect(finish?.type === "finish" && finish.usage).toMatchObject({
      promptTokens: 25,
      completionTokens: 9,
      cachedPromptTokens: 5,
    });
  });

  it("surfaces streamed tool calls as start plus argument fragments", async () => {
    const frames = [
      'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}\n\n',
      'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"get_weather"}}\n\n',
      'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":"}}\n\n',
      'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}\n\n',
    ];
    const { fetchImpl } = stubFetch([sseResponse(frames)]);

    const chunks = [];
    for await (const chunk of new AnthropicAdapter({ apiKey: "k", fetchImpl }).stream(
      request(),
      ANTHROPIC_SPEC,
      signal(),
    )) {
      chunks.push(chunk);
    }

    const toolChunks = chunks.filter((c) => c.type === "tool_call");
    expect(toolChunks[0]).toMatchObject({ id: "tu_1", name: "get_weather" });
    expect(toolChunks[1]).toMatchObject({ argumentsDelta: '{"city":' });
    expect(chunks.at(-1)).toMatchObject({ finishReason: "tool_calls" });
  });
});
