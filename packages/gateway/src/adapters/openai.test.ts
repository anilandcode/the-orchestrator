import { OrchestratorError } from "@orchestrator/shared";
import { describe, expect, it } from "vitest";
import { OPENAI_SPEC, jsonResponse, request, sseResponse, stubFetch } from "../test-helpers.js";
import { OpenAIAdapter } from "./openai.js";

const signal = () => new AbortController().signal;

const completion = (overrides: Record<string, unknown> = {}) =>
  jsonResponse({
    choices: [{ message: { content: "hi there" }, finish_reason: "stop" }],
    usage: { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 },
    ...overrides,
  });

describe("OpenAIAdapter.chat", () => {
  it("normalizes a plain completion", async () => {
    const { fetchImpl } = stubFetch([completion()]);
    const adapter = new OpenAIAdapter({ apiKey: "k", fetchImpl });

    const result = await adapter.chat(request(), OPENAI_SPEC, signal());

    expect(result.message).toEqual({ role: "assistant", content: "hi there" });
    expect(result.finishReason).toBe("stop");
    expect(result.usage).toEqual({
      promptTokens: 12,
      completionTokens: 4,
      totalTokens: 16,
      cachedPromptTokens: 0,
    });
  });

  it("sends the provider-native model name, not our internal id", async () => {
    const { fetchImpl, calls } = stubFetch([completion()]);
    await new OpenAIAdapter({ apiKey: "k", fetchImpl }).chat(request(), OPENAI_SPEC, signal());

    expect(calls[0]?.body.model).toBe("gpt-4o-mini");
    expect(calls[0]?.body.model).not.toBe(OPENAI_SPEC.modelId);
    expect(calls[0]?.headers.authorization).toBe("Bearer k");
  });

  it("reads cached prompt tokens so cached traffic is not overcharged", async () => {
    const { fetchImpl } = stubFetch([
      completion({
        usage: {
          prompt_tokens: 1_000,
          completion_tokens: 10,
          total_tokens: 1_010,
          prompt_tokens_details: { cached_tokens: 800 },
        },
      }),
    ]);

    const result = await new OpenAIAdapter({ apiKey: "k", fetchImpl }).chat(
      request(),
      OPENAI_SPEC,
      signal(),
    );
    expect(result.usage.cachedPromptTokens).toBe(800);
  });

  it("parses tool call arguments into an object", async () => {
    const { fetchImpl } = stubFetch([
      completion({
        choices: [
          {
            message: {
              content: null,
              tool_calls: [
                {
                  id: "call_abc",
                  function: { name: "get_weather", arguments: '{"city":"Karachi"}' },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
      }),
    ]);

    const result = await new OpenAIAdapter({ apiKey: "k", fetchImpl }).chat(
      request(),
      OPENAI_SPEC,
      signal(),
    );

    expect(result.finishReason).toBe("tool_calls");
    expect(result.message.toolCalls).toEqual([
      { id: "call_abc", name: "get_weather", arguments: { city: "Karachi" } },
    ]);
  });

  it("degrades malformed tool arguments to an empty object instead of throwing", async () => {
    const { fetchImpl } = stubFetch([
      completion({
        choices: [
          {
            message: {
              content: null,
              tool_calls: [{ id: "c1", function: { name: "f", arguments: "{not json" } }],
            },
            finish_reason: "tool_calls",
          },
        ],
      }),
    ]);

    const result = await new OpenAIAdapter({ apiKey: "k", fetchImpl }).chat(
      request(),
      OPENAI_SPEC,
      signal(),
    );
    expect(result.message.toolCalls?.[0]?.arguments).toEqual({});
  });

  it("serializes an outbound tool-calling conversation into OpenAI's shape", async () => {
    const { fetchImpl, calls } = stubFetch([completion()]);
    await new OpenAIAdapter({ apiKey: "k", fetchImpl }).chat(
      request({
        messages: [
          { role: "user", content: "weather?" },
          {
            role: "assistant",
            content: "",
            toolCalls: [{ id: "c1", name: "get_weather", arguments: { city: "Karachi" } }],
          },
          { role: "tool", content: "31C", toolCallId: "c1" },
        ],
      }),
      OPENAI_SPEC,
      signal(),
    );

    const messages = calls[0]?.body.messages as Record<string, unknown>[];
    const assistant = messages[1] as { tool_calls: { function: { arguments: string } }[] };
    expect(assistant.tool_calls[0]?.function.arguments).toBe('{"city":"Karachi"}');
    expect(messages[2]).toEqual({ role: "tool", tool_call_id: "c1", content: "31C" });
  });

  it("requests usage on streamed calls, without which cost cannot be computed", async () => {
    const { fetchImpl, calls } = stubFetch([sseResponse(["data: [DONE]\n\n"])]);
    const adapter = new OpenAIAdapter({ apiKey: "k", fetchImpl });
    for await (const _ of adapter.stream(request(), OPENAI_SPEC, signal())) {
      // drain
    }
    expect(calls[0]?.body.stream_options).toEqual({ include_usage: true });
  });
});

describe("OpenAIAdapter error classification", () => {
  it("classifies 429 as rate_limit and honours Retry-After", async () => {
    const { fetchImpl } = stubFetch([
      jsonResponse(
        { error: { message: "slow down" } },
        { status: 429, headers: { "retry-after": "2" } },
      ),
    ]);

    const err = await new OpenAIAdapter({ apiKey: "k", fetchImpl })
      .chat(request(), OPENAI_SPEC, signal())
      .catch((e: unknown) => e as OrchestratorError);

    expect(err).toBeInstanceOf(OrchestratorError);
    expect((err as OrchestratorError).errorClass).toBe("rate_limit");
    expect((err as OrchestratorError).retryAfterMs).toBe(2_000);
    expect((err as OrchestratorError).retryable).toBe(true);
  });

  it("promotes a context-length 400 above a generic invalid_request", async () => {
    // This distinction matters: a plain 400 is terminal, but an over-long prompt can succeed on a
    // larger-context model, so it must stay fallback-eligible.
    const { fetchImpl } = stubFetch([
      jsonResponse(
        {
          error: {
            message: "This model's maximum context length is 128000 tokens",
            code: "context_length_exceeded",
          },
        },
        { status: 400 },
      ),
    ]);

    const err = (await new OpenAIAdapter({ apiKey: "k", fetchImpl })
      .chat(request(), OPENAI_SPEC, signal())
      .catch((e: unknown) => e)) as OrchestratorError;

    expect(err.errorClass).toBe("context_length_exceeded");
    expect(err.fallbackEligible).toBe(true);
    expect(err.retryable).toBe(false);
  });

  it("classifies 401 as auth", async () => {
    const { fetchImpl } = stubFetch([
      jsonResponse({ error: { message: "bad key" } }, { status: 401 }),
    ]);
    const err = (await new OpenAIAdapter({ apiKey: "k", fetchImpl })
      .chat(request(), OPENAI_SPEC, signal())
      .catch((e: unknown) => e)) as OrchestratorError;
    expect(err.errorClass).toBe("auth");
  });

  it("survives a non-JSON error body from a proxy", async () => {
    const { fetchImpl } = stubFetch([
      new Response("<html>502 Bad Gateway</html>", { status: 502 }),
    ]);
    const err = (await new OpenAIAdapter({ apiKey: "k", fetchImpl })
      .chat(request(), OPENAI_SPEC, signal())
      .catch((e: unknown) => e)) as OrchestratorError;
    expect(err.errorClass).toBe("provider_unavailable");
  });
});

describe("OpenAIAdapter.stream", () => {
  it("emits text deltas, tool call fragments, and a final usage frame", async () => {
    const frames = [
      'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
      'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
      'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\\"a\\":"}}]}}]}\n\n',
      'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}\n\n',
      "data: [DONE]\n\n",
    ];
    const { fetchImpl } = stubFetch([sseResponse(frames)]);

    const chunks = [];
    for await (const chunk of new OpenAIAdapter({ apiKey: "k", fetchImpl }).stream(
      request(),
      OPENAI_SPEC,
      signal(),
    )) {
      chunks.push(chunk);
    }

    expect(chunks.filter((c) => c.type === "text").map((c) => c.delta)).toEqual(["Hel", "lo"]);
    expect(chunks.find((c) => c.type === "tool_call")).toMatchObject({
      index: 0,
      id: "c1",
      name: "f",
    });

    const finish = chunks.at(-1);
    expect(finish).toMatchObject({ type: "finish", finishReason: "tool_calls" });
    expect(finish?.type === "finish" && finish.usage?.completionTokens).toBe(7);
  });
});
