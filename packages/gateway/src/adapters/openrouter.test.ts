import { type ModelSpec, type OrchestratorError, defaultRegistry } from "@orchestrator/shared";
import { describe, expect, it } from "vitest";
import { jsonResponse, request, sseResponse, stubFetch } from "../test-helpers.js";
import { OpenAIAdapter } from "./openai.js";
import { OpenRouterAdapter } from "./openrouter.js";

const signal = () => new AbortController().signal;

/** OpenRouter model ids already use the `vendor/model` shape the registry uses. */
const SPEC: ModelSpec = {
  ...defaultRegistry.require("openai/gpt-4o-mini"),
  provider: "openrouter",
  providerModel: "openai/gpt-4o-mini",
};

const completion = (overrides: Record<string, unknown> = {}) =>
  jsonResponse({
    choices: [{ message: { content: "hi there" }, finish_reason: "stop" }],
    usage: { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 },
    ...overrides,
  });

describe("OpenRouterAdapter", () => {
  it("normalizes a completion identically to the OpenAI adapter", async () => {
    // Same wire format, so any divergence would be a bug in the shared translation — and a
    // divergence in usage alone would corrupt cost accounting for one provider and not the other.
    const openRouter = stubFetch([completion()]);
    const openAi = stubFetch([completion()]);

    const viaOpenRouter = await new OpenRouterAdapter({
      apiKey: "k",
      fetchImpl: openRouter.fetchImpl,
    }).chat(request(), SPEC, signal());

    const viaOpenAi = await new OpenAIAdapter({ apiKey: "k", fetchImpl: openAi.fetchImpl }).chat(
      request(),
      defaultRegistry.require("openai/gpt-4o-mini"),
      signal(),
    );

    expect(viaOpenRouter).toEqual(viaOpenAi);
  });

  it("posts to the OpenRouter endpoint with bearer auth", async () => {
    const { fetchImpl, calls } = stubFetch([completion()]);
    await new OpenRouterAdapter({ apiKey: "or-key", fetchImpl }).chat(request(), SPEC, signal());

    expect(calls[0]?.url).toBe("https://openrouter.ai/api/v1/chat/completions");
    expect(calls[0]?.headers.authorization).toBe("Bearer or-key");
  });

  it("sends the vendor-qualified model id", async () => {
    const { fetchImpl, calls } = stubFetch([completion()]);
    await new OpenRouterAdapter({ apiKey: "k", fetchImpl }).chat(request(), SPEC, signal());
    expect(calls[0]?.body.model).toBe("openai/gpt-4o-mini");
  });

  describe("never delegates routing", () => {
    it("omits the models[] and route fields", async () => {
      // The property this whole system rests on. OpenRouter can pick the model itself; letting it
      // would replace the decision being measured while leaving every CallEvent attributed to the
      // model we believed we asked for. The bandit would then learn from rewards earned by a model
      // it never chose — silently, and invisibly to anything downstream of the gateway.
      const { fetchImpl, calls } = stubFetch([completion()]);
      await new OpenRouterAdapter({ apiKey: "k", fetchImpl }).chat(request(), SPEC, signal());

      expect(calls[0]?.body).not.toHaveProperty("models");
      expect(calls[0]?.body).not.toHaveProperty("route");
    });

    it("omits them on streamed calls too", async () => {
      const { fetchImpl, calls } = stubFetch([sseResponse(["data: [DONE]\n\n"])]);
      const adapter = new OpenRouterAdapter({ apiKey: "k", fetchImpl });
      for await (const _ of adapter.stream(request(), SPEC, signal())) {
        // drain
      }

      expect(calls[0]?.body).not.toHaveProperty("models");
      expect(calls[0]?.body).not.toHaveProperty("route");
    });
  });

  describe("attribution headers", () => {
    it("sends them when configured", async () => {
      const { fetchImpl, calls } = stubFetch([completion()]);
      await new OpenRouterAdapter({
        apiKey: "k",
        appUrl: "https://example.test",
        appName: "The Orchestrator",
        fetchImpl,
      }).chat(request(), SPEC, signal());

      expect(calls[0]?.headers["http-referer"]).toBe("https://example.test");
      expect(calls[0]?.headers["x-title"]).toBe("The Orchestrator");
    });

    it("omits them entirely when not configured", async () => {
      const { fetchImpl, calls } = stubFetch([completion()]);
      await new OpenRouterAdapter({ apiKey: "k", fetchImpl }).chat(request(), SPEC, signal());

      expect(calls[0]?.headers["http-referer"]).toBeUndefined();
      expect(calls[0]?.headers["x-title"]).toBeUndefined();
    });
  });

  describe("error classification", () => {
    const failing = async (body: unknown, status: number) => {
      const { fetchImpl } = stubFetch([jsonResponse(body, { status })]);
      return (await new OpenRouterAdapter({ apiKey: "k", fetchImpl })
        .chat(request(), SPEC, signal())
        .catch((e: unknown) => e)) as OrchestratorError;
    };

    it("treats an upstream 5xx as provider_unavailable, not a bad request", async () => {
      // The outer status describes OpenRouter; the inner code describes whoever actually refused.
      const error = await failing(
        { error: { code: 502, message: "Provider returned error" } },
        400,
      );
      expect(error.errorClass).toBe("provider_unavailable");
      expect(error.retryable).toBe(true);
    });

    it("maps insufficient credits to auth, so it fails over rather than retrying", async () => {
      const error = await failing({ error: { message: "Insufficient credits" } }, 402);
      expect(error.errorClass).toBe("auth");
      expect(error.retryable).toBe(false);
      expect(error.fallbackEligible).toBe(true);
    });

    it("recognizes a context overflow", async () => {
      const error = await failing(
        { error: { code: 400, message: "This endpoint's maximum context length is 8192 tokens" } },
        400,
      );
      expect(error.errorClass).toBe("context_length_exceeded");
      expect(error.fallbackEligible).toBe(true);
    });

    it("treats no-available-provider as a transient outage", async () => {
      const error = await failing(
        {
          error: {
            code: 404,
            message: "No allowed providers are available for the selected model",
          },
        },
        404,
      );
      expect(error.errorClass).toBe("provider_unavailable");
    });

    it("recognizes upstream moderation", async () => {
      const error = await failing(
        { error: { code: 403, message: "Your input was flagged by content policy" } },
        403,
      );
      expect(error.errorClass).toBe("content_filter");
      // Deliberately not failed over: re-running filtered content elsewhere is evasion.
      expect(error.fallbackEligible).toBe(false);
    });

    it("names the provider in the message", async () => {
      const error = await failing({ error: { message: "something broke" } }, 500);
      expect(error.message).toMatch(/^OpenRouter: /);
    });

    it("falls back to status when the body says nothing useful", async () => {
      const error = await failing({}, 429);
      expect(error.errorClass).toBe("rate_limit");
    });

    it("survives a non-JSON error body", async () => {
      const { fetchImpl } = stubFetch([new Response("<html>504</html>", { status: 504 })]);
      const error = (await new OpenRouterAdapter({ apiKey: "k", fetchImpl })
        .chat(request(), SPEC, signal())
        .catch((e: unknown) => e)) as OrchestratorError;
      expect(error.errorClass).toBe("provider_unavailable");
    });
  });

  it("streams text deltas and a usage-bearing finish frame", async () => {
    const frames = [
      'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
      'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
      'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}\n\n',
      "data: [DONE]\n\n",
    ];
    const { fetchImpl } = stubFetch([sseResponse(frames)]);

    const chunks = [];
    for await (const chunk of new OpenRouterAdapter({ apiKey: "k", fetchImpl }).stream(
      request(),
      SPEC,
      signal(),
    )) {
      chunks.push(chunk);
    }

    expect(chunks.filter((c) => c.type === "text").map((c) => c.delta)).toEqual(["Hel", "lo"]);
    const finish = chunks.at(-1);
    expect(finish?.type === "finish" && finish.usage?.completionTokens).toBe(7);
  });

  it("requests usage on streamed calls, without which cost reads as zero", async () => {
    const { fetchImpl, calls } = stubFetch([sseResponse(["data: [DONE]\n\n"])]);
    const adapter = new OpenRouterAdapter({ apiKey: "k", fetchImpl });
    for await (const _ of adapter.stream(request(), SPEC, signal())) {
      // drain
    }
    expect(calls[0]?.body.stream_options).toEqual({ include_usage: true });
  });
});
