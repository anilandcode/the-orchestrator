import {
  type Clock,
  InMemoryCallEventSink,
  type ModelSpec,
  OrchestratorError,
  type ProviderId,
  type UnifiedChatChunk,
  type UnifiedChatRequest,
  computeCostUsd,
  createFixedClock,
  createSequentialIds,
  defaultRegistry,
} from "@orchestrator/shared";
import { beforeEach, describe, expect, it } from "vitest";
import { Gateway } from "./gateway.js";
import type { AdapterResult, ProviderAdapter } from "./provider-adapter.js";
import { request } from "./test-helpers.js";

const OPENAI_MODEL = "openai/gpt-4o-mini";
const OPENAI_BIG = "openai/gpt-4o";
const ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5";

const USAGE = {
  promptTokens: 1_000,
  completionTokens: 500,
  totalTokens: 1_500,
  cachedPromptTokens: 0,
};

type Step = "ok" | OrchestratorError;

/** A programmable adapter: each call consumes the next step of the script. */
class ScriptedAdapter implements ProviderAdapter {
  calls = 0;
  constructor(
    readonly provider: ProviderId,
    private readonly script: Step[],
    private readonly clock: Clock & { advance(ms: number): void },
    private readonly latencyMs = 100,
  ) {}

  async chat(): Promise<AdapterResult> {
    const step = this.script[this.calls++] ?? "ok";
    this.clock.advance(this.latencyMs);
    if (step !== "ok") throw step;
    return {
      message: { role: "assistant", content: "ok" },
      finishReason: "stop",
      usage: USAGE,
    };
  }

  async *stream(): AsyncGenerator<UnifiedChatChunk, void, undefined> {
    const step = this.script[this.calls++] ?? "ok";
    this.clock.advance(this.latencyMs);
    if (step !== "ok") throw step;
    yield { type: "text", delta: "ok" };
    yield { type: "finish", finishReason: "stop", usage: USAGE };
  }
}

/** Yields one chunk, then fails — the case where fallback must NOT fire. */
class MidStreamFailureAdapter implements ProviderAdapter {
  constructor(readonly provider: ProviderId) {}
  async chat(): Promise<AdapterResult> {
    throw new Error("unused");
  }
  async *stream(): AsyncGenerator<UnifiedChatChunk, void, undefined> {
    yield { type: "text", delta: "partial" };
    throw new OrchestratorError("provider_unavailable", "died mid-stream");
  }
}

describe("Gateway", () => {
  let clock: Clock & { advance(ms: number): void };
  let sink: InMemoryCallEventSink;
  let slept: number[];

  beforeEach(() => {
    clock = createFixedClock();
    sink = new InMemoryCallEventSink();
    slept = [];
  });

  const build = (adapters: ProviderAdapter[], overrides = {}) =>
    new Gateway({
      adapters,
      sink,
      clock,
      ids: createSequentialIds(),
      // Deterministic jitter and zero-cost sleeps keep retry tests instant.
      random: () => 1,
      sleep: async (ms) => {
        slept.push(ms);
      },
      ...overrides,
    });

  const plan = (modelId: string, fallbacks: string[] = []) => ({
    modelId,
    fallbacks,
    decisionId: "dec_1",
    features: [0.5, 1],
  });

  describe("happy path", () => {
    it("returns a normalized response and records exactly one event", async () => {
      const gateway = build([new ScriptedAdapter("openai", ["ok"], clock)]);
      const res = await gateway.chat(request(), plan(OPENAI_MODEL));

      expect(res.modelId).toBe(OPENAI_MODEL);
      expect(res.attempts).toBe(1);
      expect(res.latencyMs).toBe(100);
      expect(sink.events).toHaveLength(1);
      expect(sink.events[0]).toMatchObject({ status: "success", attempt: 1, latencyMs: 100 });
    });

    it("computes cost from the registry rather than trusting the provider", async () => {
      const gateway = build([new ScriptedAdapter("openai", ["ok"], clock)]);
      const res = await gateway.chat(request(), plan(OPENAI_MODEL));

      const expected = computeCostUsd(defaultRegistry.require(OPENAI_MODEL), USAGE);
      expect(res.costUsd).toBe(expected);
      expect(sink.events[0]?.costUsd).toBe(expected);
    });

    it("carries routing context onto the event so reward can be attributed", async () => {
      const gateway = build([new ScriptedAdapter("openai", ["ok"], clock)]);
      await gateway.chat(
        request({ route: { mode: "cheap", taskType: "code" } } as Partial<UnifiedChatRequest>),
        plan(OPENAI_MODEL),
      );

      expect(sink.events[0]).toMatchObject({
        routingDecisionId: "dec_1",
        features: [0.5, 1],
        taskType: "code",
        routeMode: "cheap",
      });
    });
  });

  describe("retry", () => {
    it("retries a retryable failure on the same model and records both attempts", async () => {
      const rateLimited = new OrchestratorError("rate_limit", "429");
      const gateway = build([new ScriptedAdapter("openai", [rateLimited, "ok"], clock)]);

      const res = await gateway.chat(request(), plan(OPENAI_MODEL));

      expect(res.attempts).toBe(2);
      expect(sink.events).toHaveLength(2);
      expect(sink.events[0]).toMatchObject({
        status: "error",
        errorClass: "rate_limit",
        attempt: 1,
      });
      expect(sink.events[1]).toMatchObject({ status: "success", attempt: 2 });
      // Both events name the same model: the retry was not a different arm.
      expect(new Set(sink.events.map((e) => e.modelId))).toEqual(new Set([OPENAI_MODEL]));
    });

    it("grows the backoff exponentially", async () => {
      const err = new OrchestratorError("timeout", "slow");
      const gateway = build([new ScriptedAdapter("openai", [err, err, "ok"], clock)]);
      await gateway.chat(request(), plan(OPENAI_MODEL));

      // random() is pinned to 1, so these are the un-jittered ceilings: 250ms then 500ms.
      expect(slept).toEqual([250, 500]);
    });

    it("waits at least as long as a provider-supplied Retry-After", async () => {
      const err = new OrchestratorError("rate_limit", "429", { retryAfterMs: 5_000 });
      const gateway = build([new ScriptedAdapter("openai", [err, "ok"], clock)]);
      await gateway.chat(request(), plan(OPENAI_MODEL));

      expect(slept).toEqual([5_000]);
    });

    it("does not retry a non-retryable failure", async () => {
      const auth = new OrchestratorError("auth", "bad key");
      const adapter = new ScriptedAdapter("openai", [auth, "ok"], clock);
      const gateway = build([adapter]);

      await expect(gateway.chat(request(), plan(OPENAI_MODEL))).rejects.toThrow(/bad key/);
      expect(adapter.calls).toBe(1);
      expect(slept).toEqual([]);
    });
  });

  describe("fallback", () => {
    it("moves to the next model once retries on the first are exhausted", async () => {
      const unavailable = new OrchestratorError("provider_unavailable", "503");
      const gateway = build([
        new ScriptedAdapter("openai", [unavailable, unavailable, unavailable], clock),
        new ScriptedAdapter("anthropic", ["ok"], clock),
      ]);

      const res = await gateway.chat(request(), plan(OPENAI_MODEL, [ANTHROPIC_MODEL]));

      expect(res.modelId).toBe(ANTHROPIC_MODEL);
      // Three failed attempts on the primary plus one success on the fallback.
      expect(sink.events).toHaveLength(4);
      expect(res.attempts).toBe(4);
      expect(sink.events.filter((e) => e.status === "error")).toHaveLength(3);
    });

    it("fails an over-long prompt over to a larger-context model", async () => {
      const tooLong = new OrchestratorError("context_length_exceeded", "prompt too long");
      const gateway = build([new ScriptedAdapter("openai", [tooLong, "ok"], clock)]);

      const res = await gateway.chat(request(), plan(OPENAI_MODEL, [OPENAI_BIG]));

      // Not retryable, so exactly one attempt on the primary before moving on.
      expect(res.modelId).toBe(OPENAI_BIG);
      expect(sink.events).toHaveLength(2);
      expect(slept).toEqual([]);
    });

    it("refuses to fail over a malformed request", async () => {
      const bad = new OrchestratorError("invalid_request", "messages: field required");
      const fallback = new ScriptedAdapter("anthropic", ["ok"], clock);
      const gateway = build([new ScriptedAdapter("openai", [bad], clock), fallback]);

      await expect(gateway.chat(request(), plan(OPENAI_MODEL, [ANTHROPIC_MODEL]))).rejects.toThrow(
        /field required/,
      );

      // Retrying a malformed request elsewhere would just burn money twice.
      expect(fallback.calls).toBe(0);
      expect(sink.events).toHaveLength(1);
    });

    it("refuses to fail over filtered content", async () => {
      const filtered = new OrchestratorError("content_filter", "blocked");
      const fallback = new ScriptedAdapter("anthropic", ["ok"], clock);
      const gateway = build([new ScriptedAdapter("openai", [filtered], clock), fallback]);

      await expect(gateway.chat(request(), plan(OPENAI_MODEL, [ANTHROPIC_MODEL]))).rejects.toThrow(
        /blocked/,
      );
      expect(fallback.calls).toBe(0);
    });

    it("throws the last error when every model in the chain fails", async () => {
      const err = new OrchestratorError("provider_unavailable", "everything is down");
      const gateway = build(
        [
          new ScriptedAdapter("openai", [err], clock),
          new ScriptedAdapter("anthropic", [err], clock),
        ],
        { retry: { maxAttemptsPerModel: 1 } },
      );

      await expect(gateway.chat(request(), plan(OPENAI_MODEL, [ANTHROPIC_MODEL]))).rejects.toThrow(
        /everything is down/,
      );
      expect(sink.events).toHaveLength(2);
    });
  });

  describe("unreachable models", () => {
    it("skips a model whose provider has no configured adapter", async () => {
      // The realistic case: the deployment only has one provider's key.
      const gateway = build([new ScriptedAdapter("openai", ["ok"], clock)]);
      const res = await gateway.chat(request(), plan(ANTHROPIC_MODEL, [OPENAI_MODEL]));

      expect(res.modelId).toBe(OPENAI_MODEL);
      // No call was made for the unreachable model, so no event describes one.
      expect(sink.events).toHaveLength(1);
    });

    it("skips an unknown model id", async () => {
      const gateway = build([new ScriptedAdapter("openai", ["ok"], clock)]);
      const res = await gateway.chat(request(), plan("vendor/does-not-exist", [OPENAI_MODEL]));
      expect(res.modelId).toBe(OPENAI_MODEL);
    });

    it("reports which models it can actually reach", async () => {
      const gateway = build([new ScriptedAdapter("openai", ["ok"], clock)]);
      const available = gateway.availableModels().map((m) => m.provider);
      expect(new Set(available)).toEqual(new Set(["openai"]));
    });

    it("throws when nothing in the chain is reachable", async () => {
      const gateway = build([new ScriptedAdapter("openai", ["ok"], clock)]);
      await expect(gateway.chat(request(), plan(ANTHROPIC_MODEL))).rejects.toThrow(
        /No adapter configured/,
      );
    });
  });

  describe("chain construction", () => {
    it("does not try the same model twice when it also appears in the fallbacks", async () => {
      const err = new OrchestratorError("provider_unavailable", "down");
      const adapter = new ScriptedAdapter("openai", [err], clock);
      const gateway = build([adapter], { retry: { maxAttemptsPerModel: 1 } });

      await expect(gateway.chat(request(), plan(OPENAI_MODEL, [OPENAI_MODEL]))).rejects.toThrow();
      expect(adapter.calls).toBe(1);
    });
  });

  describe("streaming", () => {
    it("records one success event with time-to-first-token", async () => {
      const gateway = build([new ScriptedAdapter("openai", ["ok"], clock)]);

      const chunks = [];
      for await (const chunk of gateway.stream(request(), plan(OPENAI_MODEL))) chunks.push(chunk);

      expect(chunks.at(-1)).toMatchObject({ type: "finish" });
      expect(sink.events).toHaveLength(1);
      expect(sink.events[0]?.status).toBe("success");
      expect(sink.events[0]?.ttftMs).toBe(100);
      expect(sink.events[0]?.costUsd).toBeGreaterThan(0);
    });

    it("falls back when the failure happens before any output", async () => {
      const err = new OrchestratorError("provider_unavailable", "503");
      const gateway = build([
        new ScriptedAdapter("openai", [err], clock),
        new ScriptedAdapter("anthropic", ["ok"], clock),
      ]);

      const chunks = [];
      for await (const chunk of gateway.stream(request(), plan(OPENAI_MODEL, [ANTHROPIC_MODEL]))) {
        chunks.push(chunk);
      }

      expect(chunks.some((c) => c.type === "text")).toBe(true);
      expect(sink.events.map((e) => e.status)).toEqual(["error", "success"]);
    });

    it("does not restart on another model once bytes have reached the caller", async () => {
      // Restarting mid-stream would duplicate output the caller has already consumed.
      const fallback = new ScriptedAdapter("anthropic", ["ok"], clock);
      const gateway = build([new MidStreamFailureAdapter("openai"), fallback]);

      const chunks = [];
      await expect(
        (async () => {
          for await (const chunk of gateway.stream(
            request(),
            plan(OPENAI_MODEL, [ANTHROPIC_MODEL]),
          )) {
            chunks.push(chunk);
          }
        })(),
      ).rejects.toThrow(/died mid-stream/);

      expect(chunks).toHaveLength(1);
      expect(fallback.calls).toBe(0);
    });
  });
});
