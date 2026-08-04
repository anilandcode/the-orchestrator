import type { AdapterResult, ProviderAdapter } from "@orchestrator/gateway";
import type { ProviderId, UnifiedChatChunk, UnifiedChatResponse } from "@orchestrator/shared";
import type { FastifyInstance } from "fastify";
import { afterEach, describe, expect, it } from "vitest";
import { loadConfig } from "./config.js";
import { type Container, buildContainer } from "./container.js";
import { buildServer } from "./server.js";

const API_KEY = "test-key";

class ReplyingAdapter implements ProviderAdapter {
  constructor(
    readonly provider: ProviderId,
    private readonly reply: string,
    private readonly toolCalls?: { id: string; name: string; arguments: Record<string, unknown> }[],
  ) {}

  async chat(): Promise<AdapterResult> {
    return {
      message: {
        role: "assistant",
        content: this.reply,
        ...(this.toolCalls ? { toolCalls: this.toolCalls } : {}),
      },
      finishReason: this.toolCalls ? "tool_calls" : "stop",
      usage: { promptTokens: 120, completionTokens: 8, totalTokens: 128, cachedPromptTokens: 0 },
    };
  }

  async *stream(): AsyncGenerator<UnifiedChatChunk, void, undefined> {
    yield { type: "finish", finishReason: "stop" };
  }
}

function makeApp(adapter: ProviderAdapter) {
  const config = loadConfig({
    ORCHESTRATOR_API_KEY: API_KEY,
    ORCHESTRATOR_DB_PATH: ":memory:",
    ROUTER_MODE: "shadow",
  } as NodeJS.ProcessEnv);

  const container = buildContainer(config, { adapters: [adapter] });
  return { app: buildServer(container), container };
}

describe("quality feedback loop", () => {
  let app!: FastifyInstance;
  let container!: Container;
  // Guards teardown for describe blocks that never build a server.
  let live = false;

  afterEach(async () => {
    if (!live) return;
    live = false;
    await app.close();
    container.close();
  });

  const post = (instance: FastifyInstance, url: string, payload: Record<string, unknown>) =>
    instance.inject({
      method: "POST",
      url,
      headers: { authorization: `Bearer ${API_KEY}` },
      payload,
    });

  describe("inline scoring", () => {
    it("grades a conforming JSON response with the schema validator, not the heuristic", async () => {
      ({ app, container } = makeApp(new ReplyingAdapter("openai", '{"city":"Karachi"}')));
      live = true;

      const response = await post(app, "/v1/chat", {
        messages: [{ role: "user", content: "extract the city" }],
        route: {
          mode: "cheap",
          taskType: "extraction",
          outputSchema: {
            type: "object",
            required: ["city"],
            properties: { city: { type: "string" } },
          },
        },
      });

      const body = response.json<UnifiedChatResponse>();
      const [event] = (container as Container).events.query({ requestId: body.requestId });

      expect(event?.qualitySource).toBe("json-schema");
      expect(event?.qualityScore).toBe(1);
      // This is the whole point: a real validator outranks the "it did not error" floor.
      expect(event?.qualityConfidence).toBeGreaterThan(0.2);
    });

    it("penalizes a response that violates the declared schema", async () => {
      ({ app, container } = makeApp(new ReplyingAdapter("openai", '{"town":"Karachi"}')));
      live = true;

      const response = await post(app, "/v1/chat", {
        messages: [{ role: "user", content: "extract the city" }],
        route: {
          taskType: "extraction",
          outputSchema: {
            type: "object",
            required: ["city"],
            properties: { city: { type: "string" } },
          },
        },
      });

      const body = response.json<UnifiedChatResponse>();
      const [event] = (container as Container).events.query({ requestId: body.requestId });

      expect(event?.qualityScore).toBe(0);
      // A schema failure must drag the reward well below a clean unschema'd success.
      expect(event?.reward).toBeLessThan(0.5);
    });

    it("grades invalid tool arguments without any caller configuration", async () => {
      ({ app, container } = makeApp(
        new ReplyingAdapter("openai", "", [{ id: "c1", name: "get_weather", arguments: {} }]),
      ));
      live = true;

      const response = await post(app, "/v1/chat", {
        messages: [{ role: "user", content: "weather?" }],
        tools: [
          {
            name: "get_weather",
            parameters: {
              type: "object",
              required: ["city"],
              properties: { city: { type: "string" } },
            },
          },
        ],
      });

      const body = response.json<UnifiedChatResponse>();
      const [event] = (container as Container).events.query({ requestId: body.requestId });

      expect(event?.qualitySource).toBe("tool-call");
      expect(event?.qualityScore).toBe(0);
    });

    it("falls back to the heuristic floor when nothing else applies", async () => {
      ({ app, container } = makeApp(new ReplyingAdapter("openai", "just some prose")));
      live = true;

      const response = await post(app, "/v1/chat", {
        messages: [{ role: "user", content: "hello" }],
      });

      const body = response.json<UnifiedChatResponse>();
      const [event] = (container as Container).events.query({ requestId: body.requestId });

      expect(event?.qualitySource).toBe("finish-reason");
      expect(event?.qualityConfidence).toBe(0.2);
    });

    it("separates two models on the same task, which is the point of the whole phase", async () => {
      // A good and a bad answer to an identical request must land on different rewards. If they do
      // not, the quality term is not carrying information and the bandit cannot learn from it.
      const schema = {
        type: "object",
        required: ["city"],
        properties: { city: { type: "string" } },
      };
      const request = {
        messages: [{ role: "user", content: "extract the city" }],
        route: { taskType: "extraction", outputSchema: schema },
      };

      const good = makeApp(new ReplyingAdapter("openai", '{"city":"Karachi"}'));
      const goodBody = (await post(good.app, "/v1/chat", request)).json<UnifiedChatResponse>();
      const goodReward = good.container.events.query({ requestId: goodBody.requestId })[0]?.reward;
      await good.app.close();
      good.container.close();

      const bad = makeApp(new ReplyingAdapter("openai", "sorry, I cannot"));
      const badBody = (await post(bad.app, "/v1/chat", request)).json<UnifiedChatResponse>();
      const badReward = bad.container.events.query({ requestId: badBody.requestId })[0]?.reward;
      await bad.app.close();
      bad.container.close();

      expect(goodReward).toBeGreaterThan(badReward as number);
      // And the gap must be substantial, not a rounding artifact.
      expect((goodReward as number) - (badReward as number)).toBeGreaterThan(0.2);
    });
  });

  describe("client feedback", () => {
    it("revises the stored reward and records human provenance", async () => {
      ({ app, container } = makeApp(new ReplyingAdapter("openai", "an answer")));
      live = true;

      const chat = await post(app, "/v1/chat", { messages: [{ role: "user", content: "hi" }] });
      const { requestId } = chat.json<UnifiedChatResponse>();
      const before = (container as Container).events.query({ requestId })[0];

      await post(app, "/v1/feedback", { requestId, quality: 0.05 });

      const after = (container as Container).events.query({ requestId })[0];
      expect(after?.reward).toBeLessThan(before?.reward as number);
      expect(after?.qualitySource).toBe("client-feedback");
      expect(after?.qualityConfidence).toBe(1);
      expect(after?.qualityRevisions).toBe(1);
    });

    it("corrects the router rather than teaching it a second time", async () => {
      // Re-teaching would count one call twice, inflating the arm's apparent confidence and
      // shrinking its exploration bonus on the strength of a single observation.
      ({ app, container } = makeApp(new ReplyingAdapter("openai", "an answer")));
      live = true;

      const chat = await post(app, "/v1/chat", { messages: [{ role: "user", content: "hi" }] });
      const body = chat.json<UnifiedChatResponse>();
      const decisionId = body.routingDecisionId as string;

      const rewardBefore = (container as Container).router.appliedRewardFor(decisionId);
      expect(rewardBefore).toBeDefined();

      await post(app, "/v1/feedback", { requestId: body.requestId, quality: 0.05 });

      const rewardAfter = (container as Container).router.appliedRewardFor(decisionId);
      expect(rewardAfter).toBeLessThan(rewardBefore as number);
    });

    it("reports how many attempts it revised", async () => {
      ({ app, container } = makeApp(new ReplyingAdapter("openai", "an answer")));
      live = true;
      const chat = await post(app, "/v1/chat", { messages: [{ role: "user", content: "hi" }] });
      const { requestId } = chat.json<UnifiedChatResponse>();

      const feedback = await post(app, "/v1/feedback", { requestId, quality: 0.9 });
      expect(feedback.json()).toMatchObject({ ok: true, scored: 1 });
    });

    it("accumulates revisions across repeated feedback", async () => {
      ({ app, container } = makeApp(new ReplyingAdapter("openai", "an answer")));
      live = true;
      const chat = await post(app, "/v1/chat", { messages: [{ role: "user", content: "hi" }] });
      const { requestId } = chat.json<UnifiedChatResponse>();

      await post(app, "/v1/feedback", { requestId, quality: 0.2 });
      await post(app, "/v1/feedback", { requestId, quality: 0.9 });

      const event = (container as Container).events.query({ requestId })[0];
      expect(event?.qualityRevisions).toBe(2);
      expect(event?.qualityScore).toBe(0.9);
    });
  });

  describe("judge configuration", () => {
    it("is disabled by default, so no unrequested spend occurs", () => {
      const config = loadConfig({} as NodeJS.ProcessEnv);
      expect(config.judgeEnabled).toBe(false);
      expect(config.judgeSampleRate).toBe(0.05);
      expect(config.judgeMaxUsdPerHour).toBe(1);
    });

    it("registers no deferred scorer when the judge is off", () => {
      ({ app, container } = makeApp(new ReplyingAdapter("openai", "x")));
      live = true;
      expect((container as Container).quality.deferredScorers).toHaveLength(0);
      expect((container as Container).quality.inlineScorers.length).toBeGreaterThan(0);
    });
  });
});
