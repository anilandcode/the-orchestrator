import type { Gateway } from "@orchestrator/gateway";
import { type UnifiedChatResponse, createFixedClock } from "@orchestrator/shared";
import { describe, expect, it, vi } from "vitest";
import { makeInput } from "../test-helpers.js";
import { LlmJudgeScorer } from "./llm-judge.js";
import { parseJudgeScore } from "./rubric.js";

const JUDGE_MODEL = "openai/gpt-4o-mini";

function fakeGateway(
  reply: string,
  costUsd = 0.0001,
): { gateway: Gateway; chat: ReturnType<typeof vi.fn> } {
  const chat = vi.fn(
    async (): Promise<UnifiedChatResponse> => ({
      id: "resp_judge",
      requestId: "req_judge",
      provider: "openai",
      modelId: JUDGE_MODEL,
      message: { role: "assistant", content: reply },
      finishReason: "stop",
      usage: { promptTokens: 200, completionTokens: 3, totalTokens: 203, cachedPromptTokens: 0 },
      costUsd,
      latencyMs: 300,
      attempts: 1,
    }),
  );
  return { gateway: { chat } as unknown as Gateway, chat };
}

const judge = (overrides: Partial<ConstructorParameters<typeof LlmJudgeScorer>[0]> = {}) => {
  const { gateway, chat } = fakeGateway("0.7");
  const scorer = new LlmJudgeScorer({
    gateway,
    modelId: JUDGE_MODEL,
    sampleRate: 1,
    random: () => 0,
    clock: createFixedClock(),
    ...overrides,
  });
  return { scorer, chat };
};

describe("LlmJudgeScorer", () => {
  it("grades a sampled response", async () => {
    const { scorer } = judge();
    const assessment = await scorer.score(makeInput());

    expect(assessment?.score).toBeCloseTo(0.7, 10);
    expect(assessment?.source).toBe("llm-judge");
  });

  it("pins its own model instead of routing", async () => {
    // Routing the judge through the bandit would let the bandit influence its own grades.
    const { scorer, chat } = judge();
    await scorer.score(makeInput());

    const [request, plan] = chat.mock.calls[0] as [
      { route: { pin?: string } },
      { modelId: string },
    ];
    expect(plan.modelId).toBe(JUDGE_MODEL);
    expect(request.route.pin).toBe(JUDGE_MODEL);
  });

  it("is a deferred scorer, never inline", () => {
    // A judge on the hot path would add its latency to the call it grades, and the reward function
    // would score that as the graded model being slow.
    expect(judge().scorer.stage).toBe("deferred");
  });

  it("respects the sample rate", async () => {
    const { scorer, chat } = judge({ sampleRate: 0.1, random: () => 0.5 });
    expect(await scorer.score(makeInput())).toBeUndefined();
    expect(chat).not.toHaveBeenCalled();
  });

  it("does not grade failed calls", async () => {
    // The error class already told us what happened; grading it would just cost money.
    const { scorer, chat } = judge();
    const result = await scorer.score(
      makeInput({ event: { status: "error", errorClass: "timeout" } }),
    );
    expect(result).toBeUndefined();
    expect(chat).not.toHaveBeenCalled();
  });

  it("never grades its own traffic", async () => {
    // Without this the judge would grade its own grading calls, forever.
    const { scorer, chat } = judge();
    expect(await scorer.score(makeInput({ event: { isJudge: true } }))).toBeUndefined();
    expect(chat).not.toHaveBeenCalled();
  });

  describe("spend cap", () => {
    it("stops calling once the hourly cap is reached", async () => {
      const { gateway, chat } = fakeGateway("0.9", 0.02);
      const scorer = new LlmJudgeScorer({
        gateway,
        modelId: JUDGE_MODEL,
        sampleRate: 1,
        random: () => 0,
        maxUsdPerHour: 0.05,
        clock: createFixedClock(),
      });

      for (let i = 0; i < 3; i++) await scorer.score(makeInput());
      expect(scorer.isBreakerOpen()).toBe(true);

      const callsAtTrip = chat.mock.calls.length;
      expect(await scorer.score(makeInput())).toBeUndefined();
      // A judge billing more than the traffic it grades is a real failure mode.
      expect(chat.mock.calls.length).toBe(callsAtTrip);
    });

    it("recovers once spend ages out of the rolling window", async () => {
      const clock = createFixedClock();
      const { gateway } = fakeGateway("0.9", 0.02);
      const scorer = new LlmJudgeScorer({
        gateway,
        modelId: JUDGE_MODEL,
        sampleRate: 1,
        random: () => 0,
        maxUsdPerHour: 0.05,
        clock,
      });

      for (let i = 0; i < 3; i++) await scorer.score(makeInput());
      expect(scorer.isBreakerOpen()).toBe(true);

      clock.advance(3_600_001);
      expect(scorer.isBreakerOpen()).toBe(false);
      expect(await scorer.score(makeInput())).toBeDefined();
    });

    it("tracks spend from the gateway's computed cost", async () => {
      const { gateway } = fakeGateway("0.5", 0.003);
      const scorer = new LlmJudgeScorer({
        gateway,
        modelId: JUDGE_MODEL,
        sampleRate: 1,
        random: () => 0,
        clock: createFixedClock(),
      });

      await scorer.score(makeInput());
      await scorer.score(makeInput());
      expect(scorer.spentLastHour()).toBeCloseTo(0.006, 10);
    });
  });

  it("abstains when the judge ignores its instructions", async () => {
    // Inventing a midpoint would inject noise into the very signal this phase exists to make
    // trustworthy.
    const { gateway } = fakeGateway("I would rather not grade this.");
    const scorer = new LlmJudgeScorer({
      gateway,
      modelId: JUDGE_MODEL,
      sampleRate: 1,
      random: () => 0,
      clock: createFixedClock(),
    });
    expect(await scorer.score(makeInput())).toBeUndefined();
  });

  it("abstains when the judge call fails, rather than failing the request", async () => {
    const gateway = {
      chat: vi.fn(async () => {
        throw new Error("judge provider down");
      }),
    } as unknown as Gateway;

    const scorer = new LlmJudgeScorer({
      gateway,
      modelId: JUDGE_MODEL,
      sampleRate: 1,
      random: () => 0,
      clock: createFixedClock(),
    });
    expect(await scorer.score(makeInput())).toBeUndefined();
  });
});

describe("parseJudgeScore", () => {
  it("parses a bare decimal", () => {
    expect(parseJudgeScore("0.85")).toBeCloseTo(0.85, 10);
  });

  it("parses a number surrounded by stray text", () => {
    expect(parseJudgeScore("Score: 0.4")).toBeCloseTo(0.4, 10);
  });

  it("normalizes a 0-100 answer", () => {
    expect(parseJudgeScore("85")).toBeCloseTo(0.85, 10);
  });

  it("accepts the boundaries", () => {
    expect(parseJudgeScore("0")).toBe(0);
    expect(parseJudgeScore("1")).toBe(1);
  });

  it("returns undefined for a non-numeric reply", () => {
    expect(parseJudgeScore("excellent")).toBeUndefined();
  });

  it("rejects an out-of-range number rather than clamping it", () => {
    // Clamping would silently turn a misbehaving judge into a confident maximum score.
    expect(parseJudgeScore("420")).toBeUndefined();
  });
});
