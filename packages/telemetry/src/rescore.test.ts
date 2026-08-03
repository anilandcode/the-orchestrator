import { type CallEvent, CallEventSchema } from "@orchestrator/shared";
import { beforeEach, describe, expect, it } from "vitest";
import { RewardService } from "./reward.js";
import { InMemoryCallEventRepository } from "./store/memory.js";
import { SqliteCallEventRepository } from "./store/sqlite.js";

function event(overrides: Partial<CallEvent> = {}): CallEvent {
  return CallEventSchema.parse({
    id: "evt_1",
    tenantId: "local",
    requestId: "req_1",
    attempt: 1,
    provider: "openai",
    modelId: "openai/gpt-4o-mini",
    taskType: "general",
    routeMode: "balanced",
    promptTokens: 100,
    completionTokens: 50,
    costUsd: 0.001,
    latencyMs: 800,
    status: "success",
    finishReason: "stop",
    createdAt: 1_700_000_000_000,
    ...overrides,
  });
}

describe("RewardService.rescore", () => {
  let repository: InMemoryCallEventRepository;
  let rewards: RewardService;

  beforeEach(() => {
    repository = new InMemoryCallEventRepository();
    rewards = new RewardService(repository);
  });

  it("returns both rewards so the router can apply an exact correction", () => {
    // A caller that discards previousReward and simply re-teaches will double-count.
    const stored = event();
    repository.record(stored);
    const settled = rewards.settle(stored);

    const { previousReward, reward } = rewards.rescore({ ...stored, reward: settled }, 0.1);

    expect(previousReward).toBeCloseTo(settled, 12);
    expect(reward).toBeLessThan(previousReward);
  });

  it("raises the reward when the correction is favourable", () => {
    const stored = event({ finishReason: "length" });
    repository.record(stored);
    const settled = rewards.settle(stored);

    const { reward } = rewards.rescore({ ...stored, reward: settled }, 1);
    expect(reward).toBeGreaterThan(settled);
  });

  it("records provenance and counts the revision", () => {
    const stored = event();
    repository.record(stored);
    rewards.settle(stored, null, { source: "heuristic", confidence: 0.2 });

    expect(repository.query()[0]?.qualitySource).toBe("heuristic");
    expect(repository.query()[0]?.qualityRevisions).toBe(0);

    rewards.rescore(repository.query()[0] as CallEvent, 0.95, {
      source: "client-feedback",
      confidence: 1,
    });

    const reloaded = repository.query()[0];
    expect(reloaded?.qualitySource).toBe("client-feedback");
    expect(reloaded?.qualityConfidence).toBe(1);
    expect(reloaded?.qualityRevisions).toBe(1);
    expect(reloaded?.qualityScore).toBe(0.95);
  });

  it("accumulates the revision count across successive corrections", () => {
    const stored = event();
    repository.record(stored);
    rewards.settle(stored);

    rewards.rescore(repository.query()[0] as CallEvent, 0.5, { source: "judge", confidence: 0.6 });
    rewards.rescore(repository.query()[0] as CallEvent, 0.9, { source: "client", confidence: 1 });

    expect(repository.query()[0]?.qualityRevisions).toBe(2);
  });

  it("does not re-observe cost and latency into the normalizer", () => {
    // Only our judgement of the answer changed; the call's cost and latency did not. Counting it
    // twice would skew the rolling percentiles the reward normalizes against.
    const stored = event({ costUsd: 5, latencyMs: 60_000 });
    repository.record(stored);
    rewards.settle(stored);

    const before = rewards.statsFor("general");
    for (let i = 0; i < 50; i++) {
      rewards.rescore(repository.query()[0] as CallEvent, 0.5, {
        source: "judge",
        confidence: 0.5,
      });
    }
    expect(rewards.statsFor("general")).toEqual(before);
  });

  it("keeps a failed call at zero however generous the correction", () => {
    const stored = event({
      status: "error",
      errorClass: "timeout",
      finishReason: null,
      costUsd: 0,
    });
    repository.record(stored);
    rewards.settle(stored);

    expect(rewards.rescore(repository.query()[0] as CallEvent, 1).reward).toBe(0);
  });
});

describe("quality provenance persistence", () => {
  it("survives a SQLite round trip", () => {
    const repository = new SqliteCallEventRepository(":memory:");
    const rewards = new RewardService(repository);

    const stored = event();
    repository.record(stored);
    rewards.settle(stored, 0.7, { source: "tool-call-validator", confidence: 0.9 });

    const reloaded = repository.query()[0];
    expect(reloaded?.qualitySource).toBe("tool-call-validator");
    expect(reloaded?.qualityConfidence).toBeCloseTo(0.9, 10);
    expect(reloaded?.isJudge).toBe(false);
  });

  it("round-trips the judge flag, which keeps judge spend out of routing stats", () => {
    const repository = new SqliteCallEventRepository(":memory:");
    repository.record(event({ id: "judge_call", isJudge: true }));
    expect(repository.query()[0]?.isJudge).toBe(true);
  });

  it("defaults provenance to null on an unscored event rather than inventing one", () => {
    const repository = new SqliteCallEventRepository(":memory:");
    repository.record(event());

    const reloaded = repository.query()[0];
    expect(reloaded?.qualitySource).toBeNull();
    expect(reloaded?.qualityConfidence).toBeNull();
    expect(reloaded?.qualityRevisions).toBe(0);
  });
});
