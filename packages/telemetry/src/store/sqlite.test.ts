import { type CallEvent, CallEventSchema } from "@orchestrator/shared";
import { beforeEach, describe, expect, it } from "vitest";
import { aggregateByModel, summarize } from "../aggregate.js";
import { SqliteCallEventRepository } from "./sqlite.js";

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

describe("SqliteCallEventRepository", () => {
  let repository: SqliteCallEventRepository;

  beforeEach(() => {
    repository = new SqliteCallEventRepository(":memory:");
  });

  it("round-trips an event through storage without losing fields", () => {
    const original = event({
      routingDecisionId: "dec_1",
      features: [0.25, 1, 0],
      ttftMs: 120,
      cachedPromptTokens: 40,
      qualityScore: 0.9,
      reward: 0.77,
    });
    repository.record(original);

    const [reloaded] = repository.query();
    expect(reloaded).toEqual(original);
  });

  it("preserves sub-cent cost precision", () => {
    repository.record(event({ costUsd: 0.000_000_123 }));
    expect(repository.query()[0]?.costUsd).toBeCloseTo(0.000_000_123, 12);
  });

  it("stores every attempt of a request separately", () => {
    repository.recordMany([
      event({ id: "evt_1", attempt: 1, status: "error", errorClass: "rate_limit", costUsd: 0 }),
      event({ id: "evt_2", attempt: 2, status: "success" }),
    ]);

    expect(repository.count()).toBe(2);
    expect(repository.query().map((e) => e.attempt)).toEqual([1, 2]);
  });

  it("filters by model, task type, status, and time window", () => {
    repository.recordMany([
      event({ id: "a", modelId: "openai/gpt-4o-mini", taskType: "code", createdAt: 1_000 }),
      event({ id: "b", modelId: "anthropic/claude-haiku-4-5", taskType: "code", createdAt: 2_000 }),
      event({
        id: "c",
        modelId: "openai/gpt-4o-mini",
        taskType: "general",
        createdAt: 3_000,
        status: "error",
        errorClass: "timeout",
      }),
    ]);

    expect(repository.query({ modelId: "openai/gpt-4o-mini" })).toHaveLength(2);
    expect(repository.query({ taskType: "code" })).toHaveLength(2);
    expect(repository.query({ status: "error" })).toHaveLength(1);
    expect(repository.query({ since: 2_000 })).toHaveLength(2);
    expect(repository.query({ since: 2_000, until: 3_000 })).toHaveLength(1);
  });

  it("isolates tenants", () => {
    repository.recordMany([
      event({ id: "a", tenantId: "acme" }),
      event({ id: "b", tenantId: "globex" }),
    ]);
    expect(repository.query({ tenantId: "acme" }).map((e) => e.id)).toEqual(["a"]);
  });

  it("returns events oldest-first, which is the order replay needs", () => {
    repository.recordMany([
      event({ id: "late", createdAt: 3_000 }),
      event({ id: "early", createdAt: 1_000 }),
      event({ id: "mid", createdAt: 2_000 }),
    ]);
    expect(repository.query().map((e) => e.id)).toEqual(["early", "mid", "late"]);
  });

  it("applies a limit", () => {
    repository.recordMany([1, 2, 3, 4, 5].map((n) => event({ id: `e${n}`, createdAt: n })));
    expect(repository.query({ limit: 2 }).map((e) => e.id)).toEqual(["e1", "e2"]);
  });

  it("attaches a reward after the fact", () => {
    repository.record(event());
    repository.scoreEvent("evt_1", 0.9, 0.71);

    const [reloaded] = repository.query();
    expect(reloaded?.qualityScore).toBe(0.9);
    expect(reloaded?.reward).toBe(0.71);
  });

  it("survives reopening the same schema without re-running migrations", () => {
    const second = new SqliteCallEventRepository(repository.connection);
    second.record(event({ id: "evt_2" }));
    expect(repository.count()).toBe(1);
  });
});

describe("aggregateByModel", () => {
  it("computes success rate over attempts, not requests", () => {
    const stats = aggregateByModel([
      event({ id: "a", modelId: "m1", status: "error", errorClass: "timeout", costUsd: 0 }),
      event({ id: "b", modelId: "m1", status: "success" }),
      event({ id: "c", modelId: "m1", status: "success" }),
    ]);

    expect(stats[0]?.attempts).toBe(3);
    expect(stats[0]?.successes).toBe(2);
    expect(stats[0]?.successRate).toBeCloseTo(2 / 3, 10);
  });

  it("charges spend on failed attempts against successful ones", () => {
    const stats = aggregateByModel([
      event({ id: "a", modelId: "m1", status: "error", errorClass: "timeout", costUsd: 0.01 }),
      event({ id: "b", modelId: "m1", status: "success", costUsd: 0.01 }),
    ]);
    // $0.02 spent, one success: the honest cost per useful answer is $0.02, not $0.01.
    expect(stats[0]?.costPerSuccessUsd).toBeCloseTo(0.02, 10);
  });

  it("reports infinite cost per success when nothing succeeded", () => {
    const stats = aggregateByModel([
      event({ modelId: "m1", status: "error", errorClass: "auth", costUsd: 0.01 }),
    ]);
    expect(stats[0]?.costPerSuccessUsd).toBe(Number.POSITIVE_INFINITY);
  });

  it("breaks failures down by class", () => {
    const stats = aggregateByModel([
      event({ id: "a", modelId: "m1", status: "error", errorClass: "rate_limit" }),
      event({ id: "b", modelId: "m1", status: "error", errorClass: "rate_limit" }),
      event({ id: "c", modelId: "m1", status: "error", errorClass: "timeout" }),
    ]);
    expect(stats[0]?.errorsByClass).toEqual({ rate_limit: 2, timeout: 1 });
  });

  it("ranks models by average reward", () => {
    const stats = aggregateByModel([
      event({ id: "a", modelId: "slow", latencyMs: 9_000, costUsd: 0.04 }),
      event({ id: "b", modelId: "fast", latencyMs: 200, costUsd: 0.0001 }),
    ]);
    expect(stats[0]?.modelId).toBe("fast");
  });
});

describe("summarize", () => {
  it("counts distinct requests separately from attempts", () => {
    const summary = summarize([
      event({ id: "a", requestId: "r1", attempt: 1, status: "error", errorClass: "timeout" }),
      event({ id: "b", requestId: "r1", attempt: 2, status: "success" }),
      event({ id: "c", requestId: "r2", attempt: 1, status: "success" }),
    ]);

    expect(summary.events).toBe(3);
    expect(summary.requests).toBe(2);
    // One of two requests needed more than one attempt.
    expect(summary.retryRate).toBeCloseTo(0.5, 10);
  });

  it("handles an empty window without dividing by zero", () => {
    expect(summarize([])).toMatchObject({ events: 0, requests: 0, successRate: 0, retryRate: 0 });
  });
});
