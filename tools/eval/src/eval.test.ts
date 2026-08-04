import type { AdapterResult, ProviderAdapter } from "@orchestrator/gateway";
import { Gateway } from "@orchestrator/gateway";
import {
  type ModelSpec,
  type ProviderId,
  type UnifiedChatChunk,
  defaultRegistry,
} from "@orchestrator/shared";
import { describe, expect, it } from "vitest";
import { FixtureSchema, gradeAgainstFixture, loadFixtures } from "./fixtures.js";
import { runEval, summarizeCapabilities } from "./runner.js";

/** Replies with whatever the test dictates, so the harness is exercised with no network. */
class ScriptedAdapter implements ProviderAdapter {
  constructor(
    readonly provider: ProviderId,
    private readonly replies: Record<string, string>,
    private readonly failOn: string[] = [],
  ) {}

  async chat(
    request: { messages: { content: unknown }[] },
    spec: ModelSpec,
  ): Promise<AdapterResult> {
    if (this.failOn.includes(spec.modelId)) throw new Error("provider exploded");

    const prompt = String(request.messages.at(-1)?.content ?? "");
    const key = Object.keys(this.replies).find((needle) => prompt.includes(needle));

    return {
      message: { role: "assistant", content: key ? (this.replies[key] as string) : "" },
      finishReason: "stop",
      usage: { promptTokens: 50, completionTokens: 20, totalTokens: 70, cachedPromptTokens: 0 },
    };
  }

  async *stream(): AsyncGenerator<UnifiedChatChunk, void, undefined> {
    yield { type: "finish", finishReason: "stop" };
  }
}

const fixture = (overrides: Record<string, unknown> = {}) =>
  FixtureSchema.parse({
    id: "f1",
    taskType: "classification",
    prompt: "Classify: the delivery was late",
    mustContain: ["negative"],
    ...overrides,
  });

describe("fixtures", () => {
  it("loads and validates the shipped starter set", () => {
    const fixtures = loadFixtures();
    expect(fixtures.length).toBeGreaterThan(0);
    expect(new Set(fixtures.map((f) => f.id)).size).toBe(fixtures.length);
  });

  it("covers more than one task type, so priors are not all about one thing", () => {
    const taskTypes = new Set(loadFixtures().map((f) => f.taskType));
    expect(taskTypes.size).toBeGreaterThanOrEqual(4);
  });

  it("covers the task types benchmarks could not", () => {
    // extraction, summarization, and classification get no benchmark mapping precisely because
    // public coverage is too thin. Measuring them ourselves is the whole point of this harness.
    const taskTypes = new Set(loadFixtures().map((f) => f.taskType));
    expect(taskTypes.has("extraction")).toBe(true);
    expect(taskTypes.has("classification")).toBe(true);
    expect(taskTypes.has("summarization")).toBe(true);
  });
});

describe("gradeAgainstFixture", () => {
  it("scores a fully correct answer 1", () => {
    expect(gradeAgainstFixture(fixture(), "This is negative")).toBe(1);
  });

  it("scores a wrong answer 0", () => {
    expect(gradeAgainstFixture(fixture(), "positive")).toBe(0);
  });

  it("fails outright on forbidden content, even when required content is present", () => {
    // A right answer carrying a wrong claim is a wrong answer.
    const graded = gradeAgainstFixture(
      fixture({ mustContain: ["negative"], mustNotContain: ["positive"] }),
      "negative, though arguably positive",
    );
    expect(graded).toBe(0);
  });

  it("scores partial matches proportionally", () => {
    const graded = gradeAgainstFixture(fixture({ mustContain: ["alpha", "beta"] }), "only alpha");
    expect(graded).toBe(0.5);
  });

  it("is case-insensitive", () => {
    expect(gradeAgainstFixture(fixture(), "NEGATIVE")).toBe(1);
  });

  it("abstains when the fixture asserts nothing checkable", () => {
    // Same rule as the quality package: a fixture with no assertions cannot say whether an answer
    // was good, and scoring it anyway would manufacture a signal.
    expect(
      gradeAgainstFixture(fixture({ mustContain: [], mustNotContain: [] }), "x"),
    ).toBeUndefined();
  });
});

describe("runEval", () => {
  const models: ModelSpec[] = [
    defaultRegistry.require("openai/gpt-4o-mini"),
    defaultRegistry.require("openai/gpt-4o"),
  ];

  const build = (replies: Record<string, string>, failOn: string[] = []) =>
    new Gateway({ adapters: [new ScriptedAdapter("openai", replies, failOn)] });

  it("runs every fixture against every model", async () => {
    const gateway = build({ Classify: "negative" });
    const results = await runEval({ gateway, fixtures: [fixture()], models });

    expect(results).toHaveLength(2);
    expect(results.map((r) => r.modelId).sort()).toEqual(["openai/gpt-4o", "openai/gpt-4o-mini"]);
  });

  it("pins the model, so it measures the model rather than the router", async () => {
    // Routing here would measure the router's choice instead of each model's ability.
    const gateway = build({ Classify: "negative" });
    const results = await runEval({ gateway, fixtures: [fixture()], models });
    expect(results.every((r) => r.score === 1)).toBe(true);
  });

  it("does not let the heuristic floor cap a verified answer", async () => {
    // The heuristic returns 0.8 for any clean completion because it knows almost nothing. If it
    // were folded into the strictness rule, every correct answer would cap at 0.8 and the ranking
    // this harness exists to produce would flatten out.
    const gateway = build({ Classify: "negative" });
    const results = await runEval({
      gateway,
      fixtures: [fixture()],
      models: [models[0] as ModelSpec],
    });
    expect(results[0]?.score).toBe(1);
  });

  it("falls back to the heuristic when nothing deterministic applies", async () => {
    const gateway = build({ Anything: "a perfectly reasonable reply" });
    const results = await runEval({
      gateway,
      fixtures: [
        fixture({ id: "open", prompt: "Anything goes here", mustContain: [], mustNotContain: [] }),
      ],
      models: [models[0] as ModelSpec],
    });
    expect(results[0]?.score).toBeCloseTo(0.8, 6);
  });

  it("takes the stricter of the fixture assertion and the validator", async () => {
    // Structurally valid JSON carrying the wrong value is not a good answer, and a structural
    // scorer alone would call it one.
    const gateway = build({ invoice: '{"invoiceNumber":"WRONG","total":1,"currency":"GBP"}' });
    const results = await runEval({
      gateway,
      fixtures: [
        fixture({
          id: "schema",
          taskType: "extraction",
          prompt: "Extract the invoice fields",
          outputSchema: {
            type: "object",
            required: ["invoiceNumber"],
            properties: { invoiceNumber: { type: "string" } },
          },
          mustContain: ["INV-4471"],
        }),
      ],
      models: [models[0] as ModelSpec],
    });

    expect(results[0]?.score).toBe(0);
  });

  it("records a provider failure as a zero rather than a gap", async () => {
    // A model that errors on a task is worse at that task; omitting it would flatter the model.
    const gateway = build({ Classify: "negative" }, ["openai/gpt-4o"]);
    const results = await runEval({ gateway, fixtures: [fixture()], models });

    const failed = results.find((r) => r.modelId === "openai/gpt-4o");
    expect(failed?.score).toBe(0);
    expect(failed?.error).toMatch(/exploded/);
  });

  it("reports progress, because the real run is slow and costs money", async () => {
    const seen: string[] = [];
    await runEval({
      gateway: build({ Classify: "negative" }),
      fixtures: [fixture()],
      models,
      onProgress: (_done, _total, label) => seen.push(label),
    });
    expect(seen).toHaveLength(2);
  });
});

describe("summarizeCapabilities", () => {
  const result = (modelId: string, fixtureId: string, score: number) => ({
    fixtureId,
    modelId,
    taskType: "code" as const,
    score,
    costUsd: 0.001,
    latencyMs: 100,
    error: null,
  });

  it("averages scores per model and task", () => {
    const capabilities = summarizeCapabilities([
      result("m1", "a", 1),
      result("m1", "b", 0),
      result("m1", "c", 0.5),
    ]);

    expect(capabilities[0]?.capability).toBeCloseTo(0.5, 6);
  });

  it("weakens coverage when measured on few fixtures", () => {
    // Three examples is an anecdote; the prior weight should say so.
    const thin = summarizeCapabilities([result("m1", "a", 1)]);
    const thick = summarizeCapabilities(
      Array.from({ length: 10 }, (_, i) => result("m1", `f${i}`, 1)),
    );

    expect(thin[0]?.coverage).toBeLessThan(thick[0]?.coverage as number);
    expect(thick[0]?.coverage).toBe(1);
  });

  it("emits the shape the catalog's derivePriors already consumes", () => {
    // Measured evidence flows through the existing tested path rather than a parallel one.
    const [capability] = summarizeCapabilities([result("m1", "a", 0.8)]);
    expect(capability).toMatchObject({
      modelId: "m1",
      taskType: "code",
      contributingBenchmarks: ["a"],
    });
    expect(capability?.capability).toBeGreaterThanOrEqual(0);
    expect(capability?.capability).toBeLessThanOrEqual(1);
  });

  it("separates task types for the same model", () => {
    const capabilities = summarizeCapabilities([
      { ...result("m1", "a", 1), taskType: "code" },
      { ...result("m1", "b", 0), taskType: "reasoning" },
    ]);

    expect(capabilities).toHaveLength(2);
    expect(capabilities.find((c) => c.taskType === "code")?.capability).toBe(1);
    expect(capabilities.find((c) => c.taskType === "reasoning")?.capability).toBe(0);
  });
});
