import { describe, expect, it } from "vitest";
import { makeInput } from "../test-helpers.js";
import { CodeStructureScorer, extractCodeBlocks, findStructuralIssue } from "./code.js";
import { FinishReasonScorer } from "./finish-reason.js";
import { JsonSchemaScorer, parseJsonLoosely, validateAgainstSchema } from "./json-schema.js";
import { ToolCallScorer } from "./tool-call.js";

/**
 * The single most important property across every validator: abstaining (`undefined`) when the
 * scorer does not apply, rather than returning 0. A 0 would teach the bandit that every model fails
 * at every task the validator does not cover.
 */
describe("abstention contract", () => {
  it("tool-call abstains when no tools were offered", () => {
    expect(new ToolCallScorer().score(makeInput())).toBeUndefined();
  });

  it("tool-call abstains when tools were offered but none were called", () => {
    // Declining to call a tool is frequently the correct answer, not a failure.
    const input = makeInput({
      request: {
        messages: [{ role: "user", content: "hi" }],
        tools: [{ name: "f", parameters: {} }],
      },
    });
    expect(new ToolCallScorer().score(input)).toBeUndefined();
  });

  it("json-schema abstains when the caller declared no output schema", () => {
    expect(new JsonSchemaScorer().score(makeInput())).toBeUndefined();
  });

  it("code-structure abstains on non-code tasks", () => {
    // Prose containing a brace is not a malformed program.
    const input = makeInput({
      request: { messages: [{ role: "user", content: "hi" }], route: { taskType: "creative" } },
      response: { message: { role: "assistant", content: "a story about { braces }" } },
    });
    expect(new CodeStructureScorer().score(input)).toBeUndefined();
  });

  it("finish-reason never abstains, because something must score every call", () => {
    expect(new FinishReasonScorer().score(makeInput())).toBeDefined();
  });
});

describe("FinishReasonScorer", () => {
  it("scores a failed call 0 with high confidence", () => {
    const assessment = new FinishReasonScorer().score(
      makeInput({ event: { status: "error", errorClass: "timeout" } }),
    );
    expect(assessment.score).toBe(0);
    expect(assessment.confidence).toBeGreaterThan(0.5);
  });

  it("scores a clean completion 0.8 but with low confidence", () => {
    // "It did not error" is barely information; any real validator must outrank it.
    const assessment = new FinishReasonScorer().score(makeInput());
    expect(assessment.score).toBe(0.8);
    expect(assessment.confidence).toBe(0.2);
  });

  it("penalizes truncation", () => {
    const assessment = new FinishReasonScorer().score(
      makeInput({ response: { finishReason: "length" } }),
    );
    expect(assessment.score).toBe(0.5);
  });

  it("scores filtered content 0 with high confidence", () => {
    const assessment = new FinishReasonScorer().score(
      makeInput({ response: { finishReason: "content_filter" } }),
    );
    expect(assessment.score).toBe(0);
    expect(assessment.confidence).toBeGreaterThan(0.5);
  });
});

describe("ToolCallScorer", () => {
  const weatherTool = {
    name: "get_weather",
    parameters: {
      type: "object",
      required: ["city"],
      properties: { city: { type: "string" }, days: { type: "integer", minimum: 1 } },
    },
  };

  const withCall = (
    toolCalls: { id: string; name: string; arguments: Record<string, unknown> }[],
  ) =>
    makeInput({
      request: { messages: [{ role: "user", content: "weather?" }], tools: [weatherTool] },
      response: { message: { role: "assistant", content: "", toolCalls } },
    });

  it("scores a fully valid call 1", () => {
    const assessment = new ToolCallScorer().score(
      withCall([{ id: "c1", name: "get_weather", arguments: { city: "Karachi" } }]),
    );
    expect(assessment?.score).toBe(1);
  });

  it("catches a missing required argument", () => {
    const assessment = new ToolCallScorer().score(
      withCall([{ id: "c1", name: "get_weather", arguments: { days: 3 } }]),
    );
    expect(assessment?.score).toBe(0);
    expect(assessment?.detail).toMatch(/missing required property "city"/);
  });

  it("catches a wrong argument type", () => {
    const assessment = new ToolCallScorer().score(
      withCall([{ id: "c1", name: "get_weather", arguments: { city: 42 } }]),
    );
    expect(assessment?.score).toBe(0);
    expect(assessment?.detail).toMatch(/expected string/);
  });

  it("catches a hallucinated tool", () => {
    const assessment = new ToolCallScorer().score(
      withCall([{ id: "c1", name: "launch_missiles", arguments: {} }]),
    );
    expect(assessment?.score).toBe(0);
    expect(assessment?.detail).toMatch(/unknown tool/);
  });

  it("scores partial validity proportionally", () => {
    const assessment = new ToolCallScorer().score(
      withCall([
        { id: "c1", name: "get_weather", arguments: { city: "Karachi" } },
        { id: "c2", name: "get_weather", arguments: {} },
      ]),
    );
    expect(assessment?.score).toBe(0.5);
  });
});

describe("validateAgainstSchema", () => {
  it("accepts a conforming object", () => {
    expect(
      validateAgainstSchema(
        { name: "x", age: 3 },
        {
          type: "object",
          required: ["name"],
          properties: { name: { type: "string" }, age: { type: "integer" } },
        },
      ),
    ).toEqual([]);
  });

  it("validates nested properties", () => {
    const errors = validateAgainstSchema(
      { user: { id: "not-a-number" } },
      {
        type: "object",
        properties: { user: { type: "object", properties: { id: { type: "number" } } } },
      },
    );
    expect(errors[0]).toMatch(/user\.id/);
  });

  it("validates array items", () => {
    const errors = validateAgainstSchema(
      { tags: ["a", 2] },
      {
        type: "object",
        properties: { tags: { type: "array", items: { type: "string" } } },
      },
    );
    expect(errors[0]).toMatch(/tags\[1\]/);
  });

  it("enforces enum membership and numeric bounds", () => {
    expect(validateAgainstSchema("maybe", { enum: ["yes", "no"] })).toHaveLength(1);
    expect(validateAgainstSchema(11, { type: "number", maximum: 10 })).toHaveLength(1);
  });

  it("distinguishes integer from number", () => {
    expect(validateAgainstSchema(1.5, { type: "integer" })).toHaveLength(1);
    expect(validateAgainstSchema(2, { type: "integer" })).toEqual([]);
  });

  it("ignores unsupported keywords rather than failing them", () => {
    // The subset can only ever be too lenient, never wrongly harsh — an unknown keyword must not
    // manufacture a quality failure.
    expect(
      validateAgainstSchema({ a: 1 }, { type: "object", allOf: [{ nonsense: true }] }),
    ).toEqual([]);
  });
});

describe("parseJsonLoosely", () => {
  it("parses bare JSON", () => {
    expect(parseJsonLoosely('{"a":1}')).toEqual({ a: 1 });
  });

  it("recovers JSON from a fenced block", () => {
    expect(parseJsonLoosely('```json\n{"a":1}\n```')).toEqual({ a: 1 });
  });

  it("recovers JSON wrapped in prose", () => {
    // Recovering formatting habits is not leniency about correctness — the schema check that
    // follows is still strict.
    expect(parseJsonLoosely('Sure! Here you go: {"a":1} Hope that helps.')).toEqual({ a: 1 });
  });

  it("recovers a top-level array", () => {
    expect(parseJsonLoosely("Result: [1,2,3]")).toEqual([1, 2, 3]);
  });

  it("returns undefined when there is no JSON at all", () => {
    expect(parseJsonLoosely("no json here")).toBeUndefined();
  });
});

describe("JsonSchemaScorer", () => {
  const schema = { type: "object", required: ["city"], properties: { city: { type: "string" } } };

  const withOutput = (content: string) =>
    makeInput({
      request: {
        messages: [{ role: "user", content: "extract" }],
        route: { taskType: "extraction", outputSchema: schema },
      },
      response: { message: { role: "assistant", content } },
    });

  it("scores conforming JSON 1", () => {
    expect(new JsonSchemaScorer().score(withOutput('{"city":"Karachi"}'))?.score).toBe(1);
  });

  it("scores non-conforming JSON 0", () => {
    expect(new JsonSchemaScorer().score(withOutput('{"town":"Karachi"}'))?.score).toBe(0);
  });

  it("scores unparseable output 0", () => {
    const assessment = new JsonSchemaScorer().score(withOutput("I could not do that"));
    expect(assessment?.score).toBe(0);
    expect(assessment?.detail).toMatch(/not parseable/);
  });

  it("claims high confidence, outranking the heuristic floor", () => {
    const schemaAssessment = new JsonSchemaScorer().score(withOutput('{"city":"Karachi"}'));
    const heuristic = new FinishReasonScorer().score(makeInput());
    expect(schemaAssessment?.confidence).toBeGreaterThan(heuristic.confidence);
  });
});

describe("code structure", () => {
  const codeInput = (content: string) =>
    makeInput({
      request: { messages: [{ role: "user", content: "write code" }], route: { taskType: "code" } },
      response: { message: { role: "assistant", content } },
    });

  it("accepts a well-formed block", () => {
    expect(
      new CodeStructureScorer().score(codeInput("```ts\nfunction f() { return 1; }\n```"))?.score,
    ).toBe(1);
  });

  it("flags truncated code", () => {
    expect(
      new CodeStructureScorer().score(codeInput("```ts\nfunction f() { return 1;\n```"))?.score,
    ).toBe(0);
  });

  it("penalizes a code request answered with no code", () => {
    const assessment = new CodeStructureScorer().score(codeInput("You should use a for loop."));
    expect(assessment?.score).toBe(0.3);
    // Low confidence: an explanation may legitimately be what was wanted.
    expect(assessment?.confidence).toBe(0.2);
  });

  it("does not treat braces inside strings as imbalance", () => {
    // Otherwise every correct answer containing a brace in a literal gets marked down.
    expect(findStructuralIssue('const s = "{ not code }";')).toBeUndefined();
  });

  it("does not treat braces inside comments as imbalance", () => {
    expect(findStructuralIssue("// what about { this\nconst a = 1;")).toBeUndefined();
    expect(findStructuralIssue("/* { */ const a = 1;")).toBeUndefined();
  });

  it("handles escaped quotes", () => {
    expect(findStructuralIssue('const s = "he said \\"hi\\"";')).toBeUndefined();
  });

  it("detects an unterminated string", () => {
    expect(findStructuralIssue('const s = "open')).toMatch(/unterminated string/);
  });

  it("detects a mismatched closer", () => {
    expect(findStructuralIssue("function f() { return [1,2; }")).toBeTruthy();
  });

  it("extracts multiple fenced blocks", () => {
    expect(extractCodeBlocks("```js\na\n```\ntext\n```py\nb\n```")).toHaveLength(2);
  });
});
