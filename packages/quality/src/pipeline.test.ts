import { describe, expect, it } from "vitest";
import { QualityPipeline, supersedes } from "./pipeline.js";
import { CONFIDENCE, type QualityAssessment, type QualityScorer } from "./scorer.js";
import { makeInput } from "./test-helpers.js";

function scorer(
  name: string,
  stage: "inline" | "deferred",
  assessment: QualityAssessment | undefined,
): QualityScorer {
  return { name, stage, score: () => assessment };
}

const assessment = (score: number, confidence: number, source = "s"): QualityAssessment => ({
  score,
  confidence,
  source,
});

describe("QualityPipeline", () => {
  it("prefers the highest-confidence opinion rather than averaging", async () => {
    // Averaging a confident human score with a vague heuristic yields a number describing neither.
    const pipeline = new QualityPipeline([
      scorer("heuristic", "inline", assessment(0.8, CONFIDENCE.heuristic, "heuristic")),
      scorer("validator", "inline", assessment(0.1, CONFIDENCE.deterministic, "validator")),
    ]);

    const best = await pipeline.assessInline(makeInput());
    expect(best?.score).toBe(0.1);
    expect(best?.source).toBe("validator");
  });

  it("ignores abstaining scorers entirely", async () => {
    const pipeline = new QualityPipeline([
      scorer("abstainer", "inline", undefined),
      scorer("heuristic", "inline", assessment(0.8, CONFIDENCE.heuristic, "heuristic")),
    ]);

    const best = await pipeline.assessInline(makeInput());
    // An abstention must not be read as a zero, nor block a lower-confidence real opinion.
    expect(best?.source).toBe("heuristic");
    expect(best?.score).toBe(0.8);
  });

  it("returns undefined when every scorer abstains", async () => {
    const pipeline = new QualityPipeline([scorer("a", "inline", undefined)]);
    expect(await pipeline.assessInline(makeInput())).toBeUndefined();
  });

  it("survives a scorer that throws", async () => {
    // A broken scorer must not fail the request it was grading, and must not be read as a zero.
    const broken: QualityScorer = {
      name: "broken",
      stage: "inline",
      score: () => {
        throw new Error("scorer exploded");
      },
    };
    const pipeline = new QualityPipeline([
      broken,
      scorer("ok", "inline", assessment(0.7, CONFIDENCE.deterministic, "ok")),
    ]);

    const best = await pipeline.assessInline(makeInput());
    expect(best?.source).toBe("ok");
  });

  it("keeps inline and deferred scorers separate", async () => {
    const pipeline = new QualityPipeline([
      scorer("inline-one", "inline", assessment(0.5, 0.9, "inline-one")),
      scorer("deferred-one", "deferred", assessment(0.9, 0.6, "deferred-one")),
    ]);

    expect(pipeline.inlineScorers).toHaveLength(1);
    expect(pipeline.deferredScorers).toHaveLength(1);
    // A deferred scorer must never be pulled onto the request path, however confident it is.
    expect((await pipeline.assessInline(makeInput()))?.source).toBe("inline-one");
    expect((await pipeline.assessDeferred(makeInput()))?.source).toBe("deferred-one");
  });

  it("ranks the standard confidence bands in the intended order", () => {
    expect(CONFIDENCE.human).toBeGreaterThan(CONFIDENCE.deterministic);
    expect(CONFIDENCE.deterministic).toBeGreaterThan(CONFIDENCE.judge);
    expect(CONFIDENCE.judge).toBeGreaterThan(CONFIDENCE.heuristic);
  });
});

describe("supersedes", () => {
  it("accepts any assessment when nothing is recorded", () => {
    expect(supersedes(assessment(0.5, CONFIDENCE.heuristic), null)).toBe(true);
  });

  it("lets a stronger signal replace a weaker one", () => {
    expect(supersedes(assessment(0.5, CONFIDENCE.human), CONFIDENCE.judge)).toBe(true);
  });

  it("refuses to let a weaker signal overwrite a stronger one", () => {
    // A sampled judge must not overwrite a human's verdict.
    expect(supersedes(assessment(0.9, CONFIDENCE.judge), CONFIDENCE.human)).toBe(false);
  });

  it("gives ties to the incumbent, avoiding pointless revision churn", () => {
    expect(supersedes(assessment(0.4, CONFIDENCE.judge), CONFIDENCE.judge)).toBe(false);
  });
});
