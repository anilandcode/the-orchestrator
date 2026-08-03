import type { QualityAssessment, QualityInput, QualityScorer } from "./scorer.js";

/**
 * Runs scorers and picks a winner.
 *
 * **Precedence, not blending.** The highest-confidence scorer that has an opinion wins outright.
 * Averaging a confident human score with a vague heuristic yields a number that describes neither —
 * and because the reward function weights quality at up to 0.85, a meaningless number there is worse
 * than an honest low-confidence one.
 */
export class QualityPipeline {
  private readonly scorers: QualityScorer[];

  constructor(scorers: QualityScorer[]) {
    this.scorers = scorers;
  }

  /** Scorers safe to run on the request path. */
  get inlineScorers(): QualityScorer[] {
    return this.scorers.filter((scorer) => scorer.stage === "inline");
  }

  /** Scorers that cost money or time and must run after the response is returned. */
  get deferredScorers(): QualityScorer[] {
    return this.scorers.filter((scorer) => scorer.stage === "deferred");
  }

  async assessInline(input: QualityInput): Promise<QualityAssessment | undefined> {
    return this.run(this.inlineScorers, input);
  }

  async assessDeferred(input: QualityInput): Promise<QualityAssessment | undefined> {
    return this.run(this.deferredScorers, input);
  }

  private async run(
    scorers: QualityScorer[],
    input: QualityInput,
  ): Promise<QualityAssessment | undefined> {
    let best: QualityAssessment | undefined;

    for (const scorer of scorers) {
      let assessment: QualityAssessment | undefined;
      try {
        assessment = await scorer.score(input);
      } catch {
        // A broken scorer must not fail the request it was grading. Abstaining is the safe
        // interpretation: we learned nothing, rather than "the model did badly".
        continue;
      }

      // undefined is abstention, not a zero. Skip without letting it influence the outcome.
      if (!assessment) continue;
      if (!best || assessment.confidence > best.confidence) best = assessment;
    }

    return best;
  }
}

/**
 * Should a new assessment replace one already recorded?
 *
 * Ties go to the incumbent so a re-run of the same scorer does not churn the stored provenance and
 * inflate the revision count for no reason.
 */
export function supersedes(
  incoming: QualityAssessment,
  existingConfidence: number | null,
): boolean {
  if (existingConfidence === null) return true;
  return incoming.confidence > existingConfidence;
}
