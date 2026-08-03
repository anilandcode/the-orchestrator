import {
  CONFIDENCE,
  type QualityAssessment,
  type QualityInput,
  type QualityScorer,
} from "../scorer.js";

/**
 * The floor.
 *
 * This is the pre-existing heuristic, now explicit about how little it knows: confidence 0.2, so any
 * real validator or judge outranks it. It exists so every call has *some* score, not because a clean
 * `finish_reason` tells us the answer was good.
 *
 * Never remove it — without a floor, calls that no specialized scorer covers would carry no quality
 * signal at all and the reward's quality term would be undefined rather than merely vague.
 */
export class FinishReasonScorer implements QualityScorer {
  readonly name = "finish-reason";
  readonly stage = "inline" as const;

  score(input: QualityInput): QualityAssessment {
    const { event, response } = input;

    if (event.status === "error") {
      return {
        score: 0,
        confidence: CONFIDENCE.deterministic,
        source: this.name,
        detail: "call failed",
      };
    }

    switch (response.finishReason) {
      case "content_filter":
        return {
          score: 0,
          confidence: CONFIDENCE.deterministic,
          source: this.name,
          detail: "content filtered",
        };
      case "length":
        return {
          score: 0.5,
          confidence: CONFIDENCE.heuristic,
          source: this.name,
          detail: "truncated before completing",
        };
      default:
        return {
          score: 0.8,
          confidence: CONFIDENCE.heuristic,
          source: this.name,
          detail: "completed without error",
        };
    }
  }
}
