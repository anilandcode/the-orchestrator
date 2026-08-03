import type { CallEvent, UnifiedChatRequest, UnifiedChatResponse } from "@orchestrator/shared";

export interface QualityInput {
  request: UnifiedChatRequest;
  response: UnifiedChatResponse;
  event: CallEvent;
}

export interface QualityAssessment {
  /** 0..1. */
  score: number;
  /**
   * 0..1 authority claim. Used to decide whether a later signal should override this one, and
   * persisted so an analyst can tell a genuinely good model from a leniently-graded one.
   */
  confidence: number;
  /** Scorer name, persisted as `quality_source`. */
  source: string;
  /** Optional human-readable justification, useful when auditing a surprising score. */
  detail?: string;
}

/**
 * A source of quality judgement.
 *
 * The return type is the important part of this contract: **`undefined` means "no opinion", and is
 * categorically different from a score of `0`.**
 *
 * A JSON-schema validator handed a creative-writing task has nothing to say. If it returned 0 the
 * bandit would learn that every model fails at creative writing — a confident, wrong signal that is
 * strictly worse than the vague heuristic it replaced. Abstention is not a fallback path here; it is
 * the common case for every specialized scorer.
 */
export interface QualityScorer {
  readonly name: string;
  /**
   * `inline` runs on the request path and must be free and synchronous-fast.
   * `deferred` runs after the response is returned and may cost money or take seconds.
   */
  readonly stage: "inline" | "deferred";
  score(
    input: QualityInput,
  ): Promise<QualityAssessment | undefined> | QualityAssessment | undefined;
}

/** Confidence bands, so scorers agree on what the numbers mean rather than each inventing a scale. */
export const CONFIDENCE = {
  /** A human said so. */
  human: 1,
  /** A deterministic check that definitively applies (schema conformance, parse success). */
  deterministic: 0.9,
  /** A model's opinion. Real signal, but noisy and not reproducible. */
  judge: 0.6,
  /** "The call did not error." Barely information — the floor, not a judgement. */
  heuristic: 0.2,
} as const;

/** Extract plain text from a response, whatever content shape the provider used. */
export function responseText(response: UnifiedChatResponse): string {
  const { content } = response.message;
  if (typeof content === "string") return content;
  return content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("");
}
