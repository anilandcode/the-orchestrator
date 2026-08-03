import type { Gateway } from "@orchestrator/gateway";
import {
  type CallEventSink,
  type Clock,
  UnifiedChatRequestSchema,
  systemClock,
} from "@orchestrator/shared";
import {
  CONFIDENCE,
  type QualityAssessment,
  type QualityInput,
  type QualityScorer,
  responseText,
} from "../scorer.js";
import { JUDGE_SYSTEM_PROMPT, buildJudgePrompt, parseJudgeScore } from "./rubric.js";

export interface LlmJudgeConfig {
  gateway: Gateway;
  /** Pinned deliberately — see the note on `score`. */
  modelId: string;
  /** Fraction of calls to grade, 0..1. */
  sampleRate?: number;
  /** Hard ceiling on judge spend in any rolling hour. */
  maxUsdPerHour?: number;
  /** Judge output is a single number; this caps a runaway reply. */
  maxTokens?: number;
  clock?: Clock;
  random?: () => number;
  /** Judge traffic is recorded here so its spend is visible but separable. */
  sink?: CallEventSink;
}

interface Spend {
  at: number;
  usd: number;
}

/**
 * Grades a sample of responses with a cheap model.
 *
 * Three properties matter more than the grading itself:
 *
 *   1. **Pinned model, never routed.** The judge calls a fixed model directly. Routing it through the
 *      bandit would let the bandit influence its own grades — the arm that gets picked to judge would
 *      shape the rewards that decide which arm gets picked.
 *   2. **Deferred.** It runs after the response has already been returned to the caller. A judge on
 *      the hot path would add its own latency to every request it grades, which the reward function
 *      would then score as the *graded model* being slow.
 *   3. **Capped.** A judge billing more than the traffic it grades is a real failure mode. Spend is
 *      tracked in a rolling window and the circuit breaker opens rather than degrading quietly.
 */
export class LlmJudgeScorer implements QualityScorer {
  readonly name = "llm-judge";
  readonly stage = "deferred" as const;

  private readonly gateway: Gateway;
  private readonly modelId: string;
  private readonly sampleRate: number;
  private readonly maxUsdPerHour: number;
  private readonly maxTokens: number;
  private readonly clock: Clock;
  private readonly random: () => number;

  private readonly spend: Spend[] = [];
  private breakerOpenedAt: number | undefined;

  constructor(config: LlmJudgeConfig) {
    this.gateway = config.gateway;
    this.modelId = config.modelId;
    this.sampleRate = config.sampleRate ?? 0.05;
    this.maxUsdPerHour = config.maxUsdPerHour ?? 1;
    this.maxTokens = config.maxTokens ?? 8;
    this.clock = config.clock ?? systemClock;
    this.random = config.random ?? Math.random;
  }

  /** Judge spend in the trailing hour. */
  spentLastHour(): number {
    this.prune();
    return this.spend.reduce((total, entry) => total + entry.usd, 0);
  }

  isBreakerOpen(): boolean {
    return this.spentLastHour() >= this.maxUsdPerHour;
  }

  async score(input: QualityInput): Promise<QualityAssessment | undefined> {
    // Grading a failed call tells us nothing the error class did not already say, and still costs
    // money.
    if (input.event.status !== "success") return undefined;
    // Never grade the judge's own traffic — that recursion would bill forever.
    if (input.event.isJudge) return undefined;
    if (this.random() >= this.sampleRate) return undefined;

    if (this.isBreakerOpen()) {
      if (this.breakerOpenedAt === undefined) this.breakerOpenedAt = this.clock.now();
      // Abstain rather than throw: the cap is a budget decision, not an error, and the pipeline
      // should fall through to a cheaper scorer.
      return undefined;
    }
    this.breakerOpenedAt = undefined;

    const request = UnifiedChatRequestSchema.parse({
      tenantId: input.request.tenantId,
      messages: [
        { role: "system", content: JUDGE_SYSTEM_PROMPT },
        {
          role: "user",
          content: buildJudgePrompt(
            input.request.route.taskType,
            promptOf(input),
            responseText(input.response),
          ),
        },
      ],
      maxTokens: this.maxTokens,
      temperature: 0,
      route: { pin: this.modelId, mode: "cheap" },
    });

    try {
      const judgement = await this.gateway.chat(request, { modelId: this.modelId });
      this.spend.push({ at: this.clock.now(), usd: judgement.costUsd });

      const score = parseJudgeScore(responseText(judgement));
      // A judge that ignored its instructions has told us nothing. Abstaining is honest; inventing
      // a midpoint would inject noise into the signal this phase exists to make trustworthy.
      if (score === undefined) return undefined;

      return {
        score,
        confidence: CONFIDENCE.judge,
        source: this.name,
        detail: `judged by ${this.modelId}`,
      };
    } catch {
      // A judge outage must never fail the request it was grading.
      return undefined;
    }
  }

  private prune(): void {
    const cutoff = this.clock.now() - 3_600_000;
    while (this.spend.length > 0 && (this.spend[0] as Spend).at < cutoff) this.spend.shift();
  }
}

function promptOf(input: QualityInput): string {
  return input.request.messages
    .filter((message) => message.role === "user")
    .map((message) =>
      typeof message.content === "string" ? message.content : JSON.stringify(message.content),
    )
    .join("\n\n");
}
