import { z } from "zod";
import { ERROR_CLASSES } from "../errors.js";
import { FinishReasonSchema, ProviderIdSchema, RouteModeSchema, TaskTypeSchema } from "./chat.js";

/**
 * The join point between the gateway and the router, and the only durable record of what actually
 * happened on a call.
 *
 * One event per **attempt**, never per request. A request that fails over twice writes three events.
 * Collapsing them would attribute a successful outcome to the model that failed, which is precisely
 * the signal the bandit learns from.
 */
export const CallStatusSchema = z.enum(["success", "error"]);
export type CallStatus = z.infer<typeof CallStatusSchema>;

export const ErrorClassSchema = z.enum(ERROR_CLASSES);

export const CallEventSchema = z.object({
  id: z.string(),
  tenantId: z.string(),
  /** Shared across every attempt belonging to one caller-facing request. */
  requestId: z.string(),
  /** Links back to the RoutingDecision that picked this model, so reward can be attributed. */
  routingDecisionId: z.string().nullable().default(null),
  /** 1-based. Attempt 2+ means a retry or a fallback fired. */
  attempt: z.number().int().positive(),

  provider: ProviderIdSchema,
  modelId: z.string(),
  taskType: TaskTypeSchema,
  routeMode: RouteModeSchema,
  /** The feature vector the router saw. Stored so replay can score counterfactually. */
  features: z.array(z.number()).default([]),

  promptTokens: z.number().int().nonnegative(),
  completionTokens: z.number().int().nonnegative(),
  cachedPromptTokens: z.number().int().nonnegative().default(0),
  costUsd: z.number().nonnegative(),
  latencyMs: z.number().nonnegative(),
  ttftMs: z.number().nonnegative().nullable().default(null),

  status: CallStatusSchema,
  errorClass: ErrorClassSchema.nullable().default(null),
  finishReason: FinishReasonSchema.nullable().default(null),

  /** 0..1. Populated by a QualityScorer or by explicit feedback; null until then. */
  qualityScore: z.number().min(0).max(1).nullable().default(null),
  /**
   * Which scorer produced `qualityScore`. Without this you cannot tell a genuinely better model from
   * one that happened to be graded by a more lenient scorer.
   */
  qualitySource: z.string().nullable().default(null),
  /** 0..1. How much authority the scorer claims; a strict validator outranks the heuristic floor. */
  qualityConfidence: z.number().min(0).max(1).nullable().default(null),
  /** How many times a later signal corrected this event's reward. */
  qualityRevisions: z.number().int().nonnegative().default(0),
  /**
   * True for calls the LLM judge itself made. Real spend, but must never be routed on or counted as
   * user traffic — otherwise the judge pollutes the statistics it exists to produce.
   */
  isJudge: z.boolean().default(false),
  /** 0..1. Computed by the telemetry reward function once quality is known. */
  reward: z.number().min(0).max(1).nullable().default(null),

  /** Epoch milliseconds. */
  createdAt: z.number().int().nonnegative(),
});

export type CallEvent = z.infer<typeof CallEventSchema>;
export type CallEventInput = z.input<typeof CallEventSchema>;

/** The gateway emits through this. Keeping it an interface is why gateway does not import telemetry. */
export interface CallEventSink {
  record(event: CallEvent): void;
}

/** Collects events in memory. Used by tests and by the replay simulator. */
export class InMemoryCallEventSink implements CallEventSink {
  readonly events: CallEvent[] = [];
  record(event: CallEvent): void {
    this.events.push(event);
  }
}

export const NOOP_CALL_EVENT_SINK: CallEventSink = { record: () => {} };
