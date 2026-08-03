import { z } from "zod";
import { RouteModeSchema, type TaskType, TaskTypeSchema } from "./chat.js";

export const RoutingStrategySchema = z.enum(["static", "adaptive", "pinned"]);
export type RoutingStrategy = z.infer<typeof RoutingStrategySchema>;

export const RoutingDecisionSchema = z.object({
  decisionId: z.string(),
  /** The model that will actually be called. */
  modelId: z.string(),
  /** Ordered chain tried on fallback-eligible failures. */
  fallbacks: z.array(z.string()),
  strategy: RoutingStrategySchema,
  /** Human-readable justification. Routing decisions must be explainable, not just correct. */
  reason: z.string(),
  /**
   * In shadow mode the bandit's pick is recorded here while the static pick executes. Comparing the
   * two columns over real traffic is what earns the right to flip ROUTER_MODE to `adaptive`.
   */
  shadowModelId: z.string().nullable().default(null),
  features: z.array(z.number()).default([]),
  taskType: TaskTypeSchema,
  routeMode: RouteModeSchema,
  createdAt: z.number().int().nonnegative(),
});

export type RoutingDecision = z.infer<typeof RoutingDecisionSchema>;

/** What the router learns from after a call settles. */
export interface RoutingOutcome {
  modelId: string;
  features: number[];
  /** 0..1, from the telemetry reward function. */
  reward: number;
  /** Carried explicitly so the router's cold-start gate stays per-task without re-deriving it. */
  taskType?: TaskType;
  /**
   * 0..1 — how much authority the scorer behind `reward` claimed.
   *
   * This is what lets the router notice that a task type is only ever graded by "the call did not
   * error", and decline to steer it. Omitting it leaves the router unable to tell an informative
   * reward from an uninformative one, so it will assume the signal is usable.
   */
  qualityConfidence?: number;
  /** Links back to the decision this outcome settles, when the caller still has it. */
  decisionId?: string;
}
