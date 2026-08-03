import { z } from "zod";
import type { Variables } from "./guards.js";

/**
 * Run state is *derived* from an append-only event log, never patched in place.
 *
 * The reason is resumption. A run that crashed mid-step must restart from what actually happened,
 * and a mutable snapshot cannot distinguish "the model call succeeded and we crashed before saving"
 * from "the model call never ran" — the difference between a duplicate paid call and a lost one.
 * The log makes that unambiguous.
 */

export const RunStatusSchema = z.enum(["running", "paused", "completed", "failed"]);
export type RunStatus = z.infer<typeof RunStatusSchema>;

export const RunEventSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("run_started"),
    at: z.number().int(),
    workflowId: z.string(),
    entry: z.string(),
    input: z.record(z.string(), z.unknown()),
  }),
  z.object({
    type: z.literal("node_started"),
    at: z.number().int(),
    nodeId: z.string(),
    attempt: z.number().int().positive(),
  }),
  z.object({
    type: z.literal("node_succeeded"),
    at: z.number().int(),
    nodeId: z.string(),
    /** Variables this node set. Applied over the run's variables when state is rebuilt. */
    output: z.record(z.string(), z.unknown()),
    costUsd: z.number().nonnegative().default(0),
    latencyMs: z.number().nonnegative().default(0),
    /** Present for model nodes, so a run can be traced back to its routing decisions. */
    decisionId: z.string().nullable().default(null),
    modelId: z.string().nullable().default(null),
  }),
  z.object({
    type: z.literal("node_failed"),
    at: z.number().int(),
    nodeId: z.string(),
    attempt: z.number().int().positive(),
    error: z.string(),
    errorClass: z.string().nullable().default(null),
    /** False once attempts are exhausted — the point the run itself fails. */
    willRetry: z.boolean(),
  }),
  z.object({
    type: z.literal("run_paused"),
    at: z.number().int(),
    nodeId: z.string(),
    prompt: z.string(),
  }),
  z.object({
    type: z.literal("run_resumed"),
    at: z.number().int(),
    nodeId: z.string(),
    input: z.record(z.string(), z.unknown()),
  }),
  z.object({
    type: z.literal("run_completed"),
    at: z.number().int(),
    terminalNodeId: z.string(),
  }),
  z.object({
    type: z.literal("run_failed"),
    at: z.number().int(),
    nodeId: z.string().nullable(),
    error: z.string(),
  }),
]);

export type RunEvent = z.infer<typeof RunEventSchema>;

export interface RunState {
  runId: string;
  workflowId: string;
  tenantId: string;
  status: RunStatus;
  /** The node awaiting execution, or the one that paused/failed the run. */
  currentNodeId: string | null;
  variables: Variables;
  steps: number;
  totalCostUsd: number;
  error: string | null;
  /** Set while paused, so a caller knows what the run is waiting for. */
  pendingPrompt: string | null;
  createdAt: number;
  updatedAt: number;
}

/**
 * Rebuild state by folding the event log.
 *
 * This function is the definition of what a run *is*. Anything that mutates state elsewhere is a bug
 * waiting to diverge from the durable record.
 */
export function deriveState(runId: string, tenantId: string, events: RunEvent[]): RunState {
  const state: RunState = {
    runId,
    workflowId: "",
    tenantId,
    status: "running",
    currentNodeId: null,
    variables: {},
    steps: 0,
    totalCostUsd: 0,
    error: null,
    pendingPrompt: null,
    createdAt: events[0]?.at ?? 0,
    updatedAt: events.at(-1)?.at ?? 0,
  };

  for (const event of events) {
    switch (event.type) {
      case "run_started":
        state.workflowId = event.workflowId;
        state.currentNodeId = event.entry;
        state.variables = { ...event.input };
        state.status = "running";
        break;

      case "node_started":
        state.currentNodeId = event.nodeId;
        // Steps count executed attempts, not distinct nodes: a retrying node consumes budget too,
        // and maxSteps exists to bound total work rather than graph depth.
        state.steps += 1;
        break;

      case "node_succeeded":
        state.variables = { ...state.variables, ...event.output };
        state.totalCostUsd += event.costUsd;
        break;

      case "node_failed":
        if (!event.willRetry) {
          state.error = event.error;
        }
        break;

      case "run_paused":
        state.status = "paused";
        state.currentNodeId = event.nodeId;
        state.pendingPrompt = event.prompt;
        break;

      case "run_resumed":
        state.status = "running";
        state.variables = { ...state.variables, ...event.input };
        state.pendingPrompt = null;
        break;

      case "run_completed":
        state.status = "completed";
        state.currentNodeId = null;
        state.pendingPrompt = null;
        break;

      case "run_failed":
        state.status = "failed";
        state.error = event.error;
        state.pendingPrompt = null;
        break;
    }
  }

  return state;
}
