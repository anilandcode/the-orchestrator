import type { Variables } from "./guards.js";
import type { NodeType, WorkflowNode } from "./schema.js";

/**
 * How a single node actually gets performed.
 *
 * This interface is the reason the orchestrator imports neither the gateway nor the router. An
 * orchestrator that knew how to call a provider would be a second gateway; one that knew how to pick
 * a model would make routing unswappable. It sequences steps and decides what happens on failure —
 * nothing more.
 */
export interface NodeExecutionContext {
  runId: string;
  tenantId: string;
  node: WorkflowNode;
  variables: Variables;
  /** 1-based. Passed through so an executor can vary behaviour on retry if it wants to. */
  attempt: number;
  signal: AbortSignal;
}

export interface NodeExecutionResult {
  /** Variables to merge into the run. */
  output: Variables;
  costUsd?: number;
  latencyMs?: number;
  /** Set by model nodes so a run can be traced back to the routing decisions inside it. */
  decisionId?: string | null;
  modelId?: string | null;
  /**
   * Set by a checkpoint node to suspend the run. The runner persists and stops; nothing further
   * executes until `resume` is called.
   */
  pause?: { prompt: string };
}

export interface NodeExecutor {
  execute(context: NodeExecutionContext): Promise<NodeExecutionResult>;
}

export type NodeExecutorMap = Partial<Record<NodeType, NodeExecutor>>;
