import {
  type Clock,
  type IdGenerator,
  OrchestratorError,
  systemClock,
  systemIds,
  toOrchestratorError,
} from "@orchestrator/shared";
import type { NodeExecutorMap } from "./executor.js";
import { evaluateCondition } from "./guards.js";
import type { WorkflowDefinition, WorkflowNode } from "./schema.js";
import { type RunEvent, type RunState, deriveState } from "./state.js";
import type { RunStore } from "./store/runs.js";

export interface GraphRunnerConfig {
  executors: NodeExecutorMap;
  store: RunStore;
  clock?: Clock;
  ids?: IdGenerator;
  /** Per-node wall-clock ceiling. */
  nodeTimeoutMs?: number;
}

export interface StartOptions {
  tenantId?: string;
  input?: Record<string, unknown>;
  runId?: string;
}

/**
 * Executes a workflow graph.
 *
 * Three properties are the point of this class:
 *
 *   - **Durable.** Every step appends to the run log before the next one begins, so a crash resumes
 *     from what actually happened rather than from the beginning.
 *   - **Resumable.** A checkpoint node suspends the run; `resume` continues it, possibly days later
 *     in a different process.
 *   - **Bounded.** `maxSteps` caps total executed attempts. A guard that never goes false would
 *     otherwise loop against a paid provider indefinitely.
 */
export class GraphRunner {
  private readonly executors: NodeExecutorMap;
  private readonly store: RunStore;
  private readonly clock: Clock;
  private readonly ids: IdGenerator;
  private readonly nodeTimeoutMs: number;

  constructor(config: GraphRunnerConfig) {
    this.executors = config.executors;
    this.store = config.store;
    this.clock = config.clock ?? systemClock;
    this.ids = config.ids ?? systemIds;
    this.nodeTimeoutMs = config.nodeTimeoutMs ?? 120_000;
  }

  async start(workflow: WorkflowDefinition, options: StartOptions = {}): Promise<RunState> {
    const runId = options.runId ?? this.ids.generate("run");
    const tenantId = options.tenantId ?? "local";
    const now = this.clock.now();

    this.store.create({
      runId,
      tenantId,
      workflowId: workflow.id,
      status: "running",
      createdAt: now,
      updatedAt: now,
    });

    this.store.append(runId, [
      {
        type: "run_started",
        at: now,
        workflowId: workflow.id,
        entry: workflow.entry,
        input: options.input ?? {},
      },
    ]);

    return this.drive(workflow, runId, tenantId);
  }

  /**
   * Continue a paused run.
   *
   * `input` lands in the run's variables, which is how a human's answer at a checkpoint reaches the
   * rest of the workflow.
   */
  async resume(
    workflow: WorkflowDefinition,
    runId: string,
    input: Record<string, unknown> = {},
  ): Promise<RunState> {
    const record = this.store.get(runId);
    if (!record) throw new OrchestratorError("invalid_request", `Unknown run: ${runId}`);

    const state = this.load(runId, record.tenantId);
    if (state.status !== "paused") {
      throw new OrchestratorError(
        "invalid_request",
        `Run ${runId} is ${state.status}, not paused; only a paused run can be resumed`,
      );
    }

    this.store.append(runId, [
      {
        type: "run_resumed",
        at: this.clock.now(),
        nodeId: state.currentNodeId ?? "",
        input,
      },
    ]);

    return this.drive(workflow, runId, record.tenantId, { skipCurrent: true });
  }

  getState(runId: string): RunState | undefined {
    const record = this.store.get(runId);
    return record ? this.load(runId, record.tenantId) : undefined;
  }

  // --- internals ------------------------------------------------------------

  private load(runId: string, tenantId: string): RunState {
    return deriveState(runId, tenantId, this.store.events(runId));
  }

  /**
   * The step loop.
   *
   * `skipCurrent` is set on resume: the checkpoint node that paused the run has already done its
   * job, so execution advances past it rather than pausing again immediately.
   */
  private async drive(
    workflow: WorkflowDefinition,
    runId: string,
    tenantId: string,
    options: { skipCurrent?: boolean } = {},
  ): Promise<RunState> {
    let state = this.load(runId, tenantId);
    let advancePast = options.skipCurrent ?? false;

    while (state.status === "running") {
      const nodeId = state.currentNodeId;
      if (!nodeId) {
        state = this.fail(runId, tenantId, null, "Run has no current node");
        break;
      }

      if (advancePast) {
        advancePast = false;
        const next = this.nextNodeId(workflow, nodeId, state);
        state = next ? this.moveTo(runId, tenantId, next) : this.complete(runId, tenantId, nodeId);
        continue;
      }

      if (state.steps >= workflow.maxSteps) {
        // A cycle that never resolves is the failure mode this exists to stop.
        state = this.fail(
          runId,
          tenantId,
          nodeId,
          `Run exceeded maxSteps (${workflow.maxSteps}); the graph is probably looping`,
        );
        break;
      }

      const node = workflow.nodes.find((candidate) => candidate.id === nodeId);
      if (!node) {
        state = this.fail(runId, tenantId, nodeId, `Unknown node: ${nodeId}`);
        break;
      }

      const outcome = await this.runNode(runId, tenantId, node, state, workflow.maxSteps);
      state = outcome.state;

      if (outcome.status === "failed" || outcome.status === "paused") break;

      const next = this.nextNodeId(workflow, node.id, state);
      state = next ? this.moveTo(runId, tenantId, next) : this.complete(runId, tenantId, node.id);
    }

    return state;
  }

  private async runNode(
    runId: string,
    tenantId: string,
    node: WorkflowNode,
    state: RunState,
    maxSteps: number,
  ): Promise<{ state: RunState; status: "ok" | "failed" | "paused" }> {
    const executor = this.executors[node.type];
    if (!executor) {
      return {
        state: this.fail(
          runId,
          tenantId,
          node.id,
          `No executor registered for node type: ${node.type}`,
        ),
        status: "failed",
      };
    }

    let current = state;

    for (let attempt = 1; attempt <= node.maxAttempts; attempt++) {
      // The step budget binds retries too, not just node transitions. Checking only in the outer
      // loop would let a node with a high maxAttempts burn its full quota of paid provider calls
      // after the run had already exhausted its budget — which is the exact runaway maxSteps exists
      // to stop.
      if (current.steps >= maxSteps) {
        return {
          state: this.fail(
            runId,
            tenantId,
            node.id,
            `Run exceeded maxSteps (${maxSteps}) while retrying ${node.id}`,
          ),
          status: "failed",
        };
      }

      this.store.append(runId, [
        { type: "node_started", at: this.clock.now(), nodeId: node.id, attempt },
      ]);
      const startedAt = this.clock.monotonic();

      try {
        const result = await executor.execute({
          runId,
          tenantId,
          node,
          variables: current.variables,
          attempt,
          signal: AbortSignal.timeout(this.nodeTimeoutMs),
        });

        if (result.pause) {
          this.store.append(runId, [
            {
              type: "run_paused",
              at: this.clock.now(),
              nodeId: node.id,
              prompt: result.pause.prompt,
            },
          ]);
          const paused = this.load(runId, tenantId);
          this.store.updateStatus(runId, "paused", this.clock.now());
          return { state: paused, status: "paused" };
        }

        this.store.append(runId, [
          {
            type: "node_succeeded",
            at: this.clock.now(),
            nodeId: node.id,
            output: result.output,
            costUsd: result.costUsd ?? 0,
            latencyMs: result.latencyMs ?? this.clock.monotonic() - startedAt,
            decisionId: result.decisionId ?? null,
            modelId: result.modelId ?? null,
          },
        ]);

        return { state: this.load(runId, tenantId), status: "ok" };
      } catch (raw) {
        const error = toOrchestratorError(raw);
        const willRetry = attempt < node.maxAttempts && error.retryable;

        this.store.append(runId, [
          {
            type: "node_failed",
            at: this.clock.now(),
            nodeId: node.id,
            attempt,
            error: error.message,
            errorClass: error.errorClass,
            willRetry,
          },
        ]);

        current = this.load(runId, tenantId);
        if (willRetry) continue;

        return {
          state: this.fail(runId, tenantId, node.id, error.message),
          status: "failed",
        };
      }
    }

    return {
      state: this.fail(runId, tenantId, node.id, `Node ${node.id} exhausted its attempts`),
      status: "failed",
    };
  }

  /**
   * Pick the next node.
   *
   * Edges are evaluated in declaration order and the first matching guard wins, so an unconditional
   * edge acts as a default when listed last. No matching edge means this node was terminal.
   */
  private nextNodeId(
    workflow: WorkflowDefinition,
    fromId: string,
    state: RunState,
  ): string | undefined {
    for (const edge of workflow.edges) {
      if (edge.from !== fromId) continue;
      if (!edge.when || evaluateCondition(edge.when, state.variables)) return edge.to;
    }
    return undefined;
  }

  private moveTo(runId: string, tenantId: string, nodeId: string): RunState {
    const state = this.load(runId, tenantId);
    state.currentNodeId = nodeId;
    return state;
  }

  private complete(runId: string, tenantId: string, terminalNodeId: string): RunState {
    this.store.append(runId, [{ type: "run_completed", at: this.clock.now(), terminalNodeId }]);
    this.store.updateStatus(runId, "completed", this.clock.now());
    return this.load(runId, tenantId);
  }

  private fail(runId: string, tenantId: string, nodeId: string | null, error: string): RunState {
    this.store.append(runId, [{ type: "run_failed", at: this.clock.now(), nodeId, error }]);
    this.store.updateStatus(runId, "failed", this.clock.now());
    return this.load(runId, tenantId);
  }
}
