import { OrchestratorError, createFixedClock, createSequentialIds } from "@orchestrator/shared";
import { beforeEach, describe, expect, it } from "vitest";
import type { NodeExecutionContext, NodeExecutionResult, NodeExecutor } from "./executor.js";
import { CheckpointExecutor, TransformExecutor } from "./executors/builtin.js";
import { GraphRunner } from "./runner.js";
import { type WorkflowDefinitionInput, WorkflowDefinitionSchema } from "./schema.js";
import { InMemoryRunStore } from "./store/runs.js";

/** Records what it was asked to do, and can be told to fail a given number of times first. */
class ScriptedExecutor implements NodeExecutor {
  calls: { nodeId: string; attempt: number; variables: Record<string, unknown> }[] = [];

  constructor(
    private readonly output: Record<string, unknown> = { result: "ok" },
    private readonly failures = 0,
    private readonly errorClass: "timeout" | "invalid_request" = "timeout",
  ) {}

  execute(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    this.calls.push({
      nodeId: context.node.id,
      attempt: context.attempt,
      variables: { ...context.variables },
    });

    if (this.calls.length <= this.failures) {
      return Promise.reject(new OrchestratorError(this.errorClass, `boom ${this.calls.length}`));
    }
    return Promise.resolve({ output: this.output, costUsd: 0.01, modelId: "test/model" });
  }
}

const workflow = (overrides: Partial<WorkflowDefinitionInput> = {}) =>
  WorkflowDefinitionSchema.parse({
    id: "wf",
    entry: "a",
    nodes: [{ id: "a", type: "model", config: {} }],
    ...overrides,
  });

describe("GraphRunner", () => {
  let store: InMemoryRunStore;
  let clock: ReturnType<typeof createFixedClock>;

  beforeEach(() => {
    store = new InMemoryRunStore();
    clock = createFixedClock();
  });

  const runner = (executors: Record<string, NodeExecutor>) =>
    new GraphRunner({
      executors: executors as never,
      store,
      clock,
      ids: createSequentialIds(),
    });

  describe("linear execution", () => {
    it("runs a single node and completes", async () => {
      const model = new ScriptedExecutor({ answer: "42" });
      const state = await runner({ model }).start(workflow(), { input: { question: "x" } });

      expect(state.status).toBe("completed");
      expect(state.variables).toEqual({ question: "x", answer: "42" });
      expect(state.totalCostUsd).toBeCloseTo(0.01, 10);
    });

    it("threads variables from one node into the next", async () => {
      const model = new ScriptedExecutor({ draft: "hello" });
      const transform = new TransformExecutor();

      const state = await runner({ model, transform }).start(
        workflow({
          nodes: [
            { id: "a", type: "model", config: {} },
            { id: "b", type: "transform", config: { set: { shout: "{{draft}}!" } } },
          ],
          edges: [{ from: "a", to: "b" }],
        }),
      );

      expect(state.status).toBe("completed");
      expect(state.variables.shout).toBe("hello!");
    });

    it("treats a node with no outgoing edge as terminal", async () => {
      const model = new ScriptedExecutor();
      const state = await runner({ model }).start(
        workflow({
          nodes: [
            { id: "a", type: "model", config: {} },
            { id: "orphan", type: "model", config: {} },
          ],
        }),
      );

      expect(state.status).toBe("completed");
      expect(model.calls.map((c) => c.nodeId)).toEqual(["a"]);
    });
  });

  describe("branching", () => {
    const branching = (score: number) =>
      workflow({
        entry: "score",
        nodes: [
          { id: "score", type: "transform", config: { set: { score: String(score) } } },
          { id: "high", type: "transform", config: { set: { path: "high" } } },
          { id: "low", type: "transform", config: { set: { path: "low" } } },
        ],
        edges: [
          { from: "score", to: "high", when: { path: "score", op: "gte", value: 7 } },
          { from: "score", to: "low" },
        ],
      });

    it("follows the first edge whose guard passes", async () => {
      const state = await runner({ transform: new TransformExecutor() }).start(branching(9));
      expect(state.variables.path).toBe("high");
    });

    it("falls through to an unconditional edge listed last", async () => {
      // Declaration order is the tie-break, so an unguarded edge acts as a default.
      const state = await runner({ transform: new TransformExecutor() }).start(branching(3));
      expect(state.variables.path).toBe("low");
    });
  });

  describe("retries", () => {
    it("retries a retryable failure up to maxAttempts", async () => {
      const model = new ScriptedExecutor({ answer: "ok" }, 2, "timeout");
      const state = await runner({ model }).start(
        workflow({ nodes: [{ id: "a", type: "model", config: {}, maxAttempts: 3 }] }),
      );

      expect(state.status).toBe("completed");
      expect(model.calls.map((c) => c.attempt)).toEqual([1, 2, 3]);
    });

    it("fails the run once attempts are exhausted", async () => {
      const model = new ScriptedExecutor({}, 5, "timeout");
      const state = await runner({ model }).start(
        workflow({ nodes: [{ id: "a", type: "model", config: {}, maxAttempts: 2 }] }),
      );

      expect(state.status).toBe("failed");
      expect(state.error).toMatch(/boom/);
      expect(model.calls).toHaveLength(2);
    });

    it("does not retry a non-retryable failure", async () => {
      // A malformed request is malformed on the second attempt too.
      const model = new ScriptedExecutor({}, 5, "invalid_request");
      const state = await runner({ model }).start(
        workflow({ nodes: [{ id: "a", type: "model", config: {}, maxAttempts: 5 }] }),
      );

      expect(state.status).toBe("failed");
      expect(model.calls).toHaveLength(1);
    });

    it("fails the run when no executor is registered for a node type", async () => {
      const state = await runner({}).start(workflow());
      expect(state.status).toBe("failed");
      expect(state.error).toMatch(/No executor registered/);
    });
  });

  describe("loop bounds", () => {
    it("stops a workflow that cycles forever", async () => {
      // The failure mode this exists to prevent: a guard that never goes false, billing a provider
      // on every pass.
      const model = new ScriptedExecutor();
      const state = await runner({ model }).start(
        workflow({
          maxSteps: 6,
          nodes: [
            { id: "a", type: "model", config: {} },
            { id: "b", type: "model", config: {} },
          ],
          edges: [
            { from: "a", to: "b" },
            { from: "b", to: "a" },
          ],
        }),
      );

      expect(state.status).toBe("failed");
      expect(state.error).toMatch(/exceeded maxSteps/);
      expect(model.calls.length).toBeLessThanOrEqual(6);
    });

    it("counts retries against the step budget", async () => {
      // Retries consume provider budget too, so bounding distinct nodes alone would not bound work.
      const model = new ScriptedExecutor({}, 10, "timeout");
      const state = await runner({ model }).start(
        workflow({ maxSteps: 3, nodes: [{ id: "a", type: "model", config: {}, maxAttempts: 10 }] }),
      );

      expect(state.status).toBe("failed");
      expect(model.calls.length).toBeLessThanOrEqual(4);
    });
  });

  describe("pause and resume", () => {
    const approval = workflow({
      entry: "draft",
      nodes: [
        { id: "draft", type: "model", config: {} },
        { id: "approve", type: "checkpoint", config: { prompt: "Approve: {{answer}}?" } },
        { id: "publish", type: "transform", config: { set: { published: "yes" } } },
      ],
      edges: [
        { from: "draft", to: "approve" },
        { from: "approve", to: "publish", when: { path: "decision", op: "eq", value: "approve" } },
      ],
    });

    it("suspends at a checkpoint with an interpolated prompt", async () => {
      const graph = runner({
        model: new ScriptedExecutor({ answer: "the draft" }),
        checkpoint: new CheckpointExecutor(),
        transform: new TransformExecutor(),
      });

      const state = await graph.start(approval);
      expect(state.status).toBe("paused");
      expect(state.pendingPrompt).toBe("Approve: the draft?");
      expect(state.variables.published).toBeUndefined();
    });

    it("continues from the checkpoint when resumed", async () => {
      const graph = runner({
        model: new ScriptedExecutor({ answer: "the draft" }),
        checkpoint: new CheckpointExecutor(),
        transform: new TransformExecutor(),
      });

      const paused = await graph.start(approval);
      const resumed = await graph.resume(approval, paused.runId, { decision: "approve" });

      expect(resumed.status).toBe("completed");
      expect(resumed.variables.published).toBe("yes");
    });

    it("does not re-pause on the checkpoint it just resumed from", async () => {
      const checkpoint = new CheckpointExecutor();
      const graph = runner({
        model: new ScriptedExecutor({ answer: "d" }),
        checkpoint,
        transform: new TransformExecutor(),
      });

      const paused = await graph.start(approval);
      const resumed = await graph.resume(approval, paused.runId, { decision: "approve" });
      expect(resumed.status).toBe("completed");
    });

    it("completes rather than hanging when no edge matches after resume", async () => {
      const graph = runner({
        model: new ScriptedExecutor({ answer: "d" }),
        checkpoint: new CheckpointExecutor(),
        transform: new TransformExecutor(),
      });

      const paused = await graph.start(approval);
      const resumed = await graph.resume(approval, paused.runId, { decision: "reject" });

      expect(resumed.status).toBe("completed");
      expect(resumed.variables.published).toBeUndefined();
    });

    it("refuses to resume a run that is not paused", async () => {
      const graph = runner({ model: new ScriptedExecutor() });
      const finished = await graph.start(workflow());

      await expect(graph.resume(workflow(), finished.runId)).rejects.toThrow(/not paused/);
    });

    it("rejects an unknown run id", async () => {
      await expect(runner({}).resume(workflow(), "run_nope")).rejects.toThrow(/Unknown run/);
    });
  });

  describe("durability", () => {
    it("rebuilds identical state from the event log alone", async () => {
      // State is a fold over events; if a second reader disagrees with the runner, resumption is
      // unsound.
      const graph = runner({ model: new ScriptedExecutor({ answer: "x" }) });
      const state = await graph.start(workflow(), { input: { seed: 1 } });

      const reloaded = graph.getState(state.runId);
      expect(reloaded).toEqual(state);
    });

    it("survives being resumed by a different runner instance", async () => {
      // The realistic case: the process that paused the run is long gone.
      const first = runner({
        model: new ScriptedExecutor({ answer: "d" }),
        checkpoint: new CheckpointExecutor(),
        transform: new TransformExecutor(),
      });

      const approval = workflow({
        entry: "gate",
        nodes: [
          { id: "gate", type: "checkpoint", config: { prompt: "ok?" } },
          { id: "done", type: "transform", config: { set: { finished: "yes" } } },
        ],
        edges: [{ from: "gate", to: "done" }],
      });

      const paused = await first.start(approval);

      const second = new GraphRunner({
        executors: { checkpoint: new CheckpointExecutor(), transform: new TransformExecutor() },
        store,
        clock,
        ids: createSequentialIds(),
      });

      const resumed = await second.resume(approval, paused.runId, { answer: "yes" });
      expect(resumed.status).toBe("completed");
      expect(resumed.variables.finished).toBe("yes");
    });

    it("records every attempt in the log, including failures", async () => {
      const graph = runner({ model: new ScriptedExecutor({}, 1, "timeout") });
      const state = await graph.start(
        workflow({ nodes: [{ id: "a", type: "model", config: {}, maxAttempts: 2 }] }),
      );

      const types = store.events(state.runId).map((event) => event.type);
      expect(types).toContain("node_failed");
      expect(types).toContain("node_succeeded");
      expect(types.filter((t) => t === "node_started")).toHaveLength(2);
    });

    it("tracks run status in the store, not only in derived state", async () => {
      const graph = runner({ model: new ScriptedExecutor() });
      const state = await graph.start(workflow());
      expect(store.get(state.runId)?.status).toBe("completed");
    });

    it("accumulates cost across nodes", async () => {
      const graph = runner({ model: new ScriptedExecutor() });
      const state = await graph.start(
        workflow({
          nodes: [
            { id: "a", type: "model", config: {} },
            { id: "b", type: "model", config: {} },
          ],
          edges: [{ from: "a", to: "b" }],
        }),
      );
      expect(state.totalCostUsd).toBeCloseTo(0.02, 10);
    });
  });
});
