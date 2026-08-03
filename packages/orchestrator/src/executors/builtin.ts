import type { NodeExecutionContext, NodeExecutionResult, NodeExecutor } from "../executor.js";
import { interpolate } from "../guards.js";
import { CheckpointNodeConfigSchema, TransformNodeConfigSchema } from "../schema.js";

/**
 * Executors with no I/O, so they belong in this package rather than being injected.
 *
 * The model and tool executors deliberately do not live here: they need the gateway and the router,
 * and an orchestrator that imported those would stop being swappable.
 */

/** Reshapes run variables from templates. Pure — no network, no clock, no randomness. */
export class TransformExecutor implements NodeExecutor {
  execute(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    const config = TransformNodeConfigSchema.parse(context.node.config);

    const output: Record<string, unknown> = {};
    for (const [target, template] of Object.entries(config.set)) {
      output[target] = interpolate(template, context.variables);
    }

    return Promise.resolve({ output, costUsd: 0 });
  }
}

/**
 * Suspends the run for human input.
 *
 * The node itself does almost nothing — the interesting behaviour is that the runner persists and
 * stops, and `resume` picks up possibly days later in a different process. That is the entire reason
 * run state is durable and event-sourced rather than held in memory.
 */
export class CheckpointExecutor implements NodeExecutor {
  execute(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    const config = CheckpointNodeConfigSchema.parse(context.node.config);

    return Promise.resolve({
      output: {},
      costUsd: 0,
      pause: { prompt: interpolate(config.prompt, context.variables) },
    });
  }
}
