import type { ToolRegistry } from "@orchestrator/mcp";
import {
  type NodeExecutionContext,
  type NodeExecutionResult,
  type NodeExecutor,
  ToolNodeConfigSchema,
  interpolate,
} from "@orchestrator/orchestrator";

/**
 * Runs a `tool` workflow node against the MCP registry.
 *
 * Policy is enforced inside the registry rather than here, so a workflow cannot reach a tool its
 * tenant was never granted simply by naming it. The executor's only job is to interpolate arguments
 * and hand the result back as a run variable.
 */
export class ToolNodeExecutor implements NodeExecutor {
  constructor(private readonly registry: ToolRegistry) {}

  async execute(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    const config = ToolNodeConfigSchema.parse(context.node.config);

    const args: Record<string, unknown> = {};
    for (const [key, template] of Object.entries(config.arguments)) {
      args[key] = interpolate(template, context.variables);
    }

    const result = await this.registry.call({
      tenantId: context.tenantId,
      qualifiedName: config.tool,
      arguments: args,
      runId: context.runId,
    });

    // A tool reporting its own failure is not a node failure: the workflow may well have a guard
    // that branches on it, and throwing here would deny it that choice.
    return {
      output: {
        [config.outputVar]: result.text,
        [`${config.outputVar}_isError`]: result.isError,
      },
      costUsd: 0,
    };
  }
}
