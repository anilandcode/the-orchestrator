import type { ExecutionPlan, Gateway } from "@orchestrator/gateway";
import {
  ModelNodeConfigSchema,
  type NodeExecutionContext,
  type NodeExecutionResult,
  type NodeExecutor,
  interpolate,
} from "@orchestrator/orchestrator";
import type { AdaptiveRouter } from "@orchestrator/router";
import {
  type Message,
  UnifiedChatRequestSchema,
  type UnifiedChatResponse,
  systemIds,
} from "@orchestrator/shared";
import type { RoutingDecisionRepository } from "@orchestrator/telemetry";

/**
 * Runs one model node of a workflow.
 *
 * This is where the router is consulted **per node** rather than once per workflow — the property
 * that makes multi-step orchestration worth pairing with adaptive routing. A pipeline can classify
 * cheaply, reason expensively, and summarize cheaply again, with each step routed on its own merits
 * and its own reward.
 *
 * It lives in `apps/api` rather than in the orchestrator package because it needs both the gateway
 * and the router, and the orchestrator is forbidden from importing either.
 */
export class ModelNodeExecutor implements NodeExecutor {
  constructor(
    private readonly deps: {
      gateway: Gateway;
      router: AdaptiveRouter;
      decisions: RoutingDecisionRepository;
      /** Called after the response lands, so the run's calls feed the same learning loop. */
      onSettled: (
        requestId: string,
        request: unknown,
        response: UnifiedChatResponse,
      ) => Promise<void>;
      estimatePromptTokens: (messages: { content: unknown }[]) => number;
    },
  ) {}

  async execute(context: NodeExecutionContext): Promise<NodeExecutionResult> {
    const config = ModelNodeConfigSchema.parse(context.node.config);

    const messages: Message[] = [];
    if (config.system) {
      messages.push({ role: "system", content: interpolate(config.system, context.variables) });
    }
    messages.push({ role: "user", content: interpolate(config.prompt, context.variables) });

    const request = UnifiedChatRequestSchema.parse({
      tenantId: context.tenantId,
      requestId: systemIds.generate("req"),
      messages,
      ...(config.tools ? { tools: config.tools } : {}),
      ...(config.maxTokens !== undefined ? { maxTokens: config.maxTokens } : {}),
      ...(config.temperature !== undefined ? { temperature: config.temperature } : {}),
      route: config.route,
    });

    const decision = this.deps.router.select({
      request,
      available: this.deps.gateway.availableModels(),
      estimatedPromptTokens: this.deps.estimatePromptTokens(request.messages),
    });
    this.deps.decisions.record(decision);

    const plan: ExecutionPlan = {
      modelId: decision.modelId,
      fallbacks: decision.fallbacks,
      decisionId: decision.decisionId,
      features: decision.features,
    };

    const response = await this.deps.gateway.chat(request, plan);
    // Workflow calls are ordinary traffic: they must be scored and fed back exactly like a direct
    // /v1/chat call, or the router learns from only half of what the system does.
    await this.deps.onSettled(request.requestId as string, request, response);

    return {
      output: { [config.outputVar]: textOf(response) },
      costUsd: response.costUsd,
      latencyMs: response.latencyMs,
      decisionId: decision.decisionId,
      modelId: response.modelId,
    };
  }
}

function textOf(response: UnifiedChatResponse): string {
  const { content } = response.message;
  if (typeof content === "string") return content;
  return content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("");
}
