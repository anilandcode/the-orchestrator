import type { ExecutionPlan } from "@orchestrator/gateway";
import type { RoutingContext } from "@orchestrator/router";
import {
  type OrchestratorError,
  UnifiedChatRequestSchema,
  systemIds,
  toOrchestratorError,
} from "@orchestrator/shared";
import { aggregateByModel, summarize } from "@orchestrator/telemetry";
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from "fastify";
import { z } from "zod";
import type { Container } from "./container.js";

/** Rough token estimate. Routing happens before the provider counts anything, so an estimate is
 *  all that exists at decision time — deliberately conservative rather than clever. */
function estimatePromptTokens(messages: { content: unknown }[]): number {
  const characters = messages.reduce((total, message) => {
    if (typeof message.content === "string") return total + message.content.length;
    return total + JSON.stringify(message.content).length;
  }, 0);
  return Math.ceil(characters / 4);
}

const FeedbackSchema = z.object({
  requestId: z.string(),
  /** 0..1. The caller's own judgement of the answer. */
  quality: z.number().min(0).max(1),
});

export function buildServer(container: Container): FastifyInstance {
  const app = Fastify({ logger: { level: process.env.LOG_LEVEL ?? "info" } });

  app.addHook("onRequest", async (request: FastifyRequest, reply: FastifyReply) => {
    if (request.url === "/healthz") return;

    const header = request.headers.authorization ?? "";
    const token = header.startsWith("Bearer ") ? header.slice(7) : "";
    if (token !== container.config.apiKey) {
      await reply.status(401).send({ error: { message: "Invalid API key", type: "auth" } });
    }
  });

  app.get("/healthz", async () => ({
    status: "ok",
    routerMode: container.config.routerMode,
    reachableModels: container.gateway.availableModels().length,
  }));

  app.get("/v1/models", async () => ({
    // Only what the gateway can actually reach given the configured keys — listing models we
    // cannot call would just produce confusing failures.
    data: container.gateway.availableModels().map((spec) => ({
      id: spec.modelId,
      provider: spec.provider,
      tier: spec.tier,
      contextWindow: spec.contextWindow,
      inputCostPerMTok: spec.inputCostPerMTok,
      outputCostPerMTok: spec.outputCostPerMTok,
      capabilities: spec.capabilities,
    })),
  }));

  app.post("/v1/chat", async (request: FastifyRequest, reply: FastifyReply) => {
    const parsed = UnifiedChatRequestSchema.safeParse({
      tenantId: container.config.defaultTenantId,
      ...(request.body as Record<string, unknown>),
    });

    if (!parsed.success) {
      return reply.status(400).send({
        error: { message: parsed.error.message, type: "invalid_request" },
      });
    }

    const chatRequest = {
      ...parsed.data,
      requestId: parsed.data.requestId ?? systemIds.generate("req"),
    };

    const routingContext: RoutingContext = {
      request: chatRequest,
      available: container.gateway.availableModels(),
      estimatedPromptTokens: estimatePromptTokens(chatRequest.messages),
    };

    try {
      const decision = container.router.select(routingContext);
      // Persisted before the call: a decision that produced a crash is still evidence.
      container.decisions.record(decision);

      const plan: ExecutionPlan = {
        modelId: decision.modelId,
        fallbacks: decision.fallbacks,
        decisionId: decision.decisionId,
        features: decision.features,
      };

      const response = await container.gateway.chat(chatRequest, plan);
      settle(container, chatRequest.requestId);

      return reply.send(response);
    } catch (raw) {
      const error = toOrchestratorError(raw);
      // Even a total failure teaches the router something, so settle before returning.
      settle(container, chatRequest.requestId);

      return reply.status(statusFor(error)).send({
        error: {
          message: error.message,
          type: error.errorClass,
          retryable: error.retryable,
        },
      });
    }
  });

  app.post("/v1/feedback", async (request: FastifyRequest, reply: FastifyReply) => {
    const parsed = FeedbackSchema.safeParse(request.body);
    if (!parsed.success) {
      return reply
        .status(400)
        .send({ error: { message: parsed.error.message, type: "invalid_request" } });
    }

    const events = container.events.query({ requestId: parsed.data.requestId });
    if (events.length === 0) {
      return reply
        .status(404)
        .send({ error: { message: "Unknown requestId", type: "invalid_request" } });
    }

    // An explicit quality signal beats the heuristic, so re-score and re-teach the router.
    for (const event of events) {
      if (event.status !== "success") continue;
      const reward = container.rewards.settle(event, parsed.data.quality);
      container.router.observe({
        modelId: event.modelId,
        features: event.features,
        taskType: event.taskType,
        reward,
      });
    }

    return reply.send({ ok: true, scored: events.length });
  });

  app.get("/v1/stats", async (request: FastifyRequest) => {
    const query = request.query as { since?: string };
    const since = query.since ? Number(query.since) : undefined;
    const events = container.events.query(since ? { since } : {});

    return {
      summary: summarize(events),
      models: aggregateByModel(events),
      router: {
        mode: container.config.routerMode,
        decisions: container.decisions.count(),
        disagreements: container.decisions.count({ disagreementsOnly: true }),
      },
    };
  });

  return app;
}

/**
 * Scores every attempt of a request and feeds each one back to the router.
 *
 * Per attempt, not per request: if the primary failed and a fallback succeeded, the router must
 * learn both facts. Collapsing them would credit the fallback's success to the model that broke.
 */
function settle(container: Container, requestId: string): void {
  for (const event of container.events.query({ requestId })) {
    const reward = container.rewards.settle(event);
    container.router.observe({
      modelId: event.modelId,
      features: event.features,
      taskType: event.taskType,
      reward,
    });
  }
}

function statusFor(error: OrchestratorError): number {
  switch (error.errorClass) {
    case "auth":
      return 401;
    case "invalid_request":
      return 400;
    case "context_length_exceeded":
      return 413;
    case "rate_limit":
      return 429;
    case "content_filter":
      return 422;
    case "timeout":
      return 504;
    case "provider_unavailable":
    case "network":
      return 502;
    default:
      return 500;
  }
}
