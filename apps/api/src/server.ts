import type { ExecutionPlan } from "@orchestrator/gateway";
import {
  CheckpointExecutor,
  GraphRunner,
  TransformExecutor,
  WorkflowDefinitionSchema,
} from "@orchestrator/orchestrator";
import { CONFIDENCE, supersedes } from "@orchestrator/quality";
import type { RoutingContext } from "@orchestrator/router";
import {
  type OrchestratorError,
  type UnifiedChatRequest,
  UnifiedChatRequestSchema,
  type UnifiedChatResponse,
  systemIds,
  toOrchestratorError,
} from "@orchestrator/shared";
import { aggregateByModel, summarize } from "@orchestrator/telemetry";
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from "fastify";
import { z } from "zod";
import type { Container } from "./container.js";
import { ModelNodeExecutor } from "./executors/model-node.js";
import { ToolNodeExecutor } from "./executors/tool-node.js";

/** The text a recall query is built from: the caller's most recent turn. */
function lastUserText(messages: { role: string; content: unknown }[]): string {
  const lastUser = [...messages].reverse().find((message) => message.role === "user");
  if (!lastUser) return "";
  return typeof lastUser.content === "string" ? lastUser.content : JSON.stringify(lastUser.content);
}

/**
 * Rough token estimate. Routing happens before the provider counts anything, so an estimate is all
 * that exists at decision time — deliberately conservative rather than clever.
 */
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

    // Recall happens before routing: injected context changes the prompt length the router
    // estimates from and the budget it checks against, so routing on the pre-recall prompt would
    // under-count and could blow a caller's maxCostUsd.
    if (chatRequest.memory?.recall) {
      const recalled = await container.memory.recall({
        tenantId: chatRequest.tenantId,
        sessionId: chatRequest.memory.sessionId,
        query: lastUserText(chatRequest.messages),
      });

      if (recalled.context) {
        chatRequest.messages = [
          { role: "system", content: recalled.context },
          ...chatRequest.messages,
        ];
      }
    }

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
      await settle(container, chatRequest.requestId, { request: chatRequest, response });

      if (chatRequest.memory?.write) {
        // Only the caller's own turns and the reply — never the recalled context we just injected,
        // which would compound into memory-of-memory on every turn.
        await container.memory.remember({
          tenantId: chatRequest.tenantId,
          sessionId: chatRequest.memory.sessionId,
          messages: [...parsed.data.messages, response.message],
        });
      }

      // Deferred scoring runs after the reply is sent. Awaiting it here would add the judge's
      // latency to the very call it is grading, and the reward function would read that as the
      // graded model being slow.
      void scoreDeferred(container, app, chatRequest, response).catch((error: unknown) => {
        app.log.warn({ error }, "deferred quality scoring failed");
      });

      return reply.send(response);
    } catch (raw) {
      const error = toOrchestratorError(raw);
      // Even a total failure teaches the router something, so settle before returning.
      await settle(container, chatRequest.requestId);

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

    // A human's verdict is the highest-authority signal there is, so it always supersedes.
    // This revises the existing reward rather than adding a second observation — re-teaching would
    // double-count the call and inflate the arm's apparent confidence.
    let revised = 0;
    for (const event of events) {
      if (event.status !== "success") continue;

      const { reward } = container.rewards.rescore(event, parsed.data.quality, {
        source: "client-feedback",
        confidence: CONFIDENCE.human,
      });

      if (event.routingDecisionId) {
        container.router.reviseOutcome(event.routingDecisionId, reward);
      }
      revised += 1;
    }

    return reply.send({ ok: true, scored: revised });
  });

  // --- workflows ------------------------------------------------------------

  // Executors are built here because the model node needs `settle`, which the server owns. The
  // orchestrator itself never sees the gateway or the router.
  const runner = new GraphRunner({
    executors: {
      transform: new TransformExecutor(),
      checkpoint: new CheckpointExecutor(),
      tool: new ToolNodeExecutor(container.tools),
      model: new ModelNodeExecutor({
        gateway: container.gateway,
        router: container.router,
        decisions: container.decisions,
        estimatePromptTokens,
        onSettled: async (requestId, request, response) => {
          await settle(container, requestId, {
            request: request as UnifiedChatRequest,
            response,
          });
        },
      }),
    },
    store: container.runs,
  });
  container.attachRunner(runner);

  app.post("/v1/runs", async (request: FastifyRequest, reply: FastifyReply) => {
    const body = request.body as { workflow?: unknown; input?: Record<string, unknown> };

    const workflow = WorkflowDefinitionSchema.safeParse(body.workflow);
    if (!workflow.success) {
      // Validating the whole graph up front turns a run that would die halfway through into a
      // definition error the caller can fix before spending anything.
      return reply
        .status(400)
        .send({ error: { message: workflow.error.message, type: "invalid_request" } });
    }

    try {
      const state = await runner.start(workflow.data, {
        tenantId: container.config.defaultTenantId,
        input: body.input ?? {},
      });
      return reply.send(state);
    } catch (raw) {
      const error = toOrchestratorError(raw);
      return reply.status(statusFor(error)).send({
        error: { message: error.message, type: error.errorClass, retryable: error.retryable },
      });
    }
  });

  app.post("/v1/runs/:runId/resume", async (request: FastifyRequest, reply: FastifyReply) => {
    const { runId } = request.params as { runId: string };
    const body = request.body as { workflow?: unknown; input?: Record<string, unknown> };

    const workflow = WorkflowDefinitionSchema.safeParse(body.workflow);
    if (!workflow.success) {
      return reply
        .status(400)
        .send({ error: { message: workflow.error.message, type: "invalid_request" } });
    }

    try {
      const state = await runner.resume(workflow.data, runId, body.input ?? {});
      return reply.send(state);
    } catch (raw) {
      const error = toOrchestratorError(raw);
      return reply.status(statusFor(error)).send({
        error: { message: error.message, type: error.errorClass, retryable: error.retryable },
      });
    }
  });

  app.get("/v1/runs/:runId", async (request: FastifyRequest, reply: FastifyReply) => {
    const { runId } = request.params as { runId: string };
    const state = runner.getState(runId);
    if (!state) {
      return reply.status(404).send({ error: { message: "Unknown run", type: "invalid_request" } });
    }
    // The event log is returned alongside derived state: debugging a workflow means seeing what
    // actually happened, not just where it ended up.
    return reply.send({ state, events: container.runs.events(runId) });
  });

  app.get("/v1/runs", async (request: FastifyRequest) => {
    const query = request.query as { status?: string; limit?: string };
    return {
      data: container.runs.list({
        tenantId: container.config.defaultTenantId,
        ...(query.status ? { status: query.status } : {}),
        limit: query.limit ? Number(query.limit) : 50,
      }),
    };
  });

  // --- catalog --------------------------------------------------------------

  app.get("/v1/catalog", async () => {
    const status = container.catalog.status();
    const snapshot = container.catalog.applied();

    return {
      status,
      // Every claim carries where it came from and when. "Why did the router start by preferring
      // this model?" needs an answer, and a benchmark that was retracted needs to be findable.
      priors: container.catalog.derivedPriors().map((prior) => ({
        modelId: prior.modelId,
        taskType: prior.taskType,
        routeMode: prior.routeMode,
        capability: Number(prior.capability.toFixed(4)),
        reward: Number(prior.reward.toFixed(4)),
        weight: prior.weight,
        source: prior.source,
      })),
      sources: snapshot
        ? [
            ...new Set(snapshot.entries.map((entry) => entry.provenance.source)),
            ...new Set(snapshot.scores.map((score) => score.provenance.source)),
          ]
        : [],
    };
  });

  // --- tools ----------------------------------------------------------------

  app.get("/v1/tools", async () => ({
    // Scoped to what this tenant's policy actually permits — listing tools it cannot call would
    // just invite confusing failures.
    data: container.tools.toolsFor(container.config.defaultTenantId),
  }));

  app.get("/v1/tools/audit", async (request: FastifyRequest) => {
    const query = request.query as { tool?: string; limit?: string };
    return {
      data: container.toolAudit.list({
        tenantId: container.config.defaultTenantId,
        ...(query.tool ? { tool: query.tool } : {}),
        limit: query.limit ? Number(query.limit) : 100,
      }),
    };
  });

  // --- memory ---------------------------------------------------------------

  app.post("/v1/memory/facts", async (request: FastifyRequest, reply: FastifyReply) => {
    const parsed = z
      .object({ text: z.string().min(1), metadata: z.record(z.string(), z.unknown()).optional() })
      .safeParse(request.body);

    if (!parsed.success) {
      return reply
        .status(400)
        .send({ error: { message: parsed.error.message, type: "invalid_request" } });
    }

    const [item] = await container.memory.rememberFact({
      tenantId: container.config.defaultTenantId,
      text: parsed.data.text,
      ...(parsed.data.metadata ? { metadata: parsed.data.metadata } : {}),
    });
    return reply.send(item);
  });

  app.post("/v1/memory/recall", async (request: FastifyRequest, reply: FastifyReply) => {
    const parsed = z
      .object({ sessionId: z.string().min(1), query: z.string().min(1) })
      .safeParse(request.body);

    if (!parsed.success) {
      return reply
        .status(400)
        .send({ error: { message: parsed.error.message, type: "invalid_request" } });
    }

    // Exposed so callers can inspect what memory *would* inject before trusting it with a prompt.
    return reply.send(
      await container.memory.recall({
        tenantId: container.config.defaultTenantId,
        sessionId: parsed.data.sessionId,
        query: parsed.data.query,
      }),
    );
  });

  app.delete("/v1/memory/sessions/:sessionId", async (request: FastifyRequest) => {
    const { sessionId } = request.params as { sessionId: string };
    return {
      forgotten: container.memory.forget({
        tenantId: container.config.defaultTenantId,
        sessionId,
      }),
    };
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
 *
 * Inline scorers run here — they are free and deterministic. When a successful response is available
 * they get to grade it; failed attempts skip straight to the reward function, which zeroes them.
 */
async function settle(
  container: Container,
  requestId: string,
  graded?: { request: UnifiedChatRequest; response: UnifiedChatResponse },
): Promise<void> {
  for (const event of container.events.query({ requestId })) {
    let quality: number | null | undefined;
    let provenance: { source: string; confidence: number } | undefined;

    // Only the attempt that actually produced the returned response can be graded against it.
    const isGradedAttempt =
      graded !== undefined &&
      event.status === "success" &&
      event.modelId === graded.response.modelId;

    if (isGradedAttempt) {
      const assessment = await container.quality.assessInline({
        request: graded.request,
        response: graded.response,
        event,
      });
      if (assessment) {
        quality = assessment.score;
        provenance = { source: assessment.source, confidence: assessment.confidence };
      }
    }

    const reward = container.rewards.settle(event, quality, provenance);
    container.router.observe({
      decisionId: event.routingDecisionId ?? undefined,
      modelId: event.modelId,
      features: event.features,
      taskType: event.taskType,
      reward,
      // Lets the router notice task types that only ever get the heuristic floor, and stop steering
      // them. Omitting this would leave it unable to tell an informative reward from a constant.
      qualityConfidence: provenance?.confidence,
    });
  }
}

/**
 * Runs deferred scorers (today: the sampled LLM judge) and corrects what the router already learned.
 *
 * This is a revision, not a second observation — `reviseOutcome` applies the delta so the arm ends up
 * where it would have been had the better score arrived first. Re-teaching instead would double-count.
 */
async function scoreDeferred(
  container: Container,
  app: FastifyInstance,
  request: UnifiedChatRequest,
  response: UnifiedChatResponse,
): Promise<void> {
  if (container.quality.deferredScorers.length === 0) return;

  const events = container.events.query({ requestId: response.requestId });
  const event = events.find(
    (candidate) => candidate.status === "success" && candidate.modelId === response.modelId,
  );
  if (!event) return;

  const assessment = await container.quality.assessDeferred({ request, response, event });
  if (!assessment) return;
  // A weaker signal must not overwrite a stronger one already recorded.
  if (!supersedes(assessment, event.qualityConfidence)) return;

  const { previousReward, reward } = container.rewards.rescore(event, assessment.score, {
    source: assessment.source,
    confidence: assessment.confidence,
  });

  if (event.routingDecisionId) {
    container.router.reviseOutcome(event.routingDecisionId, reward);
  }

  app.log.debug(
    { modelId: event.modelId, source: assessment.source, previousReward, reward },
    "quality revised by deferred scorer",
  );
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
