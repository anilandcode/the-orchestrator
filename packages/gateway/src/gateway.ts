import {
  type CallEvent,
  CallEventSchema,
  type CallEventSink,
  type Clock,
  type FinishReason,
  type IdGenerator,
  type ModelRegistry,
  type ModelSpec,
  NOOP_CALL_EVENT_SINK,
  OrchestratorError,
  type UnifiedChatChunk,
  type UnifiedChatRequest,
  type UnifiedChatResponse,
  type Usage,
  computeCostUsd,
  defaultRegistry,
  systemClock,
  systemIds,
  toOrchestratorError,
} from "@orchestrator/shared";
import type { ProviderAdapter } from "./provider-adapter.js";

/**
 * What the gateway is told to execute. Deliberately a plain data structure rather than a router
 * reference — the gateway must never be able to ask "which model should I use?".
 */
export interface ExecutionPlan {
  modelId: string;
  /** Tried in order when a failure is fallback-eligible. */
  fallbacks?: string[];
  /** Links emitted events back to the routing decision, so reward reaches the right arm. */
  decisionId?: string | null;
  /** The feature vector the router saw, stored for counterfactual replay. */
  features?: number[];
}

export interface RetryPolicy {
  /** Total attempts against a single model, including the first. 1 disables retry. */
  maxAttemptsPerModel: number;
  baseDelayMs: number;
  maxDelayMs: number;
}

export const DEFAULT_RETRY: RetryPolicy = {
  maxAttemptsPerModel: 3,
  baseDelayMs: 250,
  maxDelayMs: 8_000,
};

export interface GatewayConfig {
  adapters: ProviderAdapter[];
  registry?: ModelRegistry;
  sink?: CallEventSink;
  clock?: Clock;
  ids?: IdGenerator;
  retry?: Partial<RetryPolicy>;
  timeoutMs?: number;
  /** Injected so retry backoff costs no wall-clock time in tests. */
  sleep?: (ms: number) => Promise<void>;
  /** Injected so jitter is deterministic in tests. */
  random?: () => number;
}

export class Gateway {
  private readonly adapters = new Map<string, ProviderAdapter>();
  private readonly registry: ModelRegistry;
  private readonly sink: CallEventSink;
  private readonly clock: Clock;
  private readonly ids: IdGenerator;
  private readonly retry: RetryPolicy;
  private readonly timeoutMs: number;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly random: () => number;

  constructor(config: GatewayConfig) {
    for (const adapter of config.adapters) this.adapters.set(adapter.provider, adapter);
    this.registry = config.registry ?? defaultRegistry;
    this.sink = config.sink ?? NOOP_CALL_EVENT_SINK;
    this.clock = config.clock ?? systemClock;
    this.ids = config.ids ?? systemIds;
    this.retry = { ...DEFAULT_RETRY, ...config.retry };
    this.timeoutMs = config.timeoutMs ?? 60_000;
    this.sleep = config.sleep ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    this.random = config.random ?? Math.random;
  }

  /** Which models this gateway can actually reach, given the adapters that were configured. */
  availableModels(): ModelSpec[] {
    return this.registry.list().filter((spec) => this.adapters.has(spec.provider));
  }

  async chat(req: UnifiedChatRequest, plan: ExecutionPlan): Promise<UnifiedChatResponse> {
    const requestId = req.requestId ?? this.ids.generate("req");
    const chain = buildChain(plan);

    let attempt = 0;
    let lastError: OrchestratorError | undefined;

    for (const modelId of chain) {
      const resolved = this.resolve(modelId);
      if ("error" in resolved) {
        // A model we cannot reach is not a call — no event is written for it. Move down the chain.
        lastError = resolved.error;
        continue;
      }
      const { spec, adapter } = resolved;

      for (let tryIndex = 0; tryIndex < this.retry.maxAttemptsPerModel; tryIndex++) {
        attempt++;
        const startedAt = this.clock.monotonic();

        try {
          const result = await adapter.chat(req, spec, AbortSignal.timeout(this.timeoutMs));
          const latencyMs = this.clock.monotonic() - startedAt;
          const costUsd = computeCostUsd(spec, result.usage);

          this.emit({
            req,
            plan,
            requestId,
            spec,
            attempt,
            usage: result.usage,
            costUsd,
            latencyMs,
            status: "success",
            finishReason: result.finishReason,
          });

          return {
            id: this.ids.generate("resp"),
            requestId,
            provider: spec.provider,
            modelId: spec.modelId,
            message: result.message,
            finishReason: result.finishReason,
            usage: result.usage,
            costUsd,
            latencyMs,
            attempts: attempt,
            ...(plan.decisionId ? { routingDecisionId: plan.decisionId } : {}),
          };
        } catch (raw) {
          const err = toOrchestratorError(raw, "unknown", {
            provider: spec.provider,
            modelId: spec.modelId,
          });
          lastError = err;

          const latencyMs = this.clock.monotonic() - startedAt;
          // A failed attempt is still an attempt. Recording it is what stops the bandit from
          // crediting a fallback model's success to the model that actually broke.
          this.emit({
            req,
            plan,
            requestId,
            spec,
            attempt,
            usage: EMPTY_USAGE,
            costUsd: 0,
            latencyMs,
            status: "error",
            errorClass: err.errorClass,
          });

          const retriesLeft = tryIndex < this.retry.maxAttemptsPerModel - 1;
          if (err.retryable && retriesLeft) {
            await this.sleep(this.backoffMs(tryIndex, err.retryAfterMs));
            continue;
          }

          if (err.fallbackEligible) break; // next model in the chain
          throw err; // terminal: retrying anywhere would just repeat the same failure
        }
      }
    }

    throw (
      lastError ?? new OrchestratorError("unknown", `No usable model in chain: ${chain.join(", ")}`)
    );
  }

  /**
   * Streaming variant.
   *
   * Fallback only covers failures raised *before* the first chunk is yielded. Once bytes have reached
   * the caller, silently restarting on another model would duplicate output, so a mid-stream failure
   * propagates instead.
   */
  async *stream(
    req: UnifiedChatRequest,
    plan: ExecutionPlan,
  ): AsyncGenerator<UnifiedChatChunk, void, undefined> {
    const requestId = req.requestId ?? this.ids.generate("req");
    const chain = buildChain(plan);

    let attempt = 0;
    let lastError: OrchestratorError | undefined;

    for (const modelId of chain) {
      const resolved = this.resolve(modelId);
      if ("error" in resolved) {
        lastError = resolved.error;
        continue;
      }
      const { spec, adapter } = resolved;

      attempt++;
      const startedAt = this.clock.monotonic();
      let ttftMs: number | undefined;
      let started = false;
      let usage: Usage = EMPTY_USAGE;
      let finishReason: FinishReason = "stop";

      try {
        for await (const chunk of adapter.stream(req, spec, AbortSignal.timeout(this.timeoutMs))) {
          if (!started) {
            started = true;
            ttftMs = this.clock.monotonic() - startedAt;
          }
          if (chunk.type === "finish") {
            finishReason = chunk.finishReason;
            if (chunk.usage) usage = chunk.usage;
          }
          yield chunk;
        }

        this.emit({
          req,
          plan,
          requestId,
          spec,
          attempt,
          usage,
          costUsd: computeCostUsd(spec, usage),
          latencyMs: this.clock.monotonic() - startedAt,
          ...(ttftMs !== undefined ? { ttftMs } : {}),
          status: "success",
          finishReason,
        });
        return;
      } catch (raw) {
        const err = toOrchestratorError(raw, "unknown", {
          provider: spec.provider,
          modelId: spec.modelId,
        });
        lastError = err;

        this.emit({
          req,
          plan,
          requestId,
          spec,
          attempt,
          usage,
          costUsd: computeCostUsd(spec, usage),
          latencyMs: this.clock.monotonic() - startedAt,
          status: "error",
          errorClass: err.errorClass,
        });

        if (started) throw err; // caller already has partial output; do not restart
        if (err.fallbackEligible) continue;
        throw err;
      }
    }

    throw (
      lastError ?? new OrchestratorError("unknown", `No usable model in chain: ${chain.join(", ")}`)
    );
  }

  // --- internals ------------------------------------------------------------

  private resolve(
    modelId: string,
  ): { spec: ModelSpec; adapter: ProviderAdapter } | { error: OrchestratorError } {
    const spec = this.registry.get(modelId);
    if (!spec) {
      return {
        error: new OrchestratorError("invalid_request", `Unknown model: ${modelId}`, { modelId }),
      };
    }

    const adapter = this.adapters.get(spec.provider);
    if (!adapter) {
      return {
        error: new OrchestratorError(
          "auth",
          `No adapter configured for provider: ${spec.provider}`,
          { modelId, provider: spec.provider },
        ),
      };
    }

    return { spec, adapter };
  }

  /** Exponential backoff with full jitter, floored by a provider-supplied Retry-After. */
  private backoffMs(tryIndex: number, retryAfterMs: number | undefined): number {
    const exponential = Math.min(this.retry.baseDelayMs * 2 ** tryIndex, this.retry.maxDelayMs);
    const jittered = exponential * this.random();
    return Math.max(jittered, retryAfterMs ?? 0);
  }

  private emit(args: {
    req: UnifiedChatRequest;
    plan: ExecutionPlan;
    requestId: string;
    spec: ModelSpec;
    attempt: number;
    usage: Usage;
    costUsd: number;
    latencyMs: number;
    ttftMs?: number;
    status: CallEvent["status"];
    errorClass?: CallEvent["errorClass"];
    finishReason?: CallEvent["finishReason"];
  }): void {
    const event = CallEventSchema.parse({
      id: this.ids.generate("evt"),
      tenantId: args.req.tenantId,
      requestId: args.requestId,
      routingDecisionId: args.plan.decisionId ?? null,
      attempt: args.attempt,
      provider: args.spec.provider,
      modelId: args.spec.modelId,
      taskType: args.req.route.taskType,
      routeMode: args.req.route.mode,
      features: args.plan.features ?? [],
      promptTokens: args.usage.promptTokens,
      completionTokens: args.usage.completionTokens,
      cachedPromptTokens: args.usage.cachedPromptTokens,
      costUsd: args.costUsd,
      latencyMs: args.latencyMs,
      ttftMs: args.ttftMs ?? null,
      status: args.status,
      errorClass: args.errorClass ?? null,
      finishReason: args.finishReason ?? null,
      createdAt: this.clock.now(),
    });

    this.sink.record(event);
  }
}

const EMPTY_USAGE: Usage = {
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  cachedPromptTokens: 0,
};

/** Primary first, then fallbacks, with duplicates removed so a model is never tried twice in a row. */
function buildChain(plan: ExecutionPlan): string[] {
  return [...new Set([plan.modelId, ...(plan.fallbacks ?? [])])];
}
