import { CatalogService, SqliteCatalogStore, registerFromCatalog } from "@orchestrator/catalog";
import {
  AnthropicAdapter,
  Gateway,
  OpenAIAdapter,
  OpenRouterAdapter,
  type ProviderAdapter,
} from "@orchestrator/gateway";
import { SqliteToolAuditLog, ToolRegistry } from "@orchestrator/mcp";
import {
  HashingEmbedder,
  MemoryService,
  OpenAiEmbedder,
  SqliteMemoryStore,
} from "@orchestrator/memory";
import {
  CheckpointExecutor,
  type GraphRunner,
  SqliteRunStore,
  TransformExecutor,
} from "@orchestrator/orchestrator";
import {
  CodeStructureScorer,
  FinishReasonScorer,
  JsonSchemaScorer,
  LlmJudgeScorer,
  QualityPipeline,
  type QualityScorer,
  ToolCallScorer,
} from "@orchestrator/quality";
import {
  AdaptiveRouter,
  FEATURE_DIMENSION,
  LinUcbBandit,
  SqliteStateStore,
  StaticRouter,
  extractFeatures,
} from "@orchestrator/router";
import { ModelRegistry, UnifiedChatRequestSchema } from "@orchestrator/shared";
import {
  RewardService,
  RollingNormalizer,
  SqliteCallEventRepository,
  SqliteRoutingDecisionRepository,
  openDatabase,
} from "@orchestrator/telemetry";
import type { ApiConfig } from "./config.js";

/**
 * The one place the layers are wired together.
 *
 * Every dependency points one way: shared <- gateway <- telemetry <- router <- here. The gateway is
 * handed a `CallEventSink`, never the telemetry package; the router is never handed the gateway.
 */
export interface Container {
  config: ApiConfig;
  registry: ModelRegistry;
  gateway: Gateway;
  router: AdaptiveRouter;
  events: SqliteCallEventRepository;
  decisions: SqliteRoutingDecisionRepository;
  rewards: RewardService;
  quality: QualityPipeline;
  runs: SqliteRunStore;
  memory: MemoryService;
  tools: ToolRegistry;
  catalog: CatalogService;
  toolAudit: SqliteToolAuditLog;
  /** Assigned after construction: the model executor needs the settle callback the server owns. */
  runner: GraphRunner | undefined;
  attachRunner(runner: GraphRunner): void;
  close(): void;
}

export interface ContainerOverrides {
  /** Injected by tests so the API can be exercised end-to-end without touching the network. */
  adapters?: ProviderAdapter[];
}

export function buildContainer(config: ApiConfig, overrides: ContainerOverrides = {}): Container {
  const db = openDatabase(config.dbPath);
  const events = new SqliteCallEventRepository(db);
  const decisions = new SqliteRoutingDecisionRepository(db);

  // Warm the reward normalizer from history so a restart does not reset the cost/latency scales
  // and silently re-score traffic against different percentiles.
  const normalizer = new RollingNormalizer();
  normalizer.observeAll(events.query({ limit: 5_000 }));
  const rewards = new RewardService(events, normalizer);

  const registry = new ModelRegistry();

  // Ingested pricing and context windows replace the hand-written defaults where a reviewed snapshot
  // exists. Applying only what was already promoted keeps boot from re-litigating a decision that
  // was made once, deliberately, at `catalog:refresh --apply` time.
  const catalog = new CatalogService({ store: new SqliteCatalogStore(db), registry });
  catalog.applyStored();

  const adapters: ProviderAdapter[] = overrides.adapters ?? [];
  if (!overrides.adapters) {
    if (config.openaiApiKey) adapters.push(new OpenAIAdapter({ apiKey: config.openaiApiKey }));
    if (config.anthropicApiKey) {
      adapters.push(new AnthropicAdapter({ apiKey: config.anthropicApiKey }));
    }
    if (config.openrouterApiKey) {
      adapters.push(
        new OpenRouterAdapter({
          apiKey: config.openrouterApiKey,
          ...(config.openrouterAppUrl ? { appUrl: config.openrouterAppUrl } : {}),
          ...(config.openrouterAppName ? { appName: config.openrouterAppName } : {}),
        }),
      );

      // Catalog knowledge becomes callable only for allowlisted ids. Registering the whole catalog
      // would hand the bandit ~300 arms against ~2% of headroom, which costs more in exploration
      // than perfect routing could return.
      if (config.openrouterModels.length > 0) {
        const snapshot = catalog.applied();
        if (snapshot) {
          registerFromCatalog(registry, snapshot.entries, {
            allow: config.openrouterModels,
            provider: "openrouter",
          });
        }
      }
    }
  }

  const gateway = new Gateway({ adapters, registry, sink: events });

  // Order is irrelevant — the pipeline picks by confidence, not by position — but the floor is
  // listed last as a reminder that it is the fallback, not a peer of the real validators.
  const scorers: QualityScorer[] = [
    new ToolCallScorer(),
    new JsonSchemaScorer(),
    new CodeStructureScorer(),
    new FinishReasonScorer(),
  ];

  if (config.judgeEnabled) {
    scorers.push(
      new LlmJudgeScorer({
        gateway,
        modelId: config.judgeModel,
        sampleRate: config.judgeSampleRate,
        maxUsdPerHour: config.judgeMaxUsdPerHour,
      }),
    );
  }

  const quality = new QualityPipeline(scorers);

  const baseline = new StaticRouter();
  const router = new AdaptiveRouter({
    bandit: new LinUcbBandit({ dimension: FEATURE_DIMENSION, alpha: config.linucbAlpha }),
    baseline,
    mode: config.routerMode,
    coldStartPulls: config.coldStartPulls,
    stateStore: new SqliteStateStore(db),
  });

  // Priors are seeded only when explicitly enabled. The mechanism is proven; the shipped benchmark
  // data is not, and a confident wrong prior is worse than no prior at all.
  if (config.catalogPriorsEnabled) {
    const priors = catalog.routerPriors((taskType, routeMode) =>
      extractFeatures({
        request: UnifiedChatRequestSchema.parse({
          tenantId: config.defaultTenantId,
          messages: [{ role: "user", content: "" }],
          route: { mode: routeMode, taskType },
        }),
        available: gateway.availableModels(),
        estimatedPromptTokens: 1_500,
      }),
    );
    router.applyPriors(priors);
  }

  const runs = new SqliteRunStore(db);

  // Falls back to the offline embedder when no embedding model is configured, so memory works with
  // no provider account at all — lexically rather than semantically, which embedder.ts is explicit
  // about.
  const embedder =
    config.embeddingModel && config.openaiApiKey
      ? new OpenAiEmbedder({ apiKey: config.openaiApiKey, model: config.embeddingModel })
      : new HashingEmbedder();

  // Deny by default: a tenant reaches no tool until a policy grants it. MCP servers are registered
  // at runtime via /v1/tools/servers rather than baked in here.
  const toolAudit = new SqliteToolAuditLog(db);
  const tools = new ToolRegistry({ audit: toolAudit, policies: config.toolPolicies });

  const memory = new MemoryService({
    store: new SqliteMemoryStore(db),
    embedder,
    ttlMs: config.memoryTtlMs,
  });
  let closed = false;
  let runner: GraphRunner | undefined;

  return {
    config,
    registry,
    gateway,
    router,
    events,
    decisions,
    rewards,
    quality,
    runs,
    memory,
    tools,
    toolAudit,
    catalog,
    get runner() {
      return runner;
    },
    attachRunner(next: GraphRunner) {
      runner = next;
    },
    close() {
      // Idempotent: SIGINT and SIGTERM can both fire during a shutdown, and a second close must not
      // throw on an already-closed connection and mask the real reason we are shutting down.
      if (closed) return;
      closed = true;
      // Flush the last learning window; anything unpersisted is learning thrown away.
      router.persist();
      db.close();
    },
  };
}
