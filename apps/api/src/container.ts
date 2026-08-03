import {
  AnthropicAdapter,
  Gateway,
  OpenAIAdapter,
  type ProviderAdapter,
} from "@orchestrator/gateway";
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
} from "@orchestrator/router";
import { ModelRegistry } from "@orchestrator/shared";
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

  const adapters: ProviderAdapter[] = overrides.adapters ?? [];
  if (!overrides.adapters) {
    if (config.openaiApiKey) adapters.push(new OpenAIAdapter({ apiKey: config.openaiApiKey }));
    if (config.anthropicApiKey) {
      adapters.push(new AnthropicAdapter({ apiKey: config.anthropicApiKey }));
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

  let closed = false;

  return {
    config,
    registry,
    gateway,
    router,
    events,
    decisions,
    rewards,
    quality,
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
