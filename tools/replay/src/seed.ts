import {
  AdaptiveRouter,
  FEATURE_DIMENSION,
  LinUcbBandit,
  StaticRouter,
  selectCandidates,
} from "@orchestrator/router";
import {
  CallEventSchema,
  type RouteMode,
  TASK_TYPES,
  type TaskType,
  UnifiedChatRequestSchema,
  systemIds,
} from "@orchestrator/shared";
import {
  RewardService,
  SqliteCallEventRepository,
  SqliteRoutingDecisionRepository,
  openDatabase,
} from "@orchestrator/telemetry";
import { seededRandom, simulateCall, simulationModels } from "./world.js";

/**
 * Development utility: fills a database with simulated **shadow-mode** traffic.
 *
 * This exists so `pnpm replay` can be exercised — and its report reviewed — before any real traffic
 * exists. The rows are synthetic. Never draw a production conclusion from a seeded database; the
 * whole point of `replay.ts` is to analyse traffic this script did not invent.
 */

const DB_PATH = process.env.ORCHESTRATOR_DB_PATH ?? "./data/orchestrator.sqlite";
const ROUNDS = Number(process.env.SEED_ROUNDS ?? 1_500);
const ROUTE_MODES: RouteMode[] = ["cheap", "balanced", "best"];

function main(): void {
  const db = openDatabase(DB_PATH);
  const events = new SqliteCallEventRepository(db);
  const decisions = new SqliteRoutingDecisionRepository(db);
  const rewards = new RewardService(events);

  const router = new AdaptiveRouter({
    bandit: new LinUcbBandit({ dimension: FEATURE_DIMENSION, alpha: 0.5 }),
    baseline: new StaticRouter(),
    // Shadow mode is what production ships with, so that is what the seeded log should look like.
    mode: "shadow",
    coldStartPulls: 20,
  });

  const available = simulationModels();
  let written = 0;

  for (let round = 0; round < ROUNDS; round++) {
    const random = seededRandom(round * 13 + 5);
    const taskType = TASK_TYPES[Math.floor(random() * TASK_TYPES.length)] as TaskType;
    const routeMode = ROUTE_MODES[Math.floor(random() * ROUTE_MODES.length)] as RouteMode;
    const promptTokens = Math.floor(200 + random() * 3_000);
    const completionTokens = Math.floor(50 + random() * 600);

    const request = UnifiedChatRequestSchema.parse({
      tenantId: "seed",
      requestId: systemIds.generate("req"),
      messages: [{ role: "user", content: "x" }],
      route: { mode: routeMode, taskType },
    });

    const context = { request, available, estimatedPromptTokens: promptTokens };
    const decision = router.select(context);
    decisions.record(decision);

    const spec = selectCandidates(context).find((c) => c.modelId === decision.modelId);
    if (!spec) continue;

    const simulated = simulateCall({
      spec,
      taskType,
      routeMode,
      promptTokens,
      completionTokens,
      features: decision.features,
      random,
    });

    const event = CallEventSchema.parse({
      ...simulated.event,
      id: systemIds.generate("evt"),
      tenantId: "seed",
      requestId: request.requestId,
      routingDecisionId: decision.decisionId,
      createdAt: Date.now() - (ROUNDS - round) * 1_000,
    });

    events.record(event);
    const reward = rewards.settle(event, event.qualityScore);
    router.observe({
      decisionId: decision.decisionId,
      modelId: event.modelId,
      features: decision.features,
      taskType,
      reward,
    });
    written += 1;
  }

  console.log(`Seeded ${written} call events and ${ROUNDS} routing decisions into ${DB_PATH}`);
  console.log(
    "These rows are SYNTHETIC — use them to exercise `pnpm replay`, not to draw conclusions.",
  );
  events.close();
}

main();
