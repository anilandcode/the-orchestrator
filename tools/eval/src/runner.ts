import type { TaskCapability } from "@orchestrator/catalog";
import type { Gateway } from "@orchestrator/gateway";
import {
  CONFIDENCE,
  CodeStructureScorer,
  FinishReasonScorer,
  JsonSchemaScorer,
  QualityPipeline,
  ToolCallScorer,
  responseText,
} from "@orchestrator/quality";
import {
  type CallEvent,
  CallEventSchema,
  type ModelSpec,
  type TaskType,
  UnifiedChatRequestSchema,
  systemIds,
} from "@orchestrator/shared";
import { type Fixture, gradeAgainstFixture } from "./fixtures.js";

/**
 * Run every fixture against every reachable model and measure what actually happens.
 *
 * This is tier-2 evidence, and it exists because tier-1 failed. Simulation showed that seeding the
 * router from public benchmark scores made routing measurably worse — the mechanism was sound, the
 * data described a different world. Measuring the models on tasks shaped like your own traffic is
 * the response.
 *
 * Grading combines two independent signals, taking the stricter of the two:
 *
 *   - The **fixture's own assertions** (`mustContain` / `mustNotContain`), which encode what a
 *     correct answer looks like for this specific task.
 *   - The **real quality validators** from `packages/quality`, so an answer is judged by the same
 *     machinery that will judge live traffic.
 *
 * Taking the stricter is deliberate: an answer that parses as valid JSON but contains the wrong
 * invoice number is not a good answer, and a scorer that only checked structure would call it one.
 */

export interface EvalResult {
  fixtureId: string;
  modelId: string;
  taskType: TaskType;
  score: number;
  costUsd: number;
  latencyMs: number;
  error: string | null;
}

export interface EvalRunOptions {
  gateway: Gateway;
  fixtures: Fixture[];
  models?: ModelSpec[];
  tenantId?: string;
  /** Reports progress; the real thing takes minutes and spends money. */
  onProgress?: (done: number, total: number, label: string) => void;
}

export async function runEval(options: EvalRunOptions): Promise<EvalResult[]> {
  const models = options.models ?? options.gateway.availableModels();
  const pipeline = new QualityPipeline([
    new ToolCallScorer(),
    new JsonSchemaScorer(),
    new CodeStructureScorer(),
    new FinishReasonScorer(),
  ]);

  const results: EvalResult[] = [];
  const total = models.length * options.fixtures.length;
  let done = 0;

  for (const spec of models) {
    for (const fixture of options.fixtures) {
      const request = UnifiedChatRequestSchema.parse({
        tenantId: options.tenantId ?? "eval",
        requestId: systemIds.generate("eval"),
        messages: [
          ...(fixture.system ? [{ role: "system" as const, content: fixture.system }] : []),
          { role: "user" as const, content: fixture.prompt },
        ],
        ...(fixture.tools ? { tools: fixture.tools } : {}),
        ...(fixture.maxTokens !== undefined ? { maxTokens: fixture.maxTokens } : {}),
        temperature: 0,
        route: {
          // Pinned: this measures the model, not the router.
          pin: spec.modelId,
          taskType: fixture.taskType,
          ...(fixture.outputSchema ? { outputSchema: fixture.outputSchema } : {}),
        },
      });

      done += 1;
      options.onProgress?.(done, total, `${spec.modelId} / ${fixture.id}`);

      try {
        const response = await options.gateway.chat(request, { modelId: spec.modelId });
        const text = responseText(response);

        const event: CallEvent = CallEventSchema.parse({
          id: "eval",
          tenantId: "eval",
          requestId: request.requestId,
          attempt: 1,
          provider: spec.provider,
          modelId: spec.modelId,
          taskType: fixture.taskType,
          routeMode: request.route.mode,
          promptTokens: response.usage.promptTokens,
          completionTokens: response.usage.completionTokens,
          costUsd: response.costUsd,
          latencyMs: response.latencyMs,
          status: "success",
          finishReason: response.finishReason,
          createdAt: Date.now(),
        });

        const assessment = await pipeline.assessInline({ request, response, event });
        const fixtureScore = gradeAgainstFixture(fixture, text);

        /*
         * Combine only *deterministic* evidence, and take the stricter of it.
         *
         * The heuristic floor is deliberately excluded. It returns 0.8 for any clean completion
         * because it knows almost nothing — that is why `packages/quality` gives it confidence 0.2.
         * Folding it into a `min` would treat it as authoritative and cap every verified answer at
         * 0.8, flattening the very ranking this harness exists to produce.
         *
         * Where both a validator and a fixture assertion apply, the stricter wins: structurally
         * valid JSON carrying the wrong invoice number is not a good answer.
         */
        const validatorScore =
          assessment && assessment.confidence > CONFIDENCE.heuristic ? assessment.score : undefined;

        const deterministic = [validatorScore, fixtureScore].filter(
          (value): value is number => value !== undefined,
        );

        results.push({
          fixtureId: fixture.id,
          modelId: spec.modelId,
          taskType: fixture.taskType,
          // With no deterministic evidence at all, the heuristic is the only thing left to say.
          score: deterministic.length > 0 ? Math.min(...deterministic) : (assessment?.score ?? 0),
          costUsd: response.costUsd,
          latencyMs: response.latencyMs,
          error: null,
        });
      } catch (error) {
        // A failure is a measurement, not a gap: a model that errors on this task is worse at it.
        results.push({
          fixtureId: fixture.id,
          modelId: spec.modelId,
          taskType: fixture.taskType,
          score: 0,
          costUsd: 0,
          latencyMs: 0,
          error: (error as Error).message,
        });
      }
    }
  }

  return results;
}

/**
 * Collapse results into per-(model, task) capabilities.
 *
 * Emitted in the same `TaskCapability` shape the catalog uses, so measured evidence flows through
 * the existing, already-tested `derivePriors` path rather than a parallel one.
 *
 * `coverage` here means fixture coverage rather than benchmark coverage, and it does the same job:
 * a capability measured on two fixtures deserves less weight than one measured on twenty.
 */
export function summarizeCapabilities(
  results: EvalResult[],
  options: { fixturesPerTask?: Map<TaskType, number>; confidentAt?: number } = {},
): TaskCapability[] {
  const confidentAt = options.confidentAt ?? 10;
  const grouped = new Map<string, EvalResult[]>();

  for (const result of results) {
    const key = `${result.modelId}|${result.taskType}`;
    const list = grouped.get(key) ?? [];
    list.push(result);
    grouped.set(key, list);
  }

  const capabilities: TaskCapability[] = [];

  for (const [key, group] of grouped) {
    const [modelId, taskType] = key.split("|") as [string, TaskType];
    const mean = group.reduce((total, result) => total + result.score, 0) / group.length;

    capabilities.push({
      modelId,
      taskType,
      capability: Math.min(1, Math.max(0, mean)),
      contributingBenchmarks: group.map((result) => result.fixtureId),
      // Ten fixtures is treated as full confidence. Below that the prior is proportionally weaker,
      // because three examples is an anecdote.
      coverage: Math.min(1, group.length / confidentAt),
    });
  }

  return capabilities;
}
