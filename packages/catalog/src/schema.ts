import { ProviderIdSchema, TaskTypeSchema } from "@orchestrator/shared";
import { z } from "zod";

/**
 * Provenance is not optional metadata here.
 *
 * Every number in this package came from somewhere else and was true on some date. Without that,
 * "why did the router start by preferring this model?" has no answer, and a benchmark that was
 * retracted or a price that changed six months ago goes on silently steering traffic.
 */
export const ProvenanceSchema = z.object({
  source: z.string().min(1),
  sourceUrl: z.string().optional(),
  /** Epoch ms when this was fetched or last reviewed. */
  asOf: z.number().int().nonnegative(),
});
export type Provenance = z.infer<typeof ProvenanceSchema>;

/** One model as an external catalog describes it. */
export const CatalogEntrySchema = z.object({
  /** Our internal id, e.g. `openai/gpt-4o-mini`. */
  modelId: z.string().min(1),
  provider: ProviderIdSchema.nullable().default(null),
  /** The id the source used, kept so a mapping can be re-checked later. */
  sourceModelId: z.string(),
  displayName: z.string().default(""),
  inputCostPerMTok: z.number().nonnegative().nullable().default(null),
  outputCostPerMTok: z.number().nonnegative().nullable().default(null),
  contextWindow: z.number().int().positive().nullable().default(null),
  maxOutputTokens: z.number().int().positive().nullable().default(null),
  supportsTools: z.boolean().nullable().default(null),
  supportsVision: z.boolean().nullable().default(null),
  provenance: ProvenanceSchema,
});
export type CatalogEntry = z.infer<typeof CatalogEntrySchema>;

/**
 * A published benchmark result.
 *
 * `score` is on the benchmark's own scale — Arena Elo near 1200, MMLU as a percentage. Normalization
 * happens at prior-derivation time against the rest of the catalog, not here, so the stored value
 * stays checkable against the published figure.
 */
export const BenchmarkScoreSchema = z.object({
  benchmarkId: z.string().min(1),
  modelId: z.string().min(1),
  score: z.number(),
  provenance: ProvenanceSchema,
});
export type BenchmarkScore = z.infer<typeof BenchmarkScoreSchema>;

export const BenchmarkDefinitionSchema = z.object({
  id: z.string().min(1),
  name: z.string().default(""),
  /** Higher is better for nearly all of these; stated rather than assumed. */
  higherIsBetter: z.boolean().default(true),
  /** What the benchmark actually measures — the basis for any task mapping. */
  measures: z.string().default(""),
});
export type BenchmarkDefinition = z.infer<typeof BenchmarkDefinitionSchema>;

export const BenchmarkFileSchema = z.object({
  /** Bumped when the curated file's shape changes. */
  version: z.number().int().positive().default(1),
  benchmarks: z.array(BenchmarkDefinitionSchema),
  scores: z.array(BenchmarkScoreSchema),
});
export type BenchmarkFile = z.infer<typeof BenchmarkFileSchema>;

/** A model's derived capability estimate for one task type, in [0,1]. */
export const TaskCapabilitySchema = z.object({
  modelId: z.string(),
  taskType: TaskTypeSchema,
  /** Normalized 0..1 across the catalog, not the benchmark's raw scale. */
  capability: z.number().min(0).max(1),
  /** Which benchmarks contributed, so a surprising prior can be traced back. */
  contributingBenchmarks: z.array(z.string()),
  /** Falls as benchmark coverage thins; drives prior weight. */
  coverage: z.number().min(0).max(1),
});
export type TaskCapability = z.infer<typeof TaskCapabilitySchema>;

/** A complete, dated snapshot. Refresh writes a new one rather than mutating the last. */
export const CatalogSnapshotSchema = z.object({
  version: z.number().int().positive(),
  createdAt: z.number().int().nonnegative(),
  entries: z.array(CatalogEntrySchema),
  scores: z.array(BenchmarkScoreSchema),
  benchmarks: z.array(BenchmarkDefinitionSchema),
});
export type CatalogSnapshot = z.infer<typeof CatalogSnapshotSchema>;

/** How stale a snapshot may get before it is worth warning about. */
export const STALE_AFTER_MS = 30 * 24 * 60 * 60 * 1_000;

export function isStale(snapshot: CatalogSnapshot, now = Date.now()): boolean {
  return now - snapshot.createdAt > STALE_AFTER_MS;
}
