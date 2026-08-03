import { type CallEvent, CallEventSchema } from "@orchestrator/shared";
import { CALL_EVENT_MIGRATIONS } from "../migrations/call-events.js";
import { QUALITY_PROVENANCE_MIGRATIONS } from "../migrations/quality-provenance.js";
import { type Db, openDatabase, runMigrations } from "./database.js";
import type { CallEventQuery, CallEventRepository, QualityProvenance } from "./repository.js";

interface CallEventRow {
  id: string;
  tenant_id: string;
  request_id: string;
  routing_decision_id: string | null;
  attempt: number;
  provider: string;
  model_id: string;
  task_type: string;
  route_mode: string;
  features: string;
  prompt_tokens: number;
  completion_tokens: number;
  cached_prompt_tokens: number;
  cost_usd: number;
  latency_ms: number;
  ttft_ms: number | null;
  status: string;
  error_class: string | null;
  finish_reason: string | null;
  quality_score: number | null;
  quality_source: string | null;
  quality_confidence: number | null;
  quality_revisions: number;
  is_judge: number;
  reward: number | null;
  created_at: number;
}

const INSERT_SQL = `
  INSERT OR REPLACE INTO call_events (
    id, tenant_id, request_id, routing_decision_id, attempt,
    provider, model_id, task_type, route_mode, features,
    prompt_tokens, completion_tokens, cached_prompt_tokens, cost_usd, latency_ms, ttft_ms,
    status, error_class, finish_reason, quality_score, reward, created_at,
    quality_source, quality_confidence, quality_revisions, is_judge
  ) VALUES (
    @id, @tenant_id, @request_id, @routing_decision_id, @attempt,
    @provider, @model_id, @task_type, @route_mode, @features,
    @prompt_tokens, @completion_tokens, @cached_prompt_tokens, @cost_usd, @latency_ms, @ttft_ms,
    @status, @error_class, @finish_reason, @quality_score, @reward, @created_at,
    @quality_source, @quality_confidence, @quality_revisions, @is_judge
  )
`;

export class SqliteCallEventRepository implements CallEventRepository {
  private readonly db: Db;
  private readonly ownsConnection: boolean;

  constructor(source: string | Db) {
    if (typeof source === "string") {
      this.db = openDatabase(source);
      this.ownsConnection = true;
    } else {
      this.db = source;
      this.ownsConnection = false;
    }
    runMigrations(this.db, [...CALL_EVENT_MIGRATIONS, ...QUALITY_PROVENANCE_MIGRATIONS]);
  }

  get connection(): Db {
    return this.db;
  }

  record(event: CallEvent): void {
    this.db.prepare(INSERT_SQL).run(toRow(event));
  }

  recordMany(events: CallEvent[]): void {
    const statement = this.db.prepare(INSERT_SQL);
    const insertAll = this.db.transaction((batch: CallEvent[]) => {
      for (const event of batch) statement.run(toRow(event));
    });
    insertAll(events);
  }

  scoreEvent(
    id: string,
    qualityScore: number | null,
    reward: number,
    provenance?: QualityProvenance,
  ): void {
    // The revision counter increments in SQL rather than read-modify-write, so concurrent scorers
    // settling the same event cannot lose a count between the read and the write.
    this.db
      .prepare(
        `UPDATE call_events
            SET quality_score = ?,
                reward = ?,
                quality_source = COALESCE(?, quality_source),
                quality_confidence = COALESCE(?, quality_confidence),
                quality_revisions = quality_revisions + ?
          WHERE id = ?`,
      )
      .run(
        qualityScore,
        reward,
        provenance?.source ?? null,
        provenance?.confidence ?? null,
        provenance?.isRevision ? 1 : 0,
        id,
      );
  }

  query(query: CallEventQuery = {}): CallEvent[] {
    const { sql, params } = buildWhere(query);
    const limit = query.limit ? " LIMIT ?" : "";
    if (query.limit) params.push(query.limit);

    const rows = this.db
      .prepare(`SELECT * FROM call_events${sql} ORDER BY created_at ASC, rowid ASC${limit}`)
      .all(...params) as CallEventRow[];

    return rows.map(fromRow);
  }

  count(query: CallEventQuery = {}): number {
    const { sql, params } = buildWhere(query);
    const row = this.db.prepare(`SELECT COUNT(*) AS n FROM call_events${sql}`).get(...params) as {
      n: number;
    };
    return row.n;
  }

  close(): void {
    if (this.ownsConnection) this.db.close();
  }
}

function buildWhere(query: CallEventQuery): { sql: string; params: unknown[] } {
  const clauses: string[] = [];
  const params: unknown[] = [];

  if (query.tenantId) {
    clauses.push("tenant_id = ?");
    params.push(query.tenantId);
  }
  if (query.requestId) {
    clauses.push("request_id = ?");
    params.push(query.requestId);
  }
  if (query.modelId) {
    clauses.push("model_id = ?");
    params.push(query.modelId);
  }
  if (query.taskType) {
    clauses.push("task_type = ?");
    params.push(query.taskType);
  }
  if (query.status) {
    clauses.push("status = ?");
    params.push(query.status);
  }
  if (query.since !== undefined) {
    clauses.push("created_at >= ?");
    params.push(query.since);
  }
  if (query.until !== undefined) {
    clauses.push("created_at < ?");
    params.push(query.until);
  }

  return { sql: clauses.length ? ` WHERE ${clauses.join(" AND ")}` : "", params };
}

function toRow(event: CallEvent): Record<string, unknown> {
  return {
    id: event.id,
    tenant_id: event.tenantId,
    request_id: event.requestId,
    routing_decision_id: event.routingDecisionId,
    attempt: event.attempt,
    provider: event.provider,
    model_id: event.modelId,
    task_type: event.taskType,
    route_mode: event.routeMode,
    features: JSON.stringify(event.features),
    prompt_tokens: event.promptTokens,
    completion_tokens: event.completionTokens,
    cached_prompt_tokens: event.cachedPromptTokens,
    cost_usd: event.costUsd,
    latency_ms: event.latencyMs,
    ttft_ms: event.ttftMs,
    status: event.status,
    error_class: event.errorClass,
    finish_reason: event.finishReason,
    quality_score: event.qualityScore,
    quality_source: event.qualitySource,
    quality_confidence: event.qualityConfidence,
    quality_revisions: event.qualityRevisions,
    // SQLite has no boolean type; 0/1 is the storage convention.
    is_judge: event.isJudge ? 1 : 0,
    reward: event.reward,
    created_at: event.createdAt,
  };
}

function fromRow(row: CallEventRow): CallEvent {
  // Parsed rather than cast: a schema change that outruns a migration should fail loudly here, not
  // silently feed malformed rows into the router.
  return CallEventSchema.parse({
    id: row.id,
    tenantId: row.tenant_id,
    requestId: row.request_id,
    routingDecisionId: row.routing_decision_id,
    attempt: row.attempt,
    provider: row.provider,
    modelId: row.model_id,
    taskType: row.task_type,
    routeMode: row.route_mode,
    features: JSON.parse(row.features) as number[],
    promptTokens: row.prompt_tokens,
    completionTokens: row.completion_tokens,
    cachedPromptTokens: row.cached_prompt_tokens,
    costUsd: row.cost_usd,
    latencyMs: row.latency_ms,
    ttftMs: row.ttft_ms,
    status: row.status,
    errorClass: row.error_class,
    finishReason: row.finish_reason,
    qualityScore: row.quality_score,
    qualitySource: row.quality_source,
    qualityConfidence: row.quality_confidence,
    qualityRevisions: row.quality_revisions,
    isJudge: row.is_judge === 1,
    reward: row.reward,
    createdAt: row.created_at,
  });
}
