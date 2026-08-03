import { type RoutingDecision, RoutingDecisionSchema } from "@orchestrator/shared";
import { ROUTING_DECISION_MIGRATIONS } from "../migrations/routing-decisions.js";
import { type Db, runMigrations } from "./database.js";

export interface RoutingDecisionQuery {
  since?: number;
  until?: number;
  /** Only decisions where the bandit disagreed with what actually executed. */
  disagreementsOnly?: boolean;
  limit?: number;
}

export interface RoutingDecisionRepository {
  record(decision: RoutingDecision): void;
  recordMany(decisions: RoutingDecision[]): void;
  get(decisionId: string): RoutingDecision | undefined;
  query(query?: RoutingDecisionQuery): RoutingDecision[];
  count(query?: RoutingDecisionQuery): number;
}

interface DecisionRow {
  decision_id: string;
  model_id: string;
  fallbacks: string;
  strategy: string;
  reason: string;
  shadow_model_id: string | null;
  features: string;
  task_type: string;
  route_mode: string;
  created_at: number;
}

const INSERT_SQL = `
  INSERT OR REPLACE INTO routing_decisions (
    decision_id, model_id, fallbacks, strategy, reason,
    shadow_model_id, features, task_type, route_mode, created_at
  ) VALUES (
    @decision_id, @model_id, @fallbacks, @strategy, @reason,
    @shadow_model_id, @features, @task_type, @route_mode, @created_at
  )
`;

export class SqliteRoutingDecisionRepository implements RoutingDecisionRepository {
  constructor(private readonly db: Db) {
    runMigrations(this.db, ROUTING_DECISION_MIGRATIONS);
  }

  record(decision: RoutingDecision): void {
    this.db.prepare(INSERT_SQL).run(toRow(decision));
  }

  recordMany(decisions: RoutingDecision[]): void {
    const statement = this.db.prepare(INSERT_SQL);
    const insertAll = this.db.transaction((batch: RoutingDecision[]) => {
      for (const decision of batch) statement.run(toRow(decision));
    });
    insertAll(decisions);
  }

  get(decisionId: string): RoutingDecision | undefined {
    const row = this.db
      .prepare("SELECT * FROM routing_decisions WHERE decision_id = ?")
      .get(decisionId) as DecisionRow | undefined;
    return row ? fromRow(row) : undefined;
  }

  query(query: RoutingDecisionQuery = {}): RoutingDecision[] {
    const { sql, params } = buildWhere(query);
    const limit = query.limit ? " LIMIT ?" : "";
    if (query.limit) params.push(query.limit);

    const rows = this.db
      .prepare(`SELECT * FROM routing_decisions${sql} ORDER BY created_at ASC${limit}`)
      .all(...params) as DecisionRow[];
    return rows.map(fromRow);
  }

  count(query: RoutingDecisionQuery = {}): number {
    const { sql, params } = buildWhere(query);
    const row = this.db
      .prepare(`SELECT COUNT(*) AS n FROM routing_decisions${sql}`)
      .get(...params) as { n: number };
    return row.n;
  }
}

export class InMemoryRoutingDecisionRepository implements RoutingDecisionRepository {
  private readonly decisions = new Map<string, RoutingDecision>();

  record(decision: RoutingDecision): void {
    this.decisions.set(decision.decisionId, decision);
  }

  recordMany(decisions: RoutingDecision[]): void {
    for (const decision of decisions) this.record(decision);
  }

  get(decisionId: string): RoutingDecision | undefined {
    return this.decisions.get(decisionId);
  }

  query(query: RoutingDecisionQuery = {}): RoutingDecision[] {
    let matched = [...this.decisions.values()].sort((a, b) => a.createdAt - b.createdAt);

    if (query.since !== undefined)
      matched = matched.filter((d) => d.createdAt >= (query.since as number));
    if (query.until !== undefined)
      matched = matched.filter((d) => d.createdAt < (query.until as number));
    if (query.disagreementsOnly) {
      matched = matched.filter((d) => d.shadowModelId !== null && d.shadowModelId !== d.modelId);
    }

    return query.limit ? matched.slice(0, query.limit) : matched;
  }

  count(query: RoutingDecisionQuery = {}): number {
    return this.query(query).length;
  }
}

function buildWhere(query: RoutingDecisionQuery): { sql: string; params: unknown[] } {
  const clauses: string[] = [];
  const params: unknown[] = [];

  if (query.since !== undefined) {
    clauses.push("created_at >= ?");
    params.push(query.since);
  }
  if (query.until !== undefined) {
    clauses.push("created_at < ?");
    params.push(query.until);
  }
  if (query.disagreementsOnly) {
    clauses.push("shadow_model_id IS NOT NULL AND shadow_model_id != model_id");
  }

  return { sql: clauses.length ? ` WHERE ${clauses.join(" AND ")}` : "", params };
}

function toRow(decision: RoutingDecision): Record<string, unknown> {
  return {
    decision_id: decision.decisionId,
    model_id: decision.modelId,
    fallbacks: JSON.stringify(decision.fallbacks),
    strategy: decision.strategy,
    reason: decision.reason,
    shadow_model_id: decision.shadowModelId,
    features: JSON.stringify(decision.features),
    task_type: decision.taskType,
    route_mode: decision.routeMode,
    created_at: decision.createdAt,
  };
}

function fromRow(row: DecisionRow): RoutingDecision {
  return RoutingDecisionSchema.parse({
    decisionId: row.decision_id,
    modelId: row.model_id,
    fallbacks: JSON.parse(row.fallbacks) as string[],
    strategy: row.strategy,
    reason: row.reason,
    shadowModelId: row.shadow_model_id,
    features: JSON.parse(row.features) as number[],
    taskType: row.task_type,
    routeMode: row.route_mode,
    createdAt: row.created_at,
  });
}
