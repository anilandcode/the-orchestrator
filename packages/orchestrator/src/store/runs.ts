import { type Db, type Migration, runMigrations } from "@orchestrator/telemetry";
import { type RunEvent, RunEventSchema } from "../state.js";

export const RUN_MIGRATIONS: Migration[] = [
  {
    id: "004_workflow_runs",
    up: `
      CREATE TABLE IF NOT EXISTS workflow_runs (
        run_id      TEXT PRIMARY KEY,
        tenant_id   TEXT NOT NULL,
        workflow_id TEXT NOT NULL,
        status      TEXT NOT NULL,
        created_at  INTEGER NOT NULL,
        updated_at  INTEGER NOT NULL
      );

      -- Append-only. The run's state is a fold over these rows, so nothing here is ever updated.
      CREATE TABLE IF NOT EXISTS workflow_run_events (
        run_id   TEXT NOT NULL,
        seq      INTEGER NOT NULL,
        event    TEXT NOT NULL,
        PRIMARY KEY (run_id, seq)
      );

      CREATE INDEX IF NOT EXISTS idx_runs_tenant_status
        ON workflow_runs (tenant_id, status, updated_at);
    `,
  },
];

export interface RunRecord {
  runId: string;
  tenantId: string;
  workflowId: string;
  status: string;
  createdAt: number;
  updatedAt: number;
}

export interface RunStore {
  create(record: RunRecord): void;
  append(runId: string, events: RunEvent[]): void;
  events(runId: string): RunEvent[];
  get(runId: string): RunRecord | undefined;
  updateStatus(runId: string, status: string, updatedAt: number): void;
  list(query?: { tenantId?: string; status?: string; limit?: number }): RunRecord[];
}

export class SqliteRunStore implements RunStore {
  constructor(private readonly db: Db) {
    runMigrations(this.db, RUN_MIGRATIONS);
  }

  create(record: RunRecord): void {
    this.db
      .prepare(
        `INSERT OR REPLACE INTO workflow_runs
           (run_id, tenant_id, workflow_id, status, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        record.runId,
        record.tenantId,
        record.workflowId,
        record.status,
        record.createdAt,
        record.updatedAt,
      );
  }

  append(runId: string, events: RunEvent[]): void {
    if (events.length === 0) return;

    const next = this.db
      .prepare("SELECT COALESCE(MAX(seq), 0) AS max FROM workflow_run_events WHERE run_id = ?")
      .get(runId) as { max: number };

    const insert = this.db.prepare(
      "INSERT INTO workflow_run_events (run_id, seq, event) VALUES (?, ?, ?)",
    );

    // One transaction per append: a partially written step would make the fold produce a state that
    // never actually existed.
    const appendAll = this.db.transaction((batch: RunEvent[]) => {
      batch.forEach((event, index) => {
        insert.run(runId, next.max + index + 1, JSON.stringify(event));
      });
    });
    appendAll(events);
  }

  events(runId: string): RunEvent[] {
    const rows = this.db
      .prepare("SELECT event FROM workflow_run_events WHERE run_id = ? ORDER BY seq ASC")
      .all(runId) as { event: string }[];

    return rows.map((row) => RunEventSchema.parse(JSON.parse(row.event)));
  }

  get(runId: string): RunRecord | undefined {
    const row = this.db.prepare("SELECT * FROM workflow_runs WHERE run_id = ?").get(runId) as
      | {
          run_id: string;
          tenant_id: string;
          workflow_id: string;
          status: string;
          created_at: number;
          updated_at: number;
        }
      | undefined;

    return row
      ? {
          runId: row.run_id,
          tenantId: row.tenant_id,
          workflowId: row.workflow_id,
          status: row.status,
          createdAt: row.created_at,
          updatedAt: row.updated_at,
        }
      : undefined;
  }

  updateStatus(runId: string, status: string, updatedAt: number): void {
    this.db
      .prepare("UPDATE workflow_runs SET status = ?, updated_at = ? WHERE run_id = ?")
      .run(status, updatedAt, runId);
  }

  list(query: { tenantId?: string; status?: string; limit?: number } = {}): RunRecord[] {
    const clauses: string[] = [];
    const params: unknown[] = [];

    if (query.tenantId) {
      clauses.push("tenant_id = ?");
      params.push(query.tenantId);
    }
    if (query.status) {
      clauses.push("status = ?");
      params.push(query.status);
    }

    const where = clauses.length ? ` WHERE ${clauses.join(" AND ")}` : "";
    const limit = query.limit ? " LIMIT ?" : "";
    if (query.limit) params.push(query.limit);

    const rows = this.db
      .prepare(`SELECT * FROM workflow_runs${where} ORDER BY updated_at DESC${limit}`)
      .all(...params) as {
      run_id: string;
      tenant_id: string;
      workflow_id: string;
      status: string;
      created_at: number;
      updated_at: number;
    }[];

    return rows.map((row) => ({
      runId: row.run_id,
      tenantId: row.tenant_id,
      workflowId: row.workflow_id,
      status: row.status,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    }));
  }
}

/** In-memory store for tests and ephemeral runs. */
export class InMemoryRunStore implements RunStore {
  private readonly records = new Map<string, RunRecord>();
  private readonly log = new Map<string, RunEvent[]>();

  create(record: RunRecord): void {
    this.records.set(record.runId, { ...record });
  }

  append(runId: string, events: RunEvent[]): void {
    const existing = this.log.get(runId) ?? [];
    this.log.set(runId, [...existing, ...events]);
  }

  events(runId: string): RunEvent[] {
    return [...(this.log.get(runId) ?? [])];
  }

  get(runId: string): RunRecord | undefined {
    const record = this.records.get(runId);
    return record ? { ...record } : undefined;
  }

  updateStatus(runId: string, status: string, updatedAt: number): void {
    const record = this.records.get(runId);
    if (record) this.records.set(runId, { ...record, status, updatedAt });
  }

  list(query: { tenantId?: string; status?: string; limit?: number } = {}): RunRecord[] {
    let records = [...this.records.values()].sort((a, b) => b.updatedAt - a.updatedAt);
    if (query.tenantId) records = records.filter((r) => r.tenantId === query.tenantId);
    if (query.status) records = records.filter((r) => r.status === query.status);
    return query.limit ? records.slice(0, query.limit) : records;
  }
}
