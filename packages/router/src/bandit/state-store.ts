import { type Db, type Migration, runMigrations } from "@orchestrator/telemetry";
import type { BanditState } from "./bandit.js";

/**
 * Persisted learning state. A router that forgets everything on deploy is not learning — it is
 * permanently in cold start.
 *
 * The store is deliberately schemaless JSON: the router persists its bandit arms *and* its per-task
 * exploration counters together, and those shapes evolve independently of the table.
 */
export interface StateStore {
  load<T>(key: string): T | undefined;
  save(key: string, value: unknown): void;
}

/** What the adaptive router checkpoints. */
export interface RouterState {
  bandit: BanditState;
  /** Pull counts keyed `${armId}|${taskType}`. Diagnostics only. */
  contextPulls: Record<string, number>;
  /** Observations per task type. This is what the cold-start gate reads. */
  taskPulls?: Record<string, number>;
}

export const BANDIT_MIGRATIONS: Migration[] = [
  {
    id: "001_bandit_state",
    up: `
      CREATE TABLE IF NOT EXISTS router_state (
        key        TEXT PRIMARY KEY,
        state      TEXT NOT NULL,
        updated_at INTEGER NOT NULL
      );
    `,
  },
];

export class SqliteStateStore implements StateStore {
  constructor(private readonly db: Db) {
    runMigrations(this.db, BANDIT_MIGRATIONS);
  }

  load<T>(key: string): T | undefined {
    const row = this.db.prepare("SELECT state FROM router_state WHERE key = ?").get(key) as
      | { state: string }
      | undefined;
    if (!row) return undefined;

    try {
      return JSON.parse(row.state) as T;
    } catch {
      // Corrupt state is discarded rather than thrown: cold-starting the router is recoverable,
      // refusing to boot is not.
      return undefined;
    }
  }

  save(key: string, value: unknown): void {
    this.db
      .prepare(
        `INSERT INTO router_state (key, state, updated_at)
         VALUES (?, ?, ?)
         ON CONFLICT(key) DO UPDATE SET
           state = excluded.state,
           updated_at = excluded.updated_at`,
      )
      .run(key, JSON.stringify(value), Date.now());
  }
}

export class InMemoryStateStore implements StateStore {
  private readonly states = new Map<string, string>();

  load<T>(key: string): T | undefined {
    const raw = this.states.get(key);
    return raw ? (JSON.parse(raw) as T) : undefined;
  }

  save(key: string, value: unknown): void {
    // Serialized on write so tests cannot accidentally share mutable references with the bandit.
    this.states.set(key, JSON.stringify(value));
  }
}
