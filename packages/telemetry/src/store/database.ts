import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import Database from "better-sqlite3";

export type Db = Database.Database;

/**
 * A migration is an id plus idempotent DDL. Packages own their own tables: telemetry registers the
 * call-event schema, the router registers its bandit-state schema, and both run through this runner
 * against the same connection.
 */
export interface Migration {
  id: string;
  up: string;
}

export function openDatabase(path: string): Db {
  if (path !== ":memory:") {
    mkdirSync(dirname(path), { recursive: true });
  }

  const db = new Database(path);
  // WAL lets the API read while a write is in flight, which matters once events are written on
  // every request.
  db.pragma("journal_mode = WAL");
  db.pragma("synchronous = NORMAL");
  return db;
}

export function runMigrations(db: Db, migrations: Migration[]): void {
  db.exec(
    `CREATE TABLE IF NOT EXISTS _migrations (
       id TEXT PRIMARY KEY,
       applied_at INTEGER NOT NULL
     )`,
  );

  const rows = db.prepare("SELECT id FROM _migrations").all() as { id: string }[];
  const applied = new Set(rows.map((row) => row.id));
  const insert = db.prepare("INSERT INTO _migrations (id, applied_at) VALUES (?, ?)");

  const apply = db.transaction(() => {
    for (const migration of migrations) {
      if (applied.has(migration.id)) continue;
      db.exec(migration.up);
      insert.run(migration.id, Date.now());
    }
  });

  apply();
}
