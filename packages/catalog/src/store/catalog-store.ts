import { type Db, type Migration, runMigrations } from "@orchestrator/telemetry";
import { type CatalogSnapshot, CatalogSnapshotSchema } from "../schema.js";

export const CATALOG_MIGRATIONS: Migration[] = [
  {
    id: "007_catalog_snapshots",
    up: `
      CREATE TABLE IF NOT EXISTS catalog_snapshots (
        version    INTEGER PRIMARY KEY,
        snapshot   TEXT NOT NULL,
        applied    INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_catalog_applied ON catalog_snapshots (applied, version);
    `,
  },
];

/**
 * Versioned catalog storage.
 *
 * Snapshots are immutable and a refresh writes a new one rather than overwriting the last. That is
 * what makes a diff possible before anything changes, and what lets you answer "what did the router
 * believe last Tuesday?" after a routing decision turns out badly.
 */
export class SqliteCatalogStore {
  constructor(private readonly db: Db) {
    runMigrations(this.db, CATALOG_MIGRATIONS);
  }

  /** Write a new snapshot, unapplied. Returns the assigned version. */
  write(snapshot: Omit<CatalogSnapshot, "version">): CatalogSnapshot {
    const row = this.db
      .prepare("SELECT COALESCE(MAX(version), 0) AS max FROM catalog_snapshots")
      .get() as { max: number };

    const versioned = CatalogSnapshotSchema.parse({ ...snapshot, version: row.max + 1 });

    this.db
      .prepare(
        "INSERT INTO catalog_snapshots (version, snapshot, applied, created_at) VALUES (?, ?, 0, ?)",
      )
      .run(versioned.version, JSON.stringify(versioned), versioned.createdAt);

    return versioned;
  }

  /** The snapshot currently steering the system, or undefined before anything is applied. */
  applied(): CatalogSnapshot | undefined {
    const row = this.db
      .prepare(
        "SELECT snapshot FROM catalog_snapshots WHERE applied = 1 ORDER BY version DESC LIMIT 1",
      )
      .get() as { snapshot: string } | undefined;
    return row ? CatalogSnapshotSchema.parse(JSON.parse(row.snapshot)) : undefined;
  }

  /** The newest snapshot, applied or not — what a refresh just wrote. */
  latest(): CatalogSnapshot | undefined {
    const row = this.db
      .prepare("SELECT snapshot FROM catalog_snapshots ORDER BY version DESC LIMIT 1")
      .get() as { snapshot: string } | undefined;
    return row ? CatalogSnapshotSchema.parse(JSON.parse(row.snapshot)) : undefined;
  }

  get(version: number): CatalogSnapshot | undefined {
    const row = this.db
      .prepare("SELECT snapshot FROM catalog_snapshots WHERE version = ?")
      .get(version) as { snapshot: string } | undefined;
    return row ? CatalogSnapshotSchema.parse(JSON.parse(row.snapshot)) : undefined;
  }

  /** Promote a snapshot. Only one is applied at a time. */
  apply(version: number): void {
    const promote = this.db.transaction(() => {
      this.db.prepare("UPDATE catalog_snapshots SET applied = 0").run();
      this.db.prepare("UPDATE catalog_snapshots SET applied = 1 WHERE version = ?").run(version);
    });
    promote();
  }

  list(limit = 20): { version: number; applied: boolean; createdAt: number }[] {
    const rows = this.db
      .prepare(
        "SELECT version, applied, created_at FROM catalog_snapshots ORDER BY version DESC LIMIT ?",
      )
      .all(limit) as { version: number; applied: number; created_at: number }[];

    return rows.map((row) => ({
      version: row.version,
      applied: row.applied === 1,
      createdAt: row.created_at,
    }));
  }
}
