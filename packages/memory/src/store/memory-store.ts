import { type Db, type Migration, runMigrations } from "@orchestrator/telemetry";
import { z } from "zod";
import { cosineSimilarity } from "../embedder.js";

/**
 * What kind of thing a memory is.
 *
 * `turn`    — a verbatim conversation turn, scoped to a session.
 * `summary` — a compressed stand-in for turns that have aged out of the buffer.
 * `fact`    — a durable statement about the user or their world, worth recalling across sessions.
 */
export const MemoryKindSchema = z.enum(["turn", "summary", "fact"]);
export type MemoryKind = z.infer<typeof MemoryKindSchema>;

export const MemoryItemSchema = z.object({
  id: z.string(),
  tenantId: z.string(),
  /** Null for tenant-wide facts that outlive any one conversation. */
  sessionId: z.string().nullable().default(null),
  kind: MemoryKindSchema,
  text: z.string(),
  /** Free-form provenance: role, source node, whatever the writer wants to keep. */
  metadata: z.record(z.string(), z.unknown()).default({}),
  /** Which embedder produced the vector — vectors from different models are not comparable. */
  embedder: z.string().nullable().default(null),
  createdAt: z.number().int().nonnegative(),
  /** Epoch ms after which the item is ignored and eligible for deletion. */
  expiresAt: z.number().int().nonnegative().nullable().default(null),
});

export type MemoryItem = z.infer<typeof MemoryItemSchema>;

export interface MemoryWrite extends Omit<MemoryItem, "id" | "createdAt"> {
  id?: string;
  createdAt?: number;
  embedding?: Float32Array;
}

export interface SearchOptions {
  tenantId: string;
  /** Restrict to one conversation. Omit to search everything the tenant owns. */
  sessionId?: string;
  kinds?: MemoryKind[];
  limit?: number;
  /** Results below this cosine score are dropped rather than padded in. */
  minScore?: number;
  now?: number;
}

export interface ScoredMemory {
  item: MemoryItem;
  score: number;
}

export interface MemoryStore {
  write(items: MemoryWrite[]): MemoryItem[];
  search(query: Float32Array, options: SearchOptions): ScoredMemory[];
  recent(options: { tenantId: string; sessionId: string; limit?: number }): MemoryItem[];
  forget(options: { tenantId: string; sessionId?: string }): number;
  /** Delete expired rows. Retention is a promise; it has to actually happen. */
  purgeExpired(now?: number): number;
  count(options: { tenantId: string; sessionId?: string }): number;
}

export const MEMORY_MIGRATIONS: Migration[] = [
  {
    id: "005_memory_items",
    up: `
      CREATE TABLE IF NOT EXISTS memory_items (
        id         TEXT PRIMARY KEY,
        tenant_id  TEXT NOT NULL,
        session_id TEXT,
        kind       TEXT NOT NULL,
        text       TEXT NOT NULL,
        metadata   TEXT NOT NULL DEFAULT '{}',
        embedder   TEXT,
        embedding  BLOB,
        created_at INTEGER NOT NULL,
        expires_at INTEGER
      );

      -- Every read is tenant-scoped, so the tenant column leads every index.
      CREATE INDEX IF NOT EXISTS idx_memory_tenant_session
        ON memory_items (tenant_id, session_id, created_at);
      CREATE INDEX IF NOT EXISTS idx_memory_tenant_kind
        ON memory_items (tenant_id, kind, created_at);
      CREATE INDEX IF NOT EXISTS idx_memory_expiry
        ON memory_items (expires_at);
    `,
  },
];

interface MemoryRow {
  id: string;
  tenant_id: string;
  session_id: string | null;
  kind: string;
  text: string;
  metadata: string;
  embedder: string | null;
  embedding: Buffer | null;
  created_at: number;
  expires_at: number | null;
}

/**
 * SQLite-backed memory with brute-force cosine search.
 *
 * **Scale, stated honestly.** Every search loads the tenant's candidate vectors and scores them in
 * process. That is genuinely fine into the tens of thousands of rows per tenant and keeps the system
 * dependency-free, but it is linear — past roughly 10^5 rows it needs a real index (pgvector,
 * sqlite-vec). That is a driver swap behind `MemoryStore`, which is why the interface exists.
 */
export class SqliteMemoryStore implements MemoryStore {
  constructor(
    private readonly db: Db,
    private readonly idFactory: () => string = () =>
      `mem_${Math.random().toString(36).slice(2, 14)}`,
  ) {
    runMigrations(this.db, MEMORY_MIGRATIONS);
  }

  write(items: MemoryWrite[]): MemoryItem[] {
    const statement = this.db.prepare(
      `INSERT OR REPLACE INTO memory_items
         (id, tenant_id, session_id, kind, text, metadata, embedder, embedding, created_at, expires_at)
       VALUES (@id, @tenant_id, @session_id, @kind, @text, @metadata, @embedder, @embedding, @created_at, @expires_at)`,
    );

    const written: MemoryItem[] = [];

    const insertAll = this.db.transaction((batch: MemoryWrite[]) => {
      for (const item of batch) {
        const record = MemoryItemSchema.parse({
          ...item,
          id: item.id ?? this.idFactory(),
          createdAt: item.createdAt ?? Date.now(),
        });

        statement.run({
          id: record.id,
          tenant_id: record.tenantId,
          session_id: record.sessionId,
          kind: record.kind,
          text: record.text,
          metadata: JSON.stringify(record.metadata),
          embedder: record.embedder,
          embedding: item.embedding ? Buffer.from(item.embedding.buffer.slice(0)) : null,
          created_at: record.createdAt,
          expires_at: record.expiresAt,
        });

        written.push(record);
      }
    });

    insertAll(items);
    return written;
  }

  search(query: Float32Array, options: SearchOptions): ScoredMemory[] {
    const now = options.now ?? Date.now();
    const clauses = ["tenant_id = ?", "embedding IS NOT NULL"];
    const params: unknown[] = [options.tenantId];

    if (options.sessionId) {
      // Session-scoped search still includes tenant-wide facts, which is the point of storing them
      // with a null session.
      clauses.push("(session_id = ? OR session_id IS NULL)");
      params.push(options.sessionId);
    }
    if (options.kinds?.length) {
      clauses.push(`kind IN (${options.kinds.map(() => "?").join(", ")})`);
      params.push(...options.kinds);
    }
    clauses.push("(expires_at IS NULL OR expires_at > ?)");
    params.push(now);

    const rows = this.db
      .prepare(`SELECT * FROM memory_items WHERE ${clauses.join(" AND ")}`)
      .all(...params) as MemoryRow[];

    const minScore = options.minScore ?? 0;
    const scored: ScoredMemory[] = [];

    for (const row of rows) {
      if (!row.embedding) continue;
      const score = cosineSimilarity(query, toVector(row.embedding));
      // Below the floor, an item is noise. Padding results to `limit` with weak matches is how
      // retrieval quietly poisons a prompt.
      if (score < minScore) continue;
      scored.push({ item: fromRow(row), score });
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, options.limit ?? 5);
  }

  recent(options: { tenantId: string; sessionId: string; limit?: number }): MemoryItem[] {
    const rows = this.db
      .prepare(
        `SELECT * FROM memory_items
          WHERE tenant_id = ? AND session_id = ?
          ORDER BY created_at DESC
          LIMIT ?`,
      )
      .all(options.tenantId, options.sessionId, options.limit ?? 20) as MemoryRow[];

    // Reversed so callers get chronological order, which is what a prompt needs.
    return rows.map(fromRow).reverse();
  }

  forget(options: { tenantId: string; sessionId?: string }): number {
    const result = options.sessionId
      ? this.db
          .prepare("DELETE FROM memory_items WHERE tenant_id = ? AND session_id = ?")
          .run(options.tenantId, options.sessionId)
      : this.db.prepare("DELETE FROM memory_items WHERE tenant_id = ?").run(options.tenantId);

    return result.changes;
  }

  purgeExpired(now: number = Date.now()): number {
    return this.db
      .prepare("DELETE FROM memory_items WHERE expires_at IS NOT NULL AND expires_at <= ?")
      .run(now).changes;
  }

  count(options: { tenantId: string; sessionId?: string }): number {
    const row = options.sessionId
      ? (this.db
          .prepare("SELECT COUNT(*) AS n FROM memory_items WHERE tenant_id = ? AND session_id = ?")
          .get(options.tenantId, options.sessionId) as { n: number })
      : (this.db
          .prepare("SELECT COUNT(*) AS n FROM memory_items WHERE tenant_id = ?")
          .get(options.tenantId) as { n: number });

    return row.n;
  }
}

function fromRow(row: MemoryRow): MemoryItem {
  return MemoryItemSchema.parse({
    id: row.id,
    tenantId: row.tenant_id,
    sessionId: row.session_id,
    kind: row.kind,
    text: row.text,
    metadata: JSON.parse(row.metadata) as Record<string, unknown>,
    embedder: row.embedder,
    createdAt: row.created_at,
    expiresAt: row.expires_at,
  });
}

function toVector(buffer: Buffer): Float32Array {
  return new Float32Array(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  );
}
