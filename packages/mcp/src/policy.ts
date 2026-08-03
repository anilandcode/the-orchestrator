import { type Db, type Migration, runMigrations } from "@orchestrator/telemetry";

/**
 * Who may call what.
 *
 * **Deny by default.** A tenant has no tools until a policy grants them. The opposite default fails
 * by letting one customer's agent reach another customer's integration, which is not a bug you get
 * to fix after the fact.
 */
export interface ToolPolicy {
  /** Glob-ish patterns over `server:tool`. `*` matches within a segment. */
  allow: string[];
  deny?: string[];
  /** Tools requiring explicit human approval before each call. */
  requireApproval?: string[];
}

export type ToolPolicies = Record<string, ToolPolicy>;

export type PolicyOutcome =
  | { allowed: true; requiresApproval: boolean }
  | { allowed: false; reason: string };

export function evaluatePolicy(
  qualifiedName: string,
  policy: ToolPolicy | undefined,
): PolicyOutcome {
  if (!policy) {
    return { allowed: false, reason: "no tool policy is configured for this tenant" };
  }

  // Deny wins over allow, always. A policy where the ordering could be argued is a policy nobody
  // can reason about in an incident.
  if (policy.deny?.some((pattern) => matches(pattern, qualifiedName))) {
    return { allowed: false, reason: `denied by policy pattern` };
  }

  if (!policy.allow.some((pattern) => matches(pattern, qualifiedName))) {
    return { allowed: false, reason: "not covered by any allow pattern" };
  }

  return {
    allowed: true,
    requiresApproval:
      policy.requireApproval?.some((pattern) => matches(qualifiedName, pattern)) ?? false,
  };
}

/** `*` matches any run of characters within the name. Intentionally simpler than a glob library. */
export function matches(pattern: string, value: string): boolean {
  if (pattern === "*") return true;
  const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*");
  return new RegExp(`^${escaped}$`).test(value);
}

export const TOOL_AUDIT_MIGRATIONS: Migration[] = [
  {
    id: "006_tool_audit",
    up: `
      CREATE TABLE IF NOT EXISTS tool_invocations (
        id          TEXT PRIMARY KEY,
        tenant_id   TEXT NOT NULL,
        tool        TEXT NOT NULL,
        run_id      TEXT,
        allowed     INTEGER NOT NULL,
        deny_reason TEXT,
        arguments   TEXT NOT NULL,
        is_error    INTEGER NOT NULL DEFAULT 0,
        error       TEXT,
        latency_ms  REAL NOT NULL DEFAULT 0,
        created_at  INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_tool_audit_tenant_time
        ON tool_invocations (tenant_id, created_at);
      CREATE INDEX IF NOT EXISTS idx_tool_audit_tool
        ON tool_invocations (tool, created_at);
    `,
  },
];

export interface ToolInvocationRecord {
  id: string;
  tenantId: string;
  tool: string;
  runId?: string | null;
  allowed: boolean;
  denyReason?: string | null;
  arguments: Record<string, unknown>;
  isError?: boolean;
  error?: string | null;
  latencyMs?: number;
  createdAt: number;
}

export interface ToolAuditLog {
  record(entry: ToolInvocationRecord): void;
  list(query?: { tenantId?: string; tool?: string; limit?: number }): ToolInvocationRecord[];
}

/**
 * Durable audit of every invocation, **including the ones that were denied**.
 *
 * Denials are the more interesting half: a spike of them is how you notice a misconfigured policy or
 * an agent trying to reach something it should not. A log of successes alone answers the wrong
 * question.
 */
export class SqliteToolAuditLog implements ToolAuditLog {
  constructor(private readonly db: Db) {
    runMigrations(this.db, TOOL_AUDIT_MIGRATIONS);
  }

  record(entry: ToolInvocationRecord): void {
    this.db
      .prepare(
        `INSERT OR REPLACE INTO tool_invocations
           (id, tenant_id, tool, run_id, allowed, deny_reason, arguments, is_error, error, latency_ms, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        entry.id,
        entry.tenantId,
        entry.tool,
        entry.runId ?? null,
        entry.allowed ? 1 : 0,
        entry.denyReason ?? null,
        // Arguments are truncated, not omitted: they are the most useful field in an incident and
        // the most likely to be enormous.
        JSON.stringify(entry.arguments).slice(0, 4_000),
        entry.isError ? 1 : 0,
        entry.error ?? null,
        entry.latencyMs ?? 0,
        entry.createdAt,
      );
  }

  list(query: { tenantId?: string; tool?: string; limit?: number } = {}): ToolInvocationRecord[] {
    const clauses: string[] = [];
    const params: unknown[] = [];

    if (query.tenantId) {
      clauses.push("tenant_id = ?");
      params.push(query.tenantId);
    }
    if (query.tool) {
      clauses.push("tool = ?");
      params.push(query.tool);
    }

    const where = clauses.length ? ` WHERE ${clauses.join(" AND ")}` : "";
    params.push(query.limit ?? 100);

    const rows = this.db
      .prepare(`SELECT * FROM tool_invocations${where} ORDER BY created_at DESC LIMIT ?`)
      .all(...params) as {
      id: string;
      tenant_id: string;
      tool: string;
      run_id: string | null;
      allowed: number;
      deny_reason: string | null;
      arguments: string;
      is_error: number;
      error: string | null;
      latency_ms: number;
      created_at: number;
    }[];

    return rows.map((row) => ({
      id: row.id,
      tenantId: row.tenant_id,
      tool: row.tool,
      runId: row.run_id,
      allowed: row.allowed === 1,
      denyReason: row.deny_reason,
      arguments: JSON.parse(row.arguments) as Record<string, unknown>,
      isError: row.is_error === 1,
      error: row.error,
      latencyMs: row.latency_ms,
      createdAt: row.created_at,
    }));
  }
}

export class InMemoryToolAuditLog implements ToolAuditLog {
  readonly entries: ToolInvocationRecord[] = [];

  record(entry: ToolInvocationRecord): void {
    this.entries.push(entry);
  }

  list(query: { tenantId?: string; tool?: string; limit?: number } = {}): ToolInvocationRecord[] {
    let matched = [...this.entries].sort((a, b) => b.createdAt - a.createdAt);
    if (query.tenantId) matched = matched.filter((e) => e.tenantId === query.tenantId);
    if (query.tool) matched = matched.filter((e) => e.tool === query.tool);
    return matched.slice(0, query.limit ?? 100);
  }
}
