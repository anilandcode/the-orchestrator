import type { Migration } from "../store/database.js";

export const CALL_EVENT_MIGRATIONS: Migration[] = [
  {
    id: "001_call_events",
    up: `
      CREATE TABLE IF NOT EXISTS call_events (
        id                   TEXT PRIMARY KEY,
        tenant_id            TEXT NOT NULL,
        request_id           TEXT NOT NULL,
        routing_decision_id  TEXT,
        attempt              INTEGER NOT NULL,

        provider             TEXT NOT NULL,
        model_id             TEXT NOT NULL,
        task_type            TEXT NOT NULL,
        route_mode           TEXT NOT NULL,
        features             TEXT NOT NULL DEFAULT '[]',

        prompt_tokens        INTEGER NOT NULL,
        completion_tokens    INTEGER NOT NULL,
        cached_prompt_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd             REAL NOT NULL,
        latency_ms           REAL NOT NULL,
        ttft_ms              REAL,

        status               TEXT NOT NULL,
        error_class          TEXT,
        finish_reason        TEXT,

        quality_score        REAL,
        reward               REAL,

        created_at           INTEGER NOT NULL
      );

      -- The two access patterns that actually exist: tenant-scoped reporting, and per-model replay.
      CREATE INDEX IF NOT EXISTS idx_call_events_tenant_time
        ON call_events (tenant_id, created_at);
      CREATE INDEX IF NOT EXISTS idx_call_events_model_time
        ON call_events (model_id, created_at);
      CREATE INDEX IF NOT EXISTS idx_call_events_decision
        ON call_events (routing_decision_id);
    `,
  },
];
