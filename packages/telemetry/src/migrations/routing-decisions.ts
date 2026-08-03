import type { Migration } from "../store/database.js";

/**
 * Shadow mode is only useful if the counterfactual survives the request.
 *
 * The router records what the bandit *would* have chosen alongside what actually ran; without this
 * table that comparison lives for the duration of one HTTP request and then disappears, and the
 * evidence needed to promote `ROUTER_MODE` to `adaptive` never accumulates.
 */
export const ROUTING_DECISION_MIGRATIONS: Migration[] = [
  {
    id: "002_routing_decisions",
    up: `
      CREATE TABLE IF NOT EXISTS routing_decisions (
        decision_id      TEXT PRIMARY KEY,
        model_id         TEXT NOT NULL,
        fallbacks        TEXT NOT NULL DEFAULT '[]',
        strategy         TEXT NOT NULL,
        reason           TEXT NOT NULL,
        shadow_model_id  TEXT,
        features         TEXT NOT NULL DEFAULT '[]',
        task_type        TEXT NOT NULL,
        route_mode       TEXT NOT NULL,
        created_at       INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_routing_decisions_time
        ON routing_decisions (created_at);
      -- Finding the rounds where the bandit disagreed with what ran is the core replay query.
      CREATE INDEX IF NOT EXISTS idx_routing_decisions_shadow
        ON routing_decisions (shadow_model_id, model_id);
    `,
  },
];
