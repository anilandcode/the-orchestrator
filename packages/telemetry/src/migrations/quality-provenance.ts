import type { Migration } from "../store/database.js";

/**
 * Quality provenance.
 *
 * Without knowing *which* scorer produced a score, you cannot tell whether a model looks good because
 * it is good or because only the most lenient scorer ever rated it. A model whose traffic happens to
 * be graded by a strict code validator will look worse than one graded by the optimistic heuristic
 * fallback, and the bandit would learn that difference as if it were real quality.
 *
 * `quality_revisions` exists to make late-arriving corrections visible in the data rather than
 * silently overwriting history.
 */
export const QUALITY_PROVENANCE_MIGRATIONS: Migration[] = [
  {
    id: "003_quality_provenance",
    up: `
      ALTER TABLE call_events ADD COLUMN quality_source TEXT;
      ALTER TABLE call_events ADD COLUMN quality_confidence REAL;
      ALTER TABLE call_events ADD COLUMN quality_revisions INTEGER NOT NULL DEFAULT 0;

      -- Judge traffic is real spend but must never be routed on or counted as user traffic.
      ALTER TABLE call_events ADD COLUMN is_judge INTEGER NOT NULL DEFAULT 0;

      CREATE INDEX IF NOT EXISTS idx_call_events_quality_source
        ON call_events (quality_source);
    `,
  },
];
