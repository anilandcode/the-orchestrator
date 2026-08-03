import type { CallEvent, CallEventSink, TaskType } from "@orchestrator/shared";

export interface QualityProvenance {
  /** Name of the scorer that produced the score. */
  source: string;
  /** 0..1 authority claim, used to decide whether a later signal should override this one. */
  confidence: number;
  /** True when this replaces an earlier score rather than setting the first one. */
  isRevision?: boolean;
}

export interface CallEventQuery {
  tenantId?: string;
  /** All attempts belonging to one caller-facing request. */
  requestId?: string;
  modelId?: string;
  taskType?: TaskType;
  /** Inclusive lower bound on `createdAt`, epoch ms. */
  since?: number;
  /** Exclusive upper bound on `createdAt`, epoch ms. */
  until?: number;
  status?: CallEvent["status"];
  limit?: number;
}

/**
 * The seam that keeps SQL out of the rest of the system. Swapping SQLite for Postgres means writing a
 * second implementation of this interface and nothing else.
 */
export interface CallEventRepository extends CallEventSink {
  record(event: CallEvent): void;
  recordMany(events: CallEvent[]): void;
  /** Attaches a score after the fact, once a quality signal arrives. */
  scoreEvent(
    id: string,
    qualityScore: number | null,
    reward: number,
    provenance?: QualityProvenance,
  ): void;
  query(query?: CallEventQuery): CallEvent[];
  count(query?: CallEventQuery): number;
  close(): void;
}
