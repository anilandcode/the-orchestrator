import type { CallEvent } from "@orchestrator/shared";
import type { CallEventQuery, CallEventRepository, QualityProvenance } from "./repository.js";

/**
 * In-memory repository. Used by the replay simulator, which generates far more events than are worth
 * writing to disk, and by tests that do not care about persistence.
 */
export class InMemoryCallEventRepository implements CallEventRepository {
  private events: CallEvent[] = [];

  record(event: CallEvent): void {
    this.events.push(event);
  }

  recordMany(events: CallEvent[]): void {
    this.events.push(...events);
  }

  scoreEvent(
    id: string,
    qualityScore: number | null,
    reward: number,
    provenance?: QualityProvenance,
  ): void {
    const event = this.events.find((candidate) => candidate.id === id);
    if (!event) return;

    event.qualityScore = qualityScore;
    event.reward = reward;
    if (provenance) {
      event.qualitySource = provenance.source;
      event.qualityConfidence = provenance.confidence;
      if (provenance.isRevision) event.qualityRevisions += 1;
    }
  }

  query(query: CallEventQuery = {}): CallEvent[] {
    const matched = this.events
      .filter((event) => matches(event, query))
      .sort((a, b) => a.createdAt - b.createdAt);
    return query.limit ? matched.slice(0, query.limit) : matched;
  }

  count(query: CallEventQuery = {}): number {
    return this.events.filter((event) => matches(event, query)).length;
  }

  close(): void {
    this.events = [];
  }
}

function matches(event: CallEvent, query: CallEventQuery): boolean {
  if (query.tenantId && event.tenantId !== query.tenantId) return false;
  if (query.requestId && event.requestId !== query.requestId) return false;
  if (query.modelId && event.modelId !== query.modelId) return false;
  if (query.taskType && event.taskType !== query.taskType) return false;
  if (query.status && event.status !== query.status) return false;
  if (query.since !== undefined && event.createdAt < query.since) return false;
  if (query.until !== undefined && event.createdAt >= query.until) return false;
  return true;
}
