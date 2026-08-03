import type { CallEvent, ErrorClass } from "@orchestrator/shared";
import { computeReward, percentileOf } from "./reward.js";

export interface ModelStats {
  modelId: string;
  provider: string;
  attempts: number;
  successes: number;
  successRate: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  totalCostUsd: number;
  /** Total spend divided by successful calls — the number that actually matters commercially. */
  costPerSuccessUsd: number;
  avgReward: number;
  errorsByClass: Record<string, number>;
}

/**
 * Per-model rollup over a set of events.
 *
 * Rates are computed over *attempts*, not requests, which is why the gateway must emit one event per
 * attempt: a model that only ever succeeds as a second-choice fallback should not look identical to
 * one that succeeds first time.
 */
export function aggregateByModel(events: CallEvent[]): ModelStats[] {
  const grouped = new Map<string, CallEvent[]>();
  for (const event of events) {
    const list = grouped.get(event.modelId) ?? [];
    list.push(event);
    grouped.set(event.modelId, list);
  }

  const stats: ModelStats[] = [];

  for (const [modelId, modelEvents] of grouped) {
    const successes = modelEvents.filter((event) => event.status === "success");
    const latencies = successes.map((event) => event.latencyMs);
    const totalCostUsd = sum(modelEvents.map((event) => event.costUsd));

    const errorsByClass: Record<string, number> = {};
    for (const event of modelEvents) {
      if (event.status !== "error") continue;
      const key: ErrorClass | "unknown" = event.errorClass ?? "unknown";
      errorsByClass[key] = (errorsByClass[key] ?? 0) + 1;
    }

    const rewards = modelEvents.map((event) => event.reward ?? computeReward(event));

    stats.push({
      modelId,
      provider: modelEvents[0]?.provider ?? "unknown",
      attempts: modelEvents.length,
      successes: successes.length,
      successRate: modelEvents.length ? successes.length / modelEvents.length : 0,
      p50LatencyMs: percentileOf(latencies, 0.5),
      p95LatencyMs: percentileOf(latencies, 0.95),
      totalCostUsd,
      // Spend on failed attempts is real spend; charging it against successes is the honest view.
      costPerSuccessUsd: successes.length
        ? totalCostUsd / successes.length
        : Number.POSITIVE_INFINITY,
      avgReward: rewards.length ? sum(rewards) / rewards.length : 0,
      errorsByClass,
    });
  }

  return stats.sort((a, b) => b.avgReward - a.avgReward);
}

export interface TrafficSummary {
  events: number;
  requests: number;
  successRate: number;
  totalCostUsd: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  /** Fraction of requests that needed more than one attempt. A retry/fallback health signal. */
  retryRate: number;
}

export function summarize(events: CallEvent[]): TrafficSummary {
  const requestIds = new Set(events.map((event) => event.requestId));
  const successes = events.filter((event) => event.status === "success");
  const latencies = successes.map((event) => event.latencyMs);
  const retried = new Set(
    events.filter((event) => event.attempt > 1).map((event) => event.requestId),
  );

  return {
    events: events.length,
    requests: requestIds.size,
    successRate: events.length ? successes.length / events.length : 0,
    totalCostUsd: sum(events.map((event) => event.costUsd)),
    p50LatencyMs: percentileOf(latencies, 0.5),
    p95LatencyMs: percentileOf(latencies, 0.95),
    retryRate: requestIds.size ? retried.size / requestIds.size : 0,
  };
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
