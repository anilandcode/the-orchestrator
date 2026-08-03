import type { TaskType } from "@orchestrator/shared";

/**
 * Which benchmarks say something about which task types, and how much.
 *
 * This file is a set of opinions, deliberately collected in one place rather than scattered through
 * ingestion code, so it can be argued with in review. Every entry is a claim that a benchmark
 * measuring one thing predicts performance on another, and those claims range from reasonable
 * (SWE-bench → code) to a stretch.
 *
 * **Three task types are deliberately absent: `extraction`, `summarization`, and `classification`.**
 * Public benchmark coverage for them is thin, and the honest options were to invent a mapping from
 * general-capability scores or to abstain. Abstaining wins for the same reason it wins in
 * `packages/quality`: a confident signal with nothing behind it is worse than no signal, because the
 * bandit cannot tell the difference and will act on it. Those task types keep today's cold-start
 * behaviour until there is something real to say.
 */
export interface BenchmarkWeight {
  benchmarkId: string;
  weight: number;
}

export const TASK_BENCHMARK_MAPPING: Partial<Record<TaskType, BenchmarkWeight[]>> = {
  code: [
    // Closest thing to the real job: fix an issue in an existing repository.
    { benchmarkId: "swe-bench-verified", weight: 0.7 },
    { benchmarkId: "mmlu-pro", weight: 0.3 },
  ],
  reasoning: [
    { benchmarkId: "gpqa-diamond", weight: 0.6 },
    { benchmarkId: "mmlu-pro", weight: 0.4 },
  ],
  general: [
    // Human pairwise preference is the least unrepresentative proxy for open-ended chat.
    { benchmarkId: "arena-elo", weight: 0.6 },
    { benchmarkId: "mmlu-pro", weight: 0.4 },
  ],
  creative: [
    // Weak by admission. Arena preference captures some of it; nothing public captures much more.
    { benchmarkId: "arena-elo", weight: 1 },
  ],
};

export function mappingFor(taskType: TaskType): BenchmarkWeight[] | undefined {
  return TASK_BENCHMARK_MAPPING[taskType];
}

/** Task types this mapping is willing to make a claim about. */
export function mappedTaskTypes(): TaskType[] {
  return Object.keys(TASK_BENCHMARK_MAPPING) as TaskType[];
}

/**
 * Percentile rank of a value within a population, in [0,1].
 *
 * Benchmarks arrive on incompatible scales — Arena Elo near 1200, MMLU as a percentage, SWE-bench as
 * a percentage with a very different spread. Percentile-within-benchmark makes them comparable and,
 * unlike min-max against fixed anchors, survives a leaderboard's range shifting between refreshes.
 *
 * The cost is that it measures rank, not absolute ability: in a pool of uniformly weak models the
 * best is still scored 1. Acceptable here, because routing is a choice *among* the available pool.
 */
export function percentileRank(value: number, population: number[]): number {
  if (population.length === 0) return 0;
  if (population.length === 1) return 0.5;

  const below = population.filter((entry) => entry < value).length;
  const equal = population.filter((entry) => entry === value).length;

  // Midpoint of the tied band, so identical scores get identical ranks rather than arbitrary order.
  return (below + equal / 2) / population.length;
}
