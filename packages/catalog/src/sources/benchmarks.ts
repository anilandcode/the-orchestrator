import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { type BenchmarkFile, BenchmarkFileSchema } from "../schema.js";

/**
 * The curated benchmark file.
 *
 * Hand-maintained rather than scraped, on purpose. Leaderboard HTML changes without notice, many
 * published figures are self-reported, and a scraper gives no opportunity to notice that a number
 * moved before it starts steering traffic. A file in git is reviewed in a diff.
 */
const DEFAULT_PATH = resolve(dirname(fileURLToPath(import.meta.url)), "../data/benchmarks.json");

export function loadBenchmarkFile(path: string = DEFAULT_PATH): BenchmarkFile {
  const raw = readFileSync(path, "utf8");
  return parseBenchmarkFile(JSON.parse(raw));
}

/**
 * Parse and validate.
 *
 * Validation is strict about two things: every score must name a benchmark that the file also
 * defines, and every score must carry provenance. A score referencing an undefined benchmark is a
 * typo that would otherwise silently contribute nothing; a score without provenance is a number
 * nobody can check.
 */
export function parseBenchmarkFile(input: unknown): BenchmarkFile {
  const file = BenchmarkFileSchema.parse(input);
  const defined = new Set(file.benchmarks.map((benchmark) => benchmark.id));

  const orphans = file.scores.filter((score) => !defined.has(score.benchmarkId));
  if (orphans.length > 0) {
    const names = [...new Set(orphans.map((score) => score.benchmarkId))].join(", ");
    throw new Error(
      `Benchmark file references undefined benchmark(s): ${names}. A score whose benchmark is not declared would contribute silently to nothing.`,
    );
  }

  const unsourced = file.scores.filter((score) => !score.provenance.source);
  if (unsourced.length > 0) {
    throw new Error(
      `${unsourced.length} benchmark score(s) have no provenance. Every number must be traceable to where it came from and when, or it cannot be audited or expired.`,
    );
  }

  return file;
}

/**
 * Scores whose source is still marked as an unverified placeholder.
 *
 * Surfaced deliberately rather than filtered out: the shipped file is illustrative, and a system
 * that silently routed on made-up benchmark data would be worse than one with no priors at all.
 */
export function unverifiedScores(file: BenchmarkFile): number {
  return file.scores.filter((score) => score.provenance.source.startsWith("placeholder")).length;
}
