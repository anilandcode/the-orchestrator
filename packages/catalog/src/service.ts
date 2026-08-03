import type { ModelPrior } from "@orchestrator/router";
import type { ModelRegistry, RouteMode, TaskType } from "@orchestrator/shared";
import { type GuardrailReport, applyToRegistry, checkPricing } from "./guardrail.js";
import { type DerivedPrior, deriveCapabilities, derivePriors } from "./priors.js";
import { type CatalogSnapshot, isStale } from "./schema.js";
import { loadBenchmarkFile, unverifiedScores } from "./sources/benchmarks.js";
import { fetchOpenRouterCatalog } from "./sources/openrouter.js";
import type { SqliteCatalogStore } from "./store/catalog-store.js";

export interface CatalogServiceConfig {
  store: SqliteCatalogStore;
  registry: ModelRegistry;
  /** Injected so refresh can be tested against a fixture with no network. */
  fetchImpl?: typeof globalThis.fetch;
  benchmarkPath?: string;
  now?: () => number;
}

export interface RefreshResult {
  snapshot: CatalogSnapshot;
  guardrail: GuardrailReport;
  unverifiedBenchmarkScores: number;
}

/**
 * Ingest, version, and expose external model knowledge.
 *
 * Note what this class does *not* do: it never applies anything as a side effect of fetching.
 * `refresh` writes a snapshot and reports what would change; `apply` is a separate, deliberate act.
 * Pricing feeds cost, reward, and budget filtering, so a refresh that silently rewrote it could move
 * routing without anyone deciding to.
 */
export class CatalogService {
  private readonly store: SqliteCatalogStore;
  private readonly registry: ModelRegistry;
  private readonly fetchImpl: typeof globalThis.fetch | undefined;
  private readonly benchmarkPath: string | undefined;
  private readonly now: () => number;

  constructor(config: CatalogServiceConfig) {
    this.store = config.store;
    this.registry = config.registry;
    this.fetchImpl = config.fetchImpl;
    this.benchmarkPath = config.benchmarkPath;
    this.now = config.now ?? (() => Date.now());
  }

  /** Fetch everything into a new unapplied snapshot and report what applying it would do. */
  async refresh(): Promise<RefreshResult> {
    const entries = await fetchOpenRouterCatalog({
      ...(this.fetchImpl ? { fetchImpl: this.fetchImpl } : {}),
      now: this.now,
    });

    const benchmarkFile = this.benchmarkPath
      ? loadBenchmarkFile(this.benchmarkPath)
      : loadBenchmarkFile();

    const snapshot = this.store.write({
      createdAt: this.now(),
      entries,
      scores: benchmarkFile.scores,
      benchmarks: benchmarkFile.benchmarks,
    });

    return {
      snapshot,
      guardrail: checkPricing(entries, this.registry),
      unverifiedBenchmarkScores: unverifiedScores(benchmarkFile),
    };
  }

  /** Promote a snapshot and write its facts into the live registry. */
  apply(
    version: number,
    options: { acceptSuspicious?: boolean } = {},
  ): {
    updated: string[];
    skipped: string[];
  } {
    const snapshot = this.store.get(version);
    if (!snapshot) throw new Error(`Unknown catalog version: ${version}`);

    const guardrail = checkPricing(snapshot.entries, this.registry);
    const result = applyToRegistry(snapshot.entries, this.registry, guardrail, options);
    this.store.apply(version);

    return result;
  }

  /** Load the applied snapshot into the registry — what happens at every boot. */
  applyStored(): { updated: string[]; skipped: string[] } | undefined {
    const snapshot = this.store.applied();
    if (!snapshot) return undefined;

    const guardrail = checkPricing(snapshot.entries, this.registry);
    // Already reviewed when it was promoted; re-litigating on every boot would block startup on a
    // decision that was made once, deliberately.
    return applyToRegistry(snapshot.entries, this.registry, guardrail, { acceptSuspicious: true });
  }

  /** Derived priors for the applied snapshot, ready for `AdaptiveRouter.applyPriors`. */
  derivedPriors(): DerivedPrior[] {
    const snapshot = this.store.applied();
    if (!snapshot) return [];

    return derivePriors(deriveCapabilities(snapshot.scores), this.registry);
  }

  /**
   * Priors in router form.
   *
   * The feature vector is built by the caller, because `extractFeatures` lives in the router package
   * and this one must not depend on it in the other direction.
   */
  routerPriors(
    buildFeatures: (taskType: TaskType, routeMode: RouteMode) => number[],
  ): ModelPrior[] {
    return this.derivedPriors().map((prior) => ({
      modelId: prior.modelId,
      features: buildFeatures(prior.taskType, prior.routeMode),
      reward: prior.reward,
      weight: prior.weight,
      source: prior.source,
    }));
  }

  /** What the system currently believes, and how old it is. */
  status(): {
    version: number | null;
    createdAt: number | null;
    stale: boolean;
    models: number;
    benchmarkScores: number;
    unverifiedScores: number;
  } {
    const snapshot = this.store.applied();
    if (!snapshot) {
      return {
        version: null,
        createdAt: null,
        stale: false,
        models: 0,
        benchmarkScores: 0,
        unverifiedScores: 0,
      };
    }

    return {
      version: snapshot.version,
      createdAt: snapshot.createdAt,
      stale: isStale(snapshot, this.now()),
      models: snapshot.entries.length,
      benchmarkScores: snapshot.scores.length,
      unverifiedScores: snapshot.scores.filter((score) =>
        score.provenance.source.startsWith("placeholder"),
      ).length,
    };
  }

  applied(): CatalogSnapshot | undefined {
    return this.store.applied();
  }
}
