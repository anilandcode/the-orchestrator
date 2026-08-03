import { CatalogService, SqliteCatalogStore } from "@orchestrator/catalog";
import { ModelRegistry } from "@orchestrator/shared";
import { openDatabase } from "@orchestrator/telemetry";

/**
 * `pnpm catalog:refresh [--apply] [--accept-suspicious]`
 *
 * Two steps on purpose. Fetching writes an unapplied snapshot and prints what applying it *would*
 * do; `--apply` is a separate, deliberate act. Ingested pricing feeds cost accounting, the reward's
 * cost term, budget filtering, and `cheap`-mode ordering — a refresh that silently rewrote it could
 * move routing without anyone choosing to.
 */
const DB_PATH = process.env.ORCHESTRATOR_DB_PATH ?? "./data/orchestrator.sqlite";

async function main(): Promise<void> {
  const args = new Set(process.argv.slice(2));
  const shouldApply = args.has("--apply");
  const acceptSuspicious = args.has("--accept-suspicious");

  const db = openDatabase(DB_PATH);
  const registry = new ModelRegistry();
  const service = new CatalogService({ store: new SqliteCatalogStore(db), registry });

  console.log(`Fetching model catalog into ${DB_PATH} ...\n`);
  const { snapshot, guardrail, unverifiedBenchmarkScores } = await service.refresh();

  console.log(
    `Snapshot v${snapshot.version}: ${snapshot.entries.length} models, ${snapshot.scores.length} benchmark scores`,
  );
  console.log(`Models known locally: ${snapshot.entries.length - guardrail.unknownModels.length}`);
  console.log(`Catalog-only (no adapter here): ${guardrail.unknownModels.length}\n`);

  if (guardrail.routine.length > 0) {
    console.log(`Routine price changes (${guardrail.routine.length}):`);
    for (const change of guardrail.routine) {
      console.log(
        `  ${change.modelId} ${change.field}: ${change.from} -> ${change.to} (${change.ratio.toFixed(2)}x)`,
      );
    }
    console.log("");
  }

  if (guardrail.suspicious.length > 0) {
    console.log(
      `SUSPICIOUS price changes (${guardrail.suspicious.length}) — need --accept-suspicious:`,
    );
    for (const change of guardrail.suspicious) {
      console.log(
        `  ${change.modelId} ${change.field}: ${change.from} -> ${change.to} (${change.ratio.toFixed(2)}x)`,
      );
    }
    console.log(
      "  Providers cut prices; they rarely cut them tenfold. A large rise is usually a unit error.\n",
    );
  }

  if (guardrail.rejected.length > 0) {
    console.log(`REJECTED (${guardrail.rejected.length}):`);
    for (const rejection of guardrail.rejected) {
      console.log(`  ${rejection.modelId}: ${rejection.reason}`);
    }
    console.log("");
  }

  if (unverifiedBenchmarkScores > 0) {
    console.log(
      `WARNING: ${unverifiedBenchmarkScores} benchmark score(s) are unverified placeholders.\n` +
        "  The ingestion pipeline is sound; the shipped numbers are not. Replace them in\n" +
        "  packages/catalog/src/data/benchmarks.json with figures you have read off the cited\n" +
        "  leaderboard before letting priors influence production routing.\n",
    );
  }

  if (!shouldApply) {
    console.log(`Nothing applied. Re-run with --apply to promote v${snapshot.version}.`);
    db.close();
    return;
  }

  const result = service.apply(snapshot.version, { acceptSuspicious });
  console.log(
    `Applied v${snapshot.version}: ${result.updated.length} models updated, ${result.skipped.length} skipped.`,
  );

  const priors = service.derivedPriors();
  const tasks = [...new Set(priors.map((prior) => prior.taskType))].sort();
  console.log(`Derived ${priors.length} priors across task types: ${tasks.join(", ") || "(none)"}`);
  console.log(
    "Task types with no defensible benchmark mapping get no prior and keep normal cold-start behaviour.",
  );

  db.close();
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
