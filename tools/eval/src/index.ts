import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  AnthropicAdapter,
  Gateway,
  OpenAIAdapter,
  type ProviderAdapter,
} from "@orchestrator/gateway";
import { loadFixtures } from "./fixtures.js";
import { type EvalResult, runEval, summarizeCapabilities } from "./runner.js";

/**
 * `pnpm eval`
 *
 * **This spends real money** — one call per (model × fixture). With the starter set and six models
 * that is 60 calls, mostly on cheap models, but it is real spend and it is opt-in for that reason.
 *
 * What it produces is the thing public benchmarks could not: capability estimates measured on tasks
 * shaped like your own traffic, in the same units the router already learns in.
 */
const OUT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "../out");

async function main(): Promise<void> {
  const adapters: ProviderAdapter[] = [];
  if (process.env.OPENAI_API_KEY) {
    adapters.push(new OpenAIAdapter({ apiKey: process.env.OPENAI_API_KEY }));
  }
  if (process.env.ANTHROPIC_API_KEY) {
    adapters.push(new AnthropicAdapter({ apiKey: process.env.ANTHROPIC_API_KEY }));
  }

  if (adapters.length === 0) {
    console.error(
      "No provider keys found.\n\n" +
        "  cp .env.example .env   and set OPENAI_API_KEY and/or ANTHROPIC_API_KEY\n\n" +
        "This runs one real call per model per fixture and costs money.",
    );
    process.exit(1);
  }

  const gateway = new Gateway({ adapters, timeoutMs: 60_000 });
  const fixtures = loadFixtures();
  const models = gateway.availableModels();

  console.log(`Evaluating ${models.length} model(s) against ${fixtures.length} fixture(s).`);
  console.log(`${models.length * fixtures.length} real calls will be made.\n`);

  const results = await runEval({
    gateway,
    fixtures,
    onProgress: (done, total, label) => {
      process.stdout.write(`\r  [${String(done).padStart(3)}/${total}] ${label.padEnd(50)}`);
    },
  });
  process.stdout.write("\n\n");

  const capabilities = summarizeCapabilities(results);

  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(
    resolve(OUT_DIR, "capabilities.json"),
    JSON.stringify(capabilities, null, 2),
    "utf8",
  );
  writeFileSync(resolve(OUT_DIR, "results.json"), JSON.stringify(results, null, 2), "utf8");
  writeFileSync(resolve(OUT_DIR, "report.md"), report(results, capabilities), "utf8");

  printSummary(results, capabilities);
  console.log(`\nWritten to ${OUT_DIR}/`);
  console.log(
    "\nThese are measured on YOUR fixtures, which is what public benchmark scores were not.\n" +
      "Point MEASURED_CAPABILITIES_PATH at capabilities.json to seed the router from them.",
  );
}

function printSummary(
  results: EvalResult[],
  capabilities: ReturnType<typeof summarizeCapabilities>,
): void {
  const failures = results.filter((result) => result.error !== null).length;
  const spend = results.reduce((total, result) => total + result.costUsd, 0);

  console.log(`Calls: ${results.length}   Failures: ${failures}   Spend: $${spend.toFixed(4)}\n`);

  const byModel = new Map<string, number[]>();
  for (const capability of capabilities) {
    const scores = byModel.get(capability.modelId) ?? [];
    scores.push(capability.capability);
    byModel.set(capability.modelId, scores);
  }

  const ranked = [...byModel.entries()]
    .map(([modelId, scores]) => ({
      modelId,
      mean: scores.reduce((total, score) => total + score, 0) / scores.length,
    }))
    .sort((a, b) => b.mean - a.mean);

  console.log("Mean capability across task types:");
  for (const entry of ranked) {
    console.log(`  ${entry.modelId.padEnd(34)} ${entry.mean.toFixed(3)}`);
  }
}

function report(
  results: EvalResult[],
  capabilities: ReturnType<typeof summarizeCapabilities>,
): string {
  const lines: string[] = [];
  const models = [...new Set(results.map((result) => result.modelId))].sort();
  const tasks = [...new Set(capabilities.map((capability) => capability.taskType))].sort();

  lines.push("# Measured Model Capabilities");
  lines.push("");
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push("");
  lines.push(
    "Measured on this repository's own fixtures. This exists because seeding the router from public",
  );
  lines.push(
    "benchmark scores was measured to make routing *worse* — the mechanism was sound, the data",
  );
  lines.push(
    "described a different environment. These numbers describe the tasks you actually run.",
  );
  lines.push("");

  lines.push(`| Model | ${tasks.join(" | ")} |`);
  lines.push(`|---|${tasks.map(() => "---:").join("|")}|`);

  for (const modelId of models) {
    const cells = tasks.map((task) => {
      const match = capabilities.find(
        (capability) => capability.modelId === modelId && capability.taskType === task,
      );
      return match ? match.capability.toFixed(3) : "—";
    });
    lines.push(`| \`${modelId}\` | ${cells.join(" | ")} |`);
  }

  lines.push("");
  lines.push("## Failures");
  lines.push("");
  const failures = results.filter((result) => result.error !== null);
  if (failures.length === 0) {
    lines.push("None.");
  } else {
    // A failure is a measurement — a model that errors on a task is worse at that task.
    for (const failure of failures) {
      lines.push(`- \`${failure.modelId}\` on \`${failure.fixtureId}\`: ${failure.error}`);
    }
  }

  return lines.join("\n");
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
