import {
  AnthropicAdapter,
  Gateway,
  OpenAIAdapter,
  type ProviderAdapter,
} from "@orchestrator/gateway";
import {
  InMemoryCallEventSink,
  UnifiedChatRequestSchema,
  type UnifiedChatResponse,
} from "@orchestrator/shared";

/**
 * Live smoke test. **This spends real money** — a few hundredths of a cent per provider.
 *
 * Opt-in on purpose: the entire unit suite runs offline against mocked fetch, so this exists only to
 * answer the one question mocks cannot: do the real APIs still behave the way the adapters assume?
 * Wire formats drift, and a green mocked suite is no evidence against that.
 */

const PROMPT = "Reply with exactly the word: pong";

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
        "This script makes real API calls and costs a small amount of money.",
    );
    process.exit(1);
  }

  const sink = new InMemoryCallEventSink();
  const gateway = new Gateway({ adapters, sink, timeoutMs: 30_000 });

  const targets = gateway.availableModels().filter((spec) => spec.tier === "economy");
  console.log(`Calling ${targets.length} model(s) with an identical unified request.\n`);

  const responses: UnifiedChatResponse[] = [];
  let failures = 0;

  for (const spec of targets) {
    const request = UnifiedChatRequestSchema.parse({
      messages: [{ role: "user", content: PROMPT }],
      maxTokens: 16,
      route: { pin: spec.modelId },
    });

    try {
      const response = await gateway.chat(request, { modelId: spec.modelId });
      responses.push(response);
      console.log(
        `  OK   ${spec.modelId.padEnd(32)} ${String(response.latencyMs.toFixed(0)).padStart(6)}ms  ` +
          `$${response.costUsd.toFixed(8)}  ${response.usage.promptTokens}->${response.usage.completionTokens} tok  ` +
          `"${String(response.message.content).trim().slice(0, 40)}"`,
      );
    } catch (error) {
      failures += 1;
      console.error(`  FAIL ${spec.modelId.padEnd(32)} ${(error as Error).message}`);
    }
  }

  console.log(`\nEvents recorded: ${sink.events.length} (one per attempt, including retries)`);

  // The actual assertion: both providers must normalize into the same response shape. If this
  // drifts, everything above the gateway silently starts comparing unlike things.
  const shapesAgree = responses.every(
    (response) =>
      response.message.role === "assistant" &&
      typeof response.usage.promptTokens === "number" &&
      response.usage.promptTokens > 0 &&
      response.costUsd >= 0 &&
      response.attempts >= 1,
  );

  if (!shapesAgree) {
    console.error("\nFAIL: responses did not normalize to a consistent shape.");
    process.exit(1);
  }
  if (failures > 0) {
    console.error(`\nFAIL: ${failures} provider call(s) failed.`);
    process.exit(1);
  }

  const totalCost = responses.reduce((total, response) => total + response.costUsd, 0);
  console.log(`Total spend: $${totalCost.toFixed(8)}`);
  console.log("\nPASS: every provider normalized to the same response shape.");
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
