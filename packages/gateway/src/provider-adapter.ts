import type {
  FinishReason,
  Message,
  ModelSpec,
  ProviderId,
  UnifiedChatChunk,
  UnifiedChatRequest,
  Usage,
} from "@orchestrator/shared";

/** Injected so no test ever touches the network. */
export type FetchLike = typeof globalThis.fetch;

/**
 * What an adapter returns: normalized content and token counts, nothing else.
 *
 * Cost, latency, ids, and attempt counting are deliberately absent — the gateway computes those
 * identically for every provider. An adapter that priced its own calls would let two providers drift
 * apart in the one number the router compares them on.
 */
export interface AdapterResult {
  message: Message;
  finishReason: FinishReason;
  usage: Usage;
}

export interface ProviderAdapter {
  readonly provider: ProviderId;

  chat(req: UnifiedChatRequest, spec: ModelSpec, signal: AbortSignal): Promise<AdapterResult>;

  stream(
    req: UnifiedChatRequest,
    spec: ModelSpec,
    signal: AbortSignal,
  ): AsyncIterable<UnifiedChatChunk>;
}

export interface AdapterConfig {
  apiKey: string;
  baseUrl?: string;
  fetchImpl?: FetchLike;
}

/** Flatten unified content into the plain string most provider text fields expect. */
export function contentToText(content: Message["content"]): string {
  if (typeof content === "string") return content;
  return content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("");
}

/**
 * Tool arguments arrive from providers as JSON strings and are exposed as parsed objects. A model can
 * emit malformed JSON, and that must surface as an empty object rather than crashing the request —
 * the finish reason and the caller's own validation are what should catch it.
 */
export function safeParseToolArguments(raw: string | undefined): Record<string, unknown> {
  if (!raw) return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}
