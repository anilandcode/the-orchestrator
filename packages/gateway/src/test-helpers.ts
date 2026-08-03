import type { ModelSpec, UnifiedChatRequest } from "@orchestrator/shared";
import { UnifiedChatRequestSchema, defaultRegistry } from "@orchestrator/shared";
import type { FetchLike } from "./provider-adapter.js";

/** Shared fixtures for offline adapter and gateway tests. */

export const OPENAI_SPEC: ModelSpec = defaultRegistry.require("openai/gpt-4o-mini");
export const ANTHROPIC_SPEC: ModelSpec = defaultRegistry.require("anthropic/claude-haiku-4-5");

export function request(overrides: Partial<UnifiedChatRequest> = {}): UnifiedChatRequest {
  return UnifiedChatRequestSchema.parse({
    messages: [{ role: "user", content: "hello" }],
    ...overrides,
  });
}

export function jsonResponse(
  body: unknown,
  init: { status?: number; headers?: Record<string, string> } = {},
): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { "content-type": "application/json", ...init.headers },
  });
}

export function sseResponse(frames: string[]): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      for (const frame of frames) controller.enqueue(encoder.encode(frame));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

export interface CapturedRequest {
  url: string;
  headers: Record<string, string>;
  body: Record<string, unknown>;
}

/** A fetch stub that records what was sent and replays queued responses in order. */
export function stubFetch(responses: Response[]): {
  fetchImpl: FetchLike;
  calls: CapturedRequest[];
} {
  const calls: CapturedRequest[] = [];
  const queue = [...responses];

  const fetchImpl: FetchLike = async (input, init) => {
    const headers: Record<string, string> = {};
    for (const [key, value] of Object.entries(init?.headers ?? {})) {
      headers[key.toLowerCase()] = String(value);
    }

    calls.push({
      url: String(input),
      headers,
      body: JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>,
    });

    const next = queue.shift();
    if (!next) throw new Error("stubFetch: no response queued for this call");
    return next;
  };

  return { fetchImpl, calls };
}
