import {
  type ContentPart,
  type FinishReason,
  type Message,
  type ModelSpec,
  OrchestratorError,
  type ProviderId,
  type ToolCall,
  type UnifiedChatChunk,
  type UnifiedChatRequest,
  type Usage,
  classifyHttpStatus,
} from "@orchestrator/shared";
import {
  type AdapterConfig,
  type AdapterResult,
  type FetchLike,
  type ProviderAdapter,
  contentToText,
} from "../provider-adapter.js";
import { parseSseFrames } from "../sse.js";

// --- provider wire shapes (only the fields we consume) ----------------------

interface AnthropicContentBlock {
  type: string;
  text?: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
}

interface AnthropicUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

interface AnthropicMessageResponse {
  content?: AnthropicContentBlock[];
  stop_reason?: string | null;
  usage?: AnthropicUsage;
}

interface AnthropicStreamEvent {
  type?: string;
  index?: number;
  message?: { usage?: AnthropicUsage };
  delta?: {
    type?: string;
    text?: string;
    partial_json?: string;
    stop_reason?: string | null;
  };
  content_block?: AnthropicContentBlock;
  usage?: AnthropicUsage;
}

interface AnthropicErrorBody {
  error?: { type?: string; message?: string };
}

const DEFAULT_BASE_URL = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION = "2023-06-01";
/** Anthropic requires max_tokens. This is the ceiling used when the caller does not specify one. */
const DEFAULT_MAX_TOKENS = 4_096;

export class AnthropicAdapter implements ProviderAdapter {
  readonly provider: ProviderId = "anthropic";
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly fetchImpl: FetchLike;

  constructor(config: AdapterConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = (config.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, "");
    this.fetchImpl = config.fetchImpl ?? globalThis.fetch;
  }

  async chat(
    req: UnifiedChatRequest,
    spec: ModelSpec,
    signal: AbortSignal,
  ): Promise<AdapterResult> {
    const res = await this.post(this.buildBody(req, spec, false), signal, spec);
    const body = (await res.json()) as AnthropicMessageResponse;

    return {
      message: toUnifiedMessage(body.content ?? []),
      finishReason: toFinishReason(body.stop_reason),
      usage: toUsage(body.usage),
    };
  }

  async *stream(
    req: UnifiedChatRequest,
    spec: ModelSpec,
    signal: AbortSignal,
  ): AsyncGenerator<UnifiedChatChunk, void, undefined> {
    const res = await this.post(this.buildBody(req, spec, true), signal, spec);
    if (!res.body) {
      throw new OrchestratorError("unknown", "Anthropic stream returned no body", {
        provider: this.provider,
        modelId: spec.modelId,
      });
    }

    let finishReason: FinishReason = "stop";
    let inputTokens = 0;
    let cachedTokens = 0;
    let outputTokens = 0;

    for await (const frame of parseSseFrames(res.body)) {
      const event = JSON.parse(frame.data) as AnthropicStreamEvent;

      switch (event.type) {
        case "message_start": {
          // Input tokens are only reported once, at the very start.
          const usage = event.message?.usage;
          inputTokens = usage?.input_tokens ?? 0;
          cachedTokens = usage?.cache_read_input_tokens ?? 0;
          break;
        }

        case "content_block_start": {
          if (event.content_block?.type === "tool_use") {
            yield {
              type: "tool_call",
              index: event.index ?? 0,
              ...(event.content_block.id ? { id: event.content_block.id } : {}),
              ...(event.content_block.name ? { name: event.content_block.name } : {}),
            };
          }
          break;
        }

        case "content_block_delta": {
          if (event.delta?.type === "text_delta" && event.delta.text) {
            yield { type: "text", delta: event.delta.text };
          } else if (event.delta?.type === "input_json_delta" && event.delta.partial_json) {
            yield {
              type: "tool_call",
              index: event.index ?? 0,
              argumentsDelta: event.delta.partial_json,
            };
          }
          break;
        }

        case "message_delta": {
          if (event.delta?.stop_reason) finishReason = toFinishReason(event.delta.stop_reason);
          if (event.usage?.output_tokens !== undefined) outputTokens = event.usage.output_tokens;
          break;
        }

        default:
          break;
      }
    }

    // Same accounting as the non-streaming path: input_tokens excludes cache reads, but promptTokens
    // reports everything the model saw. Diverging here would under-cost cached streaming traffic.
    yield {
      type: "finish",
      finishReason,
      usage: toUsage({
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        cache_read_input_tokens: cachedTokens,
      }),
    };
  }

  // --- internals ------------------------------------------------------------

  private buildBody(
    req: UnifiedChatRequest,
    spec: ModelSpec,
    stream: boolean,
  ): Record<string, unknown> {
    const { system, messages } = splitSystem(req.messages);

    const body: Record<string, unknown> = {
      model: spec.providerModel,
      messages,
      // Required by the API, unlike OpenAI. Cap at the model's own limit so a caller asking for more
      // than the model supports fails our validation rather than the provider's.
      max_tokens: Math.min(req.maxTokens ?? DEFAULT_MAX_TOKENS, spec.maxOutputTokens),
    };

    if (system) body.system = system;
    if (req.temperature !== undefined) {
      // Anthropic's temperature range is 0..1; the unified contract allows 0..2.
      body.temperature = Math.min(req.temperature, 1);
    }
    if (req.tools?.length) {
      body.tools = req.tools.map((tool) => ({
        name: tool.name,
        ...(tool.description ? { description: tool.description } : {}),
        input_schema: tool.parameters,
      }));
    }
    if (stream) body.stream = true;

    return body;
  }

  private async post(
    body: Record<string, unknown>,
    signal: AbortSignal,
    spec: ModelSpec,
  ): Promise<Response> {
    const res = await this.fetchImpl(`${this.baseUrl}/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": this.apiKey,
        "anthropic-version": ANTHROPIC_VERSION,
      },
      body: JSON.stringify(body),
      signal,
    });

    if (!res.ok) throw await this.toError(res, spec);
    return res;
  }

  private async toError(res: Response, spec: ModelSpec): Promise<OrchestratorError> {
    const raw = await res.text().catch(() => "");
    let parsed: AnthropicErrorBody = {};
    try {
      parsed = JSON.parse(raw) as AnthropicErrorBody;
    } catch {
      // Non-JSON body; status alone drives classification.
    }

    const message = parsed.error?.message ?? raw.slice(0, 500) ?? res.statusText;
    const type = parsed.error?.type ?? "";

    let errorClass = classifyHttpStatus(res.status);
    if (type === "overloaded_error") errorClass = "provider_unavailable";
    else if (type === "rate_limit_error") errorClass = "rate_limit";
    else if (type === "authentication_error" || type === "permission_error") errorClass = "auth";

    // An over-long prompt arrives as a generic invalid_request_error, but it is fallback-eligible
    // onto a larger-context model where a genuinely malformed request is not.
    if (/prompt is too long|exceeds the maximum|too many tokens/i.test(message)) {
      errorClass = "context_length_exceeded";
    }

    const retryAfter = res.headers.get("retry-after");

    return new OrchestratorError(errorClass, `Anthropic: ${message}`, {
      provider: this.provider,
      modelId: spec.modelId,
      status: res.status,
      ...(retryAfter ? { retryAfterMs: Number(retryAfter) * 1000 } : {}),
    });
  }
}

// --- translation ------------------------------------------------------------

/**
 * Anthropic takes the system prompt as a top-level field rather than a message, and only accepts
 * alternating user/assistant roles. Tool results become `tool_result` blocks on a user message;
 * consecutive ones are merged, because two adjacent user messages are rejected.
 */
export function splitSystem(messages: Message[]): {
  system: string;
  messages: Record<string, unknown>[];
} {
  const systemParts: string[] = [];
  const out: Record<string, unknown>[] = [];

  for (const message of messages) {
    if (message.role === "system") {
      systemParts.push(contentToText(message.content));
      continue;
    }

    if (message.role === "tool") {
      const block = {
        type: "tool_result",
        tool_use_id: message.toolCallId,
        content: contentToText(message.content),
      };

      const previous = out.at(-1);
      if (previous && previous.role === "user" && Array.isArray(previous.content)) {
        (previous.content as unknown[]).push(block);
      } else {
        out.push({ role: "user", content: [block] });
      }
      continue;
    }

    out.push({ role: message.role, content: toAnthropicContent(message) });
  }

  return { system: systemParts.join("\n\n"), messages: out };
}

function toAnthropicContent(message: Message): unknown {
  const blocks: unknown[] = [];

  if (typeof message.content === "string") {
    if (message.content) blocks.push({ type: "text", text: message.content });
  } else {
    for (const part of message.content as ContentPart[]) {
      blocks.push(
        part.type === "text"
          ? { type: "text", text: part.text }
          : {
              type: "image",
              source: { type: "base64", media_type: part.mediaType, data: part.dataBase64 },
            },
      );
    }
  }

  for (const call of message.toolCalls ?? []) {
    blocks.push({ type: "tool_use", id: call.id, name: call.name, input: call.arguments });
  }

  // An assistant turn that is purely a tool call still needs a non-empty content array.
  return blocks.length > 0 ? blocks : [{ type: "text", text: "" }];
}

function toUnifiedMessage(blocks: AnthropicContentBlock[]): Message {
  const text = blocks
    .filter((block) => block.type === "text")
    .map((block) => block.text ?? "")
    .join("");

  const toolCalls: ToolCall[] = blocks
    .filter((block) => block.type === "tool_use")
    .map((block, index) => ({
      id: block.id ?? `call_${index}`,
      name: block.name ?? "",
      arguments: block.input ?? {},
    }));

  return {
    role: "assistant",
    content: text,
    ...(toolCalls.length ? { toolCalls } : {}),
  };
}

function toFinishReason(reason: string | null | undefined): FinishReason {
  switch (reason) {
    case "end_turn":
    case "stop_sequence":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_calls";
    case "refusal":
      return "content_filter";
    default:
      return "stop";
  }
}

function toUsage(usage: AnthropicUsage | undefined): Usage {
  // input_tokens excludes cache reads, but the unified contract reports promptTokens as the total
  // the model actually saw — otherwise cost math would undercount cached traffic.
  const cached = usage?.cache_read_input_tokens ?? 0;
  const uncached = usage?.input_tokens ?? 0;
  const promptTokens = uncached + cached;
  const completionTokens = usage?.output_tokens ?? 0;

  return {
    promptTokens,
    completionTokens,
    totalTokens: promptTokens + completionTokens,
    cachedPromptTokens: cached,
  };
}
