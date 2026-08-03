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
  safeParseToolArguments,
} from "../provider-adapter.js";
import { parseSseFrames } from "../sse.js";

// --- provider wire shapes (only the fields we consume) ----------------------

interface OpenAiToolCall {
  id?: string;
  index?: number;
  function?: { name?: string; arguments?: string };
}

interface OpenAiMessage {
  content?: string | null;
  tool_calls?: OpenAiToolCall[];
}

interface OpenAiChoice {
  message?: OpenAiMessage;
  delta?: OpenAiMessage;
  finish_reason?: string | null;
}

interface OpenAiUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
}

interface OpenAiChatResponse {
  choices?: OpenAiChoice[];
  usage?: OpenAiUsage;
}

interface OpenAiErrorBody {
  error?: { message?: string; type?: string; code?: string };
}

const DEFAULT_BASE_URL = "https://api.openai.com/v1";

export class OpenAIAdapter implements ProviderAdapter {
  readonly provider: ProviderId = "openai";
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
    const body = (await res.json()) as OpenAiChatResponse;

    const choice = body.choices?.[0];
    if (!choice) {
      throw new OrchestratorError("unknown", "OpenAI returned no choices", {
        provider: this.provider,
        modelId: spec.modelId,
      });
    }

    return {
      message: toUnifiedMessage(choice.message ?? {}),
      finishReason: toFinishReason(choice.finish_reason),
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
      throw new OrchestratorError("unknown", "OpenAI stream returned no body", {
        provider: this.provider,
        modelId: spec.modelId,
      });
    }

    let finishReason: FinishReason = "stop";
    let usage: Usage | undefined;

    for await (const frame of parseSseFrames(res.body)) {
      if (frame.data === "[DONE]") break;

      const payload = JSON.parse(frame.data) as OpenAiChatResponse;
      if (payload.usage) usage = toUsage(payload.usage);

      const choice = payload.choices?.[0];
      if (!choice) continue;

      if (choice.finish_reason) finishReason = toFinishReason(choice.finish_reason);

      const delta = choice.delta;
      if (delta?.content) {
        yield { type: "text", delta: delta.content };
      }

      for (const call of delta?.tool_calls ?? []) {
        yield {
          type: "tool_call",
          index: call.index ?? 0,
          ...(call.id ? { id: call.id } : {}),
          ...(call.function?.name ? { name: call.function.name } : {}),
          ...(call.function?.arguments ? { argumentsDelta: call.function.arguments } : {}),
        };
      }
    }

    yield { type: "finish", finishReason, ...(usage ? { usage } : {}) };
  }

  // --- internals ------------------------------------------------------------

  private buildBody(
    req: UnifiedChatRequest,
    spec: ModelSpec,
    stream: boolean,
  ): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model: spec.providerModel,
      messages: req.messages.map(toOpenAiMessage),
    };

    if (req.maxTokens !== undefined) body.max_completion_tokens = req.maxTokens;
    if (req.temperature !== undefined) body.temperature = req.temperature;
    if (req.tools?.length) {
      body.tools = req.tools.map((tool) => ({
        type: "function",
        function: {
          name: tool.name,
          ...(tool.description ? { description: tool.description } : {}),
          parameters: tool.parameters,
        },
      }));
    }
    if (stream) {
      body.stream = true;
      // Without this OpenAI omits usage entirely from streamed responses, and a call with no token
      // counts cannot be costed — which would silently poison the router's cost signal.
      body.stream_options = { include_usage: true };
    }

    return body;
  }

  private async post(
    body: Record<string, unknown>,
    signal: AbortSignal,
    spec: ModelSpec,
  ): Promise<Response> {
    const res = await this.fetchImpl(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
      signal,
    });

    if (!res.ok) throw await this.toError(res, spec);
    return res;
  }

  private async toError(res: Response, spec: ModelSpec): Promise<OrchestratorError> {
    const raw = await res.text().catch(() => "");
    let parsed: OpenAiErrorBody = {};
    try {
      parsed = JSON.parse(raw) as OpenAiErrorBody;
    } catch {
      // Non-JSON error body (proxy/CDN pages). Status alone drives classification.
    }

    const message = parsed.error?.message ?? raw.slice(0, 500) ?? res.statusText;
    const code = parsed.error?.code ?? "";

    // Refine before falling back to status: a context-length overflow arrives as a plain 400, but
    // unlike other 400s it is fallback-eligible onto a larger-context model.
    let errorClass = classifyHttpStatus(res.status);
    if (code === "context_length_exceeded" || /maximum context length/i.test(message)) {
      errorClass = "context_length_exceeded";
    } else if (code === "content_filter" || parsed.error?.type === "content_policy_violation") {
      errorClass = "content_filter";
    }

    const retryAfter = res.headers.get("retry-after");

    return new OrchestratorError(errorClass, `OpenAI: ${message}`, {
      provider: this.provider,
      modelId: spec.modelId,
      status: res.status,
      ...(retryAfter ? { retryAfterMs: Number(retryAfter) * 1000 } : {}),
    });
  }
}

// --- translation ------------------------------------------------------------

function toOpenAiMessage(message: Message): Record<string, unknown> {
  if (message.role === "tool") {
    return {
      role: "tool",
      tool_call_id: message.toolCallId,
      content: contentToText(message.content),
    };
  }

  const out: Record<string, unknown> = {
    role: message.role,
    content: toOpenAiContent(message.content),
  };

  if (message.toolCalls?.length) {
    out.tool_calls = message.toolCalls.map((call) => ({
      id: call.id,
      type: "function",
      function: { name: call.name, arguments: JSON.stringify(call.arguments) },
    }));
  }
  if (message.name) out.name = message.name;

  return out;
}

function toOpenAiContent(content: Message["content"]): unknown {
  if (typeof content === "string") return content;
  return content.map((part: ContentPart) =>
    part.type === "text"
      ? { type: "text", text: part.text }
      : {
          type: "image_url",
          image_url: { url: `data:${part.mediaType};base64,${part.dataBase64}` },
        },
  );
}

function toUnifiedMessage(message: OpenAiMessage): Message {
  const toolCalls: ToolCall[] = (message.tool_calls ?? []).map((call, index) => ({
    id: call.id ?? `call_${index}`,
    name: call.function?.name ?? "",
    arguments: safeParseToolArguments(call.function?.arguments),
  }));

  return {
    role: "assistant",
    content: message.content ?? "",
    ...(toolCalls.length ? { toolCalls } : {}),
  };
}

function toFinishReason(reason: string | null | undefined): FinishReason {
  switch (reason) {
    case "stop":
      return "stop";
    case "length":
      return "length";
    case "tool_calls":
    case "function_call":
      return "tool_calls";
    case "content_filter":
      return "content_filter";
    default:
      return "stop";
  }
}

function toUsage(usage: OpenAiUsage | undefined): Usage {
  const promptTokens = usage?.prompt_tokens ?? 0;
  const completionTokens = usage?.completion_tokens ?? 0;
  return {
    promptTokens,
    completionTokens,
    totalTokens: usage?.total_tokens ?? promptTokens + completionTokens,
    cachedPromptTokens: usage?.prompt_tokens_details?.cached_tokens ?? 0,
  };
}
