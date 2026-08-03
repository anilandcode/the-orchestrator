import { z } from "zod";

/**
 * The unified wire contract. Every adapter translates to and from these shapes; nothing above the
 * gateway ever sees a provider-native payload.
 */

export const ProviderIdSchema = z.enum(["openai", "anthropic"]);
export type ProviderId = z.infer<typeof ProviderIdSchema>;

export const RoleSchema = z.enum(["system", "user", "assistant", "tool"]);
export type Role = z.infer<typeof RoleSchema>;

export const TextPartSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
});

export const ImagePartSchema = z.object({
  type: z.literal("image"),
  mediaType: z.string(),
  dataBase64: z.string(),
});

export const ContentPartSchema = z.discriminatedUnion("type", [TextPartSchema, ImagePartSchema]);
export type ContentPart = z.infer<typeof ContentPartSchema>;

/**
 * Tool call arguments are held as a parsed object, not a JSON string. OpenAI streams them as string
 * fragments and Anthropic sends structured input; normalizing to a parsed object here means callers
 * never have to know which one they were talking to.
 */
export const ToolCallSchema = z.object({
  id: z.string(),
  name: z.string(),
  arguments: z.record(z.string(), z.unknown()),
});
export type ToolCall = z.infer<typeof ToolCallSchema>;

export const MessageSchema = z.object({
  role: RoleSchema,
  content: z.union([z.string(), z.array(ContentPartSchema)]),
  /** Present on assistant messages that requested tools. */
  toolCalls: z.array(ToolCallSchema).optional(),
  /** Required on `role: "tool"` messages — links the result back to the call. */
  toolCallId: z.string().optional(),
  name: z.string().optional(),
});
export type Message = z.infer<typeof MessageSchema>;

export const ToolDefinitionSchema = z.object({
  name: z.string(),
  description: z.string().optional(),
  /** JSON Schema for the tool's parameters. Passed through to providers largely untouched. */
  parameters: z.record(z.string(), z.unknown()),
});
export type ToolDefinition = z.infer<typeof ToolDefinitionSchema>;

/** How the caller wants cost, quality, and speed traded off. Becomes the reward weights. */
export const RouteModeSchema = z.enum(["cheap", "balanced", "best"]);
export type RouteMode = z.infer<typeof RouteModeSchema>;

export const TaskTypeSchema = z.enum([
  "general",
  "code",
  "extraction",
  "summarization",
  "classification",
  "reasoning",
  "creative",
]);
export type TaskType = z.infer<typeof TaskTypeSchema>;
export const TASK_TYPES = TaskTypeSchema.options;

/**
 * The caller declares *intent*, not a model. This is the whole reason routing is possible: an API that
 * takes `model: "gpt-4o"` has already made the decision the router exists to make.
 */
export const RouteHintSchema = z.object({
  mode: RouteModeSchema.default("balanced"),
  taskType: TaskTypeSchema.default("general"),
  /** Hard ceiling for a single call. Models whose worst case exceeds it are filtered out. */
  maxCostUsd: z.number().positive().optional(),
  maxLatencyMs: z.number().positive().optional(),
  /** Escape hatch: bypass routing entirely and use this model. */
  pin: z.string().optional(),
  /**
   * JSON Schema the caller expects the response to satisfy.
   *
   * Optional, and purely a *quality signal* — the gateway does not enforce it. Declaring it lets the
   * schema validator grade the answer deterministically instead of falling back to the heuristic,
   * which is the difference between the bandit learning something and learning a constant.
   */
  outputSchema: z.record(z.string(), z.unknown()).optional(),
});
export type RouteHint = z.infer<typeof RouteHintSchema>;

export const UnifiedChatRequestSchema = z.object({
  tenantId: z.string().min(1).default("local"),
  requestId: z.string().optional(),
  messages: z.array(MessageSchema).min(1),
  tools: z.array(ToolDefinitionSchema).optional(),
  maxTokens: z.number().int().positive().optional(),
  temperature: z.number().min(0).max(2).optional(),
  stream: z.boolean().default(false),
  route: RouteHintSchema.default({}),
  /**
   * Opt-in conversational memory.
   *
   * Deliberately explicit rather than automatic: silently prepending recalled context to a prompt
   * changes what the model sees and what it costs, and a caller that did not ask for that should
   * not get it.
   */
  memory: z
    .object({
      sessionId: z.string().min(1),
      /** Set false to write this turn to memory without recalling anything into it. */
      recall: z.boolean().default(true),
      /** Set false to use memory for this turn without persisting it. */
      write: z.boolean().default(true),
    })
    .optional(),
  metadata: z.record(z.string(), z.string()).optional(),
});
export type UnifiedChatRequest = z.infer<typeof UnifiedChatRequestSchema>;
export type UnifiedChatRequestInput = z.input<typeof UnifiedChatRequestSchema>;

export const UsageSchema = z.object({
  promptTokens: z.number().int().nonnegative(),
  completionTokens: z.number().int().nonnegative(),
  totalTokens: z.number().int().nonnegative(),
  /** Cache-read prompt tokens, billed at a discount by both providers. */
  cachedPromptTokens: z.number().int().nonnegative().default(0),
});
export type Usage = z.infer<typeof UsageSchema>;

export const FinishReasonSchema = z.enum([
  "stop",
  "length",
  "tool_calls",
  "content_filter",
  "error",
]);
export type FinishReason = z.infer<typeof FinishReasonSchema>;

export const UnifiedChatResponseSchema = z.object({
  id: z.string(),
  requestId: z.string(),
  provider: ProviderIdSchema,
  modelId: z.string(),
  message: MessageSchema,
  finishReason: FinishReasonSchema,
  usage: UsageSchema,
  /** Always computed from the model registry, never read from the provider payload. */
  costUsd: z.number().nonnegative(),
  latencyMs: z.number().nonnegative(),
  /** Time to first token. Only populated for streamed calls. */
  ttftMs: z.number().nonnegative().optional(),
  /** How many provider calls it took, including retries and fallbacks. 1 means clean first try. */
  attempts: z.number().int().positive(),
  routingDecisionId: z.string().optional(),
});
export type UnifiedChatResponse = z.infer<typeof UnifiedChatResponseSchema>;

export const UnifiedChatChunkSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("text"), delta: z.string() }),
  z.object({
    type: z.literal("tool_call"),
    index: z.number().int().nonnegative(),
    id: z.string().optional(),
    name: z.string().optional(),
    /** Raw JSON fragment. Accumulate across chunks, then parse at `finish`. */
    argumentsDelta: z.string().optional(),
  }),
  z.object({
    type: z.literal("finish"),
    finishReason: FinishReasonSchema,
    usage: UsageSchema.optional(),
  }),
]);
export type UnifiedChatChunk = z.infer<typeof UnifiedChatChunkSchema>;
