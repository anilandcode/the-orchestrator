import {
  type CallEvent,
  CallEventSchema,
  type UnifiedChatRequestInput,
  UnifiedChatRequestSchema,
  type UnifiedChatResponse,
  UnifiedChatResponseSchema,
} from "@orchestrator/shared";
import type { QualityInput } from "./scorer.js";

export function makeInput(
  overrides: {
    request?: UnifiedChatRequestInput;
    response?: Partial<UnifiedChatResponse>;
    event?: Partial<CallEvent>;
  } = {},
): QualityInput {
  const request = UnifiedChatRequestSchema.parse(
    overrides.request ?? { messages: [{ role: "user", content: "hi" }] },
  );

  const response = UnifiedChatResponseSchema.parse({
    id: "resp_1",
    requestId: "req_1",
    provider: "openai",
    modelId: "openai/gpt-4o-mini",
    message: { role: "assistant", content: "hello" },
    finishReason: "stop",
    usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15, cachedPromptTokens: 0 },
    costUsd: 0.0001,
    latencyMs: 400,
    attempts: 1,
    ...overrides.response,
  });

  const event = CallEventSchema.parse({
    id: "evt_1",
    tenantId: "local",
    requestId: "req_1",
    attempt: 1,
    provider: "openai",
    modelId: "openai/gpt-4o-mini",
    taskType: request.route.taskType,
    routeMode: request.route.mode,
    promptTokens: 10,
    completionTokens: 5,
    costUsd: 0.0001,
    latencyMs: 400,
    status: "success",
    finishReason: "stop",
    createdAt: 1_700_000_000_000,
    ...overrides.event,
  });

  return { request, response, event };
}
