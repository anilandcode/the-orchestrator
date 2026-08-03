import {
  type ModelSpec,
  type UnifiedChatRequestInput,
  UnifiedChatRequestSchema,
  defaultRegistry,
} from "@orchestrator/shared";
import type { RoutingContext } from "./router.js";

export const ALL_MODELS: ModelSpec[] = defaultRegistry.list();

export const CHEAPEST = "openai/gpt-4o-mini";
export const PREMIUM = "anthropic/claude-opus-5";
export const HUGE_CONTEXT = "openai/gpt-4.1";

export function context(
  request: UnifiedChatRequestInput = { messages: [{ role: "user", content: "hi" }] },
  overrides: Partial<RoutingContext> = {},
): RoutingContext {
  return {
    request: UnifiedChatRequestSchema.parse(request),
    available: ALL_MODELS,
    estimatedPromptTokens: 500,
    ...overrides,
  };
}
