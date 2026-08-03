import type {
  ModelSpec,
  RoutingDecision,
  RoutingOutcome,
  UnifiedChatRequest,
} from "@orchestrator/shared";

export interface RoutingContext {
  request: UnifiedChatRequest;
  /** Models the gateway can actually reach. Routing to an unreachable model wastes an attempt. */
  available: ModelSpec[];
  /** Estimated, not exact — real counts only exist after the provider replies. */
  estimatedPromptTokens: number;
  /** How deep into a conversation this is. A weak signal that longer threads are harder. */
  turnIndex?: number;
  /** Set when this request already failed elsewhere, so the router can avoid repeating a choice. */
  priorAttemptFailed?: boolean;
}

export interface Router {
  /** Runs on the request path, ahead of the model call. Must be cheap and synchronous. */
  select(context: RoutingContext): RoutingDecision;
  /** Feeds a settled outcome back. A no-op for the static router. */
  observe(outcome: RoutingOutcome): void;
}

/** Per-tenant overrides applied before any other rule. */
export interface TenantPolicy {
  /** When set, only these models may be used. */
  allowModels?: string[];
  denyModels?: string[];
  /** Force a model for a given task type regardless of mode. */
  pinByTaskType?: Partial<Record<string, string>>;
}

export type TenantPolicies = Record<string, TenantPolicy>;
