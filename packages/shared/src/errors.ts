/**
 * Normalized error taxonomy.
 *
 * Provider-specific status codes must be translated into one of these classes inside the adapter that
 * produced them. Nothing above the gateway should ever branch on an HTTP status: the retry loop and the
 * router's fallback chain both key off the policy table below.
 */

export const ERROR_CLASSES = [
  "rate_limit",
  "timeout",
  "auth",
  "invalid_request",
  "provider_unavailable",
  "content_filter",
  "context_length_exceeded",
  "network",
  "unknown",
] as const;

export type ErrorClass = (typeof ERROR_CLASSES)[number];

export interface ErrorPolicy {
  /** Retry the *same* model. Only for faults that are plausibly transient. */
  retryable: boolean;
  /** Advance to the next model in the fallback chain. */
  fallbackEligible: boolean;
}

export const ERROR_POLICY: Readonly<Record<ErrorClass, ErrorPolicy>> = Object.freeze({
  rate_limit: { retryable: true, fallbackEligible: true },
  timeout: { retryable: true, fallbackEligible: true },
  network: { retryable: true, fallbackEligible: true },
  provider_unavailable: { retryable: true, fallbackEligible: true },

  // A bad key will not fix itself on retry, but a different provider may well be configured correctly.
  auth: { retryable: false, fallbackEligible: true },

  // A malformed request is malformed everywhere. Failing it over just burns money twice.
  invalid_request: { retryable: false, fallbackEligible: false },

  // Deliberately NOT fallback-eligible: automatically re-running filtered content against a second
  // provider is filter evasion, and it is not a default this system should ship with.
  content_filter: { retryable: false, fallbackEligible: false },

  // Not retryable on the same model, but a larger-context model can genuinely serve this.
  context_length_exceeded: { retryable: false, fallbackEligible: true },

  unknown: { retryable: false, fallbackEligible: true },
});

export interface OrchestratorErrorOptions {
  provider?: string;
  modelId?: string;
  /** Original HTTP status, kept for logging only — never branch on it upstream. */
  status?: number;
  /** Honoured by the retry loop when a provider tells us how long to wait. */
  retryAfterMs?: number;
  cause?: unknown;
}

export class OrchestratorError extends Error {
  readonly errorClass: ErrorClass;
  readonly provider: string | undefined;
  readonly modelId: string | undefined;
  readonly status: number | undefined;
  readonly retryAfterMs: number | undefined;

  constructor(errorClass: ErrorClass, message: string, options: OrchestratorErrorOptions = {}) {
    super(message, options.cause !== undefined ? { cause: options.cause } : undefined);
    this.name = "OrchestratorError";
    this.errorClass = errorClass;
    this.provider = options.provider;
    this.modelId = options.modelId;
    this.status = options.status;
    this.retryAfterMs = options.retryAfterMs;
  }

  get retryable(): boolean {
    return ERROR_POLICY[this.errorClass].retryable;
  }

  get fallbackEligible(): boolean {
    return ERROR_POLICY[this.errorClass].fallbackEligible;
  }
}

/** Coerce anything thrown into a classified error, so callers never see a raw provider exception. */
export function toOrchestratorError(
  err: unknown,
  fallbackClass: ErrorClass = "unknown",
  context: OrchestratorErrorOptions = {},
): OrchestratorError {
  if (err instanceof OrchestratorError) return err;

  if (err instanceof Error) {
    // AbortController surfaces timeouts as AbortError / TimeoutError.
    if (err.name === "AbortError" || err.name === "TimeoutError") {
      return new OrchestratorError("timeout", err.message || "request aborted", {
        ...context,
        cause: err,
      });
    }
    // undici throws TypeError('fetch failed') with the real reason on .cause.
    if (err.name === "TypeError" && /fetch failed|network|socket/i.test(err.message)) {
      return new OrchestratorError("network", err.message, { ...context, cause: err });
    }
    return new OrchestratorError(fallbackClass, err.message, { ...context, cause: err });
  }

  return new OrchestratorError(fallbackClass, String(err), { ...context, cause: err });
}

/**
 * Map an HTTP status to an error class. Adapters should refine this using the provider's own error
 * body (e.g. distinguishing a context-length 400 from a plain malformed-payload 400) before falling
 * back to this table.
 */
export function classifyHttpStatus(status: number): ErrorClass {
  if (status === 401 || status === 403) return "auth";
  if (status === 408) return "timeout";
  if (status === 429) return "rate_limit";
  if (status >= 500) return "provider_unavailable";
  if (status >= 400) return "invalid_request";
  return "unknown";
}
