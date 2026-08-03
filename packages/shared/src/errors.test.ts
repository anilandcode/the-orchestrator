import { describe, expect, it } from "vitest";
import {
  ERROR_CLASSES,
  ERROR_POLICY,
  OrchestratorError,
  classifyHttpStatus,
  toOrchestratorError,
} from "./errors.js";

describe("error policy", () => {
  it("defines a policy for every error class", () => {
    for (const cls of ERROR_CLASSES) {
      expect(ERROR_POLICY[cls]).toBeDefined();
    }
  });

  it("does not retry or fail over a malformed request — it is malformed everywhere", () => {
    expect(ERROR_POLICY.invalid_request).toEqual({ retryable: false, fallbackEligible: false });
  });

  it("does not fail content-filtered requests over to another provider", () => {
    // Automatically routing filtered content elsewhere is filter evasion, not resilience.
    expect(ERROR_POLICY.content_filter.fallbackEligible).toBe(false);
  });

  it("fails auth errors over but does not retry them", () => {
    // A bad key will not fix itself; a different provider may be configured correctly.
    expect(ERROR_POLICY.auth).toEqual({ retryable: false, fallbackEligible: true });
  });

  it("fails an over-long context over to a larger-context model", () => {
    expect(ERROR_POLICY.context_length_exceeded).toEqual({
      retryable: false,
      fallbackEligible: true,
    });
  });

  it("retries transient faults", () => {
    for (const cls of ["rate_limit", "timeout", "network", "provider_unavailable"] as const) {
      expect(ERROR_POLICY[cls].retryable).toBe(true);
    }
  });
});

describe("OrchestratorError", () => {
  it("exposes policy through the instance", () => {
    const err = new OrchestratorError("rate_limit", "slow down", { retryAfterMs: 1_200 });
    expect(err.retryable).toBe(true);
    expect(err.fallbackEligible).toBe(true);
    expect(err.retryAfterMs).toBe(1_200);
  });
});

describe("toOrchestratorError", () => {
  it("passes an already-classified error through untouched", () => {
    const original = new OrchestratorError("auth", "bad key");
    expect(toOrchestratorError(original)).toBe(original);
  });

  it("classifies an aborted request as a timeout", () => {
    const abort = new Error("The operation was aborted");
    abort.name = "AbortError";
    expect(toOrchestratorError(abort).errorClass).toBe("timeout");
  });

  it("classifies a failed fetch as a network fault, which is retryable", () => {
    const netErr = new TypeError("fetch failed");
    const classified = toOrchestratorError(netErr);
    expect(classified.errorClass).toBe("network");
    expect(classified.retryable).toBe(true);
  });

  it("falls back to the supplied class for anything unrecognized", () => {
    expect(toOrchestratorError(new Error("weird"), "provider_unavailable").errorClass).toBe(
      "provider_unavailable",
    );
  });

  it("handles non-Error throws", () => {
    expect(toOrchestratorError("just a string").message).toBe("just a string");
  });
});

describe("classifyHttpStatus", () => {
  it.each([
    [401, "auth"],
    [403, "auth"],
    [408, "timeout"],
    [429, "rate_limit"],
    [400, "invalid_request"],
    [422, "invalid_request"],
    [500, "provider_unavailable"],
    [503, "provider_unavailable"],
  ] as const)("maps %i to %s", (status, expected) => {
    expect(classifyHttpStatus(status)).toBe(expected);
  });
});
