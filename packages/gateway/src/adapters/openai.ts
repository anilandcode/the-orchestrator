import type { AdapterConfig } from "../provider-adapter.js";
import { OpenAiCompatibleAdapter, type OpenAiCompatibleProfile } from "./openai-compatible.js";

/**
 * OpenAI.
 *
 * The wire format lives in `openai-compatible.ts` because several providers speak it. What remains
 * here is only what OpenAI does differently from the others.
 */
export const OPENAI_PROFILE: OpenAiCompatibleProfile = {
  provider: "openai",
  label: "OpenAI",
  defaultBaseUrl: "https://api.openai.com/v1",

  authHeaders: (apiKey) => ({ authorization: `Bearer ${apiKey}` }),

  refineError: (body, _status, message) => {
    const code = String(body.error?.code ?? "");

    // A context-length overflow arrives as a plain 400, but unlike other 400s it is
    // fallback-eligible onto a larger-context model — so it must not be classified by status alone.
    if (code === "context_length_exceeded" || /maximum context length/i.test(message)) {
      return "context_length_exceeded";
    }
    if (code === "content_filter" || body.error?.type === "content_policy_violation") {
      return "content_filter";
    }
    return undefined;
  },

  decorateBody: (body, _request, stream) => {
    if (!stream) return;
    // Without this OpenAI omits usage entirely from streamed responses, and a call with no token
    // counts cannot be costed — which would silently poison the router's cost signal.
    body.stream_options = { include_usage: true };
  },
};

export class OpenAIAdapter extends OpenAiCompatibleAdapter {
  constructor(config: AdapterConfig) {
    super(config, OPENAI_PROFILE);
  }
}
