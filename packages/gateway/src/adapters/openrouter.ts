import type { AdapterConfig } from "../provider-adapter.js";
import { OpenAiCompatibleAdapter, type OpenAiCompatibleProfile } from "./openai-compatible.js";

export interface OpenRouterAdapterConfig extends AdapterConfig {
  /** Optional attribution shown on OpenRouter's public leaderboards. */
  appUrl?: string;
  appName?: string;
}

/**
 * OpenRouter — one key, several hundred models across providers.
 *
 * It speaks the OpenAI chat-completions format, so the wire translation is shared. What differs is
 * error shape (upstream failures arrive wrapped), billing-specific statuses, and one thing that
 * matters more than either.
 *
 * **This adapter never delegates routing, and that is deliberate.**
 *
 * OpenRouter accepts `models: [...]` and `route: "fallback"`, which make *it* choose the model. Using
 * them would be actively destructive here. The entire premise of this system is that model selection
 * is a learned decision with measurable regret; handing it upstream would replace the thing being
 * measured while leaving every `CallEvent` attributed to whichever model we believed we asked for.
 * The bandit would then learn from rewards earned by a model it never selected — silently, and in a
 * way no test downstream of the gateway could detect.
 *
 * So the body is built by `OpenAiCompatibleAdapter` and neither field is ever added.
 * `openrouter.test.ts` asserts their absence rather than trusting this comment.
 */
export const OPENROUTER_PROFILE: OpenAiCompatibleProfile = {
  provider: "openrouter",
  label: "OpenRouter",
  defaultBaseUrl: "https://openrouter.ai/api/v1",

  authHeaders: (apiKey) => ({ authorization: `Bearer ${apiKey}` }),

  refineError: (body, status, message) => {
    // Insufficient credits. Not retryable and not a malformed request; failing over to another
    // provider is the useful behaviour, which is what the `auth` class already encodes.
    if (status === 402) return "auth";

    // OpenRouter wraps the upstream provider's failure, so the outer status describes OpenRouter
    // while the inner code describes whoever actually refused. A 502 from an upstream is a
    // transient upstream fault, not a bad request to OpenRouter.
    const upstream = Number(body.error?.code);
    if (Number.isFinite(upstream) && upstream >= 500) return "provider_unavailable";

    if (/context length|too many tokens|maximum context/i.test(message)) {
      return "context_length_exceeded";
    }
    if (/moderation|flagged|content policy/i.test(message)) return "content_filter";

    // No provider on OpenRouter can serve this model right now — worth failing over, not retrying.
    if (/no (allowed |)providers/i.test(message)) return "provider_unavailable";

    return undefined;
  },

  decorateBody: (body, _request, stream) => {
    if (!stream) return;
    // Mirrors the OpenAI-compatible convention. Unverified against the live API — this repo has
    // never made a real call — and `pnpm smoke` is what would confirm it. If usage came back empty
    // on streamed calls, cost would silently read as zero, so it is worth checking first.
    body.stream_options = { include_usage: true };
  },
};

export class OpenRouterAdapter extends OpenAiCompatibleAdapter {
  constructor(config: OpenRouterAdapterConfig) {
    super(config, {
      ...OPENROUTER_PROFILE,
      extraHeaders: {
        ...(config.appUrl ? { "HTTP-Referer": config.appUrl } : {}),
        ...(config.appName ? { "X-Title": config.appName } : {}),
      },
    });
  }
}
