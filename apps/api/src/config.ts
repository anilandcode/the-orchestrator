import type { RouterMode } from "@orchestrator/router";

export interface ApiConfig {
  port: number;
  apiKey: string;
  defaultTenantId: string;
  dbPath: string;
  routerMode: RouterMode;
  linucbAlpha: number;
  coldStartPulls: number;
  openaiApiKey: string | undefined;
  anthropicApiKey: string | undefined;
  /** The judge costs real money per graded call, so it is opt-in. */
  judgeEnabled: boolean;
  judgeModel: string;
  judgeSampleRate: number;
  judgeMaxUsdPerHour: number;
  /** Memory retention. Null keeps items indefinitely. */
  memoryTtlMs: number | null;
  /** Uses the offline hashing embedder unless a real embedding model is configured. */
  embeddingModel: string | undefined;
  /** Per-tenant tool allow/deny. Absent means the tenant gets no tools at all. */
  toolPolicies: Record<string, { allow: string[]; deny?: string[] }>;
  /**
   * Seed the bandit from catalog benchmark priors.
   *
   * **Off by default, on evidence.** Simulation showed the seeding mechanism works — priors drawn
   * from ground truth cut early regret 27% — but that seeding from the shipped benchmark numbers
   * made routing worse, because those numbers do not describe the environment being routed in.
   * Enable this only once the benchmark file holds figures you have validated against your own
   * traffic. See `pnpm replay:simulate`.
   */
  catalogPriorsEnabled: boolean;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): ApiConfig {
  return {
    port: Number(env.PORT ?? 3000),
    apiKey: env.ORCHESTRATOR_API_KEY ?? "local-dev-key",
    defaultTenantId: env.DEFAULT_TENANT_ID ?? "local",
    dbPath: env.ORCHESTRATOR_DB_PATH ?? "./data/orchestrator.sqlite",
    // Defaults to shadow: the bandit does not steer real traffic until replay justifies it.
    routerMode: parseRouterMode(env.ROUTER_MODE),
    linucbAlpha: Number(env.ROUTER_LINUCB_ALPHA ?? 0.6),
    coldStartPulls: Number(env.ROUTER_COLD_START_PULLS ?? 25),
    openaiApiKey: env.OPENAI_API_KEY || undefined,
    anthropicApiKey: env.ANTHROPIC_API_KEY || undefined,
    // Off by default: enabling it starts billing for calls the user did not make.
    judgeEnabled: env.QUALITY_JUDGE_ENABLED === "true",
    judgeModel: env.QUALITY_JUDGE_MODEL ?? "openai/gpt-4o-mini",
    judgeSampleRate: Number(env.QUALITY_JUDGE_SAMPLE_RATE ?? 0.05),
    judgeMaxUsdPerHour: Number(env.QUALITY_JUDGE_MAX_USD_PER_HOUR ?? 1),
    memoryTtlMs: env.MEMORY_TTL_MS ? Number(env.MEMORY_TTL_MS) : null,
    embeddingModel: env.MEMORY_EMBEDDING_MODEL || undefined,
    toolPolicies: parseToolPolicies(env.TOOL_POLICIES),
    catalogPriorsEnabled: env.CATALOG_PRIORS_ENABLED === "true",
  };
}

function parseRouterMode(raw: string | undefined): RouterMode {
  if (raw === "static" || raw === "shadow" || raw === "adaptive") return raw;
  if (raw) {
    throw new Error(`Invalid ROUTER_MODE: ${raw}. Expected static | shadow | adaptive.`);
  }
  return "shadow";
}

/**
 * Tool policies as JSON, e.g. `{"local":{"allow":["files:read_*"]}}`.
 *
 * A malformed policy fails startup rather than falling back to a permissive default — silently
 * granting broader tool access than intended is the failure worth being loud about.
 */
function parseToolPolicies(
  raw: string | undefined,
): Record<string, { allow: string[]; deny?: string[] }> {
  if (!raw) return {};
  try {
    return JSON.parse(raw) as Record<string, { allow: string[]; deny?: string[] }>;
  } catch (error) {
    throw new Error(`Invalid TOOL_POLICIES JSON: ${(error as Error).message}`);
  }
}
