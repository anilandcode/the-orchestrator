import {
  type Clock,
  type IdGenerator,
  OrchestratorError,
  type ToolDefinition,
  systemClock,
  systemIds,
} from "@orchestrator/shared";
import type { McpClient } from "./client.js";
import { type ToolAuditLog, type ToolPolicies, evaluatePolicy } from "./policy.js";
import { type CallToolResult, type McpTool, contentToText } from "./protocol.js";

export interface RegisteredTool {
  /** `server:tool` — unique across servers. */
  qualifiedName: string;
  serverName: string;
  tool: McpTool;
}

export interface ToolRegistryConfig {
  audit: ToolAuditLog;
  policies?: ToolPolicies;
  clock?: Clock;
  ids?: IdGenerator;
  /** Cap on tool-result text handed back. Unbounded output would blow the context window. */
  maxResultChars?: number;
}

/** Descriptions are third-party text that ends up in prompts, so they are length-capped. */
const MAX_DESCRIPTION_CHARS = 1_000;

/**
 * Many MCP servers behind one catalog.
 *
 * Names are namespaced as `server:tool` because two servers will eventually both expose `search`,
 * and silently resolving that collision one way or the other is how an agent ends up calling the
 * wrong integration.
 */
export class ToolRegistry {
  private readonly clients = new Map<string, McpClient>();
  private readonly tools = new Map<string, RegisteredTool>();
  private readonly audit: ToolAuditLog;
  private readonly policies: ToolPolicies;
  private readonly clock: Clock;
  private readonly ids: IdGenerator;
  private readonly maxResultChars: number;

  constructor(config: ToolRegistryConfig) {
    this.audit = config.audit;
    this.policies = config.policies ?? {};
    this.clock = config.clock ?? systemClock;
    this.ids = config.ids ?? systemIds;
    this.maxResultChars = config.maxResultChars ?? 20_000;
  }

  /** Connect a server and index its catalog. Failure is contained to that server. */
  async register(client: McpClient): Promise<RegisteredTool[]> {
    if (!client.isReady) await client.connect();

    const serverName = client.name;
    this.clients.set(serverName, client);

    const discovered = await client.listTools();
    const registered: RegisteredTool[] = [];

    for (const tool of discovered) {
      const qualifiedName = `${serverName}:${tool.name}`;
      const entry: RegisteredTool = {
        qualifiedName,
        serverName,
        tool: {
          ...tool,
          ...(tool.description
            ? { description: tool.description.slice(0, MAX_DESCRIPTION_CHARS) }
            : {}),
        },
      };

      this.tools.set(qualifiedName, entry);
      registered.push(entry);
    }

    return registered;
  }

  /** The tools a tenant may actually use, in the shape the gateway expects. */
  toolsFor(tenantId: string): ToolDefinition[] {
    const policy = this.policies[tenantId];

    return [...this.tools.values()]
      .filter((entry) => evaluatePolicy(entry.qualifiedName, policy).allowed)
      .map((entry) => ({
        name: entry.qualifiedName,
        ...(entry.tool.description ? { description: entry.tool.description } : {}),
        parameters: entry.tool.inputSchema,
      }));
  }

  list(): RegisteredTool[] {
    return [...this.tools.values()];
  }

  /**
   * Invoke a tool on a tenant's behalf.
   *
   * Policy is checked here rather than at the caller, and the denial is audited before it throws —
   * a refused call is the one most worth having a record of.
   */
  async call(options: {
    tenantId: string;
    qualifiedName: string;
    arguments: Record<string, unknown>;
    runId?: string;
  }): Promise<{ text: string; isError: boolean; raw: CallToolResult }> {
    const startedAt = this.clock.monotonic();
    const decision = evaluatePolicy(options.qualifiedName, this.policies[options.tenantId]);

    if (!decision.allowed) {
      this.audit.record({
        id: this.ids.generate("tool"),
        tenantId: options.tenantId,
        tool: options.qualifiedName,
        runId: options.runId ?? null,
        allowed: false,
        denyReason: decision.reason,
        arguments: options.arguments,
        createdAt: this.clock.now(),
      });

      throw new OrchestratorError(
        "invalid_request",
        `Tool ${options.qualifiedName} is not permitted for this tenant: ${decision.reason}`,
      );
    }

    const entry = this.tools.get(options.qualifiedName);
    if (!entry) {
      throw new OrchestratorError("invalid_request", `Unknown tool: ${options.qualifiedName}`);
    }

    const client = this.clients.get(entry.serverName);
    if (!client) {
      throw new OrchestratorError(
        "provider_unavailable",
        `MCP server ${entry.serverName} is not connected`,
      );
    }

    try {
      const raw = await client.callTool(entry.tool.name, options.arguments);
      const text = contentToText(raw).slice(0, this.maxResultChars);

      this.audit.record({
        id: this.ids.generate("tool"),
        tenantId: options.tenantId,
        tool: options.qualifiedName,
        runId: options.runId ?? null,
        allowed: true,
        arguments: options.arguments,
        // A tool that reports its own failure is still a successful protocol exchange; the
        // distinction matters when reading the audit log.
        isError: raw.isError,
        latencyMs: this.clock.monotonic() - startedAt,
        createdAt: this.clock.now(),
      });

      return { text, isError: raw.isError, raw };
    } catch (error) {
      this.audit.record({
        id: this.ids.generate("tool"),
        tenantId: options.tenantId,
        tool: options.qualifiedName,
        runId: options.runId ?? null,
        allowed: true,
        arguments: options.arguments,
        isError: true,
        error: (error as Error).message,
        latencyMs: this.clock.monotonic() - startedAt,
        createdAt: this.clock.now(),
      });
      throw error;
    }
  }

  async close(): Promise<void> {
    await Promise.all([...this.clients.values()].map((client) => client.close()));
    this.clients.clear();
    this.tools.clear();
  }
}
