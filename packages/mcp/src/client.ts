import { OrchestratorError } from "@orchestrator/shared";
import {
  type CallToolResult,
  CallToolResultSchema,
  InitializeResultSchema,
  JSONRPC_VERSION,
  JsonRpcResponseSchema,
  ListToolsResultSchema,
  MCP_PROTOCOL_VERSION,
  type McpTool,
} from "./protocol.js";
import type { McpTransport } from "./transport/transport.js";

export interface McpClientConfig {
  transport: McpTransport;
  /** Per-request ceiling. A hung server would otherwise hold a workflow node open. */
  requestTimeoutMs?: number;
  clientName?: string;
  clientVersion?: string;
}

interface Pending {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

/**
 * A client for one MCP server.
 *
 * Everything this returns originated with a third party, so nothing is trusted: responses are parsed
 * through schemas rather than cast, and every request is bounded by a timeout. A tool server that
 * hangs is a far more common failure than one that misbehaves outright.
 */
export class McpClient {
  private readonly transport: McpTransport;
  private readonly requestTimeoutMs: number;
  private readonly clientName: string;
  private readonly clientVersion: string;

  private readonly pending = new Map<string | number, Pending>();
  private nextId = 1;
  private ready = false;
  private serverName = "";

  constructor(config: McpClientConfig) {
    this.transport = config.transport;
    this.requestTimeoutMs = config.requestTimeoutMs ?? 30_000;
    this.clientName = config.clientName ?? "the-orchestrator";
    this.clientVersion = config.clientVersion ?? "0.1.0";

    this.transport.onMessage((raw) => this.receive(raw));
    this.transport.onClose((reason) => this.failAll(reason));
  }

  get name(): string {
    return this.serverName || this.transport.name;
  }

  get isReady(): boolean {
    return this.ready;
  }

  async connect(): Promise<void> {
    await this.transport.start();

    const result = InitializeResultSchema.parse(
      await this.request("initialize", {
        protocolVersion: MCP_PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: { name: this.clientName, version: this.clientVersion },
      }),
    );

    this.serverName = result.serverInfo.name || this.transport.name;

    // The spec requires this notification before normal operation; skipping it leaves some servers
    // refusing every subsequent call.
    await this.transport.send({ jsonrpc: JSONRPC_VERSION, method: "notifications/initialized" });
    this.ready = true;
  }

  /** Full catalog, following pagination. */
  async listTools(): Promise<McpTool[]> {
    const tools: McpTool[] = [];
    let cursor: string | undefined;

    do {
      const page = ListToolsResultSchema.parse(
        await this.request("tools/list", cursor ? { cursor } : {}),
      );
      tools.push(...page.tools);
      cursor = page.nextCursor;
      // A server that returns its own cursor forever would loop here; bound it.
      if (tools.length > 1_000) break;
    } while (cursor);

    return tools;
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<CallToolResult> {
    return CallToolResultSchema.parse(await this.request("tools/call", { name, arguments: args }));
  }

  async close(): Promise<void> {
    this.failAll("client closed");
    this.ready = false;
    await this.transport.close();
  }

  // --- internals ------------------------------------------------------------

  private request(method: string, params: unknown): Promise<unknown> {
    const id = this.nextId++;

    return new Promise<unknown>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(
          new OrchestratorError(
            "timeout",
            `MCP request ${method} to ${this.name} timed out after ${this.requestTimeoutMs}ms`,
          ),
        );
      }, this.requestTimeoutMs);

      // Unref so a pending tool call cannot keep the process alive on shutdown.
      timer.unref?.();
      this.pending.set(id, { resolve, reject, timer });

      this.transport
        .send({ jsonrpc: JSONRPC_VERSION, id, method, params })
        .catch((error: unknown) => {
          clearTimeout(timer);
          this.pending.delete(id);
          reject(error as Error);
        });
    });
  }

  private receive(raw: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      // A server writing non-JSON to stdout is misbehaving, but killing the client over it would
      // take down every in-flight call.
      return;
    }

    const response = JsonRpcResponseSchema.safeParse(parsed);
    // Server-initiated requests and notifications are not supported; ignoring them is correct
    // rather than an error, since we advertise no capabilities that would invite them.
    if (!response.success) return;

    const pending = this.pending.get(response.data.id);
    if (!pending) return;

    clearTimeout(pending.timer);
    this.pending.delete(response.data.id);

    if (response.data.error) {
      pending.reject(
        new OrchestratorError(
          "provider_unavailable",
          `MCP error from ${this.name}: ${response.data.error.message}`,
        ),
      );
      return;
    }

    pending.resolve(response.data.result);
  }

  /** Fail everything in flight. A dead transport will never answer, so waiting is pointless. */
  private failAll(reason: string): void {
    for (const [id, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(
        new OrchestratorError("provider_unavailable", `MCP transport closed: ${reason}`),
      );
      this.pending.delete(id);
    }
    this.ready = false;
  }
}
