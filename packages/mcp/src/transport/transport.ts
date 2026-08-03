import type { JsonRpcNotification, JsonRpcRequest } from "../protocol.js";

/**
 * How bytes reach a server.
 *
 * Kept behind an interface because the two transports have genuinely different failure modes — a
 * spawned process can die, an HTTP endpoint can 503 — and because tests need a third that does
 * neither.
 */
export interface McpTransport {
  readonly name: string;
  start(): Promise<void>;
  send(message: JsonRpcRequest | JsonRpcNotification): Promise<void>;
  /** Registers the sink for server->client messages. Called once, before `start`. */
  onMessage(handler: (raw: string) => void): void;
  /** Called when the transport dies on its own, so the client can fail pending requests. */
  onClose(handler: (reason: string) => void): void;
  close(): Promise<void>;
}

/**
 * In-process transport backed by a handler function.
 *
 * Not only a test double: it is how this system exposes its *own* capabilities as MCP tools without
 * spawning a process or opening a socket to talk to itself.
 */
export class InProcessTransport implements McpTransport {
  readonly name = "in-process";

  private messageHandler: ((raw: string) => void) | undefined;
  private closeHandler: ((reason: string) => void) | undefined;
  private closed = false;

  constructor(private readonly handle: (request: JsonRpcRequest) => Promise<unknown>) {}

  start(): Promise<void> {
    this.closed = false;
    return Promise.resolve();
  }

  async send(message: JsonRpcRequest | JsonRpcNotification): Promise<void> {
    if (this.closed) throw new Error("Transport is closed");
    // Notifications carry no id and expect no reply.
    if (!("id" in message)) return;

    const request = message;
    try {
      const result = await this.handle(request);
      this.deliver({ jsonrpc: "2.0", id: request.id, result });
    } catch (error) {
      this.deliver({
        jsonrpc: "2.0",
        id: request.id,
        error: { code: -32_603, message: (error as Error).message },
      });
    }
  }

  onMessage(handler: (raw: string) => void): void {
    this.messageHandler = handler;
  }

  onClose(handler: (reason: string) => void): void {
    this.closeHandler = handler;
  }

  close(): Promise<void> {
    this.closed = true;
    this.closeHandler?.("closed");
    return Promise.resolve();
  }

  private deliver(payload: unknown): void {
    // Async so a handler cannot re-enter the client synchronously mid-send.
    queueMicrotask(() => this.messageHandler?.(JSON.stringify(payload)));
  }
}
