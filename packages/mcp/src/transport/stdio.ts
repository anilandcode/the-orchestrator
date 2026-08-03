import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import type { JsonRpcNotification, JsonRpcRequest } from "../protocol.js";
import type { McpTransport } from "./transport.js";

export interface StdioTransportConfig {
  command: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
}

/**
 * Newline-delimited JSON over a child process's stdio — the standard MCP local transport.
 *
 * Two things this deliberately does:
 *
 *   - **Does not inherit the parent environment.** A tool server gets exactly the variables it was
 *     configured with. Passing `process.env` wholesale would hand every provider key in this process
 *     to a third-party binary.
 *   - **Surfaces stderr rather than swallowing it.** A server that fails during startup usually says
 *     why on stderr and then exits; without this the only symptom is a handshake timeout.
 */
export class StdioTransport implements McpTransport {
  readonly name: string;

  private child: ChildProcessWithoutNullStreams | undefined;
  private buffer = "";
  private messageHandler: ((raw: string) => void) | undefined;
  private closeHandler: ((reason: string) => void) | undefined;
  private stderrTail: string[] = [];

  constructor(private readonly config: StdioTransportConfig) {
    this.name = `stdio:${config.command}`;
  }

  start(): Promise<void> {
    const child = spawn(this.config.command, this.config.args ?? [], {
      // Explicit, minimal environment — never the parent's.
      env: { PATH: process.env.PATH ?? "", ...this.config.env },
      ...(this.config.cwd ? { cwd: this.config.cwd } : {}),
      stdio: ["pipe", "pipe", "pipe"],
    }) as ChildProcessWithoutNullStreams;

    this.child = child;

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      this.buffer += chunk;

      let newline = this.buffer.indexOf("\n");
      while (newline !== -1) {
        const line = this.buffer.slice(0, newline).trim();
        this.buffer = this.buffer.slice(newline + 1);
        if (line) this.messageHandler?.(line);
        newline = this.buffer.indexOf("\n");
      }
    });

    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      // Bounded: a chatty server must not grow this without limit.
      this.stderrTail.push(chunk);
      if (this.stderrTail.length > 20) this.stderrTail.shift();
    });

    child.on("exit", (code) => {
      const detail = this.stderrTail.join("").trim().slice(-500);
      this.closeHandler?.(`server exited with code ${code}${detail ? `: ${detail}` : ""}`);
    });

    child.on("error", (error) => this.closeHandler?.(error.message));

    return Promise.resolve();
  }

  send(message: JsonRpcRequest | JsonRpcNotification): Promise<void> {
    const child = this.child;
    if (!child || child.exitCode !== null) {
      return Promise.reject(new Error(`MCP transport ${this.name} is not running`));
    }

    child.stdin.write(`${JSON.stringify(message)}\n`);
    return Promise.resolve();
  }

  onMessage(handler: (raw: string) => void): void {
    this.messageHandler = handler;
  }

  onClose(handler: (reason: string) => void): void {
    this.closeHandler = handler;
  }

  close(): Promise<void> {
    this.child?.kill();
    this.child = undefined;
    return Promise.resolve();
  }
}
