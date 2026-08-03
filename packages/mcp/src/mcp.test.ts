import { OrchestratorError, createFixedClock, createSequentialIds } from "@orchestrator/shared";
import { beforeEach, describe, expect, it } from "vitest";
import { McpClient } from "./client.js";
import { InMemoryToolAuditLog, evaluatePolicy, matches } from "./policy.js";
import type { JsonRpcRequest } from "./protocol.js";
import { ToolRegistry } from "./registry.js";
import { InProcessTransport } from "./transport/transport.js";

/** A minimal conforming MCP server, so the client is tested against the protocol, not a mock. */
function fakeServer(
  options: {
    name?: string;
    tools?: { name: string; description?: string; inputSchema?: Record<string, unknown> }[];
    onCall?: (name: string, args: Record<string, unknown>) => unknown;
    pages?: boolean;
  } = {},
) {
  const tools = options.tools ?? [
    { name: "search", description: "Search the corpus", inputSchema: { type: "object" } },
  ];
  const calls: { name: string; args: Record<string, unknown> }[] = [];

  const transport = new InProcessTransport(async (request: JsonRpcRequest) => {
    switch (request.method) {
      case "initialize":
        return {
          protocolVersion: "2025-06-18",
          capabilities: { tools: {} },
          serverInfo: { name: options.name ?? "fake", version: "1.0.0" },
        };

      case "tools/list": {
        if (!options.pages) return { tools };
        const cursor = (request.params as { cursor?: string } | undefined)?.cursor;
        return cursor
          ? { tools: [{ name: "second-page-tool", inputSchema: {} }] }
          : { tools, nextCursor: "page-2" };
      }

      case "tools/call": {
        const params = request.params as { name: string; arguments: Record<string, unknown> };
        calls.push({ name: params.name, args: params.arguments });
        const result = options.onCall?.(params.name, params.arguments);
        if (result instanceof Error) throw result;
        return result ?? { content: [{ type: "text", text: "done" }] };
      }

      default:
        throw new Error(`unexpected method ${request.method}`);
    }
  });

  return { transport, calls };
}

describe("McpClient", () => {
  it("completes the initialize handshake and reports the server name", async () => {
    const { transport } = fakeServer({ name: "corpus-server" });
    const client = new McpClient({ transport });

    await client.connect();
    expect(client.isReady).toBe(true);
    expect(client.name).toBe("corpus-server");
  });

  it("lists tools", async () => {
    const { transport } = fakeServer();
    const client = new McpClient({ transport });
    await client.connect();

    const tools = await client.listTools();
    expect(tools.map((t) => t.name)).toEqual(["search"]);
  });

  it("follows pagination", async () => {
    const { transport } = fakeServer({ pages: true });
    const client = new McpClient({ transport });
    await client.connect();

    expect((await client.listTools()).map((t) => t.name)).toEqual(["search", "second-page-tool"]);
  });

  it("calls a tool and returns its content", async () => {
    const { transport, calls } = fakeServer({
      onCall: () => ({ content: [{ type: "text", text: "42 results" }] }),
    });
    const client = new McpClient({ transport });
    await client.connect();

    const result = await client.callTool("search", { q: "orchestrator" });
    expect(result.content[0]?.text).toBe("42 results");
    expect(calls[0]?.args).toEqual({ q: "orchestrator" });
  });

  it("preserves isError, which is a tool failure rather than a protocol failure", async () => {
    // MCP reports tool-level failures in the result so the model can react to them. Turning that
    // into a thrown error would hide a recoverable failure from the model.
    const { transport } = fakeServer({
      onCall: () => ({ content: [{ type: "text", text: "no such record" }], isError: true }),
    });
    const client = new McpClient({ transport });
    await client.connect();

    const result = await client.callTool("search", {});
    expect(result.isError).toBe(true);
  });

  it("surfaces a JSON-RPC error as a classified error", async () => {
    const { transport } = fakeServer({ onCall: () => new Error("upstream exploded") });
    const client = new McpClient({ transport });
    await client.connect();

    await expect(client.callTool("search", {})).rejects.toThrow(/upstream exploded/);
  });

  it("times out rather than hanging a workflow node forever", async () => {
    // The most common tool-server failure is not misbehaviour, it is silence.
    const transport = new InProcessTransport(async (request) => {
      if (request.method === "initialize") {
        return { protocolVersion: "2025-06-18", capabilities: {}, serverInfo: { name: "slow" } };
      }
      return new Promise(() => {
        // never resolves
      });
    });

    const client = new McpClient({ transport, requestTimeoutMs: 40 });
    await client.connect();

    const error = (await client.callTool("x", {}).catch((e: unknown) => e)) as OrchestratorError;
    expect(error).toBeInstanceOf(OrchestratorError);
    expect(error.errorClass).toBe("timeout");
  });

  it("fails in-flight requests when the transport dies", async () => {
    const transport = new InProcessTransport(async (request) => {
      if (request.method === "initialize") {
        return { protocolVersion: "2025-06-18", capabilities: {}, serverInfo: { name: "dying" } };
      }
      return new Promise(() => {});
    });

    const client = new McpClient({ transport, requestTimeoutMs: 5_000 });
    await client.connect();

    const pending = client.callTool("x", {});
    await transport.close();

    // A dead transport will never answer; waiting for the timeout would waste the node's budget.
    await expect(pending).rejects.toThrow(/transport closed/i);
  });

  it("ignores non-JSON output rather than killing every in-flight call", async () => {
    // Servers do log stray text to stdout. Treating that as fatal would take down every call
    // already in flight over what is, at worst, a cosmetic bug in the server.
    let deliver: ((raw: string) => void) | undefined;

    const transport = new InProcessTransport(async (request) => {
      if (request.method === "initialize") {
        return { protocolVersion: "2025-06-18", capabilities: {}, serverInfo: { name: "chatty" } };
      }
      return { tools: [{ name: "search", inputSchema: {} }] };
    });

    const originalOnMessage = transport.onMessage.bind(transport);
    transport.onMessage = (handler) => {
      deliver = handler;
      originalOnMessage(handler);
    };

    const client = new McpClient({ transport });
    await client.connect();

    // Simulate the server writing a log line to stdout mid-session.
    expect(() => deliver?.("starting up, listening on stdio")).not.toThrow();
    expect(await client.listTools()).toHaveLength(1);
  });
});

describe("policy matching", () => {
  it("matches exact names and wildcards", () => {
    expect(matches("github:create_issue", "github:create_issue")).toBe(true);
    expect(matches("github:*", "github:create_issue")).toBe(true);
    expect(matches("*", "anything:at_all")).toBe(true);
    expect(matches("github:*", "gitlab:create_issue")).toBe(false);
  });

  it("does not let regex metacharacters in a pattern widen it", () => {
    // A tool literally named "a.b" must not be matched by the pattern "a.b" interpreted as regex.
    expect(matches("srv:a.b", "srv:axb")).toBe(false);
    expect(matches("srv:a.b", "srv:a.b")).toBe(true);
  });
});

describe("evaluatePolicy", () => {
  it("denies everything when a tenant has no policy", () => {
    // Deny by default: the opposite lets one customer's agent reach another's integration.
    const outcome = evaluatePolicy("github:create_issue", undefined);
    expect(outcome.allowed).toBe(false);
  });

  it("denies anything not covered by an allow pattern", () => {
    const outcome = evaluatePolicy("github:delete_repo", { allow: ["github:read_*"] });
    expect(outcome.allowed).toBe(false);
  });

  it("allows a covered tool", () => {
    expect(evaluatePolicy("github:read_issue", { allow: ["github:read_*"] }).allowed).toBe(true);
  });

  it("lets deny override allow", () => {
    // A policy whose precedence could be argued is one nobody can reason about in an incident.
    const outcome = evaluatePolicy("github:delete_repo", {
      allow: ["github:*"],
      deny: ["github:delete_*"],
    });
    expect(outcome.allowed).toBe(false);
  });

  it("flags tools that require approval", () => {
    const outcome = evaluatePolicy("bank:transfer", {
      allow: ["bank:*"],
      requireApproval: ["bank:transfer"],
    });
    expect(outcome).toEqual({ allowed: true, requiresApproval: true });
  });
});

describe("ToolRegistry", () => {
  let audit: InMemoryToolAuditLog;

  const build = (policies = { acme: { allow: ["*"] } }) =>
    new ToolRegistry({
      audit,
      policies,
      clock: createFixedClock(),
      ids: createSequentialIds(),
    });

  beforeEach(() => {
    audit = new InMemoryToolAuditLog();
  });

  it("namespaces tools by server so two servers can expose the same name", async () => {
    // Silently resolving a collision is how an agent calls the wrong integration.
    const registry = build();
    await registry.register(new McpClient({ transport: fakeServer({ name: "github" }).transport }));
    await registry.register(new McpClient({ transport: fakeServer({ name: "gitlab" }).transport }));

    expect(
      registry
        .list()
        .map((t) => t.qualifiedName)
        .sort(),
    ).toEqual(["github:search", "gitlab:search"]);
  });

  it("exposes only the tools a tenant's policy allows", async () => {
    const registry = build({ acme: { allow: ["github:*"] } });
    await registry.register(new McpClient({ transport: fakeServer({ name: "github" }).transport }));
    await registry.register(
      new McpClient({ transport: fakeServer({ name: "secrets" }).transport }),
    );

    expect(registry.toolsFor("acme").map((t) => t.name)).toEqual(["github:search"]);
    // A tenant with no policy sees nothing at all.
    expect(registry.toolsFor("unknown-tenant")).toEqual([]);
  });

  it("calls an allowed tool and returns flattened text", async () => {
    const registry = build();
    const server = fakeServer({
      name: "github",
      onCall: () => ({ content: [{ type: "text", text: "issue #4 created" }] }),
    });
    await registry.register(new McpClient({ transport: server.transport }));

    const result = await registry.call({
      tenantId: "acme",
      qualifiedName: "github:search",
      arguments: { q: "bug" },
    });

    expect(result.text).toBe("issue #4 created");
    expect(server.calls[0]?.name).toBe("search");
  });

  it("refuses a tool outside the policy and audits the refusal", async () => {
    const registry = build({ acme: { allow: ["github:read_*"] } });
    await registry.register(new McpClient({ transport: fakeServer({ name: "github" }).transport }));

    await expect(
      registry.call({ tenantId: "acme", qualifiedName: "github:search", arguments: {} }),
    ).rejects.toThrow(/not permitted/);

    // Denials are the more interesting half of the audit log.
    const entry = audit.entries[0];
    expect(entry?.allowed).toBe(false);
    expect(entry?.denyReason).toMatch(/allow pattern/);
    expect(entry?.tool).toBe("github:search");
  });

  it("audits successful calls with latency", async () => {
    const registry = build();
    await registry.register(new McpClient({ transport: fakeServer({ name: "github" }).transport }));

    await registry.call({
      tenantId: "acme",
      qualifiedName: "github:search",
      arguments: { q: "x" },
      runId: "run_1",
    });

    const entry = audit.entries[0];
    expect(entry?.allowed).toBe(true);
    expect(entry?.isError).toBe(false);
    expect(entry?.runId).toBe("run_1");
  });

  it("audits a tool that fails, then rethrows", async () => {
    const registry = build();
    await registry.register(
      new McpClient({
        transport: fakeServer({ name: "github", onCall: () => new Error("boom") }).transport,
      }),
    );

    await expect(
      registry.call({ tenantId: "acme", qualifiedName: "github:search", arguments: {} }),
    ).rejects.toThrow(/boom/);

    expect(audit.entries[0]?.isError).toBe(true);
    expect(audit.entries[0]?.error).toMatch(/boom/);
  });

  it("rejects an unknown tool", async () => {
    const registry = build();
    await expect(
      registry.call({ tenantId: "acme", qualifiedName: "ghost:tool", arguments: {} }),
    ).rejects.toThrow(/Unknown tool/);
  });

  it("caps oversized tool output so it cannot blow the context window", async () => {
    const registry = new ToolRegistry({
      audit,
      policies: { acme: { allow: ["*"] } },
      clock: createFixedClock(),
      ids: createSequentialIds(),
      maxResultChars: 50,
    });

    await registry.register(
      new McpClient({
        transport: fakeServer({
          name: "noisy",
          onCall: () => ({ content: [{ type: "text", text: "x".repeat(5_000) }] }),
        }).transport,
      }),
    );

    const result = await registry.call({
      tenantId: "acme",
      qualifiedName: "noisy:search",
      arguments: {},
    });
    expect(result.text).toHaveLength(50);
  });

  it("truncates a hostile tool description rather than passing it to a model whole", async () => {
    // Descriptions are third-party text that lands in prompts.
    const registry = build();
    await registry.register(
      new McpClient({
        transport: fakeServer({
          name: "evil",
          tools: [{ name: "t", description: "z".repeat(50_000), inputSchema: {} }],
        }).transport,
      }),
    );

    expect(registry.list()[0]?.tool.description?.length).toBeLessThanOrEqual(1_000);
  });
});
