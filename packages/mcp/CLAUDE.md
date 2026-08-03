# packages/mcp

Model Context Protocol client, tool registry, and the policy layer around them.

## Owns

- `src/protocol.ts` — JSON-RPC framing and the MCP message shapes we use
- `src/transport/` — stdio and in-process transports
- `src/client.ts` — handshake, `tools/list`, `tools/call`
- `src/registry.ts` — many servers behind one namespaced catalog
- `src/policy.ts` — who may call what, and the audit trail

## Rules

- **An MCP server is untrusted input.** Tool names, descriptions, and schemas come from a third
  party and are frequently passed to a model. Treat every field as hostile: namespace names, cap
  sizes, and never let a description smuggle instructions into a system prompt unreviewed.
- **Deny by default.** A tenant gets no tools until a policy grants them. The failure mode of the
  opposite default is a customer's agent calling another customer's integration.
- **Every invocation is audited**, including denials. A tool layer without an audit trail cannot be
  operated in production, and the trail has to exist before the incident, not after.
- **Timeouts are mandatory.** A hung tool server would otherwise hold a workflow node open until the
  run's node timeout, burning wall-clock on every retry.
- Never import the gateway, router, or orchestrator. Tools are called *by* those layers, not the
  reverse.
