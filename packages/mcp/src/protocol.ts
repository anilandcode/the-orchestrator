import { z } from "zod";

/**
 * The subset of MCP this client speaks.
 *
 * Deliberately partial: tool discovery and invocation. Resources, prompts, and sampling are not
 * implemented, and their absence is stated rather than stubbed — a stub that silently returns empty
 * is worse than a method that does not exist.
 */

export const MCP_PROTOCOL_VERSION = "2025-06-18";
export const JSONRPC_VERSION = "2.0";

export const JsonRpcErrorSchema = z.object({
  code: z.number(),
  message: z.string(),
  data: z.unknown().optional(),
});

export const JsonRpcResponseSchema = z.object({
  jsonrpc: z.literal(JSONRPC_VERSION),
  id: z.union([z.string(), z.number()]),
  result: z.unknown().optional(),
  error: JsonRpcErrorSchema.optional(),
});

export type JsonRpcResponse = z.infer<typeof JsonRpcResponseSchema>;

export interface JsonRpcRequest {
  jsonrpc: typeof JSONRPC_VERSION;
  id: string | number;
  method: string;
  params?: unknown;
}

export interface JsonRpcNotification {
  jsonrpc: typeof JSONRPC_VERSION;
  method: string;
  params?: unknown;
}

/** A tool as the server describes it. Every field here is third-party input. */
export const McpToolSchema = z.object({
  name: z.string().min(1),
  description: z.string().optional(),
  inputSchema: z.record(z.string(), z.unknown()).default({}),
});
export type McpTool = z.infer<typeof McpToolSchema>;

export const ListToolsResultSchema = z.object({
  tools: z.array(McpToolSchema).default([]),
  nextCursor: z.string().optional(),
});

export const ContentBlockSchema = z.object({
  type: z.string(),
  text: z.string().optional(),
  data: z.string().optional(),
  mimeType: z.string().optional(),
});

export const CallToolResultSchema = z.object({
  content: z.array(ContentBlockSchema).default([]),
  /**
   * MCP reports tool-level failures here rather than as a JSON-RPC error, because the model is
   * meant to see and react to them. Conflating the two would hide recoverable failures from the
   * model and surface protocol bugs to it instead.
   */
  isError: z.boolean().default(false),
});
export type CallToolResult = z.infer<typeof CallToolResultSchema>;

export const InitializeResultSchema = z.object({
  protocolVersion: z.string(),
  capabilities: z.record(z.string(), z.unknown()).default({}),
  serverInfo: z
    .object({ name: z.string().default(""), version: z.string().default("") })
    .default({ name: "", version: "" }),
});

/** Flatten tool output to text for a model or a workflow variable. */
export function contentToText(result: CallToolResult): string {
  return result.content
    .map((block) => {
      if (block.type === "text") return block.text ?? "";
      // Non-text blocks are summarized rather than inlined: dumping base64 image bytes into a
      // prompt burns the context window for no benefit.
      return `[${block.type}${block.mimeType ? ` ${block.mimeType}` : ""}]`;
    })
    .join("\n")
    .trim();
}
