import { RouteHintSchema, ToolDefinitionSchema } from "@orchestrator/shared";
import { z } from "zod";

/**
 * Workflow definitions are data, not code.
 *
 * They get stored, versioned, sent over the wire, and edited by people who are not deploying this
 * repo. That rules out functions anywhere in the definition — including guards, which is why
 * conditions below are a small declarative schema rather than an expression language.
 */

/** Comparison operators for edge guards. Deliberately few — see `guards.ts` for why. */
export const ConditionOperatorSchema = z.enum([
  "eq",
  "neq",
  "gt",
  "gte",
  "lt",
  "lte",
  "contains",
  "matches",
  "exists",
  "empty",
]);
export type ConditionOperator = z.infer<typeof ConditionOperatorSchema>;

export const ComparisonSchema = z.object({
  /** Dot path into run variables, e.g. `draft.wordCount`. */
  path: z.string().min(1),
  op: ConditionOperatorSchema,
  value: z.unknown().optional(),
});
export type Comparison = z.infer<typeof ComparisonSchema>;

export type Condition =
  | { all: Condition[] }
  | { any: Condition[] }
  | { not: Condition }
  | Comparison;

// Recursive schemas need the explicit annotation; zod cannot infer through the lazy reference.
export const ConditionSchema: z.ZodType<Condition> = z.lazy(() =>
  z.union([
    z.object({ all: z.array(ConditionSchema) }),
    z.object({ any: z.array(ConditionSchema) }),
    z.object({ not: ConditionSchema }),
    ComparisonSchema,
  ]),
);

/**
 * Node types.
 *
 * `model`      — one routed model call. The router is consulted here, per node, which is the point:
 *                a workflow can be cheap at one step and premium at the next.
 * `transform`  — a pure, declarative reshaping of run variables. No I/O.
 * `checkpoint` — pauses the run for human input. The reason durable state exists.
 * `tool`       — an external call. Phase 7 backs this with MCP; the node type exists now so
 *                workflows written today do not need rewriting then.
 */
export const NodeTypeSchema = z.enum(["model", "transform", "checkpoint", "tool"]);
export type NodeType = z.infer<typeof NodeTypeSchema>;

export const ModelNodeConfigSchema = z.object({
  /** Prompt template. `{{path}}` interpolates a run variable. */
  prompt: z.string(),
  system: z.string().optional(),
  /** Per-node routing intent — the whole reason the router is consulted at each step. */
  route: RouteHintSchema.default({}),
  tools: z.array(ToolDefinitionSchema).optional(),
  maxTokens: z.number().int().positive().optional(),
  temperature: z.number().min(0).max(2).optional(),
  /** Run variable to store the response text in. */
  outputVar: z.string().min(1),
});

export const TransformNodeConfigSchema = z.object({
  /** `{ target: "{{template}}" }` — each value is interpolated against current variables. */
  set: z.record(z.string(), z.string()),
});

export const CheckpointNodeConfigSchema = z.object({
  /** Shown to whoever has to approve or supply input. */
  prompt: z.string(),
  /** Variable the human's input lands in when the run resumes. */
  outputVar: z.string().min(1).optional(),
});

export const ToolNodeConfigSchema = z.object({
  tool: z.string().min(1),
  /** Argument templates, interpolated like `transform`. */
  arguments: z.record(z.string(), z.string()).default({}),
  outputVar: z.string().min(1),
});

export const WorkflowNodeSchema = z.object({
  id: z.string().min(1),
  type: NodeTypeSchema,
  config: z.record(z.string(), z.unknown()),
  /** Attempts for this node before the run fails. Independent of the gateway's own provider retry. */
  maxAttempts: z.number().int().positive().default(1),
});
export type WorkflowNode = z.infer<typeof WorkflowNodeSchema>;

export const WorkflowEdgeSchema = z.object({
  from: z.string().min(1),
  to: z.string().min(1),
  /** Absent means unconditional. Edges are evaluated in declaration order; first match wins. */
  when: ConditionSchema.optional(),
});
export type WorkflowEdge = z.infer<typeof WorkflowEdgeSchema>;

export const WorkflowDefinitionSchema = z
  .object({
    id: z.string().min(1),
    name: z.string().default(""),
    version: z.number().int().positive().default(1),
    entry: z.string().min(1),
    nodes: z.array(WorkflowNodeSchema).min(1),
    edges: z.array(WorkflowEdgeSchema).default([]),
    /**
     * Hard bound on executed steps. Not optional paranoia: a guard that never goes false would
     * otherwise loop against a paid provider indefinitely.
     */
    maxSteps: z.number().int().positive().default(50),
  })
  .superRefine((workflow, ctx) => {
    const ids = new Set<string>();
    for (const node of workflow.nodes) {
      if (ids.has(node.id)) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Duplicate node id: ${node.id}` });
      }
      ids.add(node.id);
    }

    if (!ids.has(workflow.entry)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: `Entry node does not exist: ${workflow.entry}`,
      });
    }

    // A dangling edge is a workflow that fails halfway through a run rather than at definition
    // time. Catching it here turns a production incident into a validation error.
    for (const edge of workflow.edges) {
      if (!ids.has(edge.from)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: `Edge from unknown node: ${edge.from}`,
        });
      }
      if (!ids.has(edge.to)) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Edge to unknown node: ${edge.to}` });
      }
    }
  });

export type WorkflowDefinition = z.infer<typeof WorkflowDefinitionSchema>;
export type WorkflowDefinitionInput = z.input<typeof WorkflowDefinitionSchema>;
