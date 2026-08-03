import type { ToolDefinition } from "@orchestrator/shared";
import {
  CONFIDENCE,
  type QualityAssessment,
  type QualityInput,
  type QualityScorer,
} from "../scorer.js";
import { validateAgainstSchema } from "./json-schema.js";

/**
 * Validates emitted tool calls against the caller's own tool definitions.
 *
 * This is the highest-value deterministic validator because it needs no configuration — any request
 * with `tools` gets graded for free — and it catches a genuine, common failure mode: a model emitting
 * confident-looking arguments that do not satisfy the schema, inventing a tool that was never offered,
 * or omitting a required field.
 */
export class ToolCallScorer implements QualityScorer {
  readonly name = "tool-call";
  readonly stage = "inline" as const;

  score(input: QualityInput): QualityAssessment | undefined {
    const tools = input.request.tools;
    const toolCalls = input.response.message.toolCalls;

    // Abstain: no tools offered, so there is nothing this scorer can judge.
    if (!tools?.length) return undefined;

    // The model was offered tools and chose not to call any. That is frequently correct — declining
    // to call a tool is a legitimate answer — so this scorer has no opinion.
    if (!toolCalls?.length) return undefined;

    const byName = new Map<string, ToolDefinition>(tools.map((tool) => [tool.name, tool]));
    const problems: string[] = [];
    let valid = 0;

    for (const call of toolCalls) {
      const definition = byName.get(call.name);
      if (!definition) {
        // Calling a tool that was never offered is unambiguously wrong.
        problems.push(`unknown tool "${call.name}"`);
        continue;
      }

      const errors = validateAgainstSchema(call.arguments, definition.parameters);
      if (errors.length === 0) {
        valid += 1;
      } else {
        problems.push(`${call.name}: ${errors.join("; ")}`);
      }
    }

    return {
      score: valid / toolCalls.length,
      confidence: CONFIDENCE.deterministic,
      source: this.name,
      detail:
        problems.length > 0
          ? `${valid}/${toolCalls.length} valid — ${problems.join(" | ")}`
          : `${valid}/${toolCalls.length} tool calls valid`,
    };
  }
}
