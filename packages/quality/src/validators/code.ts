import {
  CONFIDENCE,
  type QualityAssessment,
  type QualityInput,
  type QualityScorer,
  responseText,
} from "../scorer.js";

/**
 * Structural checks on code responses.
 *
 * **Parse-only, by design.** Executing model-generated code needs a real sandbox — process
 * isolation, filesystem and network restrictions, resource limits — and getting that wrong turns a
 * quality scorer into remote code execution. That work is deliberately out of scope here; this
 * checks structure only, and never runs anything.
 *
 * What it can still catch cheaply: no code block at all, unbalanced delimiters, and obvious
 * truncation mid-construct. Those are common real failures on cheap models.
 */
export class CodeStructureScorer implements QualityScorer {
  readonly name = "code-structure";
  readonly stage = "inline" as const;

  score(input: QualityInput): QualityAssessment | undefined {
    // Abstain on anything that is not a code task — prose containing a brace is not a failure.
    if (input.request.route.taskType !== "code") return undefined;

    const text = responseText(input.response);
    const blocks = extractCodeBlocks(text);

    if (blocks.length === 0) {
      // A code request answered with no code at all. Not necessarily wrong (it may be an
      // explanation), so confidence stays below a definitive check.
      return {
        score: 0.3,
        confidence: CONFIDENCE.heuristic,
        source: this.name,
        detail: "no fenced code block in response",
      };
    }

    const problems: string[] = [];
    let balanced = 0;

    for (const block of blocks) {
      const issue = findStructuralIssue(block);
      if (issue) problems.push(issue);
      else balanced += 1;
    }

    return {
      score: balanced / blocks.length,
      confidence: CONFIDENCE.deterministic,
      source: this.name,
      detail:
        problems.length > 0
          ? `${balanced}/${blocks.length} well-formed — ${problems.join("; ")}`
          : `${blocks.length} code block(s) structurally sound`,
    };
  }
}

export function extractCodeBlocks(text: string): string[] {
  const blocks: string[] = [];
  const pattern = /```[a-zA-Z0-9+#-]*\n([\s\S]*?)```/g;

  let match = pattern.exec(text);
  while (match !== null) {
    if (match[1]?.trim()) blocks.push(match[1]);
    match = pattern.exec(text);
  }

  return blocks;
}

/**
 * Delimiter balance, ignoring anything inside strings or comments — otherwise a brace in a string
 * literal reads as an imbalance and every correct answer containing one gets marked down.
 */
export function findStructuralIssue(code: string): string | undefined {
  const pairs: Record<string, string> = { ")": "(", "]": "[", "}": "{" };
  const opens = new Set(["(", "[", "{"]);
  const stack: string[] = [];

  let inString: string | undefined;
  let inLineComment = false;
  let inBlockComment = false;

  for (let i = 0; i < code.length; i++) {
    const char = code[i] as string;
    const next = code[i + 1];

    if (inLineComment) {
      if (char === "\n") inLineComment = false;
      continue;
    }
    if (inBlockComment) {
      if (char === "*" && next === "/") {
        inBlockComment = false;
        i++;
      }
      continue;
    }
    if (inString) {
      if (char === "\\") i++;
      else if (char === inString) inString = undefined;
      continue;
    }

    if (char === "/" && next === "/") {
      inLineComment = true;
      i++;
      continue;
    }
    if (char === "/" && next === "*") {
      inBlockComment = true;
      i++;
      continue;
    }
    if (char === "#") {
      inLineComment = true;
      continue;
    }
    if (char === '"' || char === "'" || char === "`") {
      inString = char;
      continue;
    }

    if (opens.has(char)) {
      stack.push(char);
    } else if (pairs[char]) {
      const expected = pairs[char];
      if (stack.pop() !== expected) return `unbalanced "${char}"`;
    }
  }

  if (inString) return "unterminated string literal";
  if (inBlockComment) return "unterminated block comment";
  if (stack.length > 0) return `${stack.length} unclosed "${stack[stack.length - 1]}"`;

  return undefined;
}
