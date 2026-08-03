import type { Comparison, Condition } from "./schema.js";

/**
 * Declarative edge conditions.
 *
 * This is deliberately *not* an expression language. A workflow definition is stored, versioned, and
 * frequently authored by someone who is not deploying this repo — so evaluating arbitrary
 * expressions from it would be executing untrusted input. A dozen comparison operators cover the
 * branching real workflows need ("did it succeed", "is the score above 7", "does it mention X")
 * without that exposure.
 *
 * When something genuinely does not fit, the escape hatch is a `transform` node computing the value
 * first, not a bigger condition grammar.
 */
export type Variables = Record<string, unknown>;

export function evaluateCondition(condition: Condition, variables: Variables): boolean {
  if ("all" in condition) {
    return condition.all.every((child) => evaluateCondition(child, variables));
  }
  if ("any" in condition) {
    return condition.any.some((child) => evaluateCondition(child, variables));
  }
  if ("not" in condition) {
    return !evaluateCondition(condition.not, variables);
  }
  return evaluateComparison(condition, variables);
}

function evaluateComparison(comparison: Comparison, variables: Variables): boolean {
  const actual = readPath(variables, comparison.path);
  const expected = comparison.value;

  switch (comparison.op) {
    case "exists":
      return actual !== undefined && actual !== null;

    case "empty":
      return isEmpty(actual);

    case "eq":
      return looseEquals(actual, expected);

    case "neq":
      return !looseEquals(actual, expected);

    case "gt":
    case "gte":
    case "lt":
    case "lte": {
      const left = Number(actual);
      const right = Number(expected);
      // A non-numeric comparison is a definition bug, but failing the guard is safer than throwing
      // mid-run: the run takes another edge rather than dying with a half-finished workflow.
      if (Number.isNaN(left) || Number.isNaN(right)) return false;
      if (comparison.op === "gt") return left > right;
      if (comparison.op === "gte") return left >= right;
      if (comparison.op === "lt") return left < right;
      return left <= right;
    }

    case "contains": {
      if (Array.isArray(actual)) return actual.some((entry) => looseEquals(entry, expected));
      if (typeof actual === "string") return actual.includes(String(expected));
      return false;
    }

    case "matches": {
      if (typeof actual !== "string" || typeof expected !== "string") return false;
      try {
        // Case-insensitive by default: guards are usually matching model prose, where casing is
        // incidental.
        return new RegExp(expected, "i").test(actual);
      } catch {
        // An invalid pattern is a definition bug; failing the guard beats killing the run.
        return false;
      }
    }
  }
}

/** Dot-path read, e.g. `review.score` or `items.0.name`. */
export function readPath(source: unknown, path: string): unknown {
  let current: unknown = source;

  for (const segment of path.split(".")) {
    if (current === null || current === undefined) return undefined;

    if (Array.isArray(current)) {
      const index = Number(segment);
      current = Number.isInteger(index) ? current[index] : undefined;
      continue;
    }

    if (typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[segment];
  }

  return current;
}

/**
 * `{{path}}` interpolation for prompts and transforms.
 *
 * A missing path renders as an empty string rather than the literal `{{path}}`. Sending a model the
 * text `{{summary}}` because an earlier node failed to set it produces confidently wrong output;
 * an empty slot is at least visibly wrong.
 */
export function interpolate(template: string, variables: Variables): string {
  return template.replace(/\{\{\s*([\w.[\]]+)\s*\}\}/g, (_match, path: string) => {
    const value = readPath(variables, path);
    if (value === undefined || value === null) return "";
    return typeof value === "string" ? value : JSON.stringify(value);
  });
}

function looseEquals(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  // Values arriving from JSON are frequently strings where the definition wrote numbers.
  if (typeof a === "number" && typeof b === "string") return a === Number(b);
  if (typeof a === "string" && typeof b === "number") return Number(a) === b;
  if (typeof a === "object" && typeof b === "object" && a !== null && b !== null) {
    return JSON.stringify(a) === JSON.stringify(b);
  }
  return false;
}

function isEmpty(value: unknown): boolean {
  if (value === undefined || value === null) return true;
  if (typeof value === "string") return value.trim().length === 0;
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === "object") return Object.keys(value as object).length === 0;
  return false;
}
