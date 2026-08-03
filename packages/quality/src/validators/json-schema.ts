import {
  CONFIDENCE,
  type QualityAssessment,
  type QualityInput,
  type QualityScorer,
  responseText,
} from "../scorer.js";

/**
 * A deliberately small JSON Schema subset validator.
 *
 * Scope is `type`, `required`, `properties`, `enum`, `items`, and the numeric/string bounds — the
 * parts models actually violate. A full JSON Schema implementation (refs, allOf/anyOf/oneOf,
 * conditionals) is a dependency, not a hundred lines, and pulling one in to grade tool arguments
 * would be a poor trade this early. Unsupported keywords are ignored rather than failed, so this
 * validator can only ever be too lenient, never wrongly harsh.
 */
export function validateAgainstSchema(
  value: unknown,
  schema: Record<string, unknown>,
  path = "",
): string[] {
  const errors: string[] = [];
  const at = path || "root";

  const type = schema.type as string | undefined;
  if (type && !matchesType(value, type)) {
    return [`${at}: expected ${type}, got ${describe(value)}`];
  }

  const enumValues = schema.enum as unknown[] | undefined;
  if (enumValues && !enumValues.some((candidate) => deepEqual(candidate, value))) {
    errors.push(`${at}: ${JSON.stringify(value)} is not one of ${JSON.stringify(enumValues)}`);
  }

  if (type === "object" || (!type && isPlainObject(value))) {
    const object = value as Record<string, unknown>;
    const properties = (schema.properties ?? {}) as Record<string, Record<string, unknown>>;

    for (const key of (schema.required ?? []) as string[]) {
      if (!(key in object)) errors.push(`${at}: missing required property "${key}"`);
    }

    for (const [key, subSchema] of Object.entries(properties)) {
      if (key in object && isPlainObject(subSchema)) {
        errors.push(
          ...validateAgainstSchema(object[key], subSchema, path ? `${path}.${key}` : key),
        );
      }
    }
  }

  if (type === "array" && Array.isArray(value)) {
    const items = schema.items as Record<string, unknown> | undefined;
    if (items && isPlainObject(items)) {
      value.forEach((entry, index) => {
        errors.push(...validateAgainstSchema(entry, items, `${path}[${index}]`));
      });
    }
  }

  if (typeof value === "number") {
    const min = schema.minimum as number | undefined;
    const max = schema.maximum as number | undefined;
    if (min !== undefined && value < min) errors.push(`${at}: ${value} < minimum ${min}`);
    if (max !== undefined && value > max) errors.push(`${at}: ${value} > maximum ${max}`);
  }

  if (typeof value === "string") {
    const minLength = schema.minLength as number | undefined;
    if (minLength !== undefined && value.length < minLength) {
      errors.push(`${at}: string shorter than minLength ${minLength}`);
    }
  }

  return errors;
}

/**
 * Grades a response against an output schema the caller declared on `route.outputSchema`.
 *
 * Abstains unless a schema was declared — most traffic is free-form text, and demanding JSON of it
 * would be nonsense.
 */
export class JsonSchemaScorer implements QualityScorer {
  readonly name = "json-schema";
  readonly stage = "inline" as const;

  score(input: QualityInput): QualityAssessment | undefined {
    const schema = input.request.route.outputSchema;
    if (!schema) return undefined;

    const text = responseText(input.response).trim();
    const parsed = parseJsonLoosely(text);

    if (parsed === undefined) {
      return {
        score: 0,
        confidence: CONFIDENCE.deterministic,
        source: this.name,
        detail: "response is not parseable JSON",
      };
    }

    const errors = validateAgainstSchema(parsed, schema);
    return {
      score: errors.length === 0 ? 1 : 0,
      confidence: CONFIDENCE.deterministic,
      source: this.name,
      detail: errors.length === 0 ? "conforms to declared schema" : errors.slice(0, 3).join("; "),
    };
  }
}

/**
 * Models routinely wrap JSON in prose or a fenced block despite instructions. Recovering it is not
 * being lenient about correctness — the schema check that follows is still strict — it just avoids
 * scoring formatting habits as if they were content failures.
 */
export function parseJsonLoosely(text: string): unknown {
  const candidates = [text];

  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced?.[1]) candidates.push(fenced[1].trim());

  const firstBrace = text.indexOf("{");
  const lastBrace = text.lastIndexOf("}");
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    candidates.push(text.slice(firstBrace, lastBrace + 1));
  }

  const firstBracket = text.indexOf("[");
  const lastBracket = text.lastIndexOf("]");
  if (firstBracket !== -1 && lastBracket > firstBracket) {
    candidates.push(text.slice(firstBracket, lastBracket + 1));
  }

  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate) as unknown;
    } catch {
      // Try the next extraction strategy.
    }
  }

  return undefined;
}

function matchesType(value: unknown, type: string): boolean {
  switch (type) {
    case "object":
      return isPlainObject(value);
    case "array":
      return Array.isArray(value);
    case "string":
      return typeof value === "string";
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "integer":
      return typeof value === "number" && Number.isInteger(value);
    case "boolean":
      return typeof value === "boolean";
    case "null":
      return value === null;
    default:
      // Unknown type keyword: abstain rather than fail.
      return true;
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function describe(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

function deepEqual(a: unknown, b: unknown): boolean {
  return a === b || JSON.stringify(a) === JSON.stringify(b);
}
