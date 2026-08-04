import { readFileSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { TaskTypeSchema, ToolDefinitionSchema } from "@orchestrator/shared";
import { z } from "zod";

/**
 * Evaluation fixtures.
 *
 * This exists because of a measured finding: seeding the router from public benchmark scores made
 * routing *worse*, because those scores describe a different environment than the one being routed
 * in. The answer to "benchmarks do not transfer" is not a better benchmark — it is measuring the
 * models yourself, on tasks shaped like your traffic.
 *
 * A fixture is therefore not a benchmark question. It is a small, representative example of work you
 * actually send, paired with something checkable about a good answer.
 */
export const FixtureSchema = z.object({
  id: z.string().min(1),
  taskType: TaskTypeSchema,
  prompt: z.string().min(1),
  system: z.string().optional(),
  tools: z.array(ToolDefinitionSchema).optional(),
  /** Declaring a schema lets the deterministic validator grade this instead of the heuristic. */
  outputSchema: z.record(z.string(), z.unknown()).optional(),
  /** Substrings a correct answer must contain, matched case-insensitively. */
  mustContain: z.array(z.string()).default([]),
  /** Substrings whose presence marks the answer wrong. */
  mustNotContain: z.array(z.string()).default([]),
  maxTokens: z.number().int().positive().optional(),
});

export type Fixture = z.infer<typeof FixtureSchema>;

const DEFAULT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "../fixtures");

export function loadFixtures(dir: string = DEFAULT_DIR): Fixture[] {
  const files = readdirSync(dir).filter((name) => name.endsWith(".json"));
  const fixtures: Fixture[] = [];

  for (const file of files) {
    const parsed = JSON.parse(readFileSync(join(dir, file), "utf8")) as unknown;
    const entries = Array.isArray(parsed) ? parsed : [parsed];
    for (const entry of entries) fixtures.push(FixtureSchema.parse(entry));
  }

  const ids = new Set<string>();
  for (const fixture of fixtures) {
    if (ids.has(fixture.id)) throw new Error(`Duplicate fixture id: ${fixture.id}`);
    ids.add(fixture.id);
  }

  return fixtures;
}

/**
 * Grade a response against a fixture's own expectations.
 *
 * Returns `undefined` when the fixture declares nothing checkable — the same abstention rule the
 * quality package follows. A fixture with no assertions cannot say whether an answer was good, and
 * scoring it anyway would manufacture a signal.
 */
export function gradeAgainstFixture(fixture: Fixture, text: string): number | undefined {
  if (fixture.mustContain.length === 0 && fixture.mustNotContain.length === 0) return undefined;

  const haystack = text.toLowerCase();
  const required = fixture.mustContain.filter((needle) =>
    haystack.includes(needle.toLowerCase()),
  ).length;
  const forbidden = fixture.mustNotContain.filter((needle) =>
    haystack.includes(needle.toLowerCase()),
  ).length;

  // Any forbidden content fails outright: a right answer with a wrong claim in it is a wrong answer.
  if (forbidden > 0) return 0;
  if (fixture.mustContain.length === 0) return 1;

  return required / fixture.mustContain.length;
}
