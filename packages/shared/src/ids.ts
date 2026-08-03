import { randomUUID } from "node:crypto";

/**
 * Injectable id source. Ids appear in stored `CallEvent` rows and in replay reports, so tests need
 * stable values to assert against.
 */
export interface IdGenerator {
  generate(prefix: string): string;
}

export const systemIds: IdGenerator = {
  generate: (prefix) => `${prefix}_${randomUUID().replaceAll("-", "").slice(0, 24)}`,
};

/** Monotonic counter ids for tests: `evt_1`, `evt_2`, ... */
export function createSequentialIds(): IdGenerator {
  const counters = new Map<string, number>();
  return {
    generate(prefix) {
      const next = (counters.get(prefix) ?? 0) + 1;
      counters.set(prefix, next);
      return `${prefix}_${next}`;
    },
  };
}
