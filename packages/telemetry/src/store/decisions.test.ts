import { type RoutingDecision, RoutingDecisionSchema } from "@orchestrator/shared";
import { beforeEach, describe, expect, it } from "vitest";
import { openDatabase } from "./database.js";
import {
  InMemoryRoutingDecisionRepository,
  type RoutingDecisionRepository,
  SqliteRoutingDecisionRepository,
} from "./decisions.js";

function decision(overrides: Partial<RoutingDecision> = {}): RoutingDecision {
  return RoutingDecisionSchema.parse({
    decisionId: "dec_1",
    modelId: "openai/gpt-4o-mini",
    fallbacks: ["anthropic/claude-haiku-4-5"],
    strategy: "static",
    reason: "cheap mode",
    taskType: "general",
    routeMode: "cheap",
    createdAt: 1_000,
    ...overrides,
  });
}

// The same contract must hold for both implementations, or replay results depend on the backend.
const implementations: [string, () => RoutingDecisionRepository][] = [
  ["sqlite", () => new SqliteRoutingDecisionRepository(openDatabase(":memory:"))],
  ["in-memory", () => new InMemoryRoutingDecisionRepository()],
];

for (const [name, create] of implementations) {
  describe(`RoutingDecisionRepository (${name})`, () => {
    let repository: RoutingDecisionRepository;

    beforeEach(() => {
      repository = create();
    });

    it("round-trips a decision", () => {
      const original = decision({
        features: [0.5, 1, 0],
        shadowModelId: "anthropic/claude-opus-5",
      });
      repository.record(original);
      expect(repository.get("dec_1")).toEqual(original);
    });

    it("returns undefined for an unknown id", () => {
      expect(repository.get("dec_missing")).toBeUndefined();
    });

    it("isolates the rounds where the bandit disagreed with what ran", () => {
      // This is the core replay query: shadow mode is only useful if disagreements are findable.
      repository.recordMany([
        decision({ decisionId: "agree", shadowModelId: "openai/gpt-4o-mini" }),
        decision({ decisionId: "no_shadow", shadowModelId: null }),
        decision({ decisionId: "disagree", shadowModelId: "anthropic/claude-opus-5" }),
      ]);

      const disagreements = repository.query({ disagreementsOnly: true });
      expect(disagreements.map((d) => d.decisionId)).toEqual(["disagree"]);
      expect(repository.count()).toBe(3);
      expect(repository.count({ disagreementsOnly: true })).toBe(1);
    });

    it("does not count a null shadow as a disagreement", () => {
      // A decision made in `static` mode never consulted the bandit; that is not a disagreement.
      repository.record(decision({ shadowModelId: null }));
      expect(repository.count({ disagreementsOnly: true })).toBe(0);
    });

    it("filters by time window and returns oldest-first", () => {
      repository.recordMany([
        decision({ decisionId: "c", createdAt: 3_000 }),
        decision({ decisionId: "a", createdAt: 1_000 }),
        decision({ decisionId: "b", createdAt: 2_000 }),
      ]);

      expect(repository.query().map((d) => d.decisionId)).toEqual(["a", "b", "c"]);
      expect(repository.query({ since: 2_000 }).map((d) => d.decisionId)).toEqual(["b", "c"]);
      expect(repository.query({ since: 2_000, until: 3_000 }).map((d) => d.decisionId)).toEqual([
        "b",
      ]);
    });

    it("preserves the fallback chain and feature vector", () => {
      repository.record(decision({ fallbacks: ["a", "b"], features: [1, 0.25] }));
      const reloaded = repository.get("dec_1");
      expect(reloaded?.fallbacks).toEqual(["a", "b"]);
      expect(reloaded?.features).toEqual([1, 0.25]);
    });
  });
}
