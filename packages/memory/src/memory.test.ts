import { openDatabase } from "@orchestrator/telemetry";
import { beforeEach, describe, expect, it } from "vitest";
import { HashingEmbedder, cosineSimilarity } from "./embedder.js";
import { MemoryService } from "./service.js";
import { SqliteMemoryStore } from "./store/memory-store.js";

// 512 is the package default, chosen because smaller spaces let hash-collision noise reach the
// relevance floor — see the measurement table in embedder.ts.
const embedder = new HashingEmbedder(512);

function makeService(
  overrides: ConstructorParameters<typeof MemoryService>[0] extends infer T
    ? Partial<T>
    : never = {},
) {
  let counter = 0;
  const store = new SqliteMemoryStore(openDatabase(":memory:"), () => `mem_${++counter}`);
  const service = new MemoryService({ store, embedder, ...overrides });
  return { store, service };
}

describe("HashingEmbedder", () => {
  it("is deterministic across calls", async () => {
    const [a] = await embedder.embed(["refund my order"]);
    const [b] = await embedder.embed(["refund my order"]);
    expect(Array.from(a as Float32Array)).toEqual(Array.from(b as Float32Array));
  });

  it("scores related text above unrelated text", async () => {
    const [query, related, unrelated] = await embedder.embed([
      "how do I get a refund for my order",
      "refund policy for orders is 30 days",
      "the weather in Karachi is hot today",
    ]);

    expect(cosineSimilarity(query as Float32Array, related as Float32Array)).toBeGreaterThan(
      cosineSimilarity(query as Float32Array, unrelated as Float32Array),
    );
  });

  it("produces unit vectors, so cosine is a plain dot product", async () => {
    const [vector] = await embedder.embed(["some text of moderate length here"]);
    const magnitude = Math.sqrt(
      Array.from(vector as Float32Array).reduce((total, value) => total + value * value, 0),
    );
    expect(magnitude).toBeCloseTo(1, 5);
  });

  it("handles empty input without producing NaN", async () => {
    const [vector] = await embedder.embed([""]);
    expect(Array.from(vector as Float32Array).every(Number.isFinite)).toBe(true);
  });
});

describe("SqliteMemoryStore", () => {
  let store: SqliteMemoryStore;

  beforeEach(() => {
    let counter = 0;
    store = new SqliteMemoryStore(openDatabase(":memory:"), () => `mem_${++counter}`);
  });

  it("round-trips an item with its vector", async () => {
    const [embedding] = await embedder.embed(["hello world"]);
    store.write([
      {
        tenantId: "acme",
        sessionId: "s1",
        kind: "turn",
        text: "hello world",
        metadata: { role: "user" },
        embedder: embedder.name,
        embedding,
        expiresAt: null,
      },
    ]);

    const results = store.search(embedding as Float32Array, { tenantId: "acme" });
    expect(results[0]?.item.text).toBe("hello world");
    expect(results[0]?.score).toBeCloseTo(1, 5);
  });

  describe("tenant isolation", () => {
    it("never returns another tenant's memories", async () => {
      // The worst failure this system can have is leaking one customer's context into another's
      // prompt, so the scoping is enforced in the store rather than trusted to callers.
      const [embedding] = await embedder.embed(["acme internal roadmap"]);

      store.write([
        {
          tenantId: "acme",
          sessionId: null,
          kind: "fact",
          text: "acme internal roadmap",
          metadata: {},
          embedder: embedder.name,
          embedding,
          expiresAt: null,
        },
      ]);

      expect(store.search(embedding as Float32Array, { tenantId: "globex" })).toHaveLength(0);
      expect(store.search(embedding as Float32Array, { tenantId: "acme" })).toHaveLength(1);
    });

    it("scopes recent() and count() by tenant too", async () => {
      store.write([
        {
          tenantId: "acme",
          sessionId: "s",
          kind: "turn",
          text: "a",
          metadata: {},
          embedder: null,
          expiresAt: null,
        },
        {
          tenantId: "globex",
          sessionId: "s",
          kind: "turn",
          text: "b",
          metadata: {},
          embedder: null,
          expiresAt: null,
        },
      ]);

      expect(store.recent({ tenantId: "acme", sessionId: "s" })).toHaveLength(1);
      expect(store.count({ tenantId: "acme" })).toBe(1);
    });

    it("forgets only the requested tenant", async () => {
      store.write([
        {
          tenantId: "acme",
          sessionId: "s",
          kind: "turn",
          text: "a",
          metadata: {},
          embedder: null,
          expiresAt: null,
        },
        {
          tenantId: "globex",
          sessionId: "s",
          kind: "turn",
          text: "b",
          metadata: {},
          embedder: null,
          expiresAt: null,
        },
      ]);

      expect(store.forget({ tenantId: "acme" })).toBe(1);
      expect(store.count({ tenantId: "globex" })).toBe(1);
    });
  });

  describe("session scoping", () => {
    it("includes tenant-wide facts in a session-scoped search", async () => {
      // A fact is about the user, not the conversation, so it must surface in every session.
      const [embedding] = await embedder.embed(["prefers concise answers"]);

      store.write([
        {
          tenantId: "acme",
          sessionId: null,
          kind: "fact",
          text: "prefers concise answers",
          metadata: {},
          embedder: embedder.name,
          embedding,
          expiresAt: null,
        },
      ]);

      expect(
        store.search(embedding as Float32Array, { tenantId: "acme", sessionId: "any-session" }),
      ).toHaveLength(1);
    });

    it("excludes turns from a different session", async () => {
      const [embedding] = await embedder.embed(["session one detail"]);
      store.write([
        {
          tenantId: "acme",
          sessionId: "s1",
          kind: "turn",
          text: "session one detail",
          metadata: {},
          embedder: embedder.name,
          embedding,
          expiresAt: null,
        },
      ]);

      expect(
        store.search(embedding as Float32Array, { tenantId: "acme", sessionId: "s2" }),
      ).toHaveLength(0);
    });
  });

  describe("relevance floor", () => {
    it("drops weak matches rather than padding to the limit", async () => {
      // Padding a prompt with near-irrelevant memories is how retrieval quietly makes answers worse.
      const [related, unrelated, query] = await embedder.embed([
        "billing invoice payment receipt",
        "quantum chromodynamics lattice gauge",
        "billing invoice question",
      ]);

      store.write([
        {
          tenantId: "t",
          sessionId: null,
          kind: "fact",
          text: "billing",
          metadata: {},
          embedder: embedder.name,
          embedding: related,
          expiresAt: null,
        },
        {
          tenantId: "t",
          sessionId: null,
          kind: "fact",
          text: "physics",
          metadata: {},
          embedder: embedder.name,
          embedding: unrelated,
          expiresAt: null,
        },
      ]);

      const results = store.search(query as Float32Array, {
        tenantId: "t",
        limit: 5,
        minScore: 0.2,
      });
      expect(results).toHaveLength(1);
      expect(results[0]?.item.text).toBe("billing");
    });
  });

  describe("retention", () => {
    it("hides expired items from search", async () => {
      const [embedding] = await embedder.embed(["stale"]);
      store.write([
        {
          tenantId: "t",
          sessionId: null,
          kind: "fact",
          text: "stale",
          metadata: {},
          embedder: embedder.name,
          embedding,
          expiresAt: 1_000,
        },
      ]);

      expect(store.search(embedding as Float32Array, { tenantId: "t", now: 500 })).toHaveLength(1);
      expect(store.search(embedding as Float32Array, { tenantId: "t", now: 2_000 })).toHaveLength(
        0,
      );
    });

    it("actually deletes expired rows, because retention is a promise", async () => {
      store.write([
        {
          tenantId: "t",
          sessionId: null,
          kind: "fact",
          text: "stale",
          metadata: {},
          embedder: null,
          expiresAt: 1_000,
        },
        {
          tenantId: "t",
          sessionId: null,
          kind: "fact",
          text: "kept",
          metadata: {},
          embedder: null,
          expiresAt: null,
        },
      ]);

      expect(store.purgeExpired(2_000)).toBe(1);
      expect(store.count({ tenantId: "t" })).toBe(1);
    });
  });
});

describe("MemoryService write policy", () => {
  it("skips pleasantries", () => {
    const { service } = makeService();
    expect(service.shouldRemember({ role: "user", content: "thanks!" })).toBe(false);
    expect(service.shouldRemember({ role: "user", content: "ok" })).toBe(false);
    expect(service.shouldRemember({ role: "user", content: "hi" })).toBe(false);
  });

  it("skips anything too short to be worth recalling", () => {
    const { service } = makeService();
    expect(service.shouldRemember({ role: "user", content: "what about it" })).toBe(false);
  });

  it("keeps substantive turns", () => {
    const { service } = makeService();
    expect(
      service.shouldRemember({
        role: "user",
        content: "My account number is 4471 and I was charged twice in March",
      }),
    ).toBe(true);
  });

  it("skips tool payloads, which are large and stale by the next session", () => {
    const { service } = makeService();
    expect(
      service.shouldRemember({
        role: "tool",
        content: "a".repeat(200),
        toolCallId: "c1",
      }),
    ).toBe(false);
  });
});

describe("MemoryService recall", () => {
  it("returns recent turns in chronological order", async () => {
    const { service } = makeService();
    await service.remember({
      tenantId: "t",
      sessionId: "s",
      messages: [
        { role: "user", content: "First substantive message about billing problems" },
        { role: "assistant", content: "Second substantive reply regarding your invoice" },
      ],
    });

    const result = await service.recall({ tenantId: "t", sessionId: "s", query: "billing" });
    expect(result.buffer.map((item) => item.text)).toEqual([
      "First substantive message about billing problems",
      "Second substantive reply regarding your invoice",
    ]);
  });

  it("does not repeat buffered items in the semantic results", async () => {
    // Paying twice for the same text wastes budget that could hold something new.
    const { service } = makeService();
    await service.remember({
      tenantId: "t",
      sessionId: "s",
      messages: [{ role: "user", content: "My subscription renewal date is the 14th of June" }],
    });

    const result = await service.recall({ tenantId: "t", sessionId: "s", query: "renewal date" });
    const bufferIds = new Set(result.buffer.map((item) => item.id));
    expect(result.recalled.every((scored) => !bufferIds.has(scored.item.id))).toBe(true);
  });

  it("surfaces a cross-session fact", async () => {
    const { service } = makeService();
    await service.rememberFact({
      tenantId: "t",
      text: "The customer prefers responses in Urdu when discussing billing",
    });

    const result = await service.recall({
      tenantId: "t",
      sessionId: "brand-new-session",
      query: "billing question in which language",
    });

    expect(result.recalled.some((s) => s.item.kind === "fact")).toBe(true);
    expect(result.context).toMatch(/Urdu/);
  });

  it("returns empty context when there is nothing worth recalling", async () => {
    const { service } = makeService();
    const result = await service.recall({ tenantId: "t", sessionId: "s", query: "anything" });
    expect(result.context).toBe("");
  });

  it("caps injected context so memory cannot crowd out the request", async () => {
    const { service } = makeService({ retrievalPolicy: { maxChars: 200, bufferSize: 50 } });

    await service.remember({
      tenantId: "t",
      sessionId: "s",
      messages: Array.from({ length: 20 }, (_, i) => ({
        role: "user" as const,
        content: `Message number ${i} with enough length to be stored by the write policy`,
      })),
    });

    const result = await service.recall({ tenantId: "t", sessionId: "s", query: "message" });
    expect(result.context.length).toBeLessThanOrEqual(202);
  });

  it("keeps the most recent turns when truncating", async () => {
    const { service } = makeService({
      retrievalPolicy: { maxChars: 150, bufferSize: 50, semanticLimit: 0 },
    });

    await service.remember({
      tenantId: "t",
      sessionId: "s",
      messages: [
        { role: "user", content: "OLDEST message that should be dropped when truncating happens" },
        { role: "user", content: "NEWEST message that must survive truncation of the context" },
      ],
    });

    const result = await service.recall({ tenantId: "t", sessionId: "s", query: "message" });
    expect(result.context).toMatch(/NEWEST/);
  });

  it("forgets a session on request", async () => {
    const { service } = makeService();
    await service.remember({
      tenantId: "t",
      sessionId: "s",
      messages: [{ role: "user", content: "Something substantive worth storing in memory here" }],
    });

    expect(service.forget({ tenantId: "t", sessionId: "s" })).toBe(1);
    const result = await service.recall({ tenantId: "t", sessionId: "s", query: "something" });
    expect(result.buffer).toHaveLength(0);
  });

  it("applies a TTL when one is configured", async () => {
    let now = 1_000;
    const { store, service } = makeService({ ttlMs: 500, now: () => now });

    await service.remember({
      tenantId: "t",
      sessionId: "s",
      messages: [{ role: "user", content: "An expiring message about my account details here" }],
    });

    now = 2_000;
    expect(service.purgeExpired()).toBe(1);
    expect(store.count({ tenantId: "t" })).toBe(0);
  });
});
