import { describe, expect, it } from "vitest";
import { HashingEmbedder, OpenAiEmbedder, cosineSimilarity } from "./embedder.js";

/**
 * Guards the property the relevance floor depends on: unrelated text must score well below the
 * default `minScore` of 0.25, or memory retrieval quietly starts injecting noise into prompts.
 */
describe("HashingEmbedder separation", () => {
  const pairs: { query: string; related: string; unrelated: string }[] = [
    {
      query: "billing invoice question",
      related: "billing invoice payment receipt",
      unrelated: "quantum chromodynamics lattice gauge",
    },
    {
      query: "how do I get a refund for my order",
      related: "refund policy for orders is 30 days",
      unrelated: "the weather in Karachi is hot today",
    },
  ];

  it("keeps unrelated similarity below the default relevance floor at the default dimension", async () => {
    const embedder = new HashingEmbedder();

    for (const pair of pairs) {
      const [query, related, unrelated] = await embedder.embed([
        pair.query,
        pair.related,
        pair.unrelated,
      ]);

      const unrelatedScore = cosineSimilarity(query as Float32Array, unrelated as Float32Array);
      const relatedScore = cosineSimilarity(query as Float32Array, related as Float32Array);

      expect(unrelatedScore).toBeLessThan(0.25);
      expect(relatedScore).toBeGreaterThan(unrelatedScore);
    }
  });

  it("degrades at small dimensions, which is why 512 is the default", async () => {
    // Documents the failure rather than hiding it: at 128 dimensions collision noise is high enough
    // that unrelated content clears the floor.
    const small = new HashingEmbedder(128);
    const [query, , unrelated] = await small.embed([
      "billing invoice question",
      "billing invoice payment receipt",
      "quantum chromodynamics lattice gauge",
    ]);

    expect(cosineSimilarity(query as Float32Array, unrelated as Float32Array)).toBeGreaterThan(
      0.25,
    );
  });

  it("reports the dimension it actually produces", async () => {
    const embedder = new HashingEmbedder(256);
    const [vector] = await embedder.embed(["x"]);
    expect(vector).toHaveLength(256);
    expect(embedder.dimension).toBe(256);
  });
});

describe("cosineSimilarity", () => {
  it("returns 0 for mismatched dimensions rather than throwing", () => {
    // Vectors from two different embedders are not comparable; scoring them 0 keeps a model swap
    // from crashing retrieval.
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([1, 0, 0]))).toBe(0);
  });

  it("returns 0 for a zero vector instead of NaN", () => {
    expect(cosineSimilarity(new Float32Array([0, 0]), new Float32Array([1, 1]))).toBe(0);
  });

  it("scores identical vectors at 1 and opposite at -1", () => {
    expect(cosineSimilarity(new Float32Array([1, 2]), new Float32Array([1, 2]))).toBeCloseTo(1, 6);
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([-1, 0]))).toBeCloseTo(
      -1,
      6,
    );
  });
});

describe("OpenAiEmbedder", () => {
  it("indexes results rather than assuming response order", async () => {
    // The API does not guarantee ordering, and silently mismatching text to vectors would corrupt
    // every future retrieval in a way that is very hard to notice.
    const fetchImpl = (async () =>
      new Response(
        JSON.stringify({
          data: [
            { index: 1, embedding: [0, 1] },
            { index: 0, embedding: [1, 0] },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      )) as typeof globalThis.fetch;

    const embedder = new OpenAiEmbedder({ apiKey: "k", dimension: 2, fetchImpl });
    const [first, second] = await embedder.embed(["a", "b"]);

    expect(Array.from(first as Float32Array)).toEqual([1, 0]);
    expect(Array.from(second as Float32Array)).toEqual([0, 1]);
  });

  it("surfaces a provider error instead of returning empty vectors", async () => {
    const fetchImpl = (async () =>
      new Response("rate limited", { status: 429 })) as typeof globalThis.fetch;

    const embedder = new OpenAiEmbedder({ apiKey: "k", fetchImpl });
    await expect(embedder.embed(["a"])).rejects.toThrow(/429/);
  });

  it("skips the network entirely for an empty batch", async () => {
    let called = false;
    const fetchImpl = (async () => {
      called = true;
      return new Response("{}");
    }) as typeof globalThis.fetch;

    const embedder = new OpenAiEmbedder({ apiKey: "k", fetchImpl });
    expect(await embedder.embed([])).toEqual([]);
    expect(called).toBe(false);
  });
});
