/**
 * Embeddings.
 *
 * The interface exists so the store never knows where vectors come from, and so the whole package
 * can be tested and run offline. `HashingEmbedder` is not a stub — it is a real, if unsophisticated,
 * lexical embedder that makes memory usable with no provider account at all.
 */
export interface Embedder {
  readonly name: string;
  readonly dimension: number;
  embed(texts: string[]): Promise<Float32Array[]>;
}

/**
 * Deterministic local embedder: hashed character trigrams plus word unigrams, L2-normalized.
 *
 * Be clear about what this is. It captures *lexical* overlap, not meaning: "cancel my subscription"
 * and "how do I unsubscribe" score poorly because they share few tokens. A provider embedder is
 * strictly better for semantic recall.
 *
 * It earns its place regardless — no key, no network, no cost, deterministic across runs — which
 * makes it the default for tests, local development, and the retrieval-quality harness, where a
 * reproducible baseline matters more than absolute recall.
 *
 * **On the default dimension.** Measured cosine between a related and an unrelated pair:
 *
 *   dim  128:  related 0.63 / 0.36   unrelated 0.34 / 0.18   <- noise overlaps signal
 *   dim  256:  related 0.56 / 0.29   unrelated 0.25 / 0.12   <- noise sits on the 0.25 floor
 *   dim  512:  related 0.56 / 0.29   unrelated 0.12 / 0.10   <- clean separation
 *
 * Hash collisions give unrelated texts a similarity floor that shrinks as the space grows. At 256 it
 * lands right on the default `minScore`, so unrelated memories would routinely clear it. 512 is the
 * smallest size where the gap is unambiguous, which is why it is the default.
 *
 * Note the weaker related pair still only reaches ~0.29: paraphrases that share few tokens score
 * poorly no matter the dimension. That is the lexical ceiling, and the reason to use a provider
 * embedder when semantic recall actually matters.
 */
export class HashingEmbedder implements Embedder {
  readonly name = "hashing-local";
  readonly dimension: number;

  constructor(dimension = 512) {
    this.dimension = dimension;
  }

  embed(texts: string[]): Promise<Float32Array[]> {
    return Promise.resolve(texts.map((text) => this.embedOne(text)));
  }

  private embedOne(text: string): Float32Array {
    const vector = new Float32Array(this.dimension);
    const normalized = text.toLowerCase().replace(/\s+/g, " ").trim();

    // Word unigrams carry topic; trigrams give robustness to inflection and typos.
    for (const word of normalized.split(" ")) {
      if (!word) continue;
      addFeature(vector, `w:${word}`, 1);
      const padded = ` ${word} `;
      for (let i = 0; i < padded.length - 2; i++) {
        addFeature(vector, `t:${padded.slice(i, i + 3)}`, 0.5);
      }
    }

    // L2 normalize so cosine similarity is a plain dot product and long texts do not dominate.
    let norm = 0;
    for (let i = 0; i < vector.length; i++) norm += (vector[i] ?? 0) ** 2;
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let i = 0; i < vector.length; i++) vector[i] = (vector[i] ?? 0) / norm;
    }

    return vector;
  }
}

export interface OpenAiEmbedderConfig {
  apiKey: string;
  model?: string;
  dimension?: number;
  baseUrl?: string;
  fetchImpl?: typeof globalThis.fetch;
}

/**
 * Provider-backed embedder.
 *
 * Deliberately uses its own `fetch` rather than the gateway: embeddings are not chat, they must never
 * be routed, and routing them would let memory influence the bandit that memory is meant to serve.
 */
export class OpenAiEmbedder implements Embedder {
  readonly name: string;
  readonly dimension: number;

  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof globalThis.fetch;

  constructor(config: OpenAiEmbedderConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model ?? "text-embedding-3-small";
    this.dimension = config.dimension ?? 1_536;
    this.name = `openai:${this.model}`;
    this.baseUrl = (config.baseUrl ?? "https://api.openai.com/v1").replace(/\/$/, "");
    this.fetchImpl = config.fetchImpl ?? globalThis.fetch;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];

    const response = await this.fetchImpl(`${this.baseUrl}/embeddings`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ model: this.model, input: texts, dimensions: this.dimension }),
    });

    if (!response.ok) {
      throw new Error(`Embedding request failed (${response.status}): ${await response.text()}`);
    }

    const body = (await response.json()) as { data?: { embedding: number[]; index: number }[] };
    const rows = body.data ?? [];
    // Order is not guaranteed by the API contract, so index rather than assume.
    const out = new Array<Float32Array>(texts.length);
    for (const row of rows) out[row.index] = Float32Array.from(row.embedding);

    return out;
  }
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;

  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const left = a[i] ?? 0;
    const right = b[i] ?? 0;
    dot += left * right;
    normA += left * left;
    normB += right * right;
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}

function addFeature(vector: Float32Array, token: string, weight: number): void {
  // FNV-1a: fast, well-distributed, and stable across processes — which matters because stored
  // vectors must stay comparable to ones computed later.
  let hash = 2_166_136_261;
  for (let i = 0; i < token.length; i++) {
    hash ^= token.charCodeAt(i);
    hash = Math.imul(hash, 16_777_619);
  }

  const index = Math.abs(hash) % vector.length;
  // Sign from a separate bit keeps unrelated tokens from all pushing the same direction.
  const sign = (hash & 1) === 0 ? 1 : -1;
  vector[index] = (vector[index] ?? 0) + weight * sign;
}
