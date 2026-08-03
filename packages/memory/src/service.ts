import type { Message } from "@orchestrator/shared";
import type { Embedder } from "./embedder.js";
import type { MemoryItem, MemoryStore, MemoryWrite, ScoredMemory } from "./store/memory-store.js";

/**
 * Write policy: what is worth remembering.
 *
 * "Store everything" is the default that makes memory useless. A store full of "ok", "thanks", and
 * "can you try again" retrieves noise, and noise in a prompt is worse than no memory at all because
 * it displaces the actual question. These rules are a product decision, stated explicitly so they
 * can be argued with rather than discovered.
 */
export interface WritePolicy {
  /** Below this, a turn is pleasantries. */
  minLength: number;
  /** Roles worth persisting at all. */
  roles: Message["role"][];
  /** Turns matching these are almost never worth recalling. */
  ignorePatterns: RegExp[];
}

export const DEFAULT_WRITE_POLICY: WritePolicy = {
  minLength: 24,
  // Tool payloads are usually large, structured, and stale by the next session.
  roles: ["user", "assistant"],
  ignorePatterns: [
    /^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope)\b[.!]?$/i,
    /^(hi|hello|hey|good morning|good evening)\b[.!]?$/i,
  ],
};

export interface RetrievalPolicy {
  /** How many recent turns to include verbatim. */
  bufferSize: number;
  /** How many semantically retrieved items to add. */
  semanticLimit: number;
  /**
   * Similarity floor. Weak matches are dropped rather than used to fill the quota — a prompt padded
   * with near-irrelevant memories is measurably worse than a short one.
   */
  minScore: number;
  /** Hard cap on injected characters, so memory cannot crowd out the actual request. */
  maxChars: number;
}

export const DEFAULT_RETRIEVAL_POLICY: RetrievalPolicy = {
  bufferSize: 8,
  semanticLimit: 5,
  minScore: 0.25,
  maxChars: 4_000,
};

export interface MemoryServiceConfig {
  store: MemoryStore;
  embedder: Embedder;
  writePolicy?: Partial<WritePolicy>;
  retrievalPolicy?: Partial<RetrievalPolicy>;
  /** Default retention. Null keeps memories indefinitely. */
  ttlMs?: number | null;
  now?: () => number;
}

export interface RecallResult {
  /** Recent turns, chronological. */
  buffer: MemoryItem[];
  /** Semantically retrieved items, most relevant first. */
  recalled: ScoredMemory[];
  /** Ready to prepend as a system message, or empty when nothing cleared the bar. */
  context: string;
}

export class MemoryService {
  private readonly store: MemoryStore;
  private readonly embedder: Embedder;
  private readonly writePolicy: WritePolicy;
  private readonly retrievalPolicy: RetrievalPolicy;
  private readonly ttlMs: number | null;
  private readonly now: () => number;

  constructor(config: MemoryServiceConfig) {
    this.store = config.store;
    this.embedder = config.embedder;
    this.writePolicy = { ...DEFAULT_WRITE_POLICY, ...config.writePolicy };
    this.retrievalPolicy = { ...DEFAULT_RETRIEVAL_POLICY, ...config.retrievalPolicy };
    this.ttlMs = config.ttlMs === undefined ? null : config.ttlMs;
    this.now = config.now ?? (() => Date.now());
  }

  /** Whether a turn clears the write policy. Exposed so the decision is testable in isolation. */
  shouldRemember(message: Message): boolean {
    if (!this.writePolicy.roles.includes(message.role)) return false;

    const text = textOf(message).trim();
    if (text.length < this.writePolicy.minLength) return false;
    if (this.writePolicy.ignorePatterns.some((pattern) => pattern.test(text))) return false;

    return true;
  }

  async remember(options: {
    tenantId: string;
    sessionId: string;
    messages: Message[];
  }): Promise<MemoryItem[]> {
    const worthKeeping = options.messages.filter((message) => this.shouldRemember(message));
    if (worthKeeping.length === 0) return [];

    const texts = worthKeeping.map((message) => textOf(message));
    const embeddings = await this.embedder.embed(texts);
    const createdAt = this.now();

    const writes: MemoryWrite[] = worthKeeping.map((message, index) => ({
      tenantId: options.tenantId,
      sessionId: options.sessionId,
      kind: "turn",
      text: texts[index] as string,
      metadata: { role: message.role },
      embedder: this.embedder.name,
      embedding: embeddings[index],
      createdAt,
      expiresAt: this.ttlMs === null ? null : createdAt + this.ttlMs,
    }));

    return this.store.write(writes);
  }

  /** Store a durable statement that should outlive the session it came from. */
  async rememberFact(options: {
    tenantId: string;
    text: string;
    metadata?: Record<string, unknown>;
    sessionId?: string;
  }): Promise<MemoryItem[]> {
    const [embedding] = await this.embedder.embed([options.text]);
    const createdAt = this.now();

    return this.store.write([
      {
        tenantId: options.tenantId,
        // Null session: a fact is about the user, not the conversation.
        sessionId: options.sessionId ?? null,
        kind: "fact",
        text: options.text,
        metadata: options.metadata ?? {},
        embedder: this.embedder.name,
        ...(embedding ? { embedding } : {}),
        createdAt,
        expiresAt: this.ttlMs === null ? null : createdAt + this.ttlMs,
      },
    ]);
  }

  /**
   * Assemble context for a request.
   *
   * Buffer first (recency is cheap and reliable), then semantic recall for anything older that is
   * actually relevant. Items already in the buffer are not repeated — paying twice for the same text
   * wastes budget that could hold something new.
   */
  async recall(options: {
    tenantId: string;
    sessionId: string;
    query: string;
  }): Promise<RecallResult> {
    const buffer = this.store.recent({
      tenantId: options.tenantId,
      sessionId: options.sessionId,
      limit: this.retrievalPolicy.bufferSize,
    });

    const [queryVector] = await this.embedder.embed([options.query]);
    const bufferIds = new Set(buffer.map((item) => item.id));

    const recalled = queryVector
      ? this.store
          .search(queryVector, {
            tenantId: options.tenantId,
            sessionId: options.sessionId,
            limit: this.retrievalPolicy.semanticLimit + bufferIds.size,
            minScore: this.retrievalPolicy.minScore,
            now: this.now(),
          })
          .filter((scored) => !bufferIds.has(scored.item.id))
          .slice(0, this.retrievalPolicy.semanticLimit)
      : [];

    return { buffer, recalled, context: this.formatContext(buffer, recalled) };
  }

  forget(options: { tenantId: string; sessionId?: string }): number {
    return this.store.forget(options);
  }

  purgeExpired(): number {
    return this.store.purgeExpired(this.now());
  }

  private formatContext(buffer: MemoryItem[], recalled: ScoredMemory[]): string {
    const sections: string[] = [];

    if (recalled.length > 0) {
      sections.push(
        ["Relevant earlier context:", ...recalled.map((s) => `- ${s.item.text}`)].join("\n"),
      );
    }
    if (buffer.length > 0) {
      sections.push(
        [
          "Recent conversation:",
          ...buffer.map((item) => `${String(item.metadata.role ?? "user")}: ${item.text}`),
        ].join("\n"),
      );
    }

    const context = sections.join("\n\n");
    if (context.length <= this.retrievalPolicy.maxChars) return context;

    // Truncate from the front: the most recent turns matter most, and a hard cap is what stops
    // memory from crowding out the request it is supposed to support.
    return `…\n${context.slice(context.length - this.retrievalPolicy.maxChars)}`;
  }
}

function textOf(message: Message): string {
  if (typeof message.content === "string") return message.content;
  return message.content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join(" ");
}
