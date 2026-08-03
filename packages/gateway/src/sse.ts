/**
 * Minimal server-sent-events frame parser.
 *
 * Both providers stream SSE but name their events differently, so this only splits frames — it makes
 * no attempt to interpret them. Interpretation belongs in each adapter.
 */
export interface SseFrame {
  event: string | undefined;
  data: string;
}

export async function* parseSseFrames(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseFrame, void, undefined> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      // Normalize CRLF so a single split rule covers both wire styles.
      buffer = buffer.replaceAll("\r\n", "\n");

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const frame = parseBlock(block);
        if (frame) yield frame;
        boundary = buffer.indexOf("\n\n");
      }
    }

    // A stream that ends without a trailing blank line still has one usable frame.
    const tail = parseBlock(buffer);
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

function parseBlock(block: string): SseFrame | undefined {
  if (!block.trim()) return undefined;

  let event: string | undefined;
  const dataLines: string[] = [];

  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue; // comment / keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    // A single leading space after the colon is part of the framing, not the value.
    const rawValue = colon === -1 ? "" : line.slice(colon + 1);
    const value = rawValue.startsWith(" ") ? rawValue.slice(1) : rawValue;

    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }

  if (dataLines.length === 0) return undefined;
  return { event, data: dataLines.join("\n") };
}
