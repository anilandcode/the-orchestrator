import { describe, expect, it } from "vitest";
import { parseSseFrames } from "./sse.js";

function streamOf(...pieces: string[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      for (const piece of pieces) controller.enqueue(encoder.encode(piece));
      controller.close();
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>) {
  const frames = [];
  for await (const frame of parseSseFrames(stream)) frames.push(frame);
  return frames;
}

describe("parseSseFrames", () => {
  it("splits frames on blank lines", async () => {
    const frames = await collect(streamOf("data: one\n\ndata: two\n\n"));
    expect(frames.map((f) => f.data)).toEqual(["one", "two"]);
  });

  it("reassembles a frame split across network chunks", async () => {
    // The realistic failure mode: TCP does not respect message boundaries.
    const frames = await collect(streamOf('data: {"a"', ":1}\n\n"));
    expect(frames).toHaveLength(1);
    expect(frames[0]?.data).toBe('{"a":1}');
  });

  it("captures the event name when present", async () => {
    const frames = await collect(streamOf("event: message_start\ndata: {}\n\n"));
    expect(frames[0]?.event).toBe("message_start");
  });

  it("handles CRLF line endings", async () => {
    const frames = await collect(streamOf("event: ping\r\ndata: x\r\n\r\n"));
    expect(frames[0]).toEqual({ event: "ping", data: "x" });
  });

  it("joins multi-line data fields", async () => {
    const frames = await collect(streamOf("data: line1\ndata: line2\n\n"));
    expect(frames[0]?.data).toBe("line1\nline2");
  });

  it("skips comment and keep-alive lines", async () => {
    const frames = await collect(streamOf(": keep-alive\n\ndata: real\n\n"));
    expect(frames).toHaveLength(1);
    expect(frames[0]?.data).toBe("real");
  });

  it("yields a trailing frame that arrived without a closing blank line", async () => {
    const frames = await collect(streamOf("data: last"));
    expect(frames.map((f) => f.data)).toEqual(["last"]);
  });

  it("strips exactly one leading space after the colon", async () => {
    const frames = await collect(streamOf("data:  two-spaces\n\n"));
    expect(frames[0]?.data).toBe(" two-spaces");
  });
});
