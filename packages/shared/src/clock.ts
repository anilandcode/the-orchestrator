/**
 * Injectable time source. Latency numbers feed the router's reward function, so tests must be able to
 * pin them exactly rather than measuring whatever the machine happened to do.
 */
export interface Clock {
  /** Wall-clock epoch milliseconds. Used for `createdAt` ordering. */
  now(): number;
  /** Monotonic milliseconds. Used for durations — immune to clock adjustment mid-request. */
  monotonic(): number;
}

export const systemClock: Clock = {
  now: () => Date.now(),
  monotonic: () => performance.now(),
};

/** Deterministic clock for tests. Both readings advance together via `advance()`. */
export function createFixedClock(
  startMs = 1_700_000_000_000,
): Clock & { advance(ms: number): void } {
  let offset = 0;
  return {
    now: () => startMs + offset,
    monotonic: () => offset,
    advance(ms: number) {
      offset += ms;
    },
  };
}
