import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const pkg = (name: string) =>
  fileURLToPath(new URL(`./packages/${name}/src/index.ts`, import.meta.url));

export default defineConfig({
  // Vite resolves its default hostname ("localhost") through DNS at startup. On hosts whose
  // /etc/hosts lacks a localhost entry that lookup fails and the runner never boots, so pin the
  // loopback address directly.
  server: { host: "127.0.0.1" },
  resolve: {
    // Alias workspace packages straight to source so `pnpm test` never
    // requires a build step. Production resolution still goes through
    // package.json exports -> dist.
    alias: {
      "@orchestrator/shared": pkg("shared"),
      "@orchestrator/gateway": pkg("gateway"),
      "@orchestrator/telemetry": pkg("telemetry"),
      "@orchestrator/router": pkg("router"),
    },
  },
  test: {
    include: ["{packages,apps,tools}/*/src/**/*.test.ts"],
    environment: "node",
    // The API's request logger is noise in test output, not signal.
    env: { LOG_LEVEL: "silent" },
    restoreMocks: true,
    unstubEnvs: true,
    unstubGlobals: true,
  },
});
