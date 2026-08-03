import { loadConfig } from "./config.js";
import { buildContainer } from "./container.js";
import { buildServer } from "./server.js";

const config = loadConfig();
const container = buildContainer(config);
const app = buildServer(container);

async function start(): Promise<void> {
  if (container.gateway.availableModels().length === 0) {
    app.log.warn(
      "No provider adapters configured — set OPENAI_API_KEY and/or ANTHROPIC_API_KEY. " +
        "The server will start, but every /v1/chat call will fail.",
    );
  }

  // 127.0.0.1 rather than localhost: this host's /etc/hosts has no localhost entry.
  await app.listen({ port: config.port, host: "127.0.0.1" });
  app.log.info(`router mode: ${config.routerMode}`);
}

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => {
    app.log.info("shutting down; flushing router state");
    // Unpersisted learning is learning thrown away.
    container.close();
    void app.close().then(() => process.exit(0));
  });
}

start().catch((error: unknown) => {
  app.log.error(error);
  process.exit(1);
});
