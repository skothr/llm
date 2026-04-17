import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// tsconfig.node.json intentionally excludes DOM/Node lib types to stay lean;
// console is a runtime global that vite-invoked node definitely has.
declare const console: { error: (...args: unknown[]) => void };

// Minimal duck-typed shape of the bits of node-http-proxy + ServerResponse
// we touch. Avoids pulling @types/node into the vite config tsconfig just
// for two methods.
type ProxyLike = {
  removeAllListeners: (event: string) => void;
  on: (event: "error", cb: (err: { code?: string; message: string }, req: unknown, res: unknown) => void) => void;
};

// Replace vite's default proxy error handler with one that stays quiet on
// ECONNREFUSED. Rationale: during dev the backend is routinely restarting
// (or started after the frontend), so a ~500 ms offline window would
// otherwise spam the console with a dozen [vite] http proxy error lines
// on every refresh. Other proxy errors still surface.
const quietOnBackendOffline = (proxy: unknown) => {
  const p = proxy as ProxyLike;
  // Vite attaches its listener before user configure runs, so we strip it
  // and re-register ours. removeAllListeners is a standard EventEmitter
  // API — safe across node-http-proxy versions.
  p.removeAllListeners("error");
  p.on("error", (err, _req, res) => {
    if (err.code === "ECONNREFUSED" || err.code === "ECONNRESET") {
      const r = res as { headersSent?: boolean; writeHead?: (s: number, h: Record<string, string>) => void; end?: (body?: string) => void } | undefined;
      if (r && !r.headersSent && r.writeHead && r.end) {
        try {
          r.writeHead(503, { "Content-Type": "application/json" });
          r.end(JSON.stringify({ detail: "backend offline" }));
        } catch {
          // Socket may already be torn down; nothing to do.
        }
      }
      return;
    }
    console.error("[proxy error]", err.message);
  });
};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        configure: quietOnBackendOffline,
      },
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
        configure: quietOnBackendOffline,
      },
    },
  },
});
