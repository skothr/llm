import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// tsconfig.node.json intentionally excludes DOM/Node lib types to stay lean;
// these runtime globals are definitely present when vite runs in node.
declare const console: { error: (...args: unknown[]) => void };
declare const process: { nextTick: (cb: () => void) => void };

type ProxyLike = {
  removeAllListeners: (event: string) => void;
  on: (event: "error", cb: (err: { code?: string; message: string }, req: unknown, res: unknown) => void) => void;
};

type ResponseLike = {
  headersSent?: boolean;
  writeHead?: (s: number, h: Record<string, string>) => void;
  end?: (body?: string) => void;
};

// Replace vite's default proxy error handler with one that stays quiet on
// ECONNREFUSED. Rationale: during dev the backend is routinely restarting
// (or started after the frontend), so a ~500 ms offline window would
// otherwise spam the console with a dozen [vite] http proxy error lines
// on every refresh.
//
// Order matters: vite registers its own 'error' listener AFTER user
// configure runs (see dep-*.js: opts.configure(proxy, opts); proxy.on(
// 'error', ...)). So stripping listeners inside configure synchronously
// removes nothing — vite's listener hasn't been installed yet. We defer
// the strip to process.nextTick so it runs once the surrounding sync
// block (including vite's proxy.on call) has completed.
//
// Verified against vite 5.4.21 + http-proxy 1.18.1 (bundled). If vite
// switches to a Promise-based configure pipeline or http-proxy changes
// listener registration order, this defer will stop working and the
// user-facing symptom is a flood of "[vite] http proxy error" lines
// during the backend-offline window. If that happens, swap
// process.nextTick for setImmediate or switch to removing the specific
// listener by reference.
const quietOnBackendOffline = (proxy: unknown) => {
  const p = proxy as ProxyLike;
  process.nextTick(() => {
    p.removeAllListeners("error");
    p.on("error", (err, _req, res) => {
      if (err.code === "ECONNREFUSED" || err.code === "ECONNRESET") {
        const r = res as ResponseLike | undefined;
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
  });
};

export default defineConfig({
  plugins: [react()],
  // Scope Vitest to src/ unit tests. Playwright's tests/e2e/ also ends in
  // .spec.ts but is not a Vitest target — importing @playwright/test under
  // Vitest triggers a "two different versions" error.
  test: {
    include: [
      "src/**/*.{test,spec}.{ts,tsx}",
      "tests/unit/**/*.{test,spec}.{ts,tsx}",
    ],
  } as unknown as Record<string, unknown>,
  server: {
    port: 5173,
    // Fail loudly if 5173 is already bound rather than silently falling
    // through to 5174 — the backend proxy config hardcodes 5173 and
    // shell scripts announce 5173, so a silent port bump would produce
    // a broken-but-looks-fine dev setup.
    strictPort: true,
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
