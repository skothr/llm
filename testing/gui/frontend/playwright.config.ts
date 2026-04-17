import { defineConfig, devices } from "@playwright/test";

// Playwright config for the LLM Surgeon GUI smoke suite. Runs against the
// vite dev server so tests exercise the real store + persistence layers
// (IndexedDB, zustand middleware) rather than a mock.
//
// We do NOT require a live backend — most tests use the experiment-import
// path to seed results into the store. The backend-offline banner may
// flash briefly but doesn't block any test assertion.
export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 30_000,
  expect: { timeout: 5_000 },
  // Serial by default: tests share localStorage/IndexedDB state for the
  // app's `llm-surgeon-gui/v1` key. beforeEach clears IDB, but running
  // them in parallel in the same browser profile would race.
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],

  use: {
    baseURL: "http://localhost:5173",
    // Retain a trace + screenshot only for failures so passing runs are
    // cheap. Traces are readable with `npx playwright show-trace`.
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
  },

  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],

  // Auto-start vite if it isn't running. reuseExistingServer=true keeps
  // local dev loops fast — if you have `npm run dev` already open,
  // Playwright piggybacks on that instead of booting a second server
  // (which strictPort:true would reject anyway).
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: true,
    timeout: 60_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
