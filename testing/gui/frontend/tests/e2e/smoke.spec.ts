import { test, expect, type Page } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FIXTURE_PATH = path.join(__dirname, "fixtures", "sample.json");
const AP_FIXTURE_PATH = path.join(__dirname, "fixtures", "activation-patching.json");
const AP_APPROX_FIXTURE_PATH = path.join(__dirname, "fixtures", "activation-patching-approx.json");
const PH_FIXTURE_PATH = path.join(__dirname, "fixtures", "activation-patching-per-head.json");
const EDGE_FIXTURE_PATH = path.join(__dirname, "fixtures", "activation-patching-edge.json");
const CIRCUIT_FIXTURE_PATH = path.join(__dirname, "fixtures", "activation-patching-circuit.json");
const PER_NEURON_FIXTURE_PATH = path.join(__dirname, "fixtures", "activation-patching-per-neuron.json");

// Wipe IDB between tests so the persistence layer doesn't carry state
// from one case into the next. Must be called while a page is loaded on
// our origin — otherwise `indexedDB.databases()` has nothing to enumerate.
async function resetStore(page: Page): Promise<void> {
  await page.evaluate(async () => {
    try {
      const dbs = await indexedDB.databases();
      await Promise.all(
        dbs.map((db) =>
          new Promise<void>((resolve) => {
            if (!db.name) return resolve();
            const req = indexedDB.deleteDatabase(db.name);
            req.onsuccess = () => resolve();
            req.onerror = () => resolve();
            req.onblocked = () => resolve();
          }),
        ),
      );
      // Also clear localStorage for symmetry — we only use IDB today,
      // but any future migration that falls back would inherit leftovers
      // otherwise.
      localStorage.clear();
    } catch {
      /* ignore */
    }
  });
}

// Accept any dialog (window.prompt / confirm) with the supplied text so
// Ctrl+S and bulk-tag flows can be driven headlessly. Call BEFORE the
// action that triggers the dialog.
function acceptNextDialog(page: Page, answer = ""): void {
  page.once("dialog", (d) => d.accept(answer));
}

test.beforeEach(async ({ page }) => {
  // First goto establishes the origin so IDB is reachable.
  await page.goto("/");
  await resetStore(page);
  // Second goto reloads without the previous state. App boots fresh.
  await page.goto("/");
});

// Heuristic: every test runs without a live backend, so 503s from the
// vite proxy and ERR_ABORTED cleanup cancellations are expected. Any
// console error matching one of these patterns is not a signal we care
// about. Real React errors and unhandled pageerrors still fail the test.
function isBackendlessNoise(text: string): boolean {
  return (
    text.includes("ECONNREFUSED") ||
    text.includes("backend offline") ||
    text.includes("503 (Service Unavailable)") ||
    text.includes("ERR_ABORTED") ||
    text.includes("/api/") ||
    text.includes("/ws/")
  );
}

test("page renders without console errors", async ({ page }) => {
  const errors: string[] = [];
  const allLogs: string[] = [];
  page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
  page.on("console", (msg) => {
    const text = msg.text();
    allLogs.push(`[${msg.type()}] ${text}`);
    if (msg.type() === "error" && !isBackendlessNoise(text)) {
      errors.push(`console.error: ${text}`);
    }
  });
  page.on("requestfailed", (req) => {
    const url = req.url();
    if (isBackendlessNoise(url)) return;
    errors.push(`requestfailed: ${url} — ${req.failure()?.errorText}`);
  });
  await page.goto("/");
  try {
    await expect(page.getByRole("heading", { name: "Visualization" })).toBeVisible({ timeout: 10_000 });
  } catch (e) {
    const html = await page.content();
    // eslint-disable-next-line no-console
    console.log("=== PAGE HTML ===\n" + html.slice(0, 2000));
    // eslint-disable-next-line no-console
    console.log("=== CONSOLE LOGS ===\n" + allLogs.join("\n"));
    throw e;
  }
  await page.waitForTimeout(500);
  expect(errors).toEqual([]);
});

test("cheat sheet opens with ? and closes with Esc", async ({ page }) => {
  await page.keyboard.press("Shift+?");
  await expect(page.getByRole("heading", { name: /Keyboard shortcuts/i })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("heading", { name: /Keyboard shortcuts/i })).not.toBeVisible();
});

test("number keys switch tabs", async ({ page }) => {
  await page.keyboard.press("2");
  // Probe tab: sampling "top_k" label is a reliable marker
  await expect(page.locator("label", { hasText: /^top_k$/ }).first()).toBeVisible();
  await page.keyboard.press("3");
  await expect(page.getByRole("button", { name: /\+ Add Intervention/i })).toBeVisible();
  await page.keyboard.press("1");
  // Sessions tab: session-name input placeholder
  await expect(page.getByPlaceholder(/Session name/i)).toBeVisible();
});

test("prompt persists across reload via IndexedDB", async ({ page }) => {
  await page.keyboard.press("2"); // Probe tab
  const textarea = page.locator("textarea").filter({ hasNot: page.locator("role=none") }).first();
  await textarea.fill("persistence sentinel value");
  // Give zustand persist middleware a moment to flush to IDB.
  await page.waitForTimeout(600);
  await page.reload();
  // Tab restoration should also land us back on Probe.
  await expect(page.locator("textarea").first()).toHaveValue("persistence sentinel value");
});

test("Ctrl+S saves current prompt to the library", async ({ page }) => {
  await page.keyboard.press("2");
  const textarea = page.locator("textarea").first();
  await textarea.fill("saved-via-shortcut");
  acceptNextDialog(page, "shortcut-name");
  await page.keyboard.press("Control+s");
  // Library select offers the saved entry
  const librarySelect = page.locator("select").filter({ hasText: /load/i }).first();
  await expect(librarySelect.locator("option", { hasText: "shortcut-name" })).toHaveCount(1);
});

test("importing an experiment populates library and results", async ({ page }) => {
  const fixture = fs.readFileSync(FIXTURE_PATH, "utf8");
  // The file input is hidden but still accepts setInputFiles.
  await page.locator('input[type="file"]').setInputFiles({
    name: "sample.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });
  await expect(page.getByText(/Imported 3 result/)).toBeVisible();
  // Two logit-lens tabs should render in the result strip
  await expect(page.getByRole("button", { name: /logit-lens \| fixture-A/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /logit-lens \| fixture-B/ })).toBeVisible();
});

test("pinning a result keeps it around after Clear All", async ({ page }) => {
  const fixture = fs.readFileSync(FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "sample.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });
  await expect(page.getByText(/Imported 3 result/)).toBeVisible();

  // Active result is res-A (tagged "capital"). Pin it via the star button
  // in ResultMetaEditor. The title starts with "Pin this result" when
  // currently unpinned.
  const pinButton = page.locator('button[title^="Pin this result"]');
  await pinButton.click();

  // Clear All removes non-pinned; the tagged fixture-A result survives.
  page.once("dialog", (d) => d.accept()); // no dialog expected, but defensive
  await page.getByRole("button", { name: "Clear All" }).click();

  // With just one result left, the tab row is hidden by design
  // (headResults.length > 1 gate). Assert by looking for the viz-body
  // heading rendered for fixture-A, and by absence of fixture-B.
  await expect(page.getByRole("heading", { name: /Logit Lens - fixture-A/ })).toBeVisible();
  // The import banner mentions missing sessions by name ("Missing sessions:
  // fixture-A, fixture-B"), so a bare text match would false-positive.
  // Restrict to the viz heading / result-tab button roles that are the
  // only places a LIVE fixture-B result would surface.
  await expect(page.getByRole("heading", { name: /fixture-B/ })).toHaveCount(0);
  await expect(page.getByRole("button", { name: /logit-lens \| fixture-B/ })).toHaveCount(0);
});

test("delete result shows undo toast and Ctrl+Z restores", async ({ page }) => {
  const fixture = fs.readFileSync(FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "sample.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });
  await expect(page.getByText(/Imported 3 result/)).toBeVisible();

  // Delete the active (fixture-A) result via the del button.
  acceptNextDialog(page, ""); // no prompt text expected
  await page.getByRole("button", { name: /^del$/ }).click();
  // Toast with Undo should appear
  await expect(page.getByRole("button", { name: /^undo$/ })).toBeVisible();

  // Ctrl+Z restores
  await page.keyboard.press("Control+z");
  await expect(page.getByRole("button", { name: /logit-lens \| fixture-A/ })).toBeVisible();
});

test("bulk tag applies to every selected result", async ({ page }) => {
  const fixture = fs.readFileSync(FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "sample.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });
  await expect(page.getByText(/Imported 3 result/)).toBeVisible();

  // Ctrl-click both logit-lens tabs to select them
  const tabA = page.getByRole("button", { name: /logit-lens \| fixture-A/ });
  const tabB = page.getByRole("button", { name: /logit-lens \| fixture-B/ });
  await tabA.click({ modifiers: ["Control"] });
  await tabB.click({ modifiers: ["Control"] });

  await expect(page.getByText(/2 selected/)).toBeVisible();

  acceptNextDialog(page, "bulk-tag-value");
  await page.getByRole("button", { name: "+ tag" }).click();

  // Filter bar should now include the new tag chip. `exact: true` is
  // critical because the bulk-tag name also appears inside the result
  // tab buttons (tab text now includes "#bulk-tag-value").
  await expect(page.getByRole("button", { name: "bulk-tag-value", exact: true })).toBeVisible();
});

test("activation-patching heatmap renders from imported fixture", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  const apFixture = fs.readFileSync(AP_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching.json",
    mimeType: "application/json",
    buffer: Buffer.from(apFixture),
  });

  await page.getByRole("heading", { name: /Activation Patching/ }).waitFor({ state: "visible", timeout: 5000 });

  const metric = page.locator("select").filter({
    has: page.locator("option", { hasText: "Logit-diff recovery" }),
  }).first();
  for (const opt of [
    "KL from clean (nats)",
    "Top-1 matches clean",
    "\u0394 p(clean top-1)",
    "Logit-diff recovery",
  ]) {
    await metric.selectOption({ label: opt });
    await page.waitForTimeout(50);
  }

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("attribution-patching heatmap renders without metric dropdown", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  const fixture = fs.readFileSync(AP_APPROX_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-approx.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByRole("heading", { name: /Attribution Patching/ }).waitFor({ state: "visible", timeout: 5000 });

  const metricSelects = page.locator("select").filter({
    has: page.locator("option", { hasText: "Logit-diff recovery" }),
  });
  await expect(metricSelects).toHaveCount(0);

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("per-head attribution heatmap renders with position selector", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  const fixture = fs.readFileSync(PH_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-per-head.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  // Heading must include "Per-head Attribution"
  await page.getByRole("heading", { name: /Per-head Attribution/ })
    .waitFor({ state: "visible", timeout: 5000 });

  // Position selector dropdown must be present (distinguishes this viz from others)
  const positionSelect = page.locator("select").filter({
    has: page.locator("option", { hasText: /^0:/ }),
  });
  await expect(positionSelect).toHaveCount(1);

  // Phase 3.10.3: IG step annotation visible when n_steps > 1
  await expect(page.getByText(/IG 5 steps/i)).toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("edge AP panel mounts without crash, tabs visible", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  const fixture = fs.readFileSync(EDGE_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-edge.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  // Heading must include "Edge Attribution"
  await page.getByRole("heading", { name: /Edge Attribution/ })
    .waitFor({ state: "visible", timeout: 5000 });

  // Position selector dropdown must be present
  const positionSelect = page.locator("select").filter({
    has: page.locator("option", { hasText: /^4:/ }),
  });
  await expect(positionSelect).toHaveCount(1);

  // Tab bar: sankey, matrix, list buttons
  await expect(page.getByRole("button", { name: "sankey" })).toBeVisible();
  await expect(page.getByRole("button", { name: "matrix" })).toBeVisible();
  await expect(page.getByRole("button", { name: "list" })).toBeVisible();

  // Phase 3.10.3: IG step annotation visible when n_steps > 1
  await expect(page.getByText(/IG 5 steps/i)).toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("circuit panel renders with τ slider and stats", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  const fixture = fs.readFileSync(CIRCUIT_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-circuit.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  // Circuit (ACDC) heading must appear
  await page.getByRole("heading", { name: /Circuit \(ACDC\)/i })
    .waitFor({ state: "visible", timeout: 5000 });

  // Stats strip shows edge count (the regex intentionally matches the stats
  // div, not the heading which also says "edges in circuit").
  await expect(page.getByText(/Edges in circuit: \d+ of \d+/i)).toBeVisible();

  // τ slider
  const tauSlider = page.locator('input[type="range"]').first();
  await expect(tauSlider).toBeVisible();

  // copy JSON export button
  await expect(page.getByRole("button", { name: /copy json/i })).toBeVisible();

  // Phase 3.10.3: IG step annotation visible when n_steps > 1
  await expect(page.getByText(/IG 5 steps/i)).toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("per-neuron FFN panel renders with table and filters", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  const fixture = fs.readFileSync(PER_NEURON_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-per-neuron.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByRole("heading", { name: /Per-Neuron FFN Attribution/i })
    .waitFor({ state: "visible", timeout: 5000 });

  // Stats strip
  await expect(page.getByText(/Showing \d+ of \d+ cells/i)).toBeVisible();

  // Table header — scope by column-header text rather than role because
  // sticky <thead> inside a scrollable <div> confuses Playwright's
  // accessibility tree ("Received: hidden" despite correct rendering).
  await expect(page.locator("th", { hasText: "ap_recovery" })).toBeVisible();
  await expect(page.locator("th", { hasText: /^neuron$/ })).toBeVisible();

  // Copy TSV button
  await expect(page.getByRole("button", { name: /copy tsv/i })).toBeVisible();

  // Phase 3.10.3: IG step annotation visible when n_steps > 1
  await expect(page.getByText(/IG 5 steps/i)).toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("per-neuron row click opens pinned card with decoded tokens", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  // Intercept the decode-neuron endpoint with a stub so this test doesn't
  // depend on a live backend.
  await page.route("**/api/sessions/*/decode-neuron", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        top_tokens: [
          { token: " Paris", logit: 2.34 },
          { token: " Lyon", logit: 1.87 },
          { token: " France", logit: 1.42 },
          { token: " French", logit: 1.10 },
          { token: " capital", logit: 0.91 },
          { token: " city", logit: 0.75 },
          { token: " Europe", logit: 0.60 },
          { token: " Seine", logit: 0.48 },
          { token: " Eiffel", logit: 0.31 },
          { token: " Louvre", logit: 0.22 },
        ],
        bottom_tokens: [
          { token: " Rome", logit: -1.89 },
          { token: " Milan", logit: -1.52 },
          { token: " Italy", logit: -1.30 },
          { token: " Italian", logit: -1.10 },
          { token: " Vatican", logit: -0.95 },
          { token: " pizza", logit: -0.81 },
          { token: " pasta", logit: -0.67 },
          { token: " Venice", logit: -0.55 },
          { token: " Colosseum", logit: -0.40 },
          { token: " Florence", logit: -0.31 },
        ],
      }),
    });
  });

  const fixture = fs.readFileSync(PER_NEURON_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-per-neuron.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByRole("heading", { name: /Per-Neuron FFN Attribution/i })
    .waitFor({ state: "visible", timeout: 5000 });

  // Click the first data row (skipping the header row).
  const firstRow = page.locator("tbody tr").first();
  await firstRow.click();

  // Pinned card assertions.
  await expect(page.getByText(/Raw W_U @ W_down/i)).toBeVisible();
  await expect(page.getByText(" Paris", { exact: true })).toBeVisible();
  await expect(page.getByText(" Rome", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /close/i })).toBeVisible();

  // Close the card.
  await page.getByRole("button", { name: /close/i }).click();
  await expect(page.getByText(/Raw W_U @ W_down/i)).not.toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("approx mode with IG shows step count alongside heatmap header", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  const fixture = fs.readFileSync(AP_APPROX_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-approx.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByRole("heading", { name: /Attribution Patching/ }).waitFor({ state: "visible", timeout: 5000 });

  await expect(page.getByText(/IG 5 steps/i)).toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("per-head pin card shows decoded tokens for attn head", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  // Intercept decode-head with a stub.
  await page.route("**/api/sessions/*/decode-head", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        top_tokens: [
          { token: " Paris", logit: 3.21 },
          { token: " Lyon", logit: 2.10 },
          { token: " France", logit: 1.85 },
          { token: " French", logit: 1.42 },
          { token: " Seine", logit: 1.05 },
        ],
        bottom_tokens: [
          { token: " Rome", logit: -2.78 },
          { token: " Milan", logit: -1.92 },
          { token: " Italy", logit: -1.60 },
          { token: " Italian", logit: -1.33 },
          { token: " Vatican", logit: -1.10 },
        ],
        singular_value_ratio: 0.58,
      }),
    });
  });

  const fixture = fs.readFileSync(PH_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-per-head.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  // Target the attn.h0 cell at layer 1 directly via data attributes —
  // row-major d3 iteration puts non-clickable (no-data) cells first, so
  // positional indexing is fragile.
  await page.locator('rect[data-layer="1"][data-unit="attn.h0"]').click();

  // Pin card assertions. Target by text — only the decode card renders this.
  await expect(page.getByText(/Dominant write direction \(sv energy ratio:/i))
    .toBeVisible({ timeout: 5000 });
  await expect(page.getByText(" Paris", { exact: true })).toBeVisible();
  await expect(page.getByText(" Rome", { exact: true })).toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("per-neuron pin card shows residual lens decode block", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  // The neuron pin card fetches BOTH decode-neuron (existing) and the
  // new decode-residual (Phase 3.11). Mock both so the test runs without
  // a live backend.
  await page.route("**/api/sessions/*/decode-neuron", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        top_tokens:    [{ token: " neuron-X", logit: 1.0 }],
        bottom_tokens: [{ token: " neuron-Y", logit: -1.0 }],
      }),
    });
  });
  await page.route("**/api/sessions/*/decode-residual", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        top_tokens: [
          { token: " Paris-lens", logit: 5.50 },
          { token: " France-lens", logit: 4.20 },
        ],
        bottom_tokens: [
          { token: " Rome-lens", logit: -2.00 },
        ],
        prompt_tokens: ["The", "capital", "of", "France", "is"],
      }),
    });
  });

  const fixture = fs.readFileSync(PER_NEURON_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-per-neuron.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByRole("heading", { name: /Per-Neuron FFN Attribution/i })
    .waitFor({ state: "visible", timeout: 5000 });

  await page.locator("tbody tr").first().click();

  // The new lens block: heading "Logit lens at (L<n>, ffn, pos <n>)"
  await expect(page.getByText(/Logit lens at \(L\d+, ffn, pos \d+\)/i))
    .toBeVisible({ timeout: 5000 });
  await expect(page.getByText(" Paris-lens", { exact: true })).toBeVisible();
  await expect(page.getByText(" Rome-lens", { exact: true })).toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("AP heatmap renders lens-trace strip with mocked grid response", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  // Mock the bulk grid endpoint (Phase 3.12). 2 layers x 2 sublayers x 5 positions.
  await page.route("**/api/sessions/*/decode-residual-grid", async (route) => {
    const cells: Array<{
      layer: number; sublayer: "attn" | "ffn"; position: number;
      tokens: Array<{ token: string; logit: number }>;
    }> = [];
    // Fixture's measurement_position is 3 (last token of "The capital of France").
    for (const layer of [0, 1]) {
      for (const sublayer of ["attn", "ffn"] as const) {
        for (let pos = 0; pos < 4; pos++) {
          const token =
            (layer === 1 && sublayer === "ffn" && pos === 3)
              ? " Paris-trace"
              : ` tok-L${layer}-${sublayer}-p${pos}`;
          cells.push({ layer, sublayer, position: pos, tokens: [{ token, logit: 1.5 }] });
        }
      }
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cells,
        prompt_tokens: ["The", "capital", "of", "France"],
        num_layers: 2,
      }),
    });
  });

  const fixture = fs.readFileSync(AP_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByRole("heading", { name: /Activation Patching/ })
    .waitFor({ state: "visible", timeout: 5000 });

  await expect(page.getByTestId("lens-trace-strip")).toBeVisible({ timeout: 5000 });
  // Default position is measurement_position (= 4 = "in") so last-layer ffn
  // shows our distinctive marker token.
  await expect(page.getByText(" Paris-trace", { exact: true })).toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("circuit panel renders causal story with mocked lens grid", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  // Mock the bulk grid endpoint. Circuit fixture's in-circuit edge is
  // (writer_layer=2, attn.h1) at position=4. Provide lens tokens for
  // (L2, attn, pos 4) so the story panel can render them.
  await page.route("**/api/sessions/*/decode-residual-grid", async (route) => {
    const cells = [
      // L2 attn position 4 — distinctive marker tokens
      { layer: 2, sublayer: "attn", position: 4, tokens: [
        { token: " Paris-story", logit: 6.0 },
        { token: " France", logit: 5.0 },
        { token: " Lyon", logit: 4.0 },
      ]},
      // Pad with empty entries for other (layer, sublayer, position) so
      // the response is still well-formed.
      { layer: 2, sublayer: "ffn", position: 4, tokens: [{ token: " other", logit: 1.0 }] },
    ];
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cells,
        prompt_tokens: ["The", "Eiffel", "Tower", "is", "in"],
        num_layers: 22,
      }),
    });
  });

  const fixture = fs.readFileSync(CIRCUIT_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-circuit.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByRole("heading", { name: /Circuit \(ACDC\)/i })
    .waitFor({ state: "visible", timeout: 5000 });

  await expect(page.getByTestId("causal-story-panel")).toBeVisible({ timeout: 5000 });
  // The lone in-circuit writer is L2 attn.h1.
  await expect(page.getByText(/L2 attn\.h1/)).toBeVisible();
  // Lens tokens render in the story.
  await expect(page.getByText(" Paris-story", { exact: true })).toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("causal story <-> sankey two-way click linking", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  await page.route("**/api/sessions/*/decode-residual-grid", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cells: [
          { layer: 2, sublayer: "attn", position: 4, tokens: [{ token: " Paris", logit: 5.0 }] },
        ],
        prompt_tokens: ["The", "Eiffel", "Tower", "is", "in"],
        num_layers: 22,
      }),
    });
  });

  const fixture = fs.readFileSync(CIRCUIT_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-circuit.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByTestId("causal-story-panel").waitFor({ state: "visible", timeout: 5000 });

  // 1) Click the story row → Sankey writer circle should grow + glow.
  const storyRow = page.getByTestId("causal-story-row-L2-attn.h1");
  const sankeyCircle = page.getByTestId("circuit-writer-L2-attn.h1");

  await expect(storyRow).toBeVisible();
  // Initial radius
  await expect(sankeyCircle).toHaveAttribute("r", "4");

  await storyRow.click();
  await expect(sankeyCircle).toHaveAttribute("r", "6");

  // 2) Click the row again → deselect → radius back to 4.
  await storyRow.click();
  await expect(sankeyCircle).toHaveAttribute("r", "4");

  // 3) Click the Sankey circle → story row picks up its selected style.
  await sankeyCircle.click();
  // The selected style applies a left-border accent. Read the inline style:
  await expect(storyRow).toHaveCSS("border-left-color", "rgb(138, 186, 255)");

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("causal story play button reveals nodes over time then resets", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  await page.route("**/api/sessions/*/decode-residual-grid", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cells: [
          { layer: 2, sublayer: "attn", position: 4, tokens: [{ token: " play-token", logit: 5.0 }] },
        ],
        prompt_tokens: ["The", "Eiffel", "Tower", "is", "in"],
        num_layers: 22,
      }),
    });
  });

  const fixture = fs.readFileSync(CIRCUIT_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-circuit.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByTestId("causal-story-panel").waitFor({ state: "visible", timeout: 5000 });

  const row = page.getByTestId("causal-story-row-L2-attn.h1");
  const playButton = page.getByTestId("causal-story-play");

  // Pre-play: row revealed (data-revealed="true").
  await expect(row).toHaveAttribute("data-revealed", "true");

  // Click play. With 1 node + 600ms interval + 600ms hold, button should
  // be aria-pressed="true" briefly.
  await playButton.click();
  await expect(playButton).toHaveAttribute("aria-pressed", "true");

  // Wait for animation to finish (600ms*2 + 200ms slack ≈ 1.5s).
  await page.waitForTimeout(1700);

  // After playback: row revealed again, button no longer pressed.
  await expect(playButton).toHaveAttribute("aria-pressed", "false");
  await expect(row).toHaveAttribute("data-revealed", "true");

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("preset library — selecting country-capitals fills textarea, auto-applies, renders heatmap", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  // Mock returns Paris for the main prompt, Roma/Tokyo/etc. for each country.
  // Specific token doesn't matter for this test — we're checking that the
  // 5-column heatmap renders, meaning all 5 preset prompts fanned out.
  await page.route("**/api/sessions/*/decode-residual-grid", async (route) => {
    const body = route.request().postDataJSON() as { prompt?: string } | null;
    const prompt = (body?.prompt ?? "").toLowerCase();
    let token = " Paris";
    if (prompt.includes("italy")) token = " Roma";
    else if (prompt.includes("japan")) token = " Tokyo";
    else if (prompt.includes("brazil")) token = " Brasilia";
    else if (prompt.includes("egypt")) token = " Cairo";
    else if (prompt.includes("germany")) token = " Berlin";
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cells: [
          { layer: 2, sublayer: "attn", position: 4, tokens: [{ token, logit: 6.0 }, { token: " other", logit: 5.0 }] },
        ],
        prompt_tokens: ["The", "Eiffel", "Tower", "is", "in"],
        num_layers: 22,
      }),
    });
  });

  const fixture = fs.readFileSync(CIRCUIT_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-circuit.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByTestId("causal-story-panel").waitFor({ state: "visible", timeout: 5000 });

  // Open the prompt-set section, then load the country-capitals preset.
  await page.getByTestId("causal-story-set-toggle").click();
  await page.getByTestId("causal-story-set-preset").selectOption("country-capitals");

  // Caption appears and references the recommended main prompt.
  const caption = page.getByTestId("causal-story-set-preset-caption");
  await expect(caption).toBeVisible();
  await expect(caption).toContainText("Country capitals");
  await expect(caption).toContainText("recommended main prompt");

  // Textarea filled with the 5 country-capitals comparison prompts.
  const textarea = page.getByTestId("causal-story-set-input");
  await expect(textarea).toHaveValue(/The capital of Italy is/);
  await expect(textarea).toHaveValue(/The capital of Japan is/);

  // Auto-applied — heatmap renders with reference + 5 comparison columns.
  const heatmap = page.getByTestId("divergence-heatmap");
  await expect(heatmap).toBeVisible({ timeout: 5000 });

  // Reference column always matches itself.
  await expect(page.getByTestId("divergence-cell-0-0"))
    .toHaveAttribute("data-match", "true", { timeout: 5000 });

  // Italy column (index 1) — token differs (Roma vs Paris) → divergence.
  await expect(page.getByTestId("divergence-cell-0-1")).toHaveAttribute("data-match", "false");

  // Manually editing the textarea clears the preset selection back to "(custom)".
  await textarea.fill("custom prompt 1\ncustom prompt 2\ncustom prompt 3");
  await expect(page.getByTestId("causal-story-set-preset")).toHaveValue("");
  await expect(caption).not.toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("divergence heatmap renders 3-prompt matrix with correct match/diverge cells", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  // Body-inspecting route: returns Paris-cells for the main prompt
  // ("Eiffel Tower"), Roma for "Italy", Madrid for "Spain", Paris-also for
  // "France replicated". Reference (col 0) = main prompt → match w/ itself.
  // Col 1 = Italy → diverges. Col 2 = Spain → diverges. Col 3 = match.
  await page.route("**/api/sessions/*/decode-residual-grid", async (route) => {
    const body = route.request().postDataJSON() as { prompt?: string } | null;
    const prompt = (body?.prompt ?? "").toLowerCase();
    let token = " Paris";
    if (prompt.includes("italy")) token = " Roma";
    else if (prompt.includes("spain")) token = " Madrid";
    // "Eiffel" / "France replicated" / unknown → Paris
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cells: [
          { layer: 2, sublayer: "attn", position: 4, tokens: [{ token, logit: 6.0 }, { token: " other", logit: 5.0 }] },
        ],
        prompt_tokens: ["The", "Eiffel", "Tower", "is", "in"],
        num_layers: 22,
      }),
    });
  });

  const fixture = fs.readFileSync(CIRCUIT_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-circuit.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByTestId("causal-story-panel").waitFor({ state: "visible", timeout: 5000 });

  // Open the prompt-set textarea and apply 3 prompts.
  await page.getByTestId("causal-story-set-toggle").click();
  await page.getByTestId("causal-story-set-input").fill(
    "The Colosseum is in Italy\nThe Sagrada Familia is in Spain\nThe Eiffel Tower is in France",
  );
  await page.getByTestId("causal-story-set-apply").click();

  // Heatmap appears (4 cols: 1 ref + 3 comparisons; 1 row: L2.attn.h1).
  const heatmap = page.getByTestId("divergence-heatmap");
  await expect(heatmap).toBeVisible({ timeout: 5000 });

  // Reference column (col 0) — always matches itself.
  await expect(page.getByTestId("divergence-cell-0-0"))
    .toHaveAttribute("data-match", "true", { timeout: 5000 });
  // Italy column → diverge.
  await expect(page.getByTestId("divergence-cell-0-1")).toHaveAttribute("data-match", "false");
  // Spain column → diverge.
  await expect(page.getByTestId("divergence-cell-0-2")).toHaveAttribute("data-match", "false");
  // "France" prompt — same Paris token → match.
  await expect(page.getByTestId("divergence-cell-0-3")).toHaveAttribute("data-match", "true");

  // Specificity strip — 1 row diverges from 2 of 3 comparisons → 0.667.
  const specEl = page.getByTestId("divergence-spec-0");
  const specValue = await specEl.getAttribute("data-spec");
  expect(specValue).not.toBeNull();
  expect(parseFloat(specValue!)).toBeCloseTo(2 / 3, 2);

  // Mode toggle: switch to Jaccard, ensure cells re-render (data-match stays
  // since the attribute is mode-independent — color changes only).
  await page.getByTestId("divergence-mode-jaccard").click();
  await expect(page.getByTestId("divergence-mode-jaccard")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("divergence-mode-top1")).toHaveAttribute("aria-pressed", "false");

  // Clear via the textarea panel — heatmap disappears.
  await page.getByTestId("causal-story-set-clear").click();
  await expect(heatmap).not.toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("comparative story renders prompt-B sub-row with divergence highlighting", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  // Single mock dispatches on the JSON body's `prompt` field so prompt A and
  // prompt B return different lens cells. The circuit fixture's lone in-circuit
  // edge is (writer_layer=2, attn.h1) at position=4.
  await page.route("**/api/sessions/*/decode-residual-grid", async (route) => {
    const body = route.request().postDataJSON() as { prompt?: string } | null;
    const isItaly = !!body?.prompt && body.prompt.toLowerCase().includes("italy");
    const tokens = isItaly
      ? [{ token: " Roma", logit: 6.0 }, { token: " Italia", logit: 5.0 }]
      : [{ token: " Paris", logit: 6.0 }, { token: " France", logit: 5.0 }];
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cells: [{ layer: 2, sublayer: "attn", position: 4, tokens }],
        prompt_tokens: ["The", "Eiffel", "Tower", "is", "in"],
        num_layers: 22,
      }),
    });
  });

  const fixture = fs.readFileSync(CIRCUIT_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-circuit.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByTestId("causal-story-panel").waitFor({ state: "visible", timeout: 5000 });

  // Initial state — A-only row (no compare mode).
  const row = page.getByTestId("causal-story-row-L2-attn.h1");
  await expect(row).toHaveAttribute("data-compare-active", "false");

  // Open the compare input, type prompt B, and apply.
  await page.getByTestId("causal-story-compare-toggle").click();
  await page.getByTestId("causal-story-compare-input").fill("The Colosseum is in Italy");
  await page.getByTestId("causal-story-compare-apply").click();

  // Compare-mode flag flips on; the per-row testid should appear.
  await expect(row).toHaveAttribute("data-compare-active", "true", { timeout: 5000 });
  const compareRow = page.getByTestId("causal-story-row-L2-attn.h1-compare");
  await expect(compareRow).toBeVisible();
  // Prompt-B tokens render with the divergent values.
  await expect(compareRow.getByText(" Roma", { exact: true })).toBeVisible();
  await expect(compareRow.getByText(" Italia", { exact: true })).toBeVisible();

  // Clear button removes compare mode.
  await page.getByTestId("causal-story-compare-clear").click();
  await expect(row).toHaveAttribute("data-compare-active", "false");

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});

test("comparative story copy-as-markdown includes A and B sub-rows", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/");

  await page.route("**/api/sessions/*/decode-residual-grid", async (route) => {
    const body = route.request().postDataJSON() as { prompt?: string } | null;
    const isItaly = !!body?.prompt && body.prompt.toLowerCase().includes("italy");
    const tokens = isItaly
      ? [{ token: " Roma", logit: 6.0 }]
      : [{ token: " Paris", logit: 6.0 }];
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cells: [{ layer: 2, sublayer: "attn", position: 4, tokens }],
        prompt_tokens: ["The", "Eiffel", "Tower", "is", "in"],
        num_layers: 22,
      }),
    });
  });

  const fixture = fs.readFileSync(CIRCUIT_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-circuit.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByTestId("causal-story-panel").waitFor({ state: "visible", timeout: 5000 });
  await page.getByTestId("causal-story-compare-toggle").click();
  await page.getByTestId("causal-story-compare-input").fill("Italy");
  await page.getByTestId("causal-story-compare-apply").click();
  await expect(page.getByTestId("causal-story-row-L2-attn.h1"))
    .toHaveAttribute("data-compare-active", "true", { timeout: 5000 });

  // Wait for both lens grids to settle so the markdown contains B tokens.
  await expect(page.getByTestId("causal-story-row-L2-attn.h1-compare")).toBeVisible();

  await page.getByTestId("causal-story-copy-md").click();
  const clipboardText = await page.evaluate(() => navigator.clipboard.readText());
  expect(clipboardText).toContain('_compare with: "Italy"_');
  expect(clipboardText).toContain("- **L2 attn.h1** — A residual:  Paris");
  expect(clipboardText).toContain("  - B residual:  Roma");
});

test("causal story copy-as-markdown writes to clipboard", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/");

  await page.route("**/api/sessions/*/decode-residual-grid", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cells: [
          { layer: 2, sublayer: "attn", position: 4, tokens: [{ token: " Paris-md", logit: 5.0 }] },
        ],
        prompt_tokens: ["The", "Eiffel", "Tower", "is", "in"],
        num_layers: 22,
      }),
    });
  });

  const fixture = fs.readFileSync(CIRCUIT_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-circuit.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByTestId("causal-story-panel").waitFor({ state: "visible", timeout: 5000 });
  await page.getByTestId("causal-story-copy-md").click();

  // Read clipboard via page.evaluate
  const clipboardText = await page.evaluate(() => navigator.clipboard.readText());
  expect(clipboardText).toContain("## Causal Story — pos 4");
  expect(clipboardText).toContain("- **L2 attn.h1** — residual:  Paris-md");
});

test("generation panel preserves leading spaces on streamed tokens", async ({ page }) => {
  // The fixture's generate result has tokens [" there", " was"] with
  // leading SP-space markers. The panel renders prompt + tokens
  // concatenated; we assert the rendered DOM text contains the full
  // sentence WITH the inter-word gap, since browsers should honor
  // white-space:pre-wrap on the generation div.
  const fixture = fs.readFileSync(FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "sample.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });
  await expect(page.getByText(/Imported 3 result/)).toBeVisible();

  // The fixture's generate result is the most-recent (timestamp 1700000002000)
  // so GenerationOutput picks it up. Locate the heading then walk to the
  // panel content so we don't accidentally match other "Once upon a time"
  // text on the page (none currently, but cheap insurance).
  const heading = page.getByRole("heading", { name: "Generation Output" });
  await expect(heading).toBeVisible();
  // The panel is the section containing the heading.
  const panel = heading.locator("xpath=..").locator("xpath=..");
  // textContent collapses inline-element boundaries into a single
  // string while preserving the underlying text characters — exactly
  // what the user sees when they read the panel.
  const text = (await panel.textContent()) ?? "";
  expect(text).toContain("Once upon a time there was");
});
