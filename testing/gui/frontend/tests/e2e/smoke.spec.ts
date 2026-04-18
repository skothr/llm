import { test, expect, type Page } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FIXTURE_PATH = path.join(__dirname, "fixtures", "sample.json");
const AP_FIXTURE_PATH = path.join(__dirname, "fixtures", "activation-patching.json");

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

  await page.getByText(/Activation Patching/).waitFor({ state: "visible", timeout: 5000 });

  const metric = page.getByRole("combobox", { name: /Metric/i }).first();
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
