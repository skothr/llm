import { useRef, useState } from "react";
import { useStore } from "../state/store";
import { downloadJSON } from "../utils/download";
import {
  buildExperimentFile,
  parseExperimentFile,
  missingSessionsFromImport,
} from "../utils/experiment";

// Export/Import control pair for the full experiment state. Sits next to
// the TabBar in App.tsx. Keeps zero local state when idle — an import
// issue banner is the only transient it owns.
export function ExperimentIO() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [banner, setBanner] = useState<{ tone: "ok" | "warn" | "err"; text: string } | null>(null);
  const [pinnedOnly, setPinnedOnly] = useState(false);

  const handleExport = () => {
    const s = useStore.getState();
    // Pinned-only narrows the results slice for sharing curated findings
    // without leaking exploratory dead-ends. Non-result fields (prompt,
    // params, library) still export in full — those are always reusable.
    const results = pinnedOnly ? s.results.filter((r) => r.pinned) : s.results;
    const file = buildExperimentFile({
      activeTab: s.activeTab,
      prompt: s.prompt,
      operation: s.operation,
      targetSession: s.targetSession,
      targetSessionB: s.targetSessionB,
      samplingParams: s.samplingParams,
      interventionSpecs: s.interventionSpecs,
      captureLogitLens: s.captureLogitLens,
      intervenePrompt: s.intervenePrompt,
      interveneSession: s.interveneSession,
      promptLibrary: s.promptLibrary,
      results,
      activeResultId: s.activeResultId && results.some((r) => r.id === s.activeResultId) ? s.activeResultId : null,
      sessions: s.sessions,
    });
    // Timestamped filename + first 24 chars of the prompt as a human hint.
    const promptSlug = (s.prompt || "experiment")
      .replace(/[^a-z0-9_-]+/gi, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 24) || "experiment";
    const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const suffix = pinnedOnly ? "_pinned" : "";
    downloadJSON(`${promptSlug}${suffix}_${ts}.json`, file);
    setBanner({
      tone: "ok",
      text: `Exported ${results.length} result(s)${pinnedOnly ? " (pinned only)" : ""}, ${s.promptLibrary.length} saved prompt(s).`,
    });
  };

  const handleImportClick = () => fileInputRef.current?.click();

  const handleFile = async (evt: React.ChangeEvent<HTMLInputElement>) => {
    const file = evt.target.files?.[0];
    // Always reset the input so re-selecting the same file re-fires change.
    evt.target.value = "";
    if (!file) return;

    let text: string;
    try { text = await file.text(); }
    catch (e) {
      setBanner({ tone: "err", text: `Could not read file: ${(e as Error).message}` });
      return;
    }
    const { file: parsed, issues } = parseExperimentFile(text);
    if (!parsed) {
      setBanner({ tone: "err", text: issues[0]?.message ?? "Invalid experiment file" });
      return;
    }

    // Apply state. This overwrites prompt/params/library/results; session
    // state and backend-fetched lists are untouched on purpose.
    useStore.setState({
      prompt: parsed.prompt,
      operation: parsed.operation,
      targetSession: parsed.targetSession,
      targetSessionB: parsed.targetSessionB,
      samplingParams: parsed.samplingParams,
      interventionSpecs: parsed.interventionSpecs,
      captureLogitLens: parsed.captureLogitLens,
      intervenePrompt: parsed.intervenePrompt,
      interveneSession: parsed.interveneSession,
      promptLibrary: parsed.promptLibrary,
      results: parsed.results,
      activeResultId: parsed.activeResultId,
      ...(parsed.activeTab ? { activeTab: parsed.activeTab } : {}),
    });

    const currentSessions = useStore.getState().sessions;
    const missing = missingSessionsFromImport(parsed, currentSessions);
    const warnCount = issues.filter((i) => i.severity === "warn").length;
    const parts: string[] = [`Imported ${parsed.results.length} result(s).`];
    if (missing.length > 0) parts.push(`Missing sessions: ${missing.join(", ")}.`);
    if (warnCount > 0) parts.push(`(${warnCount} warning${warnCount > 1 ? "s" : ""})`);
    setBanner({
      tone: missing.length > 0 ? "warn" : "ok",
      text: parts.join(" "),
    });
  };

  const bannerColor = banner == null ? undefined
    : banner.tone === "err" ? "#ff6b6b"
    : banner.tone === "warn" ? "#f0ad4e"
    : "#4ecdc4";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, padding: "2px 8px 6px 8px" }}>
      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
        <button
          onClick={handleExport}
          title={pinnedOnly
            ? "Download pinned results only, plus current prompt/params/library."
            : "Download the current prompt, params, library, and all results as one JSON file."}
          style={{ flex: 1, fontSize: 11, padding: "2px 6px" }}
        >export{pinnedOnly ? " \u2605" : ""}</button>
        <button
          onClick={handleImportClick}
          title="Load a previously-exported experiment. Overwrites current prompt/params/results; sessions untouched."
          style={{ flex: 1, fontSize: 11, padding: "2px 6px" }}
        >import</button>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/json,.json"
          onChange={handleFile}
          style={{ display: "none" }}
        />
      </div>
      <label
        style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, color: "#667", cursor: "pointer" }}
        title="Limit export to pinned (starred) results only. Prompt, params, and library always export in full."
      >
        <input
          type="checkbox"
          checked={pinnedOnly}
          onChange={(e) => setPinnedOnly(e.target.checked)}
          style={{ margin: 0 }}
        />
        pinned only
      </label>
      {banner && (
        <div
          onClick={() => setBanner(null)}
          title="Click to dismiss"
          style={{
            fontSize: 10, color: bannerColor, cursor: "pointer",
            lineHeight: 1.3, paddingTop: 2,
          }}
        >{banner.text}</div>
      )}
    </div>
  );
}
