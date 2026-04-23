import { useStore } from "../state/store";
import { LogitLensHeatmap } from "./visualizations/LogitLensHeatmap";
import { LayerInfluence } from "./visualizations/LayerInfluence";
import { ABDiff } from "./visualizations/ABDiff";
import { AttentionEntropy } from "./visualizations/AttentionEntropy";
import { ResidualNorms } from "./visualizations/ResidualNorms";
import { ActivationPatchingHeatmap } from "./visualizations/ActivationPatchingHeatmap";
import { PerHeadPatchingHeatmap } from "./visualizations/PerHeadPatchingHeatmap";
import { EdgeAttributionPanel } from "./visualizations/EdgeAttributionPanel";
import { ResultMetaEditor } from "./ResultMetaEditor";
import { ResultFilterBar, makeResultPredicate } from "./ResultFilterBar";
import { BulkActionBar } from "./BulkActionBar";
import { timeAgo, useTimeAgoTick } from "../utils/timeAgo";
import type { ProbeResult, PatchingCompleteData } from "../types/api";

function InterveneSummary({ result }: { result: ProbeResult }) {
  const completeMsg = result.data.find((m) => m.type === "complete") as
    | { type: "complete"; interventions_applied: number }
    | undefined;
  const modifiedCount = result.data.filter(
    (m) => m.type === "data" && "modified" in m && (m as { modified?: boolean }).modified
  ).length;

  return (
    <div>
      <h3 style={{ fontSize: 13, color: "#a0a0c0", marginBottom: 8 }}>
        Intervention Results - {result.sessionName}
      </h3>
      {completeMsg ? (
        <div style={{ fontSize: 13 }}>
          {completeMsg.interventions_applied} intervention(s) applied
          {modifiedCount > 0 && (
            <span style={{ color: "#8888aa" }}> ({modifiedCount} layer(s) modified)</span>
          )}
        </div>
      ) : (
        <p style={{ color: "#666" }}>Waiting for results...</p>
      )}
    </div>
  );
}

function getResult(
  results: ProbeResult[],
  pending: Record<string, ProbeResult>,
  id: string | null,
): ProbeResult | undefined {
  if (!id) return undefined;
  return pending[id] || results.find((r) => r.id === id);
}

export function VisualizationArea() {
  const results = useStore((s) => s.results);
  const pendingResults = useStore((s) => s.pendingResults);
  const activeResultId = useStore((s) => s.activeResultId);
  const setActiveResult = useStore((s) => s.setActiveResult);
  const clearResults = useStore((s) => s.clearResults);
  const deleteResult = useStore((s) => s.deleteResult);
  const recallResult = useStore((s) => s.recallResult);
  const selectedResultIds = useStore((s) => s.selectedResultIds);
  const toggleResultSelection = useStore((s) => s.toggleResultSelection);
  const selectResultRange = useStore((s) => s.selectResultRange);
  const filterTags = useStore((s) => s.filterTags);
  const filterPinnedOnly = useStore((s) => s.filterPinnedOnly);
  const filterQuery = useStore((s) => s.filterQuery);

  const now = useTimeAgoTick();

  const all = [...Object.values(pendingResults), ...results];
  const vizResults = all.filter((r) => r.operation !== "generate");
  // Count every in-flight run, including generate runs — surfaced in the
  // header so users can tell a probe/intervene panel is busy even when
  // they're looking at the viz area.
  const pendingCount = Object.keys(pendingResults).length;
  // Filter predicate is applied to non-B, non-pending results only. Pending
  // results always surface regardless of filters (a running probe should
  // never hide behind a stale filter). B-side results ride along with
  // their A in the pair view, so filtering them independently would
  // orphan the compare view.
  const predicate = makeResultPredicate({ filterTags, filterPinnedOnly, filterQuery });
  const headResults = vizResults.filter((r) => !r.isB);
  const visibleHeadResults = headResults.filter((r) => r.id in pendingResults || predicate(r));

  const rawActive = getResult(results, pendingResults, activeResultId);
  // Fall back to first visible if the active result was filtered away; the
  // user shouldn't see an empty panel because of a tag selection.
  const activeResult = rawActive && rawActive.operation !== "generate" && visibleHeadResults.includes(rawActive)
    ? rawActive
    : visibleHeadResults[0] || null;
  const isPending = activeResult ? activeResult.id in pendingResults : false;

  const abPair = activeResult && !activeResult.isB
    ? all.find((r) => r.isB && r.id === `${activeResult.id}-B`)
    : null;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8, gap: 12 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1, minWidth: 0 }}>
          <h2 style={{ margin: 0 }}>
            Visualization
            {isPending && <span style={{ color: "#4ecdc4", fontSize: 11, marginLeft: 8 }}>streaming...</span>}
            {pendingCount > 0 && (
              <span
                style={{ color: "#4ecdc4", fontSize: 11, marginLeft: 8, fontFamily: "monospace" }}
                title={`${pendingCount} run(s) currently in flight across all panels`}
              >
                {"\u25CB"} {pendingCount} running
              </span>
            )}
          </h2>
          {activeResult && <ResultMetaEditor result={activeResult} />}
        </div>
        {vizResults.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-end" }}>
            <button onClick={clearResults} style={{ fontSize: 11 }} title="Remove all non-pinned results">Clear All</button>
            {activeResult && !isPending && (
              <div style={{ display: "flex", gap: 4 }}>
                <button
                  onClick={() => recallResult(activeResult.id)}
                  style={{ fontSize: 10, padding: "1px 6px", background: "#0d2236", color: "#a0c0e0", border: "1px solid #1a5276" }}
                  title="Load this result's prompt + session + sampling params into the probe panel (doesn't auto-run)"
                >{"\u21BA"} recall</button>
                <button
                  onClick={() => deleteResult(activeResult.id)}
                  style={{ fontSize: 10, padding: "1px 6px", background: "#3a1a1a", color: "#e0a0a0" }}
                  title="Delete just this result"
                >del</button>
              </div>
            )}
          </div>
        )}
      </div>

      <ResultFilterBar />
      <BulkActionBar />

      {headResults.length > 1 && (
        <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap", alignItems: "center" }}>
          {visibleHeadResults.map((r) => {
            const isSelected = selectedResultIds.includes(r.id);
            return (
              <button
                key={r.id}
                onClick={(e) => {
                  if (e.shiftKey) {
                    e.preventDefault();
                    selectResultRange(r.id, visibleHeadResults.map((x) => x.id));
                    return;
                  }
                  if (e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    toggleResultSelection(r.id);
                    return;
                  }
                  setActiveResult(r.id);
                }}
                style={{
                  fontSize: 11,
                  background: isSelected
                    ? "#2a5e96"
                    : r.id === activeResultId ? "#1a5276" : "#0d1b2a",
                  border: isSelected
                    ? "2px solid #6ba6d4"
                    : r.pinned ? "1px solid #c08020" : undefined,
                  // Normalize padding so the 2px-border selected state
                  // doesn't shift surrounding tabs by 1px.
                  padding: isSelected ? "1px 7px" : "2px 8px",
                }}
                title={[
                  r.notes,
                  isSelected ? "(Selected — Ctrl/Cmd+click to deselect)" : "Ctrl/Cmd+click to multi-select; Shift+click for range",
                ].filter(Boolean).join("\n")}
              >
                {r.pinned && <span style={{ color: "#ffc040", marginRight: 3 }}>{"\u2605"}</span>}
                {r.operation} | {r.sessionName}
                {all.find((b) => b.isB && b.id === `${r.id}-B`) ? " (A/B)" : ""}
                {r.id in pendingResults ? " ..." : (
                  <span style={{ color: "#667", marginLeft: 4, fontSize: 10 }}>{timeAgo(r.timestamp, now)}</span>
                )}
                {r.tags && r.tags.length > 0 && (
                  <span style={{ color: "#88a0c0", marginLeft: 4 }}>#{r.tags[0]}{r.tags.length > 1 ? `+${r.tags.length - 1}` : ""}</span>
                )}
              </button>
            );
          })}
          {visibleHeadResults.length < headResults.length && (
            <span style={{ fontSize: 10, color: "#667" }}>
              ({headResults.length - visibleHeadResults.length} hidden by filter)
            </span>
          )}
        </div>
      )}

      {activeResult ? (
        abPair && activeResult.operation === "logit-lens" ? (
          <ABDiff resultA={activeResult} resultB={abPair} />
        ) : activeResult.operation === "logit-lens" ? (
          <LogitLensHeatmap result={activeResult} />
        ) : activeResult.operation === "intervene" ? (
          activeResult.data.some((m) => m.type === "data" && "predictions" in m) ? (
            <LogitLensHeatmap result={activeResult} />
          ) : (
            <InterveneSummary result={activeResult} />
          )
        ) : activeResult.operation === "influence" ? (
          <LayerInfluence result={activeResult} />
        ) : activeResult.operation === "attention" ? (
          <AttentionEntropy result={activeResult} />
        ) : activeResult.operation === "residual-norms" ? (
          <ResidualNorms result={activeResult} />
        ) : activeResult.operation === "activation-patching" ? (
          activeResult.data.find((m): m is PatchingCompleteData => m.type === "complete")?.summary.mode === "edge"
            ? <EdgeAttributionPanel result={activeResult} />
            : activeResult.data.find((m): m is PatchingCompleteData => m.type === "complete")?.summary.mode === "approx_head"
            ? <PerHeadPatchingHeatmap result={activeResult} />
            : <ActivationPatchingHeatmap result={activeResult} />
        ) : (
          <p style={{ color: "#666" }}>No visualization for {activeResult.operation}</p>
        )
      ) : (
        <p style={{ color: "#666" }}>Run a probe to see results here</p>
      )}
    </div>
  );
}
