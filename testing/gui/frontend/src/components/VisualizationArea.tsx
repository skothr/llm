import { useStore } from "../state/store";
import { LogitLensHeatmap } from "./visualizations/LogitLensHeatmap";
import { LayerInfluence } from "./visualizations/LayerInfluence";
import { ABDiff } from "./visualizations/ABDiff";
import { AttentionEntropy } from "./visualizations/AttentionEntropy";
import { ResidualNorms } from "./visualizations/ResidualNorms";
import { ResultMetaEditor } from "./ResultMetaEditor";
import { ResultFilterBar, makeResultPredicate } from "./ResultFilterBar";
import type { ProbeResult } from "../types/api";

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
  const filterTags = useStore((s) => s.filterTags);
  const filterPinnedOnly = useStore((s) => s.filterPinnedOnly);
  const filterQuery = useStore((s) => s.filterQuery);

  const all = [...Object.values(pendingResults), ...results];
  const vizResults = all.filter((r) => r.operation !== "generate");
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

      {headResults.length > 1 && (
        <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap", alignItems: "center" }}>
          {visibleHeadResults.map((r) => (
            <button
              key={r.id}
              onClick={() => setActiveResult(r.id)}
              style={{
                fontSize: 11, padding: "2px 8px",
                background: r.id === activeResultId ? "#1a5276" : "#0d1b2a",
                border: r.pinned ? "1px solid #c08020" : undefined,
              }}
              title={r.notes || undefined}
            >
              {r.pinned && <span style={{ color: "#ffc040", marginRight: 3 }}>{"\u2605"}</span>}
              {r.operation} | {r.sessionName}
              {all.find((b) => b.isB && b.id === `${r.id}-B`) ? " (A/B)" : ""}
              {r.id in pendingResults ? " ..." : ""}
              {r.tags && r.tags.length > 0 && (
                <span style={{ color: "#88a0c0", marginLeft: 4 }}>#{r.tags[0]}{r.tags.length > 1 ? `+${r.tags.length - 1}` : ""}</span>
              )}
            </button>
          ))}
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
        ) : (
          <p style={{ color: "#666" }}>No visualization for {activeResult.operation}</p>
        )
      ) : (
        <p style={{ color: "#666" }}>Run a probe to see results here</p>
      )}
    </div>
  );
}
