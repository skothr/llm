import { useStore } from "../state/store";
import { LogitLensHeatmap } from "./visualizations/LogitLensHeatmap";
import { LayerInfluence } from "./visualizations/LayerInfluence";
import { ABDiff } from "./visualizations/ABDiff";
import { AttentionEntropy } from "./visualizations/AttentionEntropy";
import { ResidualNorms } from "./visualizations/ResidualNorms";
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

  const all = [...Object.values(pendingResults), ...results];
  const vizResults = all.filter((r) => r.operation !== "generate");
  const rawActive = getResult(results, pendingResults, activeResultId);
  const activeResult = rawActive && rawActive.operation !== "generate"
    ? rawActive
    : vizResults.find((r) => !r.isB) || null;
  const isPending = activeResult ? activeResult.id in pendingResults : false;

  const abPair = activeResult && !activeResult.isB
    ? all.find((r) => r.isB && r.id === `${activeResult.id}-B`)
    : null;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <h2>
          Visualization
          {isPending && <span style={{ color: "#4ecdc4", fontSize: 11, marginLeft: 8 }}>streaming...</span>}
        </h2>
        {vizResults.length > 0 && (
          <button onClick={clearResults} style={{ fontSize: 11 }}>Clear All</button>
        )}
      </div>

      {vizResults.length > 1 && (
        <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap" }}>
          {vizResults.filter((r) => !r.isB).map((r) => (
            <button
              key={r.id}
              onClick={() => setActiveResult(r.id)}
              style={{
                fontSize: 11, padding: "2px 8px",
                background: r.id === activeResultId ? "#1a5276" : "#0d1b2a",
              }}
            >
              {r.operation} | {r.sessionName}
              {all.find((b) => b.isB && b.id === `${r.id}-B`) ? " (A/B)" : ""}
              {r.id in pendingResults ? " ..." : ""}
            </button>
          ))}
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
