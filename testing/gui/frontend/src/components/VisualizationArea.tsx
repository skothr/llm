import { useStore } from "../state/store";
import { LogitLensHeatmap } from "./visualizations/LogitLensHeatmap";
import { LayerInfluence } from "./visualizations/LayerInfluence";
import { ABDiff } from "./visualizations/ABDiff";
import { AttentionEntropy } from "./visualizations/AttentionEntropy";
import { ResidualNorms } from "./visualizations/ResidualNorms";
import type { ProbeResult } from "../types/api";

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
  const activeResult = getResult(results, pendingResults, activeResultId);
  const isPending = activeResultId ? activeResultId in pendingResults : false;

  const abPair = activeResult && !activeResult.id.includes("-B")
    ? all.find((r) => r.id === `${activeResult.id}-B`)
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
          {vizResults.filter((r) => !r.id.includes("-B")).map((r) => (
            <button
              key={r.id}
              onClick={() => setActiveResult(r.id)}
              style={{
                fontSize: 11, padding: "2px 8px",
                background: r.id === activeResultId ? "#1a5276" : "#0d1b2a",
              }}
            >
              {r.operation} | {r.sessionName}
              {all.find((b) => b.id === `${r.id}-B`) ? " (A/B)" : ""}
              {r.id in pendingResults ? " ..." : ""}
            </button>
          ))}
        </div>
      )}

      {activeResult ? (
        abPair && activeResult.operation === "logit-lens" ? (
          <ABDiff resultA={activeResult} resultB={abPair} />
        ) : activeResult.operation === "logit-lens" || activeResult.operation === "intervene" ? (
          <LogitLensHeatmap result={activeResult} />
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
