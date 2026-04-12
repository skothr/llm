import { useStore } from "../state/store";
import { LogitLensHeatmap } from "./visualizations/LogitLensHeatmap";
import { LayerInfluence } from "./visualizations/LayerInfluence";
import { ABDiff } from "./visualizations/ABDiff";

export function VisualizationArea() {
  const results = useStore((s) => s.results);
  const activeResultId = useStore((s) => s.activeResultId);
  const setActiveResult = useStore((s) => s.setActiveResult);
  const clearResults = useStore((s) => s.clearResults);

  const vizResults = results.filter((r) => r.operation !== "generate");
  const activeResult = vizResults.find((r) => r.id === activeResultId);

  const abPair = activeResult && !activeResult.id.includes("-B")
    ? vizResults.find((r) => r.id === `${activeResult.id}-B`)
    : null;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <h2>Visualization</h2>
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
                fontSize: 11,
                padding: "2px 8px",
                background: r.id === activeResultId ? "#1a5276" : "#0d1b2a",
              }}
            >
              {r.operation} | {r.sessionName}
              {vizResults.find((b) => b.id === `${r.id}-B`) ? " (A/B)" : ""}
            </button>
          ))}
        </div>
      )}

      {activeResult ? (
        abPair && activeResult.operation === "logit-lens" ? (
          <ABDiff resultA={activeResult} resultB={abPair} />
        ) : activeResult.operation === "logit-lens" ? (
          <LogitLensHeatmap result={activeResult} />
        ) : activeResult.operation === "influence" ? (
          <LayerInfluence result={activeResult} />
        ) : (
          <p style={{ color: "#666" }}>Visualization for {activeResult.operation} coming soon</p>
        )
      ) : (
        <p style={{ color: "#666" }}>Run a probe to see results here</p>
      )}
    </div>
  );
}
