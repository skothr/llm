import { useState } from "react";
import { useStore } from "../state/store";
import type { GenerateData, ProbeResult } from "../types/api";

export function GenerationOutput() {
  const results = useStore((s) => s.results);
  const [selectedToken, setSelectedToken] = useState<GenerateData | null>(null);

  const genResults = results.filter((r) => r.operation === "generate");
  if (genResults.length === 0) {
    return (
      <div>
        <h2>Generation Output</h2>
        <p style={{ color: "#666" }}>Run generate to see token stream here</p>
      </div>
    );
  }

  return (
    <div>
      <h2>Generation Output</h2>
      <div style={{ display: "flex", gap: 16 }}>
        {genResults.slice(0, 2).map((result) => (
          <GenerationPanel
            key={result.id}
            result={result}
            selectedToken={selectedToken}
            onSelectToken={setSelectedToken}
          />
        ))}
      </div>
    </div>
  );
}

function GenerationPanel({
  result,
  selectedToken,
  onSelectToken,
}: {
  result: ProbeResult;
  selectedToken: GenerateData | null;
  onSelectToken: (t: GenerateData | null) => void;
}) {
  const tokens = result.data.filter(
    (m): m is GenerateData => m.type === "data" && "token" in m && "step" in m
  );

  return (
    <div style={{ flex: 1 }}>
      <div style={{ fontSize: 12, color: "#8888aa", marginBottom: 4 }}>
        {result.sessionName} - "{result.prompt.slice(0, 30)}"
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 14, lineHeight: 1.8 }}>
        <span style={{ color: "#6688aa" }}>{result.prompt}</span>
        {tokens.map((tok) => (
          <span
            key={tok.step}
            onClick={() => onSelectToken(selectedToken?.step === tok.step ? null : tok)}
            style={{
              cursor: "pointer",
              background:
                selectedToken?.step === tok.step ? "#1a5276" : "transparent",
              borderRadius: 2,
              padding: "0 1px",
            }}
          >
            {tok.token}
          </span>
        ))}
      </div>

      {selectedToken && (
        <div
          style={{
            marginTop: 8,
            padding: 8,
            background: "#0d1b2a",
            borderRadius: 4,
            fontSize: 12,
          }}
        >
          <div style={{ color: "#a0a0c0", marginBottom: 4 }}>
            Step {selectedToken.step} - top alternatives:
          </div>
          {selectedToken.top_k.map((alt, i) => (
            <div key={i} style={{ display: "flex", gap: 8 }}>
              <span style={{ color: i === 0 ? "#4ecdc4" : "#888" }}>
                {alt.token}
              </span>
              <span style={{ color: "#666" }}>
                {(alt.prob * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
