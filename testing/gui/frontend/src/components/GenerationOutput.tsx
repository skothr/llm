import { useState } from "react";
import { useStore } from "../state/store";
import type { GenerateData, ProbeResult } from "../types/api";

function displayToken(text: string): string {
  if (text === "") return "<empty>";
  return text.replace(/\n/g, "\\n").replace(/\t/g, "\\t").replace(/\r/g, "\\r").replace(/ /g, "\u00B7");
}

export function GenerationOutput() {
  const results = useStore((s) => s.results);
  const pendingResults = useStore((s) => s.pendingResults);

  const all = [...Object.values(pendingResults), ...results];
  const genResults = all
    .filter((r) => r.operation === "generate")
    .sort((a, b) => {
      const aB = a.id.endsWith("-B") ? 1 : 0;
      const bB = b.id.endsWith("-B") ? 1 : 0;
      return aB - bB || b.timestamp - a.timestamp;
    });

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
        {genResults.slice(0, 2).map((r) => (
          <GenerationPanel key={r.id} result={r} isPending={r.id in pendingResults} />
        ))}
      </div>
    </div>
  );
}

function GenerationPanel({ result, isPending }: { result: ProbeResult; isPending: boolean }) {
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  const tokens = result.data.filter(
    (m): m is GenerateData => m.type === "data" && "token" in m && "step" in m
  );

  const selectedToken = selectedStep !== null ? tokens.find((t) => t.step === selectedStep) || null : null;

  return (
    <div style={{ flex: "1 1 0", minWidth: 0, maxWidth: "50%", overflow: "hidden" }}>
      <div style={{ fontSize: 12, color: "#8888aa", marginBottom: 4 }}>
        {result.sessionName} - "{result.prompt.slice(0, 30)}"
        {isPending && <span style={{ color: "#4ecdc4", marginLeft: 4 }}>generating...</span>}
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 14, lineHeight: 1.8, overflowWrap: "break-word", overflowY: "auto", maxHeight: 200 }}>
        <span style={{ color: "#6688aa" }}>{result.prompt}</span>
        {tokens.map((tok) =>
          tok.token === "<eos>" ? (
            <span key={tok.step} onClick={() => setSelectedStep(selectedStep === tok.step ? null : tok.step)}
              style={{ cursor: "pointer", background: selectedStep === tok.step ? "#1a5276" : "transparent", color: "#4a6a4a", fontSize: 11, padding: "1px 3px", borderRadius: 2, border: "1px solid #3a5a3a" }}>eos</span>
          ) : (
            <span key={tok.step} onClick={() => setSelectedStep(selectedStep === tok.step ? null : tok.step)}
              style={{ cursor: "pointer", background: selectedStep === tok.step ? "#1a5276" : "transparent", borderRadius: 2, padding: "0 1px", whiteSpace: "pre" }}>{tok.token}</span>
          )
        )}
        {isPending && <span className="cursor-blink" style={{ color: "#4ecdc4" }}>|</span>}
      </div>

      {selectedToken && (
        <div style={{ marginTop: 8, padding: 8, background: "#0d1b2a", borderRadius: 4, fontSize: 12 }}>
          <div style={{ color: "#a0a0c0", marginBottom: 4 }}>Step {selectedToken.step} - top alternatives:</div>
          {selectedToken.top_k.map((alt, i) => (
            <div key={i} style={{ display: "flex", gap: 8 }}>
              <span style={{ color: i === 0 ? "#4ecdc4" : "#888", fontFamily: "monospace" }}>{displayToken(alt.token)}</span>
              <span style={{ color: "#666" }}>{(alt.prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
