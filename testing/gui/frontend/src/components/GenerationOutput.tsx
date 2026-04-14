import { useState } from "react";
import { useStore } from "../state/store";
import { displayToken } from "../utils/displayToken";
import type { GenerateData, ProbeResult } from "../types/api";

export function GenerationOutput() {
  const results = useStore((s) => s.results);
  const pendingResults = useStore((s) => s.pendingResults);

  const allGen = [...Object.values(pendingResults), ...results]
    .filter((r) => r.operation === "generate");

  const latestA = allGen
    .filter((r) => !r.id.endsWith("-B"))
    .sort((a, b) => b.timestamp - a.timestamp)[0];

  const latestB = latestA
    ? allGen.find((r) => r.id === `${latestA.id}-B`)
    : undefined;

  const panels = [latestA, latestB].filter(Boolean) as ProbeResult[];

  if (panels.length === 0) {
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
        {panels.map((r) => (
          <GenerationPanel key={r.id} result={r} isPending={r.id in pendingResults} />
        ))}
      </div>
    </div>
  );
}

function formatTopK(topK: GenerateData["top_k"]): string {
  const show = topK.map((a) => displayToken(a.token));
  const maxLen = Math.max(...show.map((s) => s.length));
  return topK
    .map((alt, i) => `${show[i].padEnd(maxLen)}  ${(alt.prob * 100).toFixed(1).padStart(5)}%`)
    .join("\n");
}

function GenerationPanel({ result, isPending }: { result: ProbeResult; isPending: boolean }) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; step: number; content: string } | null>(null);

  const tokens = result.data.filter(
    (m): m is GenerateData => m.type === "data" && "token" in m && "step" in m
  );

  return (
    <div style={{ flex: "1 1 0", minWidth: 0, maxWidth: "50%", overflow: "hidden", position: "relative" }} onClick={() => setTooltip(null)}>
      <div style={{ fontSize: 12, color: "#8888aa", marginBottom: 4 }}>
        {result.sessionName} - "{result.prompt.slice(0, 30)}"
        {isPending && <span style={{ color: "#4ecdc4", marginLeft: 4 }}>generating...</span>}
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 14, lineHeight: 1.8, whiteSpace: "pre-wrap", overflowY: "auto", maxHeight: 200 }}>
        <span style={{ color: "#6688aa" }}>{result.prompt}</span>
        {tokens.map((tok) => (
          <span
            key={tok.step}
            style={{
              cursor: "pointer",
              borderRadius: 2,
              padding: "0 1px",
              background: tooltip?.step === tok.step ? "#1a5276" : "transparent",
              ...(tok.token === "<eos>" ? { color: "#4a6a4a", fontSize: 11, padding: "1px 3px", border: "1px solid #3a5a3a" } : {}),
            }}
            onClick={(e) => {
              e.stopPropagation();
              if (tooltip?.step === tok.step) {
                setTooltip(null);
              } else {
                const content = `step ${tok.step}\n${formatTopK(tok.top_k)}`;
                const estW = 220, estH = 14 * (tok.top_k.length + 1) + 20;
                let x = e.clientX + 10;
                let y = e.clientY - 10;
                if (x + estW > window.innerWidth) x = e.clientX - estW - 10;
                if (y + estH > window.innerHeight) y = window.innerHeight - estH - 10;
                if (y < 10) y = 10;
                setTooltip({ x, y, step: tok.step, content });
              }
            }}
          >
            {tok.token === "<eos>" ? "eos" : tok.token}
          </span>
        ))}
        {isPending && <span className="cursor-blink" style={{ color: "#4ecdc4" }}>|</span>}
      </div>
      {tooltip && (
        <div style={{
          position: "fixed",
          left: tooltip.x,
          top: tooltip.y,
          background: "#16213e",
          border: "1px solid #1a5276",
          borderRadius: 4,
          padding: "6px 10px",
          fontFamily: "monospace",
          fontSize: 12,
          whiteSpace: "pre",
          pointerEvents: "none",
          zIndex: 100,
        }}>
          {tooltip.content}
        </div>
      )}
    </div>
  );
}
