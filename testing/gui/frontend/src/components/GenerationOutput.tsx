import { useState, useEffect } from "react";
import { useStore } from "../state/store";
import { displayToken } from "../utils/displayToken";
import { ExportButtons } from "./ExportButtons";
import { downloadText } from "../utils/download";
import type { GenerateData, ProbeResult } from "../types/api";

function batchTokens(result: ProbeResult): string[] {
  return result.data
    .filter((m): m is GenerateData => m.type === "data" && "token" in m && "step" in m)
    .map((m) => m.token);
}

function batchStats(panels: ProbeResult[]): { uniqueCompletions: number; firstDivergence: number | null } {
  const seqs = panels.map(batchTokens);
  const completedSeqs = seqs.filter((s) => s.length > 0);
  if (completedSeqs.length < 2) {
    return { uniqueCompletions: new Set(completedSeqs.map((s) => s.join(""))).size, firstDivergence: null };
  }
  const uniqueCompletions = new Set(completedSeqs.map((s) => s.join(""))).size;
  const minLen = Math.min(...completedSeqs.map((s) => s.length));
  let firstDivergence: number | null = null;
  for (let i = 0; i < minLen; i++) {
    const ref = completedSeqs[0][i];
    if (completedSeqs.some((s) => s[i] !== ref)) { firstDivergence = i; break; }
  }
  // All compared prefixes equal but lengths differ → divergence is where
  // the shorter sequence ended.
  if (firstDivergence === null && new Set(completedSeqs.map((s) => s.length)).size > 1) {
    firstDivergence = minLen;
  }
  return { uniqueCompletions, firstDivergence };
}

export function GenerationOutput() {
  const results = useStore((s) => s.results);
  const pendingResults = useStore((s) => s.pendingResults);

  const allGen = [...Object.values(pendingResults), ...results]
    .filter((r) => r.operation === "generate");

  // Sort newest-first once; the logic below just slices out the relevant
  // group for the most recent run (either a fan-out batch or an A/B pair).
  const sorted = [...allGen].sort((a, b) => b.timestamp - a.timestamp);
  const newest = sorted[0];

  let panels: ProbeResult[] = [];
  let batchMode = false;
  if (newest?.batchId) {
    panels = sorted
      .filter((r) => r.batchId === newest.batchId)
      .sort((a, b) => (a.batchIndex ?? 0) - (b.batchIndex ?? 0));
    batchMode = true;
  } else if (newest) {
    const partner = sorted.find((r) => r.isB && r.id === `${newest.id}-B`);
    panels = [newest, partner].filter(Boolean) as ProbeResult[];
  }

  if (panels.length === 0) {
    return (
      <div>
        <h2>Generation Output</h2>
        <p style={{ color: "#666" }}>Run generate to see token stream here</p>
      </div>
    );
  }

  const stats = batchMode ? batchStats(panels) : null;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
        <h2 style={{ margin: 0 }}>Generation Output</h2>
        {batchMode && stats && (
          <span style={{ fontSize: 11, color: "#8888aa", fontFamily: "monospace" }}>
            {panels.length} seed{panels.length > 1 ? "s" : ""}
            {" · "}
            <span title="Number of distinct token sequences across the batch. 1 = every seed produced identical output.">
              {stats.uniqueCompletions} unique
            </span>
            {stats.firstDivergence !== null && (
              <>
                {" · "}
                <span title="Earliest step at which two runs produced different tokens (or ended). A low number means the batch diverges right away; high means runs mostly agree on the opening.">
                  first divergence @ step {stats.firstDivergence}
                </span>
              </>
            )}
          </span>
        )}
      </div>
      <div style={{ display: "flex", gap: 16, overflowX: batchMode ? "auto" : "visible" }}>
        {panels.map((r) => (
          <GenerationPanel
            key={r.id}
            result={r}
            isPending={r.id in pendingResults}
            narrow={batchMode && panels.length > 2}
          />
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

function exportText(result: ProbeResult, tokens: GenerateData[]): void {
  // Full text = prompt + joined token strings. Preserves whitespace exactly
  // as generated so it round-trips into an editor unchanged.
  const body = result.prompt + tokens.map((t) => t.token).join("");
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  downloadText(`generation_${result.sessionName}_${ts}.txt`, body);
}

function GenerationPanel({ result, isPending, narrow = false }: { result: ProbeResult; isPending: boolean; narrow?: boolean }) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; step: number; content: string } | null>(null);

  useEffect(() => {
    if (!tooltip) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setTooltip(null); };
    const onClick = () => setTooltip(null);
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onClick);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onClick);
    };
  }, [tooltip]);

  const tokens = result.data.filter(
    (m): m is GenerateData => m.type === "data" && "token" in m && "step" in m
  );

  const csvRows = (): (string | number)[][] => {
    const maxTopK = tokens.reduce((acc, t) => Math.max(acc, t.top_k.length), 0);
    const header = ["step", "token", "token_id"];
    for (let i = 0; i < maxTopK; i++) header.push(`top${i + 1}_token`, `top${i + 1}_prob`);
    const rows: (string | number)[][] = [header];
    for (const t of tokens) {
      const row: (string | number)[] = [t.step, t.token, t.token_id];
      for (let i = 0; i < maxTopK; i++) {
        const alt = t.top_k[i];
        row.push(alt?.token ?? "", alt?.prob ?? "");
      }
      rows.push(row);
    }
    return rows;
  };

  const seedLabel = result.seed != null ? `seed ${result.seed}` : null;
  // In batch mode we narrow each column so 4+ panels still fit. Keep the
  // wider default for the 1- or 2-panel A/B layouts so long prompts don't
  // wrap awkwardly.
  const panelStyle: React.CSSProperties = narrow
    ? { flex: "0 0 260px", minWidth: 0, maxWidth: 280, overflow: "hidden", position: "relative" }
    : { flex: "1 1 0", minWidth: 0, maxWidth: "50%", overflow: "hidden", position: "relative" };

  return (
    <div style={panelStyle}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4, gap: 8 }}>
        <div style={{ fontSize: 12, color: "#8888aa", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {seedLabel ? <span style={{ color: "#6688aa", fontFamily: "monospace" }}>{seedLabel} · </span> : null}
          {result.sessionName} - "{result.prompt.slice(0, 30)}"
          {isPending && <span style={{ color: "#4ecdc4", marginLeft: 4 }}>generating...</span>}
        </div>
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <button
            onClick={() => exportText(result, tokens)}
            title="Download the prompt+generation as a plain .txt file"
            style={{ fontSize: 10, padding: "1px 6px", background: "#0d1b2a", border: "1px solid #1a2540", color: "#a0a0c0", cursor: "pointer", borderRadius: 2 }}
          >txt</button>
          <ExportButtons
            filenameBase={`generation_${result.sessionName}`}
            getCSVRows={csvRows}
          />
        </div>
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
              ...(tok.token === "<stop>" ? { color: "#aa6a4a", fontSize: 11, padding: "1px 3px", border: "1px solid #5a3a3a" } : {}),
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
            {tok.token === "<eos>" ? "eos" : tok.token === "<stop>" ? "stop" : tok.token}
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
