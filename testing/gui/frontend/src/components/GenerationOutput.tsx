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

// Per-step, per-token occurrence counts across a batch. Used for
// divergence coloring: if at step 7 four seeds produced " the" and one
// produced " a", the counts are { " the": 4, " a": 1 } — the majority
// tokens get amber and the outlier gets red.
type AgreementMap = Map<number, Map<string, number>>;

function agreementPerStep(panels: ProbeResult[]): AgreementMap {
  const byStep: AgreementMap = new Map();
  for (const r of panels) {
    const tokens = r.data.filter(
      (m): m is GenerateData => m.type === "data" && "token" in m && "step" in m,
    );
    for (const t of tokens) {
      let m = byStep.get(t.step);
      if (!m) { m = new Map(); byStep.set(t.step, m); }
      m.set(t.token, (m.get(t.token) ?? 0) + 1);
    }
  }
  return byStep;
}

// Agreement tier for (step, token) given a batch of size N. We always
// compare against N (not the count of seeds that reached this step) so
// a token that appears in 3 of 5 seeds reads as "minority" even if
// two other seeds simply haven't arrived there yet — this is what the
// user wants while watching a stream.
type AgreementTier = "all" | "majority" | "minority" | "unique";

function agreementTier(count: number, total: number): AgreementTier {
  if (count >= total) return "all";
  if (count >= Math.ceil(total / 2)) return "majority";
  if (count <= 1) return "unique";
  return "minority";
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
  // Compute agreement once per render so all panels share a single map.
  // Only meaningful when there are >=2 siblings in the batch.
  const agreement = batchMode && panels.length >= 2 ? agreementPerStep(panels) : null;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
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
        {agreement && (
          <span
            style={{ fontSize: 10, color: "#667", fontFamily: "monospace" }}
            title="Tokens in each panel are colored by how many seeds produced that same token at that step."
          >
            <span style={{ color: "#6bc06b" }}>{"\u25A0"}</span> all agree{"  "}
            <span style={{ color: "#e0c060" }}>{"\u25A0"}</span> majority{"  "}
            <span style={{ color: "#d08060" }}>{"\u25A0"}</span> minority{"  "}
            <span style={{ color: "#c06060" }}>{"\u25A0"}</span> unique
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
            agreement={agreement}
            batchSize={panels.length}
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

const TIER_COLOR: Record<AgreementTier, string> = {
  all: "#6bc06b",       // green — full consensus
  majority: "#e0c060",  // amber — majority consensus
  minority: "#d08060",  // dim red — some siblings agree
  unique: "#c06060",    // red — this seed stands alone
};

function GenerationPanel({
  result, isPending, narrow = false, agreement = null, batchSize = 0,
}: {
  result: ProbeResult;
  isPending: boolean;
  narrow?: boolean;
  agreement?: AgreementMap | null;
  batchSize?: number;
}) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; step: number; content: string } | null>(null);
  const recallResult = useStore((s) => s.recallResult);

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

  // Prefer the explicit sweepLabel (set by the ProbePanel fan-out flow
  // for any axis) over a bare seed number. Falls back to the seed so
  // older persisted results — written before sweepLabel existed — still
  // render something meaningful.
  const seedLabel = result.sweepLabel ?? (result.seed != null ? `seed ${result.seed}` : null);
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
          {!isPending && (
            <button
              onClick={() => recallResult(result.id)}
              title={`Load this seed's prompt/session/params into the probe panel${result.runParams ? "" : " (runParams not captured — will only restore prompt + session)"}`}
              style={{ fontSize: 10, padding: "1px 6px", background: "#0d2236", border: "1px solid #1a5276", color: "#a0c0e0", cursor: "pointer", borderRadius: 2 }}
            >{"\u21BA"}</button>
          )}
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
        {tokens.map((tok) => {
          // Agreement tint is applied as text color when we have a batch.
          // Special control tokens keep their existing visual style so
          // they stay visually distinct from normal content.
          let tint: string | undefined;
          let tintTitle: string | undefined;
          if (agreement && batchSize >= 2 && tok.token !== "<eos>" && tok.token !== "<stop>") {
            const stepMap = agreement.get(tok.step);
            const count = stepMap?.get(tok.token) ?? 0;
            if (count > 0) {
              const tier = agreementTier(count, batchSize);
              tint = TIER_COLOR[tier];
              tintTitle = `step ${tok.step}: ${count}/${batchSize} seeds produced this token (${tier})`;
            }
          }
          return (
            <span
              key={tok.step}
              title={tintTitle}
              style={{
                cursor: "pointer",
                borderRadius: 2,
                padding: "0 1px",
                background: tooltip?.step === tok.step ? "#1a5276" : "transparent",
                ...(tint ? { color: tint } : {}),
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
          );
        })}
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
