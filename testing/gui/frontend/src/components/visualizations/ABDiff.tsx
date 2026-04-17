import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import * as d3 from "d3";
import { displayToken } from "../../utils/displayToken";
import { useStore } from "../../state/store";
import { useWebSocket } from "../../hooks/useWebSocket";
import { sliceHiddenStatePosition } from "../../utils/hiddenState";
import { HiddenStateBarStrip } from "./HiddenStateBarStrip";
import type {
  LogitLensData,
  ProbeResult,
  CompareLogitLensData,
  CompareCell,
  PairMetrics,
} from "../../types/api";

interface Props {
  resultA: ProbeResult;
  resultB: ProbeResult;
}

type PinnedSource =
  | { kind: "A"; msg: LogitLensData }
  | { kind: "B"; msg: LogitLensData }
  | { kind: "compare"; frame: CompareLogitLensData; cell: CompareCell };

interface Pinned {
  source: PinnedSource;
  rowLabel: string;
  posIdx: number;
  x: number;
  y: number;
}

interface AlignedRow {
  originalLayer: number;
  sublayer: string;
  label: string;
  duplicate: boolean;
  dataA: LogitLensData | null;
  dataB: LogitLensData | null;
}

function buildOriginalLookup(data: LogitLensData[]) {
  const primary = new Map<string, LogitLensData>();
  const duplicates: { originalLayer: number; sublayer: string; insertAfter: number; data: LogitLensData }[] = [];

  for (const msg of data) {
    const orig = msg.original_layer ?? msg.layer;
    const key = `${orig}.${msg.sublayer}`;
    if (primary.has(key)) {
      const prevOrig = data[data.indexOf(msg) - 1]?.original_layer ?? data[data.indexOf(msg) - 1]?.layer ?? orig;
      duplicates.push({ originalLayer: orig, sublayer: msg.sublayer, insertAfter: prevOrig, data: msg });
    } else {
      primary.set(key, msg);
    }
  }
  return { primary, duplicates };
}

type CompareMetricKey = keyof Pick<PairMetrics, "kl_ab" | "js" | "cosine" | "top1_delta_prob" | "top1_match">;

interface CompareMetricDef {
  label: string;
  description: string;
  kind: "sequential" | "diverging" | "binary";
  interpolator: (t: number) => string;
  fixedDomain?: [number, number];
  format: (v: number | boolean) => string;
}

const LN2 = Math.log(2);

const COMPARE_METRICS: Record<CompareMetricKey, CompareMetricDef> = {
  kl_ab: {
    label: "KL(A ‖ B)",
    description: "Relative entropy in nats. 0 = identical; unbounded above.",
    kind: "sequential",
    interpolator: d3.interpolateInferno,
    format: (v) => (typeof v === "number" ? v.toFixed(4) : String(v)),
  },
  js: {
    label: "Jensen–Shannon",
    description: "Symmetric divergence in nats. 0 = identical; max = ln 2 ≈ 0.693.",
    kind: "sequential",
    interpolator: d3.interpolatePlasma,
    fixedDomain: [0, LN2],
    format: (v) => (typeof v === "number" ? v.toFixed(4) : String(v)),
  },
  cosine: {
    label: "Cosine similarity",
    description: "Cosine between full-vocab distributions. 1 = identical direction.",
    kind: "sequential",
    interpolator: d3.interpolateViridis,
    fixedDomain: [0, 1],
    format: (v) => (typeof v === "number" ? v.toFixed(4) : String(v)),
  },
  top1_delta_prob: {
    label: "Δ top-1 prob (A − B)",
    description: "p_A(argmax A) − p_B(argmax B). Diverging.",
    kind: "diverging",
    interpolator: d3.interpolateRdBu,
    fixedDomain: [-1, 1],
    format: (v) => (typeof v === "number" ? `${(v * 100).toFixed(3)}%` : String(v)),
  },
  top1_match: {
    label: "Top-1 match",
    description: "Binary: do A and B agree on the argmax token?",
    kind: "binary",
    interpolator: d3.interpolateViridis,
    fixedDomain: [0, 1],
    format: (v) => (typeof v === "boolean" ? (v ? "match" : "mismatch") : v === 1 ? "match" : "mismatch"),
  },
};

export function ABDiff({ resultA, resultB }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const sessionInfo = useStore((s) => s.sessionInfo);
  const [showDiff, setShowDiff] = useState(false);
  const [compareFrames, setCompareFrames] = useState<CompareLogitLensData[]>([]);
  const [compareMetric, setCompareMetric] = useState<CompareMetricKey>("js");
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    content: string;
  } | null>(null);
  const [pinned, setPinned] = useState<Pinned | null>(null);
  const ws = useWebSocket();

  const unpin = useCallback(() => setPinned(null), []);
  useEffect(() => {
    if (!pinned) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") unpin(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pinned, unpin]);

  const dataA = useMemo(
    () => resultA.data.filter((m): m is LogitLensData => m.type === "data" && "predictions" in m),
    [resultA.data]
  );
  const dataB = useMemo(
    () => resultB.data.filter((m): m is LogitLensData => m.type === "data" && "predictions" in m),
    [resultB.data]
  );

  const compareByKey = useMemo(() => {
    const map = new Map<string, CompareCell[]>();
    for (const frame of compareFrames) {
      map.set(`${frame.original_layer}.${frame.sublayer}`, frame.cells);
    }
    return map;
  }, [compareFrames]);

  const compareFrameByKey = useMemo(() => {
    const map = new Map<string, CompareLogitLensData>();
    for (const frame of compareFrames) {
      map.set(`${frame.original_layer}.${frame.sublayer}`, frame);
    }
    return map;
  }, [compareFrames]);

  const hasCompareData = compareFrames.length > 0;

  // Any time the prompt or A/B selection changes, the existing compare data
  // no longer matches — reset it so the user sees the stale-fallback diff.
  useEffect(() => {
    setCompareFrames([]);
    setCompareError(null);
  }, [resultA.id, resultB.id, resultA.prompt, resultB.prompt]);

  const alignedRows = useMemo(() => {
    const infoA = sessionInfo[resultA.sessionName];
    const infoB = sessionInfo[resultB.sessionName];

    const lookupA = buildOriginalLookup(dataA);
    const lookupB = buildOriginalLookup(dataB);

    const numOrig = Math.max(
      infoA?.original_num_layers ?? Math.ceil(dataA.length / 2),
      infoB?.original_num_layers ?? Math.ceil(dataB.length / 2),
    );

    const rows: AlignedRow[] = [];
    for (let i = 0; i < numOrig; i++) {
      for (const sub of ["attn", "ffn"]) {
        const key = `${i}.${sub}`;
        rows.push({
          originalLayer: i,
          sublayer: sub,
          label: `L${i}.${sub}`,
          duplicate: false,
          dataA: lookupA.primary.get(key) ?? null,
          dataB: lookupB.primary.get(key) ?? null,
        });
      }

      const dupsA = lookupA.duplicates.filter((d) => d.insertAfter === i);
      const dupsB = lookupB.duplicates.filter((d) => d.insertAfter === i);
      const maxDups = Math.max(dupsA.length, dupsB.length);
      for (let d = 0; d < maxDups; d++) {
        const dupA = dupsA[d];
        const dupB = dupsB[d];
        const orig = dupA?.originalLayer ?? dupB?.originalLayer ?? i;
        const sub = dupA?.sublayer ?? dupB?.sublayer ?? "attn";
        rows.push({
          originalLayer: orig,
          sublayer: sub,
          label: `[L${orig}].${sub}`,
          duplicate: true,
          dataA: dupA?.data ?? null,
          dataB: dupB?.data ?? null,
        });
      }
    }

    return rows;
  }, [dataA, dataB, sessionInfo, resultA.sessionName, resultB.sessionName]);

  const runCompare = useCallback(() => {
    if (comparing) return;
    if (resultA.prompt !== resultB.prompt) {
      setCompareError(
        `A and B were run on different prompts (A="${resultA.prompt.slice(0, 30)}…", B="${resultB.prompt.slice(0, 30)}…"). Exact comparison requires identical prompts.`
      );
      return;
    }
    setComparing(true);
    setCompareError(null);
    setCompareFrames([]);
    setShowDiff(true);

    const key = `compare-${resultA.id}-${resultB.id}`;
    ws.connect(
      key,
      `/ws/sessions/${resultA.sessionName}/compare-logit-lens`,
      { with_session: resultB.sessionName, prompt: resultA.prompt, top_k: 10 },
      {
        onMessage: (msg) => {
          const frame = msg as unknown as CompareLogitLensData;
          if (frame.type === "data" && Array.isArray(frame.cells)) {
            setCompareFrames((prev) => [...prev, frame]);
          }
        },
        onComplete: () => {
          setComparing(false);
        },
        onError: (message) => {
          setCompareError(message || "Compare failed");
          setComparing(false);
        },
        onDisconnect: () => setComparing(false),
      }
    );
  }, [comparing, resultA, resultB, ws]);

  useEffect(() => {
    if (!svgRef.current || alignedRows.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const firstData = alignedRows.find((r) => r.dataA || r.dataB);
    const numPositions = (firstData?.dataA ?? firstData?.dataB)?.predictions.length ?? 1;

    const completeMsgA = resultA.data.find((m) => m.type === "complete") as
      | { type: "complete"; summary?: { prompt_tokens?: string[] } }
      | undefined;
    const promptTokens = completeMsgA?.summary?.prompt_tokens;

    const numRows = alignedRows.length;
    const cellW = Math.max(24, Math.min(40, 400 / numPositions));
    const cellH = 18;
    const margin = { top: promptTokens ? 50 : 30, right: 20, bottom: 30, left: 80 };
    const panelWidth = margin.left + numPositions * cellW + margin.right;
    const gap = 40;
    const totalWidth = showDiff ? panelWidth * 3 + gap * 2 : panelWidth * 2 + gap;
    const height = margin.top + numRows * cellH + margin.bottom + (showDiff ? 30 : 0);

    svg.attr("width", totalWidth).attr("height", height);

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

    function drawPanel(side: "A" | "B", offsetX: number, label: string) {
      const g = svg.append("g").attr("transform", `translate(${offsetX + margin.left},${margin.top})`);

      svg.append("text")
        .attr("x", offsetX + margin.left + (numPositions * cellW) / 2)
        .attr("y", promptTokens ? 14 : 16)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .attr("fill", "#a0a0c0")
        .text(label);

      if (promptTokens) {
        const tokG = svg.append("g").attr("transform", `translate(${offsetX + margin.left},${margin.top - 6})`);
        promptTokens.slice(0, numPositions).forEach((tok, i) => {
          tokG.append("text")
            .attr("x", i * cellW + cellW / 2)
            .attr("text-anchor", "middle")
            .attr("font-size", 9)
            .attr("font-family", "monospace")
            .attr("fill", "#6688aa")
            .text(displayToken(tok));
        });
      }

      alignedRows.forEach((row, rowIdx) => {
        const msg = side === "A" ? row.dataA : row.dataB;

        g.append("text")
          .attr("x", -4)
          .attr("y", rowIdx * cellH + cellH / 2)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 9)
          .attr("fill", row.duplicate ? "#cc88ff" : "#8888aa")
          .text(row.label);

        if (!msg) {
          g.append("rect")
            .attr("x", 0)
            .attr("y", rowIdx * cellH)
            .attr("width", numPositions * cellW - 1)
            .attr("height", cellH - 1)
            .attr("fill", "#0a0a15")
            .attr("rx", 2);
          return;
        }

        msg.predictions.forEach((posPreds, posIdx) => {
          const topPred = posPreds[0];
          if (!topPred) return;
          g.append("rect")
            .attr("x", posIdx * cellW)
            .attr("y", rowIdx * cellH)
            .attr("width", cellW - 1)
            .attr("height", cellH - 1)
            .attr("fill", colorScale(topPred.prob))
            .attr("rx", 2)
            .style("cursor", "pointer")
            .on("click", (event) => {
              setPinned({
                source: { kind: side, msg },
                rowLabel: row.label,
                posIdx,
                x: event.pageX + 10,
                y: event.pageY + 10,
              });
              setTooltip(null);
            })
            .on("mouseenter", (event) => {
              const top = posPreds.slice(0, 5);
              const show = top.map((p) => displayToken(p.token));
              const maxLen = Math.max(...show.map((s) => s.length));
              const lines = top
                .map((p, i) => `${show[i].padEnd(maxLen)}  ${(p.prob * 100).toFixed(3).padStart(7)}%`)
                .join("\n");
              setTooltip({
                x: event.pageX + 10,
                y: event.pageY - 10,
                content: `${label} ${row.label} pos ${posIdx}\n${lines}`,
              });
            })
            .on("mouseleave", () => setTooltip(null));

          if (cellW >= 28) {
            const display = displayToken(topPred.token);
            g.append("text")
              .attr("x", posIdx * cellW + cellW / 2)
              .attr("y", rowIdx * cellH + cellH / 2)
              .attr("text-anchor", "middle")
              .attr("dominant-baseline", "middle")
              .attr("font-size", 8)
              .attr("fill", topPred.prob > 0.5 ? "#000" : "#fff")
              .style("pointer-events", "none")
              .text(display.length > 5 ? display.slice(0, 4) + "\u2026" : display);
          }
        });
      });
    }

    drawPanel("A", 0, resultA.sessionName);
    drawPanel("B", panelWidth + gap, resultB.sessionName);

    if (showDiff) {
      const diffOffset = (panelWidth + gap) * 2;
      const g = svg.append("g").attr("transform", `translate(${diffOffset + margin.left},${margin.top})`);
      const def = hasCompareData ? COMPARE_METRICS[compareMetric] : null;

      svg.append("text")
        .attr("x", diffOffset + margin.left + (numPositions * cellW) / 2)
        .attr("y", promptTokens ? 14 : 16)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .attr("fill", "#a0a0c0")
        .text(hasCompareData && def ? `Diff — ${def.label}` : "Diff (top-1 delta, from top-k — run exact compare for full metrics)");

      // Compute domain for the selected compare metric if we have exact data.
      let compareDomain: [number, number] = [0, 1];
      if (hasCompareData && def) {
        if (def.fixedDomain) {
          compareDomain = def.fixedDomain;
        } else {
          let minV = Infinity;
          let maxV = -Infinity;
          compareFrames.forEach((frame) => {
            frame.cells.forEach((cell) => {
              const raw = cell.compare[compareMetric];
              const v = typeof raw === "boolean" ? (raw ? 1 : 0) : raw;
              if (!Number.isFinite(v)) return;
              if (v < minV) minV = v;
              if (v > maxV) maxV = v;
            });
          });
          if (!Number.isFinite(minV) || !Number.isFinite(maxV) || minV === maxV) {
            compareDomain = [0, 1];
          } else {
            compareDomain = [minV, maxV];
          }
        }
      }
      const divergingScale =
        def && def.kind === "diverging"
          ? d3.scaleDiverging(def.interpolator).domain([compareDomain[0], 0, compareDomain[1]])
          : null;
      const sequentialScale =
        def && def.kind !== "diverging"
          ? d3.scaleSequential(def.interpolator).domain(compareDomain)
          : null;

      alignedRows.forEach((row, rowIdx) => {
        g.append("text")
          .attr("x", -4)
          .attr("y", rowIdx * cellH + cellH / 2)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 9)
          .attr("fill", row.duplicate ? "#cc88ff" : "#8888aa")
          .text(row.label);

        if (!row.dataA || !row.dataB) {
          g.append("rect")
            .attr("x", 0)
            .attr("y", rowIdx * cellH)
            .attr("width", numPositions * cellW - 1)
            .attr("height", cellH - 1)
            .attr("fill", "#0a0a15")
            .attr("rx", 2);
          return;
        }

        const compareCells = compareByKey.get(`${row.originalLayer}.${row.sublayer}`);

        row.dataA.predictions.forEach((predsA, posIdx) => {
          const predsB = row.dataB!.predictions[posIdx];
          if (!predsA?.[0] || !predsB?.[0]) return;

          if (hasCompareData && compareCells && def) {
            const cell = compareCells[posIdx];
            if (!cell) return;
            const raw = cell.compare[compareMetric];
            const v = typeof raw === "boolean" ? (raw ? 1 : 0) : raw;

            let fill = "#333";
            if (Number.isFinite(v)) {
              if (def.kind === "binary") {
                fill = raw ? "#2d7f4a" : "#a83232";
              } else if (divergingScale) {
                fill = divergingScale(v);
              } else if (sequentialScale) {
                fill = sequentialScale(v);
              }
            }

            const frame = compareFrameByKey.get(`${row.originalLayer}.${row.sublayer}`);
            g.append("rect")
              .attr("x", posIdx * cellW)
              .attr("y", rowIdx * cellH)
              .attr("width", cellW - 1)
              .attr("height", cellH - 1)
              .attr("fill", fill)
              .attr("rx", 2)
              .attr("stroke", !cell.compare.top1_match ? "#ff6b6b" : "none")
              .attr("stroke-width", !cell.compare.top1_match ? 1.5 : 0)
              .style("cursor", "pointer")
              .on("click", (event) => {
                if (!frame) return;
                setPinned({
                  source: { kind: "compare", frame, cell },
                  rowLabel: row.label,
                  posIdx,
                  x: event.pageX + 10,
                  y: event.pageY + 10,
                });
                setTooltip(null);
              })
              .on("mouseenter", (event) => {
                const c = cell.compare;
                setTooltip({
                  x: event.pageX + 10,
                  y: event.pageY - 10,
                  content: [
                    `${row.label} pos ${posIdx}`,
                    `A top-1: ${cell.top_k_a[0]?.token ?? "?"} (${((cell.top_k_a[0]?.prob ?? 0) * 100).toFixed(3)}%)`,
                    `B top-1: ${cell.top_k_b[0]?.token ?? "?"} (${((cell.top_k_b[0]?.prob ?? 0) * 100).toFixed(3)}%)`,
                    "",
                    `KL(A‖B) = ${c.kl_ab.toFixed(5)} nats`,
                    `JS       = ${c.js.toFixed(5)} nats`,
                    `cosine   = ${c.cosine.toFixed(5)}`,
                    `Δ p₁     = ${(c.top1_delta_prob * 100).toFixed(3)}%`,
                    `top-1    = ${c.top1_match ? "match" : "MISMATCH"}`,
                  ].join("\n"),
                });
              })
              .on("mouseleave", () => setTooltip(null));
          } else {
            const diff = predsA[0].prob - predsB[0].prob;
            const tokensDiffer = predsA[0].token !== predsB[0].token;
            const diffScale = d3.scaleDiverging(d3.interpolateRdBu).domain([-1, 0, 1]);

            g.append("rect")
              .attr("x", posIdx * cellW)
              .attr("y", rowIdx * cellH)
              .attr("width", cellW - 1)
              .attr("height", cellH - 1)
              .attr("fill", diffScale(diff))
              .attr("opacity", Math.max(0.15, Math.abs(diff)))
              .attr("rx", 2)
              .attr("stroke", tokensDiffer ? "#ff6b6b" : "none")
              .attr("stroke-width", tokensDiffer ? 1.5 : 0)
              .style("cursor", "pointer")
              .on("mouseenter", (event) => {
                setTooltip({
                  x: event.pageX + 10,
                  y: event.pageY - 10,
                  content: `${row.label} pos ${posIdx}\nA: ${predsA[0].token} (${(predsA[0].prob * 100).toFixed(3)}%)\nB: ${predsB[0].token} (${(predsB[0].prob * 100).toFixed(3)}%)\n\u0394: ${(diff * 100).toFixed(3)}%`,
                });
              })
              .on("mouseleave", () => setTooltip(null));
          }
        });
      });

      // Color-scale legend for the compare panel.
      if (hasCompareData && def) {
        const legendY = numRows * cellH + 12;
        const legendX = 0;
        const legendW = Math.min(200, numPositions * cellW);
        const legendH = 8;
        const gradId = `grad-${compareMetric}-${resultA.id}-${resultB.id}`;
        const defs = svg.append("defs");
        const grad = defs.append("linearGradient").attr("id", gradId).attr("x1", "0%").attr("x2", "100%");
        const stops = 16;
        for (let i = 0; i <= stops; i++) {
          const t = i / stops;
          grad.append("stop").attr("offset", `${t * 100}%`).attr("stop-color", def.interpolator(t));
        }
        g.append("rect")
          .attr("x", legendX)
          .attr("y", legendY)
          .attr("width", legendW)
          .attr("height", legendH)
          .attr("fill", `url(#${gradId})`)
          .attr("rx", 1);
        g.append("text")
          .attr("x", legendX)
          .attr("y", legendY + legendH + 10)
          .attr("font-size", 9)
          .attr("fill", "#888")
          .text(def.format(compareDomain[0]));
        g.append("text")
          .attr("x", legendX + legendW)
          .attr("y", legendY + legendH + 10)
          .attr("text-anchor", "end")
          .attr("font-size", 9)
          .attr("fill", "#888")
          .text(def.format(compareDomain[1]));
      }
    }

  }, [alignedRows, showDiff, resultA.sessionName, resultB.sessionName, resultA.data, resultA.id, resultB.id, hasCompareData, compareByKey, compareFrameByKey, compareMetric, compareFrames]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
          A/B: {resultA.sessionName} vs {resultB.sessionName}
        </h3>
        <button onClick={() => setShowDiff(!showDiff)} style={{ fontSize: 11 }}>
          {showDiff ? "Hide Diff" : "Show Diff"}
        </button>
        <button
          onClick={runCompare}
          disabled={comparing}
          style={{
            fontSize: 11,
            background: hasCompareData ? "#1a5276" : "#2d3a4a",
            color: "#e0e0f0",
            border: "1px solid #1a5276",
            borderRadius: 3,
            padding: "2px 8px",
            cursor: comparing ? "wait" : "pointer",
          }}
          title="Streams per-cell KL/JS/cosine/Δ top-1 from full-vocab distributions on the backend."
        >
          {comparing ? "Comparing…" : hasCompareData ? "Re-run exact compare" : "Run exact compare"}
        </button>
        {hasCompareData && (
          <label style={{ fontSize: 12, color: "#8888aa", display: "flex", alignItems: "center", gap: 6 }}>
            Metric:
            <select
              value={compareMetric}
              onChange={(e) => setCompareMetric(e.target.value as CompareMetricKey)}
              style={{
                background: "#0f1626",
                color: "#e0e0f0",
                border: "1px solid #1a5276",
                borderRadius: 3,
                padding: "2px 6px",
                fontSize: 12,
              }}
              title={COMPARE_METRICS[compareMetric].description}
            >
              {(Object.keys(COMPARE_METRICS) as CompareMetricKey[]).map((k) => (
                <option key={k} value={k}>{COMPARE_METRICS[k].label}</option>
              ))}
            </select>
          </label>
        )}
        {comparing && compareFrames.length > 0 && (
          <span style={{ fontSize: 11, color: "#88aacc" }}>
            Streaming: {compareFrames.length} layer(s)
          </span>
        )}
        {compareError && (
          <span style={{ fontSize: 11, color: "#ff9090" }}>{compareError}</span>
        )}
      </div>
      <div style={{ overflowX: "auto" }}>
        <svg ref={svgRef} />
      </div>
      {tooltip && !pinned && (
        <div
          style={{
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
          }}
        >
          {tooltip.content}
        </div>
      )}
      {pinned && (
        <PinnedCardAB pinned={pinned} onClose={unpin} />
      )}
    </div>
  );
}

interface PinnedCardABProps {
  pinned: Pinned;
  onClose: () => void;
}

function PinnedCardAB({ pinned, onClose }: PinnedCardABProps) {
  const { source, rowLabel, posIdx, x, y } = pinned;

  const { vecA, vecB, title, subtitle } = useMemo(() => {
    let vecA: Float32Array | null = null;
    let vecB: Float32Array | null = null;
    let title = "";
    let subtitle = "";

    if (source.kind === "A") {
      title = `A · ${rowLabel} pos ${posIdx}`;
      if (source.msg.hidden_state) {
        try { vecA = sliceHiddenStatePosition(source.msg.hidden_state, posIdx); } catch { /* ignore */ }
      }
    } else if (source.kind === "B") {
      title = `B · ${rowLabel} pos ${posIdx}`;
      if (source.msg.hidden_state) {
        try { vecB = sliceHiddenStatePosition(source.msg.hidden_state, posIdx); } catch { /* ignore */ }
      }
    } else {
      title = `A/B · ${rowLabel} pos ${posIdx}`;
      const c = source.cell.compare;
      subtitle = `KL=${c.kl_ab.toFixed(5)}  JS=${c.js.toFixed(5)}  cos=${c.cosine.toFixed(5)}  Δp₁=${(c.top1_delta_prob * 100).toFixed(3)}%  ${c.top1_match ? "match" : "MISMATCH"}`;
      if (source.frame.hidden_state_a) {
        try { vecA = sliceHiddenStatePosition(source.frame.hidden_state_a, posIdx); } catch { /* ignore */ }
      }
      if (source.frame.hidden_state_b) {
        try { vecB = sliceHiddenStatePosition(source.frame.hidden_state_b, posIdx); } catch { /* ignore */ }
      }
    }

    return { vecA, vecB, title, subtitle };
  }, [source, rowLabel, posIdx]);

  // Shared symmetric max makes A and B bars directly comparable in magnitude
  // — important when diagnosing whether one side has amplified or dampened.
  const symmetricMax = useMemo(() => {
    let m = 0;
    if (vecA) for (let i = 0; i < vecA.length; i++) { const a = Math.abs(vecA[i]); if (a > m) m = a; }
    if (vecB) for (let i = 0; i < vecB.length; i++) { const a = Math.abs(vecB[i]); if (a > m) m = a; }
    return m;
  }, [vecA, vecB]);

  const sharedMax = source.kind === "compare" && symmetricMax > 0 ? symmetricMax : undefined;

  return (
    <div
      style={{
        position: "fixed",
        left: x,
        top: y,
        background: "#0f1626",
        border: "1px solid #1a5276",
        borderRadius: 4,
        padding: "10px 12px",
        fontFamily: "monospace",
        fontSize: 12,
        color: "#e0e0f0",
        zIndex: 200,
        boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
        maxWidth: 720,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 6 }}>
        <strong style={{ color: "#a0a0c0" }}>{title}</strong>
        <button onClick={onClose} style={{
          background: "transparent", border: "none", color: "#888", cursor: "pointer",
          fontSize: 14, padding: 0, lineHeight: 1,
        }}>×</button>
      </div>
      {subtitle && (
        <div style={{ fontSize: 11, color: "#a0a0c0", marginBottom: 8 }}>{subtitle}</div>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {vecA && <HiddenStateBarStrip data={vecA} label="A" symmetricMax={sharedMax} />}
        {vecB && <HiddenStateBarStrip data={vecB} label="B" symmetricMax={sharedMax} />}
      </div>
      {!vecA && !vecB && (
        <div style={{ fontSize: 10, color: "#888", marginTop: 6 }}>
          No hidden state available for this frame.
        </div>
      )}
    </div>
  );
}
