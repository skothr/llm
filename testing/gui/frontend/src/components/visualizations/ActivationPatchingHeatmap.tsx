import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import * as d3 from "d3";
import { ExportButtons } from "../ExportButtons";
import {
  decodeLogits, logitDiffRecovery, klFromClean, top1Match, probDelta, topK,
  type TopKEntry,
} from "../../utils/patchingMetrics";
import type { ProbeResult, PatchingBaselinesData, PatchingCellData, PatchingCompleteData } from "../../types/api";
import { ResidualDecodeBlock } from "./ResidualDecodeBlock";
import { LensTraceStrip } from "./LensTraceStrip";

interface Props {
  result: ProbeResult;
}

type MetricKey = "logit_diff_recovery" | "kl_from_clean" | "top1_match" | "prob_delta";

interface MetricDef {
  label: string;
  interpolator: (t: number) => string;
  fixedDomain?: [number, number];
  format: (v: number) => string;
}

const METRICS: Record<MetricKey, MetricDef> = {
  logit_diff_recovery: {
    label: "Logit-diff recovery",
    interpolator: d3.interpolatePiYG,
    fixedDomain: [-0.5, 1.0],
    format: (v) => v.toFixed(3),
  },
  kl_from_clean: {
    label: "KL from clean (nats)",
    interpolator: (t) => d3.interpolateInferno(1 - t),
    format: (v) => v.toFixed(3),
  },
  top1_match: {
    label: "Top-1 matches clean",
    interpolator: (t) => (t > 0.5 ? "#6bc06b" : "#333"),
    fixedDomain: [0, 1],
    format: (v) => (v > 0.5 ? "yes" : "no"),
  },
  prob_delta: {
    label: "\u0394 p(clean top-1)",
    interpolator: d3.interpolatePiYG,
    fixedDomain: [-1, 1],
    format: (v) => v.toFixed(3),
  },
};

interface PinnedCell {
  cell: PatchingCellData;
  x: number;
  y: number;
}

export function ActivationPatchingHeatmap({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [metric, setMetric] = useState<MetricKey>("logit_diff_recovery");
  const [pinned, setPinned] = useState<PinnedCell | null>(null);

  const baselines = useMemo(
    () => result.data.find((m): m is PatchingBaselinesData => m.type === "baselines"),
    [result.data]
  );
  const cells = useMemo(
    () => result.data.filter((m): m is PatchingCellData =>
      m.type === "data" && "position" in m
    ),
    [result.data]
  );
  const completeFrame = useMemo(
    () => result.data.find((m): m is PatchingCompleteData => m.type === "complete"),
    [result.data]
  );
  const rawMode = completeFrame?.summary.mode ?? "exact";
  const mode: "exact" | "approx" = rawMode === "approx" ? "approx" : "exact";

  const cleanLogits = useMemo(
    () => baselines ? decodeLogits(baselines.clean_logits) : null, [baselines]);
  const corruptedLogits = useMemo(
    () => baselines ? decodeLogits(baselines.corrupted_logits) : null, [baselines]);
  const cellLogits = useMemo(
    () => new Map(cells.filter((c) => c.patched_logits != null).map(
      (c) => [`${c.layer}.${c.sublayer}.${c.position}`, decodeLogits(c.patched_logits!)]
    )),
    [cells]
  );

  // metric passed as parameter (not closed over) so CSV exporter can
  // compute all four columns per cell without state mutation.
  const getCellValueFor = useCallback((cell: PatchingCellData, m: MetricKey): number | null => {
    if (!cleanLogits || !corruptedLogits || !baselines) return null;
    const patched = cellLogits.get(`${cell.layer}.${cell.sublayer}.${cell.position}`);
    if (!patched) return null;
    switch (m) {
      case "logit_diff_recovery":
        if (baselines.correct_token_id == null || baselines.incorrect_token_id == null) return null;
        return logitDiffRecovery(patched, cleanLogits, corruptedLogits,
          baselines.correct_token_id, baselines.incorrect_token_id);
      case "kl_from_clean":
        return klFromClean(patched, cleanLogits);
      case "top1_match":
        return top1Match(patched, cleanLogits) ? 1 : 0;
      case "prob_delta": {
        let topId = 0, topVal = -Infinity;
        for (let i = 0; i < cleanLogits.length; i++) {
          if (cleanLogits[i] > topVal) { topVal = cleanLogits[i]; topId = i; }
        }
        return probDelta(patched, corruptedLogits, topId);
      }
    }
  }, [cellLogits, cleanLogits, corruptedLogits, baselines]);

  useEffect(() => {
    if (!svgRef.current || cells.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const rowKeySet = new Set<string>();
    for (const c of cells) rowKeySet.add(`${c.layer}.${c.sublayer}`);
    const rowKeys = Array.from(rowKeySet).sort((a, b) => {
      const [la, sa] = a.split(".");
      const [lb] = b.split(".");
      if (la !== lb) return Number(la) - Number(lb);
      return sa === "attn" ? -1 : 1;
    });

    const positionSet = new Set<number>();
    for (const c of cells) positionSet.add(c.position);
    const positions = Array.from(positionSet).sort((a, b) => a - b);

    const margin = { top: 40, right: 20, bottom: 40, left: 80 };
    const cellW = Math.max(30, Math.min(60, 600 / positions.length));
    const cellH = 20;
    const width = margin.left + positions.length * cellW + margin.right;
    const height = margin.top + rowKeys.length * cellH + margin.bottom;
    svg.attr("width", width).attr("height", height);

    const effectiveMetric: MetricKey = mode === "approx" ? "logit_diff_recovery" : metric;
    const def = METRICS[effectiveMetric];

    let domain: [number, number];
    if (def.fixedDomain) {
      domain = def.fixedDomain;
    } else {
      let minV = Infinity, maxV = -Infinity;
      for (const c of cells) {
        const v = mode === "approx"
          ? (typeof c.ap_recovery === "number" ? c.ap_recovery : null)
          : getCellValueFor(c, effectiveMetric);
        if (v == null) continue;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
      domain = (Number.isFinite(minV) && Number.isFinite(maxV) && minV !== maxV) ? [minV, maxV] : [0, 1];
    }

    const colorScale = d3.scaleSequential(def.interpolator).domain(domain);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    rowKeys.forEach((rk, rowIdx) => {
      const [lStr, subStr] = rk.split(".");
      g.append("text")
        .attr("x", -4)
        .attr("y", rowIdx * cellH + cellH / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", 10)
        .attr("fill", "#8888aa")
        .text(`L${lStr}.${subStr}`);
    });

    positions.forEach((pos, colIdx) => {
      g.append("text")
        .attr("x", colIdx * cellW + cellW / 2)
        .attr("y", -6)
        .attr("text-anchor", "middle")
        .attr("font-size", 9)
        .attr("fill", "#666")
        .text(pos);
    });

    for (const cell of cells) {
      const rk = `${cell.layer}.${cell.sublayer}`;
      const rowIdx = rowKeys.indexOf(rk);
      const colIdx = positions.indexOf(cell.position);
      if (rowIdx < 0 || colIdx < 0) continue;

      const v = mode === "approx"
        ? (typeof cell.ap_recovery === "number" ? cell.ap_recovery : null)
        : getCellValueFor(cell, effectiveMetric);
      const fill = v != null ? colorScale(v) : "#222";

      g.append("rect")
        .attr("x", colIdx * cellW)
        .attr("y", rowIdx * cellH)
        .attr("width", cellW - 1)
        .attr("height", cellH - 1)
        .attr("fill", fill)
        .attr("rx", 2)
        .style("cursor", "pointer")
        .on("click", (event) => {
          // clientX/Y because PinnedCell renders with position:fixed
          // (viewport-relative); pageX/Y would drift by scroll offset.
          setPinned({ cell, x: event.clientX + 10, y: event.clientY + 10 });
        });
    }

    const legendY = rowKeys.length * cellH + 20;
    const legendW = Math.min(200, positions.length * cellW);
    const legendH = 8;
    const gradId = `ap-grad-${metric}-${result.id}`;
    const defs = svg.append("defs");
    const grad = defs.append("linearGradient").attr("id", gradId).attr("x1", "0%").attr("x2", "100%");
    for (let i = 0; i <= 16; i++) {
      grad.append("stop").attr("offset", `${(i / 16) * 100}%`).attr("stop-color", def.interpolator(i / 16));
    }
    g.append("rect")
      .attr("x", 0).attr("y", legendY).attr("width", legendW).attr("height", legendH)
      .attr("fill", `url(#${gradId})`).attr("rx", 1);
    g.append("text")
      .attr("x", 0).attr("y", legendY + legendH + 10)
      .attr("font-size", 9).attr("fill", "#888").text(def.format(domain[0]));
    g.append("text")
      .attr("x", legendW).attr("y", legendY + legendH + 10)
      .attr("text-anchor", "end").attr("font-size", 9).attr("fill", "#888").text(def.format(domain[1]));
    g.append("text")
      .attr("x", legendW / 2).attr("y", legendY - 2)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#aaa").text(def.label);
  }, [cells, metric, mode, getCellValueFor, result.id]);

  const csvRows = useCallback((): (string | number)[][] => {
    if (mode === "approx") {
      const header = ["layer", "sublayer", "position", "ap_recovery"];
      const rows: (string | number)[][] = [header];
      for (const cell of cells) {
        rows.push([cell.layer, cell.sublayer ?? "", cell.position, cell.ap_recovery ?? ""]);
      }
      return rows;
    }
    const header = ["layer", "sublayer", "position",
      "logit_diff_recovery", "kl_from_clean", "top1_match", "prob_delta"];
    const rows: (string | number)[][] = [header];
    for (const cell of cells) {
      rows.push([
        cell.layer, cell.sublayer ?? "", cell.position,
        getCellValueFor(cell, "logit_diff_recovery") ?? "",
        getCellValueFor(cell, "kl_from_clean") ?? "",
        getCellValueFor(cell, "top1_match") ?? "",
        getCellValueFor(cell, "prob_delta") ?? "",
      ]);
    }
    return rows;
  }, [cells, mode, getCellValueFor]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
          {mode === "approx" ? "Attribution Patching (\u2207)" : "Activation Patching"}
          {" \u2014 "}{result.sessionName}{" \u2014 \""}{result.prompt.slice(0, 40)}{"\""}
        </h3>
        {mode === "approx" && completeFrame?.summary?.n_steps != null && completeFrame.summary.n_steps > 1 && (
          <span style={{ color: "#a0a0c0", fontSize: 13, fontWeight: "normal" }}>
            {` — IG ${completeFrame.summary.n_steps} steps`}
          </span>
        )}
        {mode === "exact" && (
          <label style={{ fontSize: 12, color: "#8888aa", display: "flex", alignItems: "center", gap: 6 }}>
            Metric:
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value as MetricKey)}
              style={{
                background: "#0f1626", color: "#e0e0f0", border: "1px solid #1a5276",
                borderRadius: 3, padding: "2px 6px", fontSize: 12,
              }}
            >
              {(Object.keys(METRICS) as MetricKey[]).map((k) => (
                <option key={k} value={k}>{METRICS[k].label}</option>
              ))}
            </select>
          </label>
        )}
        <div style={{ marginLeft: "auto" }}>
          <ExportButtons
            filenameBase={`activation_patching_${result.sessionName}`}
            getSVG={() => svgRef.current}
            getCSVRows={csvRows}
            getJSON={() => ({
              sessionName: result.sessionName,
              prompt: result.prompt,
              timestamp: result.timestamp,
              data: result.data,
            })}
          />
        </div>
      </div>
      <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
        <div style={{ flex: 1, overflowX: "auto" }}><svg ref={svgRef} /></div>
        <LensTraceStrip
          sessionName={result.sessionName}
          prompt={result.prompt}
          promptTokens={baselines?.prompt_tokens_clean}
          initialPosition={baselines?.measurement_position}
        />
      </div>
      {pinned && (
        <PinnedCard
          cell={pinned.cell}
          x={pinned.x} y={pinned.y}
          mode={mode}
          sessionName={result.sessionName}
          prompt={result.prompt}
          baselines={baselines}
          onClose={() => setPinned(null)}
        />
      )}
    </div>
  );
}

interface PinnedCardProps {
  cell: PatchingCellData;
  x: number; y: number;
  mode: "exact" | "approx";
  sessionName: string;
  prompt: string;
  baselines: PatchingBaselinesData | undefined;
  onClose: () => void;
}

function PinnedCard({ cell, x, y, mode, sessionName, prompt, baselines, onClose }: PinnedCardProps) {
  // Exact-mode top-k decoding: compute top-5 indices client-side from the
  // cell's patched_logits and the shared clean baseline, then fetch string
  // renderings for those ids from the session tokenizer. Runs once per
  // pin-click; cached in state. Approx mode skips this entirely.
  const [patchedTop, setPatchedTop] = useState<TopKEntry[]>([]);
  const [cleanTop, setCleanTop] = useState<TopKEntry[]>([]);
  const [tokenMap, setTokenMap] = useState<Map<number, string>>(new Map());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (mode !== "exact" || !cell.patched_logits || !baselines) return;
    const patchedLogits = decodeLogits(cell.patched_logits);
    const cleanLogits = decodeLogits(baselines.clean_logits);
    const pTop = topK(patchedLogits, 5);
    const cTop = topK(cleanLogits, 5);
    setPatchedTop(pTop);
    setCleanTop(cTop);
    const uniqueIds = Array.from(new Set([...pTop, ...cTop].map((e) => e.id)));
    const ac = new AbortController();
    fetch(`/api/sessions/${sessionName}/decode-ids`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: uniqueIds }),
      signal: ac.signal,
    })
      .then((r) => r.ok ? r.json() : Promise.reject(new Error(`decode-ids ${r.status}`)))
      .then((data: { tokens: string[] }) => {
        const m = new Map<number, string>();
        uniqueIds.forEach((id, i) => m.set(id, data.tokens[i]));
        setTokenMap(m);
        setError(null);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message ?? String(e));
      });
    return () => ac.abort();
  }, [mode, cell, sessionName, baselines]);

  return (
    <div style={{
      position: "fixed", left: x, top: y, background: "#0f1626",
      border: "1px solid #1a5276", borderRadius: 4, padding: "10px 12px",
      fontFamily: "monospace", fontSize: 12, color: "#e0e0f0", zIndex: 200,
      boxShadow: "0 4px 16px rgba(0,0,0,0.5)", minWidth: 280,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 6 }}>
        <strong style={{ color: "#a0a0c0" }}>
          L{cell.layer}.{cell.sublayer} pos {cell.position}
        </strong>
        <button onClick={onClose} style={{
          background: "transparent", border: "none", color: "#888",
          cursor: "pointer", fontSize: 14, padding: 0, lineHeight: 1,
        }}>\u00d7</button>
      </div>
      {mode === "approx" && typeof cell.ap_recovery === "number" ? (
        <div style={{ marginTop: 4 }}>
          AP recovery: <strong>{cell.ap_recovery.toFixed(3)}</strong>
          <div style={{ fontSize: 10, color: "#888", marginTop: 2 }}>
            First-order approximation \u2014 run exact mode to confirm.
          </div>
        </div>
      ) : patchedTop.length > 0 ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 4 }}>
          <TopKColumn title="patched" entries={patchedTop} tokens={tokenMap} />
          <TopKColumn title="clean" entries={cleanTop} tokens={tokenMap} />
        </div>
      ) : error ? (
        <div style={{ fontSize: 10, color: "#c88" }}>decode-ids failed: {error}</div>
      ) : (
        <div style={{ fontSize: 10, color: "#888" }}>decoding top-5 tokens...</div>
      )}
      {(cell.sublayer === "attn" || cell.sublayer === "ffn") && (
        <ResidualDecodeBlock
          sessionName={sessionName}
          prompt={prompt}
          layer={cell.layer}
          sublayer={cell.sublayer}
          position={cell.position}
        />
      )}
    </div>
  );
}

function TopKColumn({ title, entries, tokens }: { title: string; entries: TopKEntry[]; tokens: Map<number, string> }) {
  return (
    <div>
      <div style={{ color: "#8888aa", fontSize: 10, marginBottom: 2 }}>{title}</div>
      {entries.map((e) => (
        <div key={e.id} style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
          <span style={{ color: "#e0e0f0" }}>{tokens.get(e.id) ?? `<${e.id}>`}</span>
          <span style={{ color: "#888" }}>{e.prob.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}
