import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import * as d3 from "d3";
import { ExportButtons } from "../ExportButtons";
import {
  decodeLogits, logitDiffRecovery, klFromClean, top1Match, probDelta,
} from "../../utils/patchingMetrics";
import type { ProbeResult, PatchingBaselinesData, PatchingCellData } from "../../types/api";

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
      m.type === "data" && "patched_logits" in m && "position" in m
    ),
    [result.data]
  );

  const cleanLogits = useMemo(
    () => baselines ? decodeLogits(baselines.clean_logits) : null, [baselines]);
  const corruptedLogits = useMemo(
    () => baselines ? decodeLogits(baselines.corrupted_logits) : null, [baselines]);
  const cellLogits = useMemo(
    () => new Map(cells.map((c) => [`${c.layer}.${c.sublayer}.${c.position}`, decodeLogits(c.patched_logits)])),
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

    const def = METRICS[metric];

    let domain: [number, number];
    if (def.fixedDomain) {
      domain = def.fixedDomain;
    } else {
      let minV = Infinity, maxV = -Infinity;
      for (const c of cells) {
        const v = getCellValueFor(c, metric);
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

      const v = getCellValueFor(cell, metric);
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
          setPinned({ cell, x: event.pageX + 10, y: event.pageY + 10 });
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
  }, [cells, metric, getCellValueFor, result.id]);

  const csvRows = useCallback((): (string | number)[][] => {
    const header = ["layer", "sublayer", "position",
      "logit_diff_recovery", "kl_from_clean", "top1_match", "prob_delta"];
    const rows: (string | number)[][] = [header];
    for (const cell of cells) {
      rows.push([
        cell.layer, cell.sublayer, cell.position,
        getCellValueFor(cell, "logit_diff_recovery") ?? "",
        getCellValueFor(cell, "kl_from_clean") ?? "",
        getCellValueFor(cell, "top1_match") ?? "",
        getCellValueFor(cell, "prob_delta") ?? "",
      ]);
    }
    return rows;
  }, [cells, getCellValueFor]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
          Activation Patching \u2014 {result.sessionName} \u2014 "{result.prompt.slice(0, 40)}"
        </h3>
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
      <div style={{ overflowX: "auto" }}><svg ref={svgRef} /></div>
      {pinned && (
        <PinnedCard cell={pinned.cell} x={pinned.x} y={pinned.y} onClose={() => setPinned(null)} />
      )}
    </div>
  );
}

function PinnedCard({ cell, x, y, onClose }: { cell: PatchingCellData; x: number; y: number; onClose: () => void }) {
  return (
    <div style={{
      position: "fixed", left: x, top: y, background: "#0f1626",
      border: "1px solid #1a5276", borderRadius: 4, padding: "10px 12px",
      fontFamily: "monospace", fontSize: 12, color: "#e0e0f0", zIndex: 200,
      boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
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
      <div style={{ fontSize: 10, color: "#888" }}>
        Click a cell to see patched logits (detailed top-k view \u2014 enhancement).
      </div>
    </div>
  );
}
