import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import * as d3 from "d3";
import { displayToken } from "../../utils/displayToken";
import { sliceHiddenStatePosition } from "../../utils/hiddenState";
import { HiddenStateBarStrip } from "./HiddenStateBarStrip";
import { DimVsLayerHeatmap } from "./DimVsLayerHeatmap";
import type { LogitLensData, ProbeResult, CellMetrics } from "../../types/api";

interface Props {
  result: ProbeResult;
}

interface PinnedCell {
  msg: LogitLensData;
  posIdx: number;
  x: number;
  y: number;
}

type MetricKey = "top1_prob" | "entropy" | "top1_margin";

interface MetricDef {
  label: string;
  shortLabel: string;
  unit: string;
  interpolator: (t: number) => string;
  fixedDomain?: [number, number];
  format: (v: number) => string;
}

const METRICS: Record<MetricKey, MetricDef> = {
  top1_prob: {
    label: "Top-1 probability",
    shortLabel: "p₁",
    unit: "",
    interpolator: d3.interpolateViridis,
    fixedDomain: [0, 1],
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  entropy: {
    label: "Entropy (nats)",
    shortLabel: "H",
    unit: "nats",
    interpolator: d3.interpolateInferno,
    format: (v) => v.toFixed(2),
  },
  top1_margin: {
    label: "Top-1 margin (p₁ − p₂)",
    shortLabel: "Δ₁₂",
    unit: "",
    interpolator: d3.interpolateCividis,
    fixedDomain: [0, 1],
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
};

export function LogitLensHeatmap({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    content: string;
  } | null>(null);
  const [pinned, setPinned] = useState<PinnedCell | null>(null);
  const [metric, setMetric] = useState<MetricKey>("top1_prob");

  const unpin = useCallback(() => setPinned(null), []);
  useEffect(() => {
    if (!pinned) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") unpin(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pinned, unpin]);

  const dataMessages = useMemo(
    () => result.data.filter((m): m is LogitLensData => m.type === "data" && "predictions" in m),
    [result.data]
  );

  const metricsAvailable = useMemo(
    () => dataMessages.some((m) => Array.isArray(m.metrics) && m.metrics.length > 0),
    [dataMessages]
  );

  useEffect(() => {
    if (!svgRef.current || dataMessages.length === 0) return;
    const firstPreds = dataMessages[0]?.predictions;
    if (!firstPreds || firstPreds.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const completeMsg = result.data.find((m) => m.type === "complete") as
      | { type: "complete"; summary?: { prompt_tokens?: string[] } }
      | undefined;
    const promptTokens = completeMsg?.summary?.prompt_tokens;

    const margin = { top: promptTokens ? 50 : 30, right: 20, bottom: 40, left: 80 };
    const numPositions = firstPreds.length;
    const numRows = dataMessages.length;

    const cellW = Math.max(30, Math.min(60, 600 / numPositions));
    const cellH = 20;
    const width = margin.left + numPositions * cellW + margin.right;
    const height = margin.top + numRows * cellH + margin.bottom;

    svg.attr("width", width).attr("height", height);

    const def = METRICS[metric];

    const getCellValue = (msg: LogitLensData, posIdx: number): number | null => {
      const m: CellMetrics | undefined = msg.metrics?.[posIdx];
      if (m && typeof m[metric] === "number") return m[metric];
      // Fallback for the pre-metrics stream (old backends or cached results).
      if (metric === "top1_prob") {
        const topPred = msg.predictions[posIdx]?.[0];
        return topPred ? topPred.prob : null;
      }
      return null;
    };

    let domain: [number, number];
    if (def.fixedDomain) {
      domain = def.fixedDomain;
    } else {
      let minV = Infinity;
      let maxV = -Infinity;
      dataMessages.forEach((msg) => {
        for (let i = 0; i < numPositions; i++) {
          const v = getCellValue(msg, i);
          if (v === null) continue;
          if (v < minV) minV = v;
          if (v > maxV) maxV = v;
        }
      });
      if (!Number.isFinite(minV) || !Number.isFinite(maxV) || minV === maxV) {
        domain = [0, 1];
      } else {
        domain = [minV, maxV];
      }
    }

    const colorScale = d3.scaleSequential(def.interpolator).domain(domain);
    const normalize = (v: number) => (v - domain[0]) / (domain[1] - domain[0] || 1);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    dataMessages.forEach((msg, rowIdx) => {
      const isModified = "modified" in msg && (msg as { modified?: boolean }).modified;
      const label = msg.original_layer != null && msg.original_layer !== msg.layer
        ? `L${msg.layer}(\u2190${msg.original_layer}).${msg.sublayer}`
        : `L${msg.layer}.${msg.sublayer}`;

      g.append("text")
        .attr("x", -4)
        .attr("y", rowIdx * cellH + cellH / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", 10)
        .attr("fill", "#8888aa")
        .text(label);

      if (isModified) {
        g.append("rect")
          .attr("x", -8)
          .attr("y", rowIdx * cellH)
          .attr("width", 3)
          .attr("height", cellH - 1)
          .attr("fill", "#ff6b6b")
          .attr("rx", 1);
      }

      msg.predictions.forEach((posPreds, posIdx) => {
        const topPred = posPreds[0];
        if (!topPred) return;

        const cellV = getCellValue(msg, posIdx);
        const fill = cellV !== null ? colorScale(cellV) : "#333";
        const brightness = cellV !== null ? normalize(cellV) : 0;

        g.append("rect")
          .attr("x", posIdx * cellW)
          .attr("y", rowIdx * cellH)
          .attr("width", cellW - 1)
          .attr("height", cellH - 1)
          .attr("fill", fill)
          .attr("rx", 2)
          .style("cursor", "pointer")
          .on("click", (event) => {
            setPinned({ msg, posIdx, x: event.pageX + 10, y: event.pageY + 10 });
            setTooltip(null);
          })
          .on("mouseenter", (event) => {
            const top = posPreds.slice(0, 5);
            const show = top.map((p) => displayToken(p.token));
            const maxLen = Math.max(...show.map((s) => s.length));
            const topkLines = top
              .map((p, i) => `${show[i].padEnd(maxLen)}  ${(p.prob * 100).toFixed(3).padStart(7)}%`)
              .join("\n");
            const cellMetrics: CellMetrics | undefined = msg.metrics?.[posIdx];
            const metricLines = cellMetrics
              ? [
                  `p₁    = ${(cellMetrics.top1_prob * 100).toFixed(3)}%`,
                  `Δ₁₂   = ${(cellMetrics.top1_margin * 100).toFixed(3)}%`,
                  `H     = ${cellMetrics.entropy.toFixed(3)} nats`,
                ].join("\n") + "\n\n"
              : "";
            setTooltip({
              x: event.pageX + 10,
              y: event.pageY - 10,
              content: `L${msg.layer}.${msg.sublayer} pos ${posIdx}\n${metricLines}${topkLines}`,
            });
          })
          .on("mouseleave", () => setTooltip(null));

        g.append("text")
          .attr("x", posIdx * cellW + cellW / 2)
          .attr("y", rowIdx * cellH + cellH / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 9)
          .attr("fill", brightness > 0.5 ? "#000" : "#fff")
          .style("pointer-events", "none")
          .text(() => { const d = displayToken(topPred.token); return d.length > 6 ? d.slice(0, 5) + "\u2026" : d; });
      });
    });

    if (promptTokens) {
      g.append("g")
        .attr("transform", "translate(0,-6)")
        .selectAll("text")
        .data(promptTokens.slice(0, numPositions))
        .join("text")
        .attr("x", (_, i) => i * cellW + cellW / 2)
        .attr("text-anchor", "middle")
        .attr("font-size", 10)
        .attr("font-family", "monospace")
        .attr("fill", "#6688aa")
        .text((d) => displayToken(d));
    }

    g.append("g")
      .attr("transform", `translate(0,${numRows * cellH + 4})`)
      .selectAll("text")
      .data(d3.range(numPositions))
      .join("text")
      .attr("x", (d) => d * cellW + cellW / 2)
      .attr("text-anchor", "middle")
      .attr("font-size", 9)
      .attr("fill", "#666")
      .text((d) => `${d}`);

    // Color-scale legend: a thin gradient strip with min/max labels.
    const legendY = numRows * cellH + 22;
    const legendX = 0;
    const legendW = Math.min(200, numPositions * cellW);
    const legendH = 8;
    const gradId = `gradient-${metric}-${result.id}`;
    const defs = svg.append("defs");
    const grad = defs.append("linearGradient").attr("id", gradId).attr("x1", "0%").attr("x2", "100%");
    const stops = 16;
    for (let i = 0; i <= stops; i++) {
      const t = i / stops;
      grad.append("stop")
        .attr("offset", `${t * 100}%`)
        .attr("stop-color", def.interpolator(t));
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
      .text(def.format(domain[0]));
    g.append("text")
      .attr("x", legendX + legendW)
      .attr("y", legendY + legendH + 10)
      .attr("text-anchor", "end")
      .attr("font-size", 9)
      .attr("fill", "#888")
      .text(def.format(domain[1]));
    g.append("text")
      .attr("x", legendX + legendW / 2)
      .attr("y", legendY - 2)
      .attr("text-anchor", "middle")
      .attr("font-size", 9)
      .attr("fill", "#aaa")
      .text(def.label);

  }, [dataMessages, result.data, result.id, metric]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
          Logit Lens - {result.sessionName} - "{result.prompt.slice(0, 40)}"
        </h3>
        {metricsAvailable && (
          <label style={{ fontSize: 12, color: "#8888aa", display: "flex", alignItems: "center", gap: 6 }}>
            Metric:
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value as MetricKey)}
              style={{
                background: "#0f1626",
                color: "#e0e0f0",
                border: "1px solid #1a5276",
                borderRadius: 3,
                padding: "2px 6px",
                fontSize: 12,
              }}
            >
              {(Object.keys(METRICS) as MetricKey[]).map((k) => (
                <option key={k} value={k}>{METRICS[k].label}</option>
              ))}
            </select>
          </label>
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
        <PinnedCard pinned={pinned} onClose={unpin} />
      )}
      <DimVsLayerHeatmap result={result} />
    </div>
  );
}

interface PinnedCardProps {
  pinned: PinnedCell;
  onClose: () => void;
}

function PinnedCard({ pinned, onClose }: PinnedCardProps) {
  const { msg, posIdx, x, y } = pinned;
  const cellMetrics: CellMetrics | undefined = msg.metrics?.[posIdx];
  const top = msg.predictions[posIdx]?.slice(0, 5) ?? [];

  const hiddenVec = useMemo(() => {
    if (!msg.hidden_state) return null;
    try {
      return sliceHiddenStatePosition(msg.hidden_state, posIdx);
    } catch {
      return null;
    }
  }, [msg.hidden_state, posIdx]);

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
        maxWidth: 520,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 6 }}>
        <strong style={{ color: "#a0a0c0" }}>
          L{msg.layer}.{msg.sublayer} pos {posIdx}
        </strong>
        <button onClick={onClose} style={{
          background: "transparent", border: "none", color: "#888", cursor: "pointer",
          fontSize: 14, padding: 0, lineHeight: 1,
        }}>×</button>
      </div>
      {cellMetrics && (
        <div style={{ fontSize: 11, color: "#a0a0c0", marginBottom: 6, whiteSpace: "pre" }}>
          {`p₁ = ${(cellMetrics.top1_prob * 100).toFixed(3)}%   Δ₁₂ = ${(cellMetrics.top1_margin * 100).toFixed(3)}%   H = ${cellMetrics.entropy.toFixed(3)} nats`}
        </div>
      )}
      <div style={{ fontSize: 11, color: "#8888aa", marginBottom: 6, whiteSpace: "pre" }}>
        {top.map((p) => `${displayToken(p.token).padEnd(8)} ${(p.prob * 100).toFixed(3).padStart(7)}%`).join("\n")}
      </div>
      {hiddenVec && (
        <div style={{ marginTop: 8 }}>
          <HiddenStateBarStrip
            data={hiddenVec}
            label={`Hidden state (${hiddenVec.length} dims, red=+, blue=−)`}
          />
        </div>
      )}
      {!hiddenVec && (
        <div style={{ fontSize: 10, color: "#888", marginTop: 6 }}>
          {msg.hidden_state
            ? `Cannot decode hidden state: shape=${msg.hidden_state.shape.join("×")}`
            : "No hidden state streamed for this frame."}
        </div>
      )}
    </div>
  );
}
