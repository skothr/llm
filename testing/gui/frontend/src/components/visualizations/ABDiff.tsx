import { useRef, useEffect, useState, useMemo } from "react";
import * as d3 from "d3";
import type { LogitLensData, ProbeResult } from "../../types/api";

interface Props {
  resultA: ProbeResult;
  resultB: ProbeResult;
}

export function ABDiff({ resultA, resultB }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [showDiff, setShowDiff] = useState(false);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    content: string;
  } | null>(null);

  const dataA = useMemo(
    () => resultA.data.filter((m): m is LogitLensData => m.type === "data" && "predictions" in m),
    [resultA.data]
  );
  const dataB = useMemo(
    () => resultB.data.filter((m): m is LogitLensData => m.type === "data" && "predictions" in m),
    [resultB.data]
  );

  useEffect(() => {
    if (!svgRef.current || dataA.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const numPositions = dataA[0]?.predictions.length || 1;
    const numRows = Math.max(dataA.length, dataB.length);
    const cellW = Math.max(24, Math.min(40, 400 / numPositions));
    const cellH = 18;
    const margin = { top: 30, right: 20, bottom: 30, left: 70 };
    const panelWidth = margin.left + numPositions * cellW + margin.right;
    const gap = 40;
    const totalWidth = showDiff ? panelWidth * 3 + gap * 2 : panelWidth * 2 + gap;
    const height = margin.top + numRows * cellH + margin.bottom;

    svg.attr("width", totalWidth).attr("height", height);

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
    const diffScale = d3.scaleDiverging(d3.interpolateRdBu).domain([-1, 0, 1]);

    function drawPanel(
      data: LogitLensData[],
      offsetX: number,
      label: string
    ) {
      const g = svg.append("g").attr("transform", `translate(${offsetX + margin.left},${margin.top})`);

      svg.append("text")
        .attr("x", offsetX + margin.left + (numPositions * cellW) / 2)
        .attr("y", 16)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .attr("fill", "#a0a0c0")
        .text(label);

      data.forEach((msg, rowIdx) => {
        g.append("text")
          .attr("x", -4)
          .attr("y", rowIdx * cellH + cellH / 2)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 9)
          .attr("fill", "#8888aa")
          .text(`L${msg.layer}.${msg.sublayer}`);

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
            .on("mouseenter", (event) => {
              const top = posPreds.slice(0, 5);
              const show = top.map((p) => p.token.replace(/ /g, "\u00B7"));
              const maxLen = Math.max(...show.map((s) => s.length));
              const lines = top
                .map((p, i) => `${show[i].padEnd(maxLen)}  ${(p.prob * 100).toFixed(1).padStart(5)}%`)
                .join("\n");
              setTooltip({
                x: event.pageX + 10,
                y: event.pageY - 10,
                content: `${label} L${msg.layer}.${msg.sublayer} pos ${posIdx}\n${lines}`,
              });
            })
            .on("mouseleave", () => setTooltip(null));

          if (cellW >= 28) {
            const display = topPred.token.replace(/ /g, "\u00B7");
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

    drawPanel(dataA, 0, resultA.sessionName);
    drawPanel(dataB, panelWidth + gap, resultB.sessionName);

    if (showDiff) {
      const diffRows = Math.min(dataA.length, dataB.length);
      const diffOffset = (panelWidth + gap) * 2;
      const g = svg.append("g").attr("transform", `translate(${diffOffset + margin.left},${margin.top})`);

      svg.append("text")
        .attr("x", diffOffset + margin.left + (numPositions * cellW) / 2)
        .attr("y", 16)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .attr("fill", "#a0a0c0")
        .text(`Diff (A - B)${dataA.length !== dataB.length ? ` [${diffRows}/${numRows} rows]` : ""}`);

      dataA.slice(0, diffRows).forEach((msgA, rowIdx) => {
        const msgB = dataB[rowIdx];
        if (!msgB) return;

        g.append("text")
          .attr("x", -4)
          .attr("y", rowIdx * cellH + cellH / 2)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 9)
          .attr("fill", "#8888aa")
          .text(`L${msgA.layer}`);

        msgA.predictions.forEach((predsA, posIdx) => {
          const predsB = msgB.predictions[posIdx];
          if (!predsA?.[0] || !predsB?.[0]) return;
          const diff = predsA[0].prob - predsB[0].prob;
          const tokensDiffer = predsA[0].token !== predsB[0].token;

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
                content: `L${msgA.layer} pos ${posIdx}\nA: ${predsA[0].token} (${(predsA[0].prob * 100).toFixed(1)}%)\nB: ${predsB[0].token} (${(predsB[0].prob * 100).toFixed(1)}%)\nd: ${(diff * 100).toFixed(1)}%`,
              });
            })
            .on("mouseleave", () => setTooltip(null));
        });
      });
    }

  }, [dataA, dataB, showDiff, resultA.sessionName, resultB.sessionName]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0" }}>
          A/B: {resultA.sessionName} vs {resultB.sessionName}
        </h3>
        <button onClick={() => setShowDiff(!showDiff)} style={{ fontSize: 11 }}>
          {showDiff ? "Hide Diff" : "Show Diff"}
        </button>
      </div>
      <div style={{ overflowX: "auto" }}>
        <svg ref={svgRef} />
      </div>
      {tooltip && (
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
    </div>
  );
}
