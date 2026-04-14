import { useRef, useEffect, useState, useMemo } from "react";
import * as d3 from "d3";
import type { LogitLensData, ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

export function LogitLensHeatmap({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    content: string;
  } | null>(null);

  const dataMessages = useMemo(
    () => result.data.filter((m): m is LogitLensData => m.type === "data" && "predictions" in m),
    [result.data]
  );

  useEffect(() => {
    if (!svgRef.current || dataMessages.length === 0) return;
    const firstPreds = dataMessages[0]?.predictions;
    if (!firstPreds || firstPreds.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 30, right: 20, bottom: 40, left: 80 };
    const numPositions = firstPreds.length;
    const numRows = dataMessages.length;

    const cellW = Math.max(30, Math.min(60, 600 / numPositions));
    const cellH = 20;
    const width = margin.left + numPositions * cellW + margin.right;
    const height = margin.top + numRows * cellH + margin.bottom;

    svg.attr("width", width).attr("height", height);

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    dataMessages.forEach((msg, rowIdx) => {
      const isModified = "modified" in msg && (msg as { modified?: boolean }).modified;
      const label = `L${msg.layer}.${msg.sublayer}`;

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

        g.append("rect")
          .attr("x", posIdx * cellW)
          .attr("y", rowIdx * cellH)
          .attr("width", cellW - 1)
          .attr("height", cellH - 1)
          .attr("fill", colorScale(topPred.prob))
          .attr("rx", 2)
          .style("cursor", "pointer")
          .on("mouseenter", (event) => {
            const lines = posPreds
              .slice(0, 5)
              .map((p) => `${p.token}: ${(p.prob * 100).toFixed(1)}%`)
              .join("\n");
            setTooltip({
              x: event.pageX + 10,
              y: event.pageY - 10,
              content: `L${msg.layer}.${msg.sublayer} pos ${posIdx}\n${lines}`,
            });
          })
          .on("mouseleave", () => setTooltip(null));

        g.append("text")
          .attr("x", posIdx * cellW + cellW / 2)
          .attr("y", rowIdx * cellH + cellH / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 9)
          .attr("fill", topPred.prob > 0.5 ? "#000" : "#fff")
          .style("pointer-events", "none")
          .text(topPred.token.length > 6 ? topPred.token.slice(0, 5) + "..." : topPred.token);
      });
    });

    g.append("g")
      .attr("transform", `translate(0,${numRows * cellH + 4})`)
      .selectAll("text")
      .data(d3.range(numPositions))
      .join("text")
      .attr("x", (d) => d * cellW + cellW / 2)
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("fill", "#8888aa")
      .text((d) => `${d}`);

  }, [dataMessages]);

  return (
    <div style={{ position: "relative" }}>
      <h3 style={{ fontSize: 13, color: "#a0a0c0", marginBottom: 8 }}>
        Logit Lens - {result.sessionName} - "{result.prompt.slice(0, 40)}"
      </h3>
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
