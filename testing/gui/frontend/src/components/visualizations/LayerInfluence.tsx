import { useRef, useEffect } from "react";
import * as d3 from "d3";
import { ExportButtons } from "../ExportButtons";
import type { ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

export function LayerInfluence({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  const completeMsg = result.data.find((m) => m.type === "complete") as
    | { type: "complete"; scores: Record<string, number> }
    | undefined;

  useEffect(() => {
    if (!svgRef.current || !completeMsg?.scores) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const entries = Object.entries(completeMsg.scores)
      .map(([layer, score]) => ({ layer: parseInt(layer), score }))
      .sort((a, b) => a.layer - b.layer);

    const margin = { top: 20, right: 20, bottom: 30, left: 50 };
    const barHeight = 22;
    const width = 500;
    const height = margin.top + entries.length * barHeight + margin.bottom;

    svg.attr("width", width).attr("height", height);

    const x = d3.scaleLinear()
      .domain([0, d3.max(entries, (d) => d.score) || 1])
      .range([0, width - margin.left - margin.right]);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const colorScale = d3.scaleSequential(d3.interpolateYlOrRd)
      .domain([0, d3.max(entries, (d) => d.score) || 1]);

    entries.forEach((d, i) => {
      g.append("rect")
        .attr("x", 0)
        .attr("y", i * barHeight)
        .attr("width", x(d.score))
        .attr("height", barHeight - 2)
        .attr("fill", colorScale(d.score))
        .attr("rx", 2);

      g.append("text")
        .attr("x", -4)
        .attr("y", i * barHeight + barHeight / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", 10)
        .attr("fill", "#8888aa")
        .text(`L${d.layer}`);

      g.append("text")
        .attr("x", x(d.score) + 4)
        .attr("y", i * barHeight + barHeight / 2)
        .attr("dominant-baseline", "middle")
        .attr("font-size", 10)
        .attr("fill", "#ccccdd")
        .text(d.score.toFixed(3));
    });

  }, [completeMsg]);

  if (!completeMsg?.scores) {
    return <p style={{ color: "#666" }}>No influence data</p>;
  }

  const scores = completeMsg.scores;
  const csvRows = (): (string | number)[][] => {
    const rows = Object.entries(scores)
      .map(([layer, score]) => [parseInt(layer), score] as (string | number)[])
      .sort((a, b) => (a[0] as number) - (b[0] as number));
    return [["layer", "score"], ...rows];
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
          Layer Influence - {result.sessionName}
        </h3>
        <ExportButtons
          filenameBase={`layer-influence_${result.sessionName}`}
          getSVG={() => svgRef.current}
          getCSVRows={csvRows}
        />
      </div>
      <svg ref={svgRef} />
    </div>
  );
}
