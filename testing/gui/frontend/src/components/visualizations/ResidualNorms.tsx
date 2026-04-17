import { useRef, useEffect } from "react";
import * as d3 from "d3";
import { ExportButtons } from "../ExportButtons";
import type { ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

export function ResidualNorms({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  const completeMsg = result.data.find((m) => m.type === "complete") as
    | { type: "complete"; norms: number[] }
    | undefined;

  useEffect(() => {
    if (!svgRef.current || !completeMsg?.norms) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const norms = completeMsg.norms;
    const margin = { top: 20, right: 20, bottom: 35, left: 60 };
    const width = 500;
    const height = 200;

    svg.attr("width", width + margin.left + margin.right)
       .attr("height", height + margin.top + margin.bottom);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, norms.length - 1]).range([0, width]);
    const y = d3.scaleLinear().domain([0, d3.max(norms) || 1]).range([height, 0]).nice();

    g.append("g").attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(Math.min(norms.length, 20)).tickFormat((d) => `${d}`))
      .selectAll("text").attr("fill", "#8888aa").attr("font-size", 10);
    g.selectAll(".domain, .tick line").attr("stroke", "#333");

    g.append("g").call(d3.axisLeft(y).ticks(5))
      .selectAll("text").attr("fill", "#8888aa").attr("font-size", 10);
    g.selectAll(".domain, .tick line").attr("stroke", "#333");

    g.append("path").datum(norms)
      .attr("d", d3.line<number>().x((_, i) => x(i)).y((d) => y(d)))
      .attr("fill", "none").attr("stroke", "#4ecdc4").attr("stroke-width", 2);

    g.selectAll("circle").data(norms).join("circle")
      .attr("cx", (_, i) => x(i)).attr("cy", (d) => y(d))
      .attr("r", 3).attr("fill", "#4ecdc4");

    g.append("text").attr("x", width / 2).attr("y", height + 30)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#8888aa").text("Layer");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", -45)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#8888aa").text("L2 Norm");
  }, [completeMsg]);

  if (!completeMsg?.norms) return <p style={{ color: "#666" }}>No residual norm data</p>;

  const norms = completeMsg.norms;
  const csvRows = (): (string | number)[][] => [
    ["layer", "l2_norm"],
    ...norms.map((n, i) => [i, n] as (string | number)[]),
  ];

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>Residual Stream Norms - {result.sessionName}</h3>
        <ExportButtons
          filenameBase={`residual-norms_${result.sessionName}`}
          getSVG={() => svgRef.current}
          getCSVRows={csvRows}
        />
      </div>
      <svg ref={svgRef} />
    </div>
  );
}
