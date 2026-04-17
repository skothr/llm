import { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import { ExportButtons } from "../ExportButtons";
import { CompareSelector, useResultById } from "../CompareSelector";
import type { ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

function normsOf(result: ProbeResult | null): number[] | null {
  if (!result) return null;
  const msg = result.data.find((m) => m.type === "complete") as
    | { type: "complete"; norms?: number[] }
    | undefined;
  return msg?.norms ?? null;
}

export function ResidualNorms({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [compareId, setCompareId] = useState<string | null>(null);
  const compareResult = useResultById(compareId);

  const normsA = normsOf(result);
  const normsB = normsOf(compareResult);

  useEffect(() => {
    if (!svgRef.current || !normsA) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // When comparing, the x-axis spans whichever series is longer so both
    // fit. Mismatched layer counts commonly mean one session had layers
    // zeroed — worth showing the shorter series truncating rather than
    // hiding the fact.
    const lenMax = Math.max(normsA.length, normsB?.length ?? 0);
    const ymax = Math.max(
      d3.max(normsA) ?? 1,
      normsB ? (d3.max(normsB) ?? 0) : 0,
    );

    const margin = { top: 20, right: 20, bottom: 35, left: 60 };
    const width = 500;
    const height = 200;
    svg.attr("width", width + margin.left + margin.right)
       .attr("height", height + margin.top + margin.bottom);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const x = d3.scaleLinear().domain([0, lenMax - 1]).range([0, width]);
    const y = d3.scaleLinear().domain([0, ymax || 1]).range([height, 0]).nice();

    g.append("g").attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(Math.min(lenMax, 20)).tickFormat((d) => `${d}`))
      .selectAll("text").attr("fill", "#8888aa").attr("font-size", 10);
    g.selectAll(".domain, .tick line").attr("stroke", "#333");

    g.append("g").call(d3.axisLeft(y).ticks(5))
      .selectAll("text").attr("fill", "#8888aa").attr("font-size", 10);
    g.selectAll(".domain, .tick line").attr("stroke", "#333");

    const lineGen = d3.line<number>().x((_, i) => x(i)).y((d) => y(d));

    // Series A in cyan (the "current" result)
    g.append("path").datum(normsA)
      .attr("d", lineGen)
      .attr("fill", "none").attr("stroke", "#4ecdc4").attr("stroke-width", 2);
    g.selectAll(".a-dot").data(normsA).join("circle")
      .attr("class", "a-dot")
      .attr("cx", (_, i) => x(i)).attr("cy", (d) => y(d))
      .attr("r", 3).attr("fill", "#4ecdc4");

    if (normsB) {
      // Series B in amber, dashed, to visually distinguish from A even in
      // the colorblind-hostile red/green corners of the palette.
      g.append("path").datum(normsB)
        .attr("d", lineGen)
        .attr("fill", "none").attr("stroke", "#f0ad4e").attr("stroke-width", 2)
        .attr("stroke-dasharray", "4,3");
      g.selectAll(".b-dot").data(normsB).join("circle")
        .attr("class", "b-dot")
        .attr("cx", (_, i) => x(i)).attr("cy", (d) => y(d))
        .attr("r", 2.5).attr("fill", "#f0ad4e");

      // Legend
      const legend = g.append("g").attr("transform", `translate(${width - 140},0)`);
      legend.append("rect").attr("x", 0).attr("y", 0).attr("width", 12).attr("height", 2).attr("fill", "#4ecdc4");
      legend.append("text").attr("x", 16).attr("y", 4).attr("font-size", 10).attr("fill", "#a0a0c0")
        .text(`A: ${result.sessionName}`);
      legend.append("rect").attr("x", 0).attr("y", 12).attr("width", 12).attr("height", 2).attr("fill", "#f0ad4e");
      legend.append("text").attr("x", 16).attr("y", 16).attr("font-size", 10).attr("fill", "#a0a0c0")
        .text(`B: ${compareResult?.sessionName ?? "compare"}`);
    }

    g.append("text").attr("x", width / 2).attr("y", height + 30)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#8888aa").text("Layer");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", -45)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#8888aa").text("L2 Norm");
  }, [normsA, normsB, result.sessionName, compareResult?.sessionName]);

  if (!normsA) return <p style={{ color: "#666" }}>No residual norm data</p>;

  const csvRows = (): (string | number)[][] => {
    if (!normsB) {
      return [["layer", "l2_norm"], ...normsA.map((n, i) => [i, n] as (string | number)[])];
    }
    // Paired wide format with explicit delta column when comparing.
    const len = Math.max(normsA.length, normsB.length);
    const rows: (string | number)[][] = [["layer", "norm_a", "norm_b", "delta"]];
    for (let i = 0; i < len; i++) {
      const a = normsA[i];
      const b = normsB[i];
      rows.push([i, a ?? "", b ?? "", a != null && b != null ? a - b : ""]);
    }
    return rows;
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8, gap: 12, flexWrap: "wrap" }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>Residual Stream Norms - {result.sessionName}</h3>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <CompareSelector
            operation="residual-norms"
            currentId={result.id}
            value={compareId}
            onChange={setCompareId}
          />
          <ExportButtons
            filenameBase={`residual-norms_${result.sessionName}${compareResult ? `_vs_${compareResult.sessionName}` : ""}`}
            getSVG={() => svgRef.current}
            getCSVRows={csvRows}
          />
        </div>
      </div>
      <svg ref={svgRef} />
    </div>
  );
}
