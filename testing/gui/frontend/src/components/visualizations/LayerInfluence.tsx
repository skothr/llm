import { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import { ExportButtons } from "../ExportButtons";
import { CompareSelector, useResultById } from "../CompareSelector";
import type { ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

function scoresOf(result: ProbeResult | null): Record<string, number> | null {
  if (!result) return null;
  const msg = result.data.find((m) => m.type === "complete") as
    | { type: "complete"; scores?: Record<string, number> }
    | undefined;
  return msg?.scores ?? null;
}

export function LayerInfluence({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [compareId, setCompareId] = useState<string | null>(null);
  const compareResult = useResultById(compareId);

  const scoresA = scoresOf(result);
  const scoresB = scoresOf(compareResult);

  useEffect(() => {
    if (!svgRef.current || !scoresA) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Merge layer index universe across A and B so a layer present in only
    // one still gets a row (useful when comparing clones with different
    // layer_map edits).
    const layers = new Set<number>();
    for (const k of Object.keys(scoresA)) layers.add(parseInt(k));
    if (scoresB) for (const k of Object.keys(scoresB)) layers.add(parseInt(k));
    const sorted = [...layers].sort((a, b) => a - b);

    const margin = { top: 20, right: 20, bottom: 30, left: 50 };
    const barHeight = 22;
    const width = 500;
    const height = margin.top + sorted.length * barHeight + margin.bottom;
    svg.attr("width", width).attr("height", height);

    const scoreMax = d3.max([
      ...sorted.map((l) => scoresA[l] ?? 0),
      ...(scoresB ? sorted.map((l) => scoresB[l] ?? 0) : []),
    ]) || 1;

    const x = d3.scaleLinear().domain([0, scoreMax]).range([0, width - margin.left - margin.right]);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const colorScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, scoreMax]);

    sorted.forEach((layer, i) => {
      const a = scoresA[layer];
      const b = scoresB?.[layer];
      const barH = scoresB ? (barHeight - 2) / 2 - 1 : barHeight - 2;

      // Layer label (left gutter)
      g.append("text")
        .attr("x", -4)
        .attr("y", i * barHeight + barHeight / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", 10)
        .attr("fill", "#8888aa")
        .text(`L${layer}`);

      if (a != null) {
        g.append("rect")
          .attr("x", 0)
          .attr("y", i * barHeight)
          .attr("width", x(a))
          .attr("height", barH)
          .attr("fill", scoresB ? "#4ecdc4" : colorScale(a))
          .attr("rx", 2);
        g.append("text")
          .attr("x", x(a) + 4)
          .attr("y", i * barHeight + barH / 2)
          .attr("dominant-baseline", "middle")
          .attr("font-size", 10)
          .attr("fill", "#ccccdd")
          .text(a.toFixed(3));
      }
      if (scoresB && b != null) {
        // Stack B under A — the same layer-axis, different rows per
        // series. Dashed outline gives a secondary cue beyond color.
        g.append("rect")
          .attr("x", 0)
          .attr("y", i * barHeight + barH + 2)
          .attr("width", x(b))
          .attr("height", barH)
          .attr("fill", "#f0ad4e")
          .attr("stroke", "#c07020")
          .attr("stroke-dasharray", "3,2")
          .attr("rx", 2);
        g.append("text")
          .attr("x", x(b) + 4)
          .attr("y", i * barHeight + barH + 2 + barH / 2)
          .attr("dominant-baseline", "middle")
          .attr("font-size", 10)
          .attr("fill", "#ccccdd")
          .text(b.toFixed(3));
      }
    });

    if (scoresB) {
      const legend = g.append("g").attr("transform", `translate(${width - margin.left - margin.right - 160},${-12})`);
      legend.append("rect").attr("x", 0).attr("y", 0).attr("width", 12).attr("height", 6).attr("fill", "#4ecdc4");
      legend.append("text").attr("x", 16).attr("y", 6).attr("font-size", 10).attr("fill", "#a0a0c0")
        .text(`A: ${result.sessionName}`);
      legend.append("rect").attr("x", 0).attr("y", 8).attr("width", 12).attr("height", 6).attr("fill", "#f0ad4e");
      legend.append("text").attr("x", 16).attr("y", 14).attr("font-size", 10).attr("fill", "#a0a0c0")
        .text(`B: ${compareResult?.sessionName ?? "compare"}`);
    }
  }, [scoresA, scoresB, result.sessionName, compareResult?.sessionName]);

  if (!scoresA) {
    return <p style={{ color: "#666" }}>No influence data</p>;
  }

  const csvRows = (): (string | number)[][] => {
    if (!scoresB) {
      const rows = Object.entries(scoresA)
        .map(([layer, score]) => [parseInt(layer), score] as (string | number)[])
        .sort((a, b) => (a[0] as number) - (b[0] as number));
      return [["layer", "score"], ...rows];
    }
    const layers = new Set<number>();
    for (const k of Object.keys(scoresA)) layers.add(parseInt(k));
    for (const k of Object.keys(scoresB)) layers.add(parseInt(k));
    const sorted = [...layers].sort((a, b) => a - b);
    return [
      ["layer", "score_a", "score_b", "delta"],
      ...sorted.map((l) => {
        const a = scoresA[l];
        const b = scoresB[l];
        return [l, a ?? "", b ?? "", a != null && b != null ? a - b : ""] as (string | number)[];
      }),
    ];
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8, gap: 12, flexWrap: "wrap" }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
          Layer Influence - {result.sessionName}
        </h3>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <CompareSelector
            operation="influence"
            currentId={result.id}
            value={compareId}
            onChange={setCompareId}
          />
          <ExportButtons
            filenameBase={`layer-influence_${result.sessionName}${compareResult ? `_vs_${compareResult.sessionName}` : ""}`}
            getSVG={() => svgRef.current}
            getCSVRows={csvRows}
          />
        </div>
      </div>
      <svg ref={svgRef} />
    </div>
  );
}
