import { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import { ExportButtons } from "../ExportButtons";
import type { ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

export function AttentionEntropy({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);

  const completeMsg = result.data.find((m) => m.type === "complete") as
    | { type: "complete"; entropy: Record<string, number[]> }
    | undefined;

  useEffect(() => {
    if (!svgRef.current || !completeMsg?.entropy) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const entries = Object.entries(completeMsg.entropy).map(([key, values]) => {
      const parts = key.split("_").map(Number);
      return { layer: parts[0], head: parts[1], values, mean: d3.mean(values) || 0 };
    }).sort((a, b) => a.layer - b.layer || a.head - b.head);

    if (entries.length === 0) return;

    const numLayers = d3.max(entries, (d) => d.layer)! + 1;
    const numHeads = d3.max(entries, (d) => d.head)! + 1;
    const cellSize = 60;
    const sparkHeight = 30;
    const margin = { top: 30, right: 20, bottom: 20, left: 40 };
    const gap = 4;
    const width = margin.left + numHeads * (cellSize + gap) + margin.right;
    const height = margin.top + numLayers * (cellSize + gap) + margin.bottom;

    svg.attr("width", width).attr("height", height);

    const maxEntropy = d3.max(entries, (d) => d.mean) || 1;
    const bgScale = d3.scaleSequential(d3.interpolateRdYlBu).domain([maxEntropy, 0]);
    const numPositions = entries[0].values.length || 1;
    const sparkX = d3.scaleLinear().domain([0, numPositions - 1]).range([2, cellSize - 2]);
    const sparkYMax = d3.max(entries, (d) => d3.max(d.values) || 0) || 1;
    const sparkY = d3.scaleLinear().domain([0, sparkYMax]).range([cellSize - 4, cellSize - sparkHeight]);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    d3.range(numHeads).forEach((h) => {
      g.append("text").attr("x", h * (cellSize + gap) + cellSize / 2).attr("y", -8)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#8888aa").text(`H${h}`);
    });
    d3.range(numLayers).forEach((l) => {
      g.append("text").attr("x", -4).attr("y", l * (cellSize + gap) + cellSize / 2)
        .attr("text-anchor", "end").attr("dominant-baseline", "middle").attr("font-size", 10).attr("fill", "#8888aa").text(`L${l}`);
    });

    entries.forEach((d) => {
      const x = d.head * (cellSize + gap);
      const y = d.layer * (cellSize + gap);

      g.append("rect").attr("x", x).attr("y", y).attr("width", cellSize).attr("height", cellSize)
        .attr("fill", bgScale(d.mean)).attr("rx", 3).attr("opacity", 0.3).style("cursor", "pointer")
        .on("mouseenter", (event) => {
          setTooltip({ x: event.pageX + 10, y: event.pageY - 10,
            content: `L${d.layer} H${d.head}\nmean: ${d.mean.toFixed(3)}\nmin: ${d3.min(d.values)!.toFixed(3)}\nmax: ${d3.max(d.values)!.toFixed(3)}` });
        })
        .on("mouseleave", () => setTooltip(null));

      const line = d3.line<number>().x((_, i) => x + sparkX(i)).y((v) => y + sparkY(v));
      g.append("path").datum(d.values).attr("d", line)
        .attr("fill", "none").attr("stroke", "#e0e0e0").attr("stroke-width", 1).style("pointer-events", "none");

      g.append("text").attr("x", x + cellSize / 2).attr("y", y + 12)
        .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#ccccdd").style("pointer-events", "none")
        .text(d.mean.toFixed(2));
    });
  }, [completeMsg]);

  if (!completeMsg?.entropy) return <p style={{ color: "#666" }}>No attention entropy data</p>;

  const entropy = completeMsg.entropy;
  const csvRows = (): (string | number)[][] => {
    // "layer,head,mean,pos0,pos1,..." wide format. Researchers importing into
    // pandas get a flat DataFrame; long format is a one-liner melt downstream.
    const entries = Object.entries(entropy).map(([key, values]) => {
      const [l, h] = key.split("_").map(Number);
      const mean = values.reduce((a, b) => a + b, 0) / (values.length || 1);
      return { layer: l, head: h, mean, values };
    }).sort((a, b) => a.layer - b.layer || a.head - b.head);
    const numPos = entries[0]?.values.length ?? 0;
    const header = ["layer", "head", "mean_entropy", ...Array.from({ length: numPos }, (_, i) => `pos${i}`)];
    const rows: (string | number)[][] = [header];
    for (const e of entries) rows.push([e.layer, e.head, e.mean, ...e.values]);
    return rows;
  };

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>Attention Entropy - {result.sessionName}</h3>
        <ExportButtons
          filenameBase={`attention-entropy_${result.sessionName}`}
          getSVG={() => svgRef.current}
          getCSVRows={csvRows}
        />
      </div>
      <div style={{ overflowX: "auto" }}><svg ref={svgRef} /></div>
      {tooltip && (
        <div style={{ position: "fixed", left: tooltip.x, top: tooltip.y, background: "#16213e", border: "1px solid #1a5276", borderRadius: 4, padding: "6px 10px", fontFamily: "monospace", fontSize: 12, whiteSpace: "pre", pointerEvents: "none", zIndex: 100 }}>
          {tooltip.content}
        </div>
      )}
    </div>
  );
}
