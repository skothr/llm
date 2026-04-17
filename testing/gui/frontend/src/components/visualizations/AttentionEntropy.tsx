import { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import { ExportButtons } from "../ExportButtons";
import { CompareSelector, useResultById } from "../CompareSelector";
import type { ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

function entropyOf(result: ProbeResult | null): Record<string, number[]> | null {
  if (!result) return null;
  const msg = result.data.find((m) => m.type === "complete") as
    | { type: "complete"; entropy?: Record<string, number[]> }
    | undefined;
  return msg?.entropy ?? null;
}

export function AttentionEntropy({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);
  const [compareId, setCompareId] = useState<string | null>(null);
  const compareResult = useResultById(compareId);

  const entropyA = entropyOf(result);
  const entropyB = entropyOf(compareResult);
  const diffing = entropyA != null && entropyB != null;

  useEffect(() => {
    if (!svgRef.current || !entropyA) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Union of L_H keys across A and B. Cells missing from one side render
    // with a dashed "N/A" overlay when comparing.
    const keys = new Set<string>(Object.keys(entropyA));
    if (entropyB) for (const k of Object.keys(entropyB)) keys.add(k);

    const entries = [...keys].map((key) => {
      const parts = key.split("_").map(Number);
      const valuesA = entropyA[key];
      const valuesB = entropyB?.[key];
      const meanA = valuesA ? (d3.mean(valuesA) ?? 0) : null;
      const meanB = valuesB ? (d3.mean(valuesB) ?? 0) : null;
      // In diff mode the cell value is A-B means; otherwise A's mean.
      const cellValue = diffing && meanA != null && meanB != null ? meanA - meanB
        : meanA;
      return { layer: parts[0], head: parts[1], valuesA, valuesB, meanA, meanB, cellValue };
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

    // Color scale:
    //   single-series → RdYlBu: lower entropy (focused head) = red,
    //                    higher entropy (diffuse head) = blue
    //   diff mode     → RdBu diverging around 0: red = A > B, blue = A < B
    const bgScale: (v: number | null) => string = (v) => {
      if (v == null) return "#1a1a2a";
      if (!diffing) {
        const scale = d3.scaleSequential(d3.interpolateRdYlBu).domain([d3.max(entries, (e) => e.meanA ?? 0)!, 0]);
        return scale(v);
      }
      const maxAbs = d3.max(entries, (e) => e.cellValue != null ? Math.abs(e.cellValue) : 0) || 1;
      const scale = d3.scaleDiverging(d3.interpolateRdBu).domain([maxAbs, 0, -maxAbs]);
      return scale(v);
    };

    // Per-position sparkline ranges use A's data in single-mode, or (A-B)
    // deltas per position in diff mode.
    const firstValues = entries.find((e) => e.valuesA)?.valuesA ?? [];
    const numPositions = firstValues.length || 1;
    const sparkX = d3.scaleLinear().domain([0, numPositions - 1]).range([2, cellSize - 2]);
    const sparkYMax = diffing
      ? d3.max(entries, (e) => {
          if (!e.valuesA || !e.valuesB) return 0;
          let m = 0;
          const n = Math.min(e.valuesA.length, e.valuesB.length);
          for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(e.valuesA[i] - e.valuesB[i]));
          return m;
        }) || 1
      : d3.max(entries, (e) => e.valuesA ? (d3.max(e.valuesA) ?? 0) : 0) || 1;
    const sparkY = d3.scaleLinear()
      .domain(diffing ? [-sparkYMax, sparkYMax] : [0, sparkYMax])
      .range([cellSize - 4, cellSize - sparkHeight]);

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
        .attr("fill", bgScale(d.cellValue))
        .attr("rx", 3).attr("opacity", 0.35).style("cursor", "pointer")
        .on("mouseenter", (event) => {
          const lines = [`L${d.layer} H${d.head}`];
          if (diffing) {
            lines.push(`A mean: ${d.meanA?.toFixed(3) ?? "n/a"}`);
            lines.push(`B mean: ${d.meanB?.toFixed(3) ?? "n/a"}`);
            lines.push(`delta:  ${d.cellValue != null ? (d.cellValue >= 0 ? "+" : "") + d.cellValue.toFixed(3) : "n/a"}`);
          } else if (d.valuesA) {
            lines.push(`mean: ${(d.meanA ?? 0).toFixed(3)}`);
            lines.push(`min:  ${(d3.min(d.valuesA) ?? 0).toFixed(3)}`);
            lines.push(`max:  ${(d3.max(d.valuesA) ?? 0).toFixed(3)}`);
          }
          setTooltip({ x: event.pageX + 10, y: event.pageY - 10, content: lines.join("\n") });
        })
        .on("mouseleave", () => setTooltip(null));

      if (diffing) {
        // Baseline at zero for the delta sparkline.
        g.append("line").attr("x1", x + sparkX(0)).attr("x2", x + sparkX(numPositions - 1))
          .attr("y1", y + sparkY(0)).attr("y2", y + sparkY(0))
          .attr("stroke", "#445566").attr("stroke-width", 0.5).style("pointer-events", "none");

        if (d.valuesA && d.valuesB) {
          const n = Math.min(d.valuesA.length, d.valuesB.length);
          const deltas = Array.from({ length: n }, (_, i) => d.valuesA![i] - d.valuesB![i]);
          const line = d3.line<number>().x((_, i) => x + sparkX(i)).y((v) => y + sparkY(v));
          g.append("path").datum(deltas).attr("d", line)
            .attr("fill", "none").attr("stroke", "#e0e0e0").attr("stroke-width", 1).style("pointer-events", "none");
        }
      } else if (d.valuesA) {
        const line = d3.line<number>().x((_, i) => x + sparkX(i)).y((v) => y + sparkY(v));
        g.append("path").datum(d.valuesA).attr("d", line)
          .attr("fill", "none").attr("stroke", "#e0e0e0").attr("stroke-width", 1).style("pointer-events", "none");
      }

      g.append("text").attr("x", x + cellSize / 2).attr("y", y + 12)
        .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#ccccdd").style("pointer-events", "none")
        .text(diffing
          ? (d.cellValue != null ? (d.cellValue >= 0 ? "+" : "") + d.cellValue.toFixed(2) : "—")
          : (d.meanA != null ? d.meanA.toFixed(2) : "—"));
    });
  }, [entropyA, entropyB, diffing]);

  if (!entropyA) return <p style={{ color: "#666" }}>No attention entropy data</p>;

  const csvRows = (): (string | number)[][] => {
    if (!entropyB) {
      const entries = Object.entries(entropyA).map(([key, values]) => {
        const [l, h] = key.split("_").map(Number);
        const mean = values.reduce((a, b) => a + b, 0) / (values.length || 1);
        return { layer: l, head: h, mean, values };
      }).sort((a, b) => a.layer - b.layer || a.head - b.head);
      const numPos = entries[0]?.values.length ?? 0;
      const header = ["layer", "head", "mean_entropy", ...Array.from({ length: numPos }, (_, i) => `pos${i}`)];
      return [header, ...entries.map((e) => [e.layer, e.head, e.mean, ...e.values] as (string | number)[])];
    }
    // Diff CSV: mean_a, mean_b, delta, per-position deltas.
    const keys = new Set<string>();
    for (const k of Object.keys(entropyA)) keys.add(k);
    for (const k of Object.keys(entropyB)) keys.add(k);
    const entries = [...keys].map((k) => {
      const [l, h] = k.split("_").map(Number);
      const va = entropyA[k];
      const vb = entropyB[k];
      return { layer: l, head: h, va, vb };
    }).sort((a, b) => a.layer - b.layer || a.head - b.head);
    const numPos = Math.max(...entries.map((e) => Math.min(e.va?.length ?? 0, e.vb?.length ?? 0)), 0);
    const header = ["layer", "head", "mean_a", "mean_b", "delta_mean", ...Array.from({ length: numPos }, (_, i) => `delta_pos${i}`)];
    const rows: (string | number)[][] = [header];
    for (const e of entries) {
      const ma = e.va ? e.va.reduce((a, b) => a + b, 0) / (e.va.length || 1) : null;
      const mb = e.vb ? e.vb.reduce((a, b) => a + b, 0) / (e.vb.length || 1) : null;
      const deltas: (string | number)[] = [];
      for (let i = 0; i < numPos; i++) {
        const a = e.va?.[i];
        const b = e.vb?.[i];
        deltas.push(a != null && b != null ? a - b : "");
      }
      rows.push([e.layer, e.head, ma ?? "", mb ?? "", ma != null && mb != null ? ma - mb : "", ...deltas]);
    }
    return rows;
  };

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8, gap: 12, flexWrap: "wrap" }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
          Attention Entropy - {result.sessionName}
          {diffing && <span style={{ color: "#8888aa", fontSize: 11 }}> {"\u2212"} {compareResult?.sessionName} ({"\u0394"} shown)</span>}
        </h3>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <CompareSelector
            operation="attention"
            currentId={result.id}
            value={compareId}
            onChange={setCompareId}
          />
          <ExportButtons
            filenameBase={`attention-entropy_${result.sessionName}${compareResult ? `_vs_${compareResult.sessionName}` : ""}`}
            getSVG={() => svgRef.current}
            getCSVRows={csvRows}
          />
        </div>
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
