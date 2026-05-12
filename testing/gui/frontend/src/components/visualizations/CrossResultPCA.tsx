import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import { pca } from "../../utils/pca";
import { decodeHiddenState } from "../../utils/hiddenState";
import { displayToken } from "../../utils/displayToken";
import { ExportButtons } from "../ExportButtons";
import type { LogitLensData, ProbeResult, EncodedHiddenStateMsg } from "../../types/api";

interface Props {
  results: ProbeResult[];
  onClose: () => void;
}

interface Sample {
  vec: Float32Array;
  resultId: string;
  sessionName: string;
  layer: number;
  position: number;
  topToken: string;
}

// Cross-result PCA: given 2+ logit-lens results, pool every
// (layer, position) hidden vector and project the union to 2D.
// Points are colored by result so the researcher can see whether
// surgeries / interventions push representations into distinct regions
// of hidden-state space or leave them clustered together.
//
// Rendered as a modal so opening it doesn't disrupt the active-result
// view in VisualizationArea.
export function CrossResultPCA({ results, onClose }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const samples = useMemo<Sample[]>(() => {
    const out: Sample[] = [];
    for (const r of results) {
      const frames = r.data.filter(
        (m): m is LogitLensData => m.type === "data" && "predictions" in m,
      );
      for (const msg of frames) {
        const hs = msg.hidden_state as EncodedHiddenStateMsg | undefined;
        if (!hs) continue;
        const [seqLen, hiddenSize] = hs.shape;
        const full = decodeHiddenState(hs);
        if (full.length !== seqLen * hiddenSize) continue;
        for (let pos = 0; pos < seqLen; pos++) {
          const vec = full.subarray(pos * hiddenSize, (pos + 1) * hiddenSize);
          out.push({
            vec: new Float32Array(vec),
            resultId: r.id,
            sessionName: r.sessionName,
            layer: msg.layer,
            position: pos,
            topToken: msg.predictions[pos]?.[0]?.token ?? "",
          });
        }
      }
    }
    return out;
  }, [results]);

  // Check that all samples share a dimension before running PCA —
  // pooling across different model sizes doesn't have a meaningful
  // joint embedding.
  const dimsConsistent = useMemo(() => {
    if (samples.length === 0) return true;
    const d0 = samples[0].vec.length;
    return samples.every((s) => s.vec.length === d0);
  }, [samples]);

  const pcaResult = useMemo(() => {
    if (samples.length < 2 || !dimsConsistent) return null;
    return pca(samples.map((s) => s.vec), 2, 40);
  }, [samples, dimsConsistent]);

  // Categorical color scheme — one color per result, stable across
  // renders because we map by result.id.
  const resultColorMap = useMemo(() => {
    const palette = d3.schemeTableau10 as readonly string[];
    const map = new Map<string, string>();
    results.forEach((r, i) => map.set(r.id, palette[i % palette.length]));
    return map;
  }, [results]);

  useEffect(() => {
    if (!svgRef.current || !pcaResult || samples.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 24, bottom: 40, left: 50 };
    const width = 600;
    const height = 400;
    svg.attr("width", width + margin.left + margin.right).attr("height", height + margin.top + margin.bottom);

    const xs = pcaResult.projections.map((p) => p[0]);
    const ys = pcaResult.projections.map((p) => p[1]);
    const x = d3.scaleLinear().domain(d3.extent(xs) as [number, number]).range([0, width]).nice();
    const y = d3.scaleLinear().domain(d3.extent(ys) as [number, number]).range([height, 0]).nice();

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    g.append("g").attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(6)).selectAll("text").attr("fill", "#8888aa").attr("font-size", 10);
    g.append("g").call(d3.axisLeft(y).ticks(6))
      .selectAll("text").attr("fill", "#8888aa").attr("font-size", 10);
    g.selectAll(".domain, .tick line").attr("stroke", "#333");

    g.append("text").attr("x", width / 2).attr("y", height + 30)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#8888aa")
      .text(`PC1 (${(pcaResult.explainedVariance[0] * 100).toFixed(1)}% var)`);
    g.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", -38)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#8888aa")
      .text(`PC2 (${(pcaResult.explainedVariance[1] * 100).toFixed(1)}% var)`);

    samples.forEach((s, i) => {
      const p = pcaResult.projections[i];
      const color = resultColorMap.get(s.resultId) ?? "#999";
      g.append("circle")
        .attr("cx", x(p[0])).attr("cy", y(p[1]))
        .attr("r", 3)
        .attr("fill", color)
        .attr("opacity", 0.7)
        .style("cursor", "pointer")
        .on("mouseenter", (event) => {
          // clientX/Y: tooltip renders with position:fixed.
          setTooltip({
            x: event.clientX + 10, y: event.clientY - 10,
            content: `${s.sessionName}\nL${s.layer} pos ${s.position}\ntoken: ${displayToken(s.topToken)}\nPC1=${p[0].toFixed(3)}  PC2=${p[1].toFixed(3)}`,
          });
        })
        .on("mouseleave", () => setTooltip(null));
    });
  }, [samples, pcaResult, resultColorMap]);

  const csvRows = (): (string | number)[][] => {
    if (!pcaResult) return [];
    const rows: (string | number)[][] = [["result_id", "session", "layer", "position", "top_token", "pc1", "pc2"]];
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      const p = pcaResult.projections[i];
      rows.push([s.resultId, s.sessionName, s.layer, s.position, s.topToken, p[0], p[1]]);
    }
    return rows;
  };

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.65)",
        display: "flex", alignItems: "center", justifyContent: "center",
        zIndex: 500,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#0f1626", border: "1px solid #1a5276", borderRadius: 6,
          padding: "14px 18px", maxWidth: 820,
          color: "#e0e0f0", boxShadow: "0 12px 40px rgba(0,0,0,0.6)",
          maxHeight: "90vh", overflow: "auto",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, gap: 12 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 13, color: "#a0a0c0" }}>
              Cross-Result PCA — {results.length} selected
            </h3>
            <div style={{ fontSize: 10, color: "#667" }}>
              {samples.length} hidden-state samples · {samples[0]?.vec.length ?? 0} dims
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {pcaResult && (
              <ExportButtons
                filenameBase="cross-pca"
                getSVG={() => svgRef.current}
                getCSVRows={csvRows}
              />
            )}
            <button
              onClick={onClose}
              style={{ background: "transparent", border: "none", color: "#888", cursor: "pointer", fontSize: 16, padding: 0 }}
            >{"\u00d7"}</button>
          </div>
        </div>

        {!dimsConsistent && (
          <p style={{ fontSize: 11, color: "#f0ad4e" }}>
            Selected results have different hidden sizes — cross-model PCA isn't meaningful. Select results from the same model family.
          </p>
        )}
        {dimsConsistent && samples.length < 2 && (
          <p style={{ fontSize: 11, color: "#f0ad4e" }}>
            Need ≥2 hidden-state samples across the selection. (Hidden states are stripped from persisted results — re-run if selected results came from a prior session.)
          </p>
        )}
        {pcaResult && (
          <>
            <svg ref={svgRef} />
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 8, fontSize: 10, fontFamily: "monospace" }}>
              {results.map((r) => (
                <span key={r.id} style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "#ccc" }}>
                  <span style={{
                    width: 10, height: 10, borderRadius: 5,
                    background: resultColorMap.get(r.id) ?? "#999",
                  }} />
                  {r.sessionName} — "{r.prompt.slice(0, 20)}{r.prompt.length > 20 ? "\u2026" : ""}"
                </span>
              ))}
            </div>
          </>
        )}
        {tooltip && (
          <div style={{
            position: "fixed", left: tooltip.x, top: tooltip.y,
            background: "#16213e", border: "1px solid #1a5276", borderRadius: 4,
            padding: "6px 10px", fontFamily: "monospace", fontSize: 11,
            whiteSpace: "pre", pointerEvents: "none", zIndex: 600,
          }}>
            {tooltip.content}
          </div>
        )}
      </div>
    </div>
  );
}
