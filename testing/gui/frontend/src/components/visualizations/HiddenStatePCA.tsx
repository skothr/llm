import { useRef, useEffect, useMemo, useState } from "react";
import * as d3 from "d3";
import { pca } from "../../utils/pca";
import { decodeHiddenState } from "../../utils/hiddenState";
import { ExportButtons } from "../ExportButtons";
import { displayToken } from "../../utils/displayToken";
import type { LogitLensData, ProbeResult, EncodedHiddenStateMsg } from "../../types/api";

interface Props {
  result: ProbeResult;
}

interface Sample {
  vec: Float32Array;
  layer: number;
  originalLayer: number;
  sublayer: string;
  position: number;
  topToken: string;
}

type ColorMode = "layer" | "position";

// Gathers every (layer, position) hidden vector from a logit-lens result
// and projects it into 2D via power-iteration PCA. Rendered as a scatter
// under the main heatmap so researchers can see how representations move
// through the network without leaving the browser.
export function HiddenStatePCA({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [computed, setComputed] = useState(false);
  const [colorMode, setColorMode] = useState<ColorMode>("layer");
  const [selectedPos, setSelectedPos] = useState<number | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);

  // Build samples from the streamed frames. Each frame is one layer; each
  // position within that frame is one sample. We only decode when the
  // user clicks "compute" — base64 decoding + Float32Array alloc adds up
  // on long prompts.
  const samples = useMemo<Sample[] | null>(() => {
    if (!computed) return null;
    const out: Sample[] = [];
    const frames = result.data.filter(
      (m): m is LogitLensData => m.type === "data" && "predictions" in m,
    );
    for (const msg of frames) {
      const hs = msg.hidden_state as EncodedHiddenStateMsg | undefined;
      if (!hs) continue;
      const [seqLen, hiddenSize] = hs.shape;
      const full = decodeHiddenState(hs);
      // Sanity: sometimes the persisted copy dropped b64 (we strip on
      // save to save space); skip without throwing.
      if (full.length !== seqLen * hiddenSize) continue;
      for (let pos = 0; pos < seqLen; pos++) {
        const vec = full.subarray(pos * hiddenSize, (pos + 1) * hiddenSize);
        // Clone so later deflation/centering can't alias back to the
        // source buffer.
        const copy = new Float32Array(vec);
        const top = msg.predictions[pos]?.[0]?.token ?? "";
        out.push({
          vec: copy,
          layer: msg.layer,
          originalLayer: msg.original_layer ?? msg.layer,
          sublayer: msg.sublayer,
          position: pos,
          topToken: top,
        });
      }
    }
    return out;
  }, [computed, result.data]);

  const pcaResult = useMemo(() => {
    if (!samples || samples.length < 2) return null;
    return pca(samples.map((s) => s.vec), 2, 40);
  }, [samples]);

  useEffect(() => {
    if (!svgRef.current || !samples || !pcaResult) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 24, bottom: 40, left: 50 };
    const width = 520;
    const height = 340;
    svg.attr("width", width + margin.left + margin.right).attr("height", height + margin.top + margin.bottom);

    const xVals = pcaResult.projections.map((p) => p[0]);
    const yVals = pcaResult.projections.map((p) => p[1]);
    const x = d3.scaleLinear().domain(d3.extent(xVals) as [number, number]).range([0, width]).nice();
    const y = d3.scaleLinear().domain(d3.extent(yVals) as [number, number]).range([height, 0]).nice();

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

    const maxLayer = Math.max(...samples.map((s) => s.layer));
    const maxPos = Math.max(...samples.map((s) => s.position));
    const layerColor = d3.scaleSequential(d3.interpolateViridis).domain([0, maxLayer || 1]);
    const posColor = d3.scaleSequential(d3.interpolateTurbo).domain([0, maxPos || 1]);
    const colorOf = (s: Sample) => colorMode === "layer" ? layerColor(s.layer) : posColor(s.position);

    // Faded points for non-selected positions when a trajectory is active.
    samples.forEach((s, i) => {
      const focused = selectedPos === null || s.position === selectedPos;
      const p = pcaResult.projections[i];
      g.append("circle")
        .attr("cx", x(p[0])).attr("cy", y(p[1]))
        .attr("r", focused ? 4 : 2)
        .attr("fill", colorOf(s))
        .attr("opacity", focused ? 0.85 : 0.15)
        .style("cursor", "pointer")
        .on("mouseenter", (event) => {
          // clientX/Y: tooltip renders with position:fixed.
          setTooltip({
            x: event.clientX + 10,
            y: event.clientY - 10,
            content: `L${s.layer}.${s.sublayer} pos ${s.position}\ntoken: ${displayToken(s.topToken)}\nPC1=${p[0].toFixed(3)}  PC2=${p[1].toFixed(3)}`,
          });
        })
        .on("mouseleave", () => setTooltip(null))
        .on("click", () => setSelectedPos((prev) => prev === s.position ? null : s.position));
    });

    // When a position is selected, connect its samples across layers with
    // a light polyline so the trajectory through layers is visible.
    if (selectedPos !== null) {
      const trajectory = samples
        .map((s, i) => ({ s, p: pcaResult.projections[i] }))
        .filter(({ s }) => s.position === selectedPos)
        .sort((a, b) => a.s.layer - b.s.layer);
      if (trajectory.length > 1) {
        const line = d3.line<{ s: Sample; p: Float32Array }>()
          .x((d) => x(d.p[0]))
          .y((d) => y(d.p[1]));
        g.append("path")
          .datum(trajectory)
          .attr("d", line)
          .attr("fill", "none")
          .attr("stroke", "#e0c040")
          .attr("stroke-width", 1.5)
          .attr("opacity", 0.7);
      }
    }
  }, [samples, pcaResult, colorMode, selectedPos]);

  const haveHiddenStates = result.data.some(
    (m) => m.type === "data" && "predictions" in m && (m as LogitLensData).hidden_state != null,
  );

  if (!haveHiddenStates) {
    return null;
  }

  const csvRows = (): (string | number)[][] => {
    if (!samples || !pcaResult) return [];
    const rows: (string | number)[][] = [["layer", "original_layer", "sublayer", "position", "top_token", "pc1", "pc2"]];
    for (let i = 0; i < samples.length; i++) {
      const s = samples[i];
      const p = pcaResult.projections[i];
      rows.push([s.layer, s.originalLayer, s.sublayer, s.position, s.topToken, p[0], p[1]]);
    }
    return rows;
  };

  return (
    <div style={{ marginTop: 24, paddingTop: 12, borderTop: "1px solid #1a2540" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8, flexWrap: "wrap" }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>Hidden-State PCA</h3>
        {!computed ? (
          <button
            onClick={() => setComputed(true)}
            style={{ fontSize: 11, padding: "2px 8px" }}
            title="Runs PCA client-side over every (layer, position) hidden vector in this result. O(N·D) per iteration; ~40 iters."
          >compute</button>
        ) : (
          <>
            <label style={{ fontSize: 11, color: "#8888aa", display: "flex", alignItems: "center", gap: 4 }}>
              color by
              <select
                value={colorMode}
                onChange={(e) => setColorMode(e.target.value as ColorMode)}
                style={{ fontSize: 11, padding: "1px 4px", background: "#0f1626", color: "#e0e0f0", border: "1px solid #1a2540", borderRadius: 3 }}
              >
                <option value="layer">layer</option>
                <option value="position">position</option>
              </select>
            </label>
            {selectedPos !== null && (
              <button
                onClick={() => setSelectedPos(null)}
                style={{ fontSize: 10, padding: "1px 6px" }}
                title="Clear position trajectory"
              >clear trace (pos {selectedPos})</button>
            )}
            <button
              onClick={() => { setComputed(false); setSelectedPos(null); }}
              style={{ fontSize: 10, padding: "1px 6px" }}
              title="Free the decoded samples"
            >close</button>
            <div style={{ marginLeft: "auto" }}>
              <ExportButtons
                filenameBase={`hidden-pca_${result.sessionName}`}
                getSVG={() => svgRef.current}
                getCSVRows={csvRows}
              />
            </div>
          </>
        )}
      </div>
      {computed && samples && samples.length < 2 && (
        <p style={{ fontSize: 11, color: "#f0ad4e" }}>
          Need ≥2 hidden-state samples to run PCA. (If this result was imported or reloaded, hidden states may have been stripped — re-run the probe.)
        </p>
      )}
      {computed && (!samples || samples.length === 0) && (
        <p style={{ fontSize: 11, color: "#f0ad4e" }}>No hidden states available (may have been stripped during persist).</p>
      )}
      {computed && pcaResult && (
        <p style={{ fontSize: 10, color: "#667", margin: "4px 0" }}>
          {samples?.length} samples × {samples?.[0]?.vec.length} dims · click a point to trace its position across layers
        </p>
      )}
      {computed && <svg ref={svgRef} />}
      {tooltip && (
        <div style={{
          position: "fixed", left: tooltip.x, top: tooltip.y,
          background: "#16213e", border: "1px solid #1a5276", borderRadius: 4,
          padding: "6px 10px", fontFamily: "monospace", fontSize: 11,
          whiteSpace: "pre", pointerEvents: "none", zIndex: 100,
        }}>
          {tooltip.content}
        </div>
      )}
    </div>
  );
}
