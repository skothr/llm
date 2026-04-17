import { useRef, useEffect, useMemo } from "react";
import * as d3 from "d3";

interface Props {
  data: Float32Array;
  numHeads: number;
  headDim: number;
  label?: string;
  cellSize?: number;
  /**
   * If provided, colors are normalized to this symmetric range. Otherwise,
   * the component normalizes per-vector to ±max(|v|), which reveals local
   * structure but makes cross-vector comparison misleading.
   */
  symmetricMax?: number;
}

export function HiddenStateHeatmap({
  data,
  numHeads,
  headDim,
  label,
  cellSize = 4,
  symmetricMax,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const stats = useMemo(() => {
    let absMax = 0;
    let mean = 0;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      const a = Math.abs(v);
      if (a > absMax) absMax = a;
      if (v < min) min = v;
      if (v > max) max = v;
      mean += v;
    }
    mean /= Math.max(1, data.length);
    return { absMax, mean, min, max };
  }, [data]);

  const effectiveMax = symmetricMax ?? stats.absMax;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = headDim * cellSize;
    canvas.height = numHeads * cellSize;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const interp = d3.interpolateRdBu;
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    const safeMax = effectiveMax > 0 ? effectiveMax : 1;

    for (let h = 0; h < numHeads; h++) {
      for (let d = 0; d < headDim; d++) {
        const v = data[h * headDim + d];
        // Map [-safeMax, +safeMax] to [0, 1], then invert because interpolateRdBu
        // goes red(low) → white(mid) → blue(high), which reads more intuitively
        // for "positive = hot". We reverse so red is positive.
        const t = 1 - (v / safeMax + 1) / 2;
        const color = d3.color(interp(Math.max(0, Math.min(1, t))));
        if (!color) continue;
        const rgb = color.rgb();
        // Paint a cellSize×cellSize block.
        const px0 = d * cellSize;
        const py0 = h * cellSize;
        for (let dy = 0; dy < cellSize; dy++) {
          for (let dx = 0; dx < cellSize; dx++) {
            const idx = ((py0 + dy) * canvas.width + (px0 + dx)) * 4;
            imgData.data[idx] = rgb.r;
            imgData.data[idx + 1] = rgb.g;
            imgData.data[idx + 2] = rgb.b;
            imgData.data[idx + 3] = 255;
          }
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [data, numHeads, headDim, cellSize, effectiveMax]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {label && (
        <div style={{ fontSize: 10, color: "#8888aa", fontFamily: "monospace" }}>
          {label}
        </div>
      )}
      <canvas
        ref={canvasRef}
        style={{
          imageRendering: "pixelated",
          border: "1px solid #1a5276",
          borderRadius: 2,
        }}
      />
      <div style={{ fontSize: 9, color: "#888", fontFamily: "monospace", whiteSpace: "nowrap" }}>
        {numHeads}×{headDim} · min={stats.min.toFixed(3)} max={stats.max.toFixed(3)} μ={stats.mean.toFixed(3)} |v|≤{effectiveMax.toFixed(3)}
      </div>
    </div>
  );
}
