import { useRef, useEffect, useMemo } from "react";

interface Props {
  data: Float32Array;
  label?: string;
  width?: number;
  height?: number;
  /**
   * If provided, the scale is pinned to ±symmetricMax. Used to make
   * side-by-side strips (e.g., A vs B) directly comparable.
   */
  symmetricMax?: number;
}

const POS_COLOR = "#4a90e2"; // blue for positive
const NEG_COLOR = "#e2524a"; // red for negative

export function HiddenStateBarStrip({
  data,
  label,
  width = 512,
  height = 80,
  symmetricMax,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const stats = useMemo(() => {
    let absMax = 0;
    let sum = 0;
    let min = Infinity;
    let max = -Infinity;
    // Track top-3 outliers by |v|
    const topAbs: { idx: number; v: number }[] = [];
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      const a = Math.abs(v);
      if (a > absMax) absMax = a;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      if (topAbs.length < 3) {
        topAbs.push({ idx: i, v });
        topAbs.sort((p, q) => Math.abs(p.v) - Math.abs(q.v));
      } else if (a > Math.abs(topAbs[0].v)) {
        topAbs[0] = { idx: i, v };
        topAbs.sort((p, q) => Math.abs(p.v) - Math.abs(q.v));
      }
    }
    topAbs.reverse();
    return { absMax, mean: sum / Math.max(1, data.length), min, max, topAbs };
  }, [data]);

  const effectiveMax = symmetricMax ?? stats.absMax;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#0a0f1e";
    ctx.fillRect(0, 0, width, height);

    const midY = height / 2;
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    const safeMax = effectiveMax > 0 ? effectiveMax : 1;
    const dimsPerCol = data.length / width;

    for (let col = 0; col < width; col++) {
      const start = Math.floor(col * dimsPerCol);
      const end = Math.max(start + 1, Math.floor((col + 1) * dimsPerCol));
      // Within each bin, keep the value with the largest magnitude — preserves
      // outliers, which are the thing that's actually interesting.
      let peak = 0;
      let peakAbs = -1;
      for (let i = start; i < end && i < data.length; i++) {
        const a = Math.abs(data[i]);
        if (a > peakAbs) {
          peakAbs = a;
          peak = data[i];
        }
      }
      if (peakAbs <= 0) continue;
      const barHeight = Math.min(midY, (peakAbs / safeMax) * midY);
      ctx.fillStyle = peak >= 0 ? POS_COLOR : NEG_COLOR;
      if (peak >= 0) {
        ctx.fillRect(col, midY - barHeight, 1, barHeight);
      } else {
        ctx.fillRect(col, midY, 1, barHeight);
      }
    }
  }, [data, width, height, effectiveMax]);

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
          display: "block",
        }}
      />
      <div style={{ fontSize: 9, color: "#888", fontFamily: "monospace", whiteSpace: "nowrap" }}>
        n={data.length} · min={stats.min.toFixed(3)} max={stats.max.toFixed(3)} μ={stats.mean.toFixed(3)} |v|≤{effectiveMax.toFixed(3)}
      </div>
      {stats.topAbs.length > 0 && (
        <div style={{ fontSize: 9, color: "#888", fontFamily: "monospace", whiteSpace: "nowrap" }}>
          top |v|: {stats.topAbs.map((t) => `[${t.idx}]=${t.v.toFixed(2)}`).join("  ")}
        </div>
      )}
    </div>
  );
}
