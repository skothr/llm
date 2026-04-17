import { useRef, useEffect, useState, useMemo } from "react";
import * as d3 from "d3";
import { decodeHiddenState } from "../../utils/hiddenState";
import type { LogitLensData, ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

const ROW_HEIGHT = 10;

export function DimVsLayerHeatmap({ result }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [position, setPosition] = useState<number | "last">("last");
  const [hover, setHover] = useState<{
    layer: string;
    dim: number;
    value: number;
    x: number;
    y: number;
  } | null>(null);

  const dataMessages = useMemo(
    () => result.data.filter((m): m is LogitLensData => m.type === "data" && "predictions" in m),
    [result.data]
  );

  const hiddenInfo = useMemo(() => {
    for (const m of dataMessages) {
      if (m.hidden_state) {
        return { seqLen: m.hidden_state.shape[0], hiddenSize: m.hidden_state.shape[1] };
      }
    }
    return null;
  }, [dataMessages]);

  const resolvedPos = useMemo(() => {
    if (!hiddenInfo) return 0;
    return position === "last" ? hiddenInfo.seqLen - 1 : position;
  }, [position, hiddenInfo]);

  const rows = useMemo(() => {
    if (!hiddenInfo) return [];
    const out: { label: string; data: Float32Array }[] = [];
    for (const msg of dataMessages) {
      if (!msg.hidden_state) continue;
      const full = decodeHiddenState(msg.hidden_state);
      const offset = resolvedPos * hiddenInfo.hiddenSize;
      out.push({
        label: `L${msg.layer}.${msg.sublayer}`,
        data: full.subarray(offset, offset + hiddenInfo.hiddenSize),
      });
    }
    return out;
  }, [dataMessages, resolvedPos, hiddenInfo]);

  const symmetricMax = useMemo(() => {
    let m = 0;
    for (const row of rows) {
      for (let i = 0; i < row.data.length; i++) {
        const a = Math.abs(row.data[i]);
        if (a > m) m = a;
      }
    }
    return m;
  }, [rows]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows.length === 0 || !hiddenInfo) return;
    const width = hiddenInfo.hiddenSize;
    const height = rows.length * ROW_HEIGHT;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const interp = d3.interpolateRdBu;
    const imgData = ctx.createImageData(width, height);
    const safeMax = symmetricMax > 0 ? symmetricMax : 1;

    for (let r = 0; r < rows.length; r++) {
      const row = rows[r];
      for (let d = 0; d < row.data.length; d++) {
        const v = row.data[d];
        // Invert so red = positive, blue = negative (same convention as the bar strip).
        const t = 1 - (v / safeMax + 1) / 2;
        const color = d3.color(interp(Math.max(0, Math.min(1, t))));
        if (!color) continue;
        const rgb = color.rgb();
        for (let dy = 0; dy < ROW_HEIGHT; dy++) {
          const y = r * ROW_HEIGHT + dy;
          const idx = (y * width + d) * 4;
          imgData.data[idx] = rgb.r;
          imgData.data[idx + 1] = rgb.g;
          imgData.data[idx + 2] = rgb.b;
          imgData.data[idx + 3] = 255;
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [rows, hiddenInfo, symmetricMax]);

  if (!hiddenInfo || rows.length === 0) return null;

  const { seqLen, hiddenSize } = hiddenInfo;

  return (
    <div style={{ marginTop: 16, borderTop: "1px solid #1a2540", paddingTop: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6, flexWrap: "wrap" }}>
        <h4 style={{ fontSize: 12, color: "#a0a0c0", margin: 0 }}>
          Hidden state evolution (layer × dim)
        </h4>
        <label style={{ fontSize: 11, color: "#8888aa", display: "flex", alignItems: "center", gap: 6 }}>
          Position:
          <select
            value={position === "last" ? "last" : String(position)}
            onChange={(e) => {
              const val = e.target.value;
              setPosition(val === "last" ? "last" : parseInt(val, 10));
            }}
            style={{
              background: "#0f1626",
              color: "#e0e0f0",
              border: "1px solid #1a5276",
              borderRadius: 3,
              padding: "2px 6px",
              fontSize: 11,
            }}
          >
            <option value="last">last</option>
            {Array.from({ length: seqLen }, (_, i) => (
              <option key={i} value={i}>{i}</option>
            ))}
          </select>
        </label>
        <span style={{ fontSize: 11, color: "#888", fontFamily: "monospace" }}>
          {rows.length} rows × {hiddenSize} dims · |v|≤{symmetricMax.toFixed(3)} · red=+, blue=−
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "flex-start", position: "relative" }}>
        <div style={{
          display: "flex", flexDirection: "column",
          paddingRight: 6, paddingTop: 0,
          fontSize: 9, fontFamily: "monospace", color: "#8888aa",
          userSelect: "none",
        }}>
          {rows.map((row, i) => (
            <div key={i} style={{ height: ROW_HEIGHT, lineHeight: `${ROW_HEIGHT}px`, textAlign: "right" }}>
              {row.label}
            </div>
          ))}
        </div>
        <div style={{ overflowX: "auto", maxWidth: "100%" }}>
          <canvas
            ref={canvasRef}
            style={{
              imageRendering: "pixelated",
              border: "1px solid #1a5276",
              borderRadius: 2,
              display: "block",
              cursor: "crosshair",
            }}
            onMouseMove={(e) => {
              const canvas = canvasRef.current;
              if (!canvas) return;
              const rect = canvas.getBoundingClientRect();
              const scaleX = canvas.width / rect.width;
              const scaleY = canvas.height / rect.height;
              const x = (e.clientX - rect.left) * scaleX;
              const y = (e.clientY - rect.top) * scaleY;
              const dim = Math.floor(x);
              const rowIdx = Math.floor(y / ROW_HEIGHT);
              if (rowIdx < 0 || rowIdx >= rows.length) return;
              const row = rows[rowIdx];
              if (dim < 0 || dim >= row.data.length) return;
              setHover({
                layer: row.label,
                dim,
                value: row.data[dim],
                x: e.pageX + 10,
                y: e.pageY + 10,
              });
            }}
            onMouseLeave={() => setHover(null)}
          />
        </div>
        {hover && (
          <div
            style={{
              position: "fixed",
              left: hover.x,
              top: hover.y,
              background: "#16213e",
              border: "1px solid #1a5276",
              borderRadius: 4,
              padding: "4px 8px",
              fontFamily: "monospace",
              fontSize: 11,
              whiteSpace: "pre",
              pointerEvents: "none",
              zIndex: 100,
              color: "#e0e0f0",
            }}
          >
            {`${hover.layer} dim ${hover.dim} = ${hover.value.toFixed(4)}`}
          </div>
        )}
      </div>
    </div>
  );
}
