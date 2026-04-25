import { useMemo, useState } from "react";
import * as d3 from "d3";
import type {
  ProbeResult,
  PatchingBaselinesData,
  PatchingCompleteData,
  EdgeCellData,
} from "../../types/api";

interface Props {
  result: ProbeResult;
}

type TabView = "sankey" | "matrix" | "list";

const WRITER_UNIT_ORDER = (u: string): number => {
  if (u === "embed") return -1;
  if (u.startsWith("attn.h")) return parseInt(u.slice(6), 10);
  if (u === "ffn") return 10000;
  return 9999;
};

const READER_UNIT_ORDER = (u: string): number => {
  if (u === "attn_in") return 0;
  if (u === "ffn_in") return 1;
  if (u === "logits") return 2;
  return 3;
};

function writerLabel(layer: number, unit: string): string {
  if (unit === "embed") return "embed";
  return `L${layer}.${unit}`;
}

function readerLabel(layer: number, unit: string): string {
  if (unit === "logits") return "logits";
  return `L${layer}.${unit}`;
}

const colorScale = d3.scaleSequential(d3.interpolatePiYG).domain([-0.5, 1.0]);

function SankeyView({ cells }: { cells: EdgeCellData[] }) {
  if (cells.length === 0) return <p style={{ color: "#666" }}>No edges at this position.</p>;

  const writerKeys = Array.from(
    new Map(cells.map((c) => [`${c.writer_layer}:${c.writer_unit}`, { layer: c.writer_layer, unit: c.writer_unit }])).values()
  ).sort((a, b) => a.layer - b.layer || WRITER_UNIT_ORDER(a.unit) - WRITER_UNIT_ORDER(b.unit));

  const readerKeys = Array.from(
    new Map(cells.map((c) => [`${c.reader_layer}:${c.reader_unit}`, { layer: c.reader_layer, unit: c.reader_unit }])).values()
  ).sort((a, b) => a.layer - b.layer || READER_UNIT_ORDER(a.unit) - READER_UNIT_ORDER(b.unit));

  const margin = { top: 20, right: 120, bottom: 20, left: 120 };
  const nodeH = 18;
  const nodeGap = 4;
  const totalH = Math.max(writerKeys.length, readerKeys.length) * (nodeH + nodeGap);
  const svgW = 520;
  const svgH = totalH + margin.top + margin.bottom;
  const colX = { writer: 0, reader: svgW - margin.left - margin.right };

  const writerY = (i: number) => i * (nodeH + nodeGap) + nodeH / 2;
  const readerY = (i: number) => i * (nodeH + nodeGap) + nodeH / 2;

  const writerIdx = new Map(writerKeys.map((w, i) => [`${w.layer}:${w.unit}`, i]));
  const readerIdx = new Map(readerKeys.map((r, i) => [`${r.layer}:${r.unit}`, i]));

  const maxAbs = Math.max(...cells.map((c) => Math.abs(c.ap_recovery)), 0.001);

  return (
    <svg width={svgW} height={svgH} style={{ overflow: "visible" }}>
      <g transform={`translate(${margin.left},${margin.top})`}>
        {cells.map((c, idx) => {
          const wi = writerIdx.get(`${c.writer_layer}:${c.writer_unit}`) ?? 0;
          const ri = readerIdx.get(`${c.reader_layer}:${c.reader_unit}`) ?? 0;
          const x0 = colX.writer;
          const x1 = colX.reader;
          const y0 = writerY(wi);
          const y1 = readerY(ri);
          const cx = (x0 + x1) / 2;
          const strokeW = Math.max(0.5, (Math.abs(c.ap_recovery) / maxAbs) * 4);
          const col = colorScale(c.ap_recovery);
          return (
            <path
              key={idx}
              d={`M ${x0},${y0} C ${cx},${y0} ${cx},${y1} ${x1},${y1}`}
              fill="none"
              stroke={col}
              strokeWidth={strokeW}
              strokeOpacity={0.7}
            />
          );
        })}
        {writerKeys.map((w, i) => (
          <text key={i} x={colX.writer - 4} y={writerY(i)} textAnchor="end"
            dominantBaseline="middle" fontSize={10} fill="#a0a0c0">
            {writerLabel(w.layer, w.unit)}
          </text>
        ))}
        {readerKeys.map((r, i) => (
          <text key={i} x={colX.reader + 4} y={readerY(i)} textAnchor="start"
            dominantBaseline="middle" fontSize={10} fill="#a0a0c0">
            {readerLabel(r.layer, r.unit)}
          </text>
        ))}
      </g>
    </svg>
  );
}

function MatrixView({ cells }: { cells: EdgeCellData[] }) {
  if (cells.length === 0) return <p style={{ color: "#666" }}>No edges at this position.</p>;

  const writerKeys = Array.from(
    new Map(cells.map((c) => [`${c.writer_layer}:${c.writer_unit}`, { layer: c.writer_layer, unit: c.writer_unit }])).values()
  ).sort((a, b) => a.layer - b.layer || WRITER_UNIT_ORDER(a.unit) - WRITER_UNIT_ORDER(b.unit));

  const readerKeys = Array.from(
    new Map(cells.map((c) => [`${c.reader_layer}:${c.reader_unit}`, { layer: c.reader_layer, unit: c.reader_unit }])).values()
  ).sort((a, b) => a.layer - b.layer || READER_UNIT_ORDER(a.unit) - READER_UNIT_ORDER(b.unit));

  const cellMap = new Map(cells.map((c) => [`${c.writer_layer}:${c.writer_unit}|${c.reader_layer}:${c.reader_unit}`, c.ap_recovery]));

  const cellW = Math.max(18, Math.min(40, 600 / readerKeys.length));
  const cellH = 16;
  const labelW = 80;
  const labelH = 60;

  const svgW = labelW + readerKeys.length * cellW + 8;
  const svgH = labelH + writerKeys.length * cellH + 8;

  return (
    <svg width={svgW} height={svgH} style={{ overflow: "visible" }}>
      {readerKeys.map((r, ci) => (
        <text
          key={ci}
          x={labelW + ci * cellW + cellW / 2}
          y={labelH - 4}
          textAnchor="end"
          fontSize={9}
          fill="#a0a0c0"
          transform={`rotate(-45, ${labelW + ci * cellW + cellW / 2}, ${labelH - 4})`}
        >
          {readerLabel(r.layer, r.unit)}
        </text>
      ))}
      {writerKeys.map((w, ri) => (
        <text key={ri} x={labelW - 4} y={labelH + ri * cellH + cellH / 2}
          textAnchor="end" dominantBaseline="middle" fontSize={9} fill="#a0a0c0">
          {writerLabel(w.layer, w.unit)}
        </text>
      ))}
      {writerKeys.map((w, ri) =>
        readerKeys.map((r, ci) => {
          const val = cellMap.get(`${w.layer}:${w.unit}|${r.layer}:${r.unit}`);
          const fill = val !== undefined ? colorScale(val) : "#222";
          return (
            <rect
              key={`${ri}-${ci}`}
              x={labelW + ci * cellW}
              y={labelH + ri * cellH}
              width={cellW - 1}
              height={cellH - 1}
              fill={fill}
            >
              {val !== undefined && <title>{`${writerLabel(w.layer, w.unit)} → ${readerLabel(r.layer, r.unit)}: ${val.toFixed(3)}`}</title>}
            </rect>
          );
        })
      )}
    </svg>
  );
}

function ListView({ cells }: { cells: EdgeCellData[] }) {
  const sorted = useMemo(
    () => [...cells].sort((a, b) => Math.abs(b.ap_recovery) - Math.abs(a.ap_recovery)),
    [cells]
  );

  const handleCopy = () => {
    const header = "Rank\tWriter\tReader\tPosition\tAP Recovery";
    const rows = sorted.map((c, i) =>
      `${i + 1}\t${writerLabel(c.writer_layer, c.writer_unit)}\t${readerLabel(c.reader_layer, c.reader_unit)}\t${c.position}\t${c.ap_recovery.toFixed(4)}`
    );
    navigator.clipboard.writeText([header, ...rows].join("\n")).catch(() => { /* ignore */ });
  };

  if (sorted.length === 0) return <p style={{ color: "#666" }}>No edges at this position.</p>;

  return (
    <div>
      <button onClick={handleCopy} style={{ fontSize: 11, marginBottom: 6 }}>Copy as TSV</button>
      <div style={{ overflowX: "auto", maxHeight: 400, overflowY: "auto" }}>
        <table style={{ fontSize: 11, fontFamily: "monospace", borderCollapse: "collapse", width: "100%" }}>
          <thead>
            <tr>
              {["#", "Writer", "Reader", "Pos", "AP Recovery"].map((h) => (
                <th key={h} style={{ textAlign: "left", color: "#8888aa", padding: "2px 8px", borderBottom: "1px solid #333" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((c, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? "transparent" : "#0a0a14" }}>
                <td style={{ padding: "1px 8px", color: "#555" }}>{i + 1}</td>
                <td style={{ padding: "1px 8px" }}>{writerLabel(c.writer_layer, c.writer_unit)}</td>
                <td style={{ padding: "1px 8px" }}>{readerLabel(c.reader_layer, c.reader_unit)}</td>
                <td style={{ padding: "1px 8px" }}>{c.position}</td>
                <td style={{ padding: "1px 8px", color: colorScale(c.ap_recovery) }}>
                  {c.ap_recovery.toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function EdgeAttributionPanel({ result }: Props) {
  const completeFrame = useMemo(
    () => result.data.find((m): m is PatchingCompleteData => m.type === "complete"),
    [result.data]
  );
  const baselines = useMemo(
    () => result.data.find((m): m is PatchingBaselinesData => m.type === "baselines"),
    [result.data]
  );

  const measPos: number = completeFrame?.summary.measurement_position ?? 0;
  const nEdges: number | undefined = completeFrame?.summary.n_edges;
  const promptTokens: string[] = baselines?.prompt_tokens_clean ?? [];

  const allEdgeCells = useMemo(
    () => result.data.filter((m): m is EdgeCellData =>
      m.type === "data" && "writer_unit" in m
    ),
    [result.data]
  );

  const positions = useMemo(
    () => Array.from(new Set(allEdgeCells.map((c) => c.position))).sort((a, b) => a - b),
    [allEdgeCells]
  );

  const [selectedPos, setSelectedPos] = useState<number>(measPos);
  const [tab, setTab] = useState<TabView>("sankey");

  const posCells = useMemo(
    () => allEdgeCells.filter((c) => c.position === selectedPos),
    [allEdgeCells, selectedPos]
  );

  const tabBtn = (t: TabView, label: string) => (
    <button
      onClick={() => setTab(t)}
      style={{
        fontSize: 12,
        background: tab === t ? "#1a5276" : "#0d1b2a",
        border: tab === t ? "1px solid #4a90d9" : "1px solid #333",
        padding: "3px 10px",
      }}
    >
      {label}
    </button>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
        Edge Attribution — {result.sessionName}
        {nEdges !== undefined && (
          <span style={{ color: "#667", fontSize: 11, marginLeft: 8 }}>
            ({nEdges.toLocaleString()} total edges)
          </span>
        )}
        {completeFrame?.summary?.n_steps != null && completeFrame.summary.n_steps > 1 && (
          <span style={{ color: "#a0a0c0", fontSize: 13, fontWeight: "normal", marginLeft: 8 }}>
            {` — IG ${completeFrame.summary.n_steps} steps`}
          </span>
        )}
      </h3>

      {positions.length > 0 && (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <label style={{ fontSize: 12, color: "#a0a0c0" }}>position:</label>
          <select
            value={selectedPos}
            onChange={(e) => setSelectedPos(Number(e.target.value))}
            style={{ fontSize: 12 }}
          >
            {positions.map((p) => (
              <option key={p} value={p}>
                {p}: {promptTokens[p] ?? `tok${p}`}
              </option>
            ))}
          </select>
        </div>
      )}

      <div style={{ display: "flex", gap: 4 }}>
        {tabBtn("sankey", "sankey")}
        {tabBtn("matrix", "matrix")}
        {tabBtn("list", "list")}
      </div>

      <div style={{ marginTop: 4 }}>
        {tab === "sankey" && <SankeyView cells={posCells} />}
        {tab === "matrix" && <MatrixView cells={posCells} />}
        {tab === "list" && <ListView cells={posCells} />}
      </div>
    </div>
  );
}
