import { useMemo, useState } from "react";
import type { PatchingCellData, PatchingCompleteData } from "../../types/api";

interface Props {
  cells: PatchingCellData[];
  complete?: PatchingCompleteData;
}

type SortKey = "ap_recovery" | "layer" | "neuron";

function apColor(ap: number): string {
  // Piecewise PiYG mapped to [-0.5, 1.0] — same as Phase 3.5+.
  const clamped = Math.max(-0.5, Math.min(1.0, ap));
  if (clamped >= 0) {
    const t = clamped;
    const g = Math.floor(200 * t + 30);
    return `rgb(30, ${g}, 60)`;
  } else {
    const t = -clamped / 0.5;
    const r = Math.floor(180 * t + 40);
    return `rgb(${r}, 40, 80)`;
  }
}

export function PerNeuronPatchingPanel({ cells, complete }: Props) {
  const neuronCells = useMemo(
    () => cells.filter((c) => c.neuron !== undefined && c.layer !== undefined),
    [cells],
  );

  const positions = useMemo(() => {
    const s = new Set<number>();
    for (const c of neuronCells) if (c.position !== undefined) s.add(c.position);
    return Array.from(s).sort((a, b) => a - b);
  }, [neuronCells]);

  const layers = useMemo(() => {
    const s = new Set<number>();
    for (const c of neuronCells) if (c.layer !== undefined) s.add(c.layer);
    return Array.from(s).sort((a, b) => a - b);
  }, [neuronCells]);

  const initialPos =
    complete?.summary?.measurement_position ?? positions[positions.length - 1] ?? 0;

  const [selectedPos, setSelectedPos] = useState<number | "all">(initialPos);
  const [selectedLayer, setSelectedLayer] = useState<number | "all">("all");
  const [neuronSearch, setNeuronSearch] = useState<string>("");
  const [sortKey, setSortKey] = useState<SortKey>("ap_recovery");
  const [sortDesc, setSortDesc] = useState<boolean>(true);

  const visible = useMemo(() => {
    let rows = neuronCells;
    if (selectedPos !== "all") rows = rows.filter((c) => c.position === selectedPos);
    if (selectedLayer !== "all") rows = rows.filter((c) => c.layer === selectedLayer);
    if (neuronSearch.trim() !== "") {
      const q = neuronSearch.trim();
      const qNum = Number(q);
      if (!Number.isNaN(qNum)) {
        rows = rows.filter((c) => c.neuron === qNum);
      }
    }
    const cmp = (a: PatchingCellData, b: PatchingCellData): number => {
      const aa = (a[sortKey] ?? 0) as number;
      const bb = (b[sortKey] ?? 0) as number;
      if (sortKey === "ap_recovery") {
        return Math.abs(bb) - Math.abs(aa);
      }
      return aa - bb;
    };
    const sorted = [...rows].sort(cmp);
    return sortDesc ? sorted : sorted.reverse();
  }, [neuronCells, selectedPos, selectedLayer, neuronSearch, sortKey, sortDesc]);

  const stats = useMemo(() => {
    if (visible.length === 0) return { min: 0, max: 0, mean: 0 };
    let min = Infinity, max = -Infinity, sum = 0;
    for (const c of visible) {
      const v = c.ap_recovery ?? 0;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    return { min, max, mean: sum / visible.length };
  }, [visible]);

  const exportTSV = () => {
    const header = "layer\tneuron\tposition\tap_recovery";
    const body = visible
      .map((c) => `${c.layer}\t${c.neuron}\t${c.position}\t${(c.ap_recovery ?? 0).toFixed(6)}`)
      .join("\n");
    navigator.clipboard?.writeText(`${header}\n${body}`).catch(() => undefined);
  };

  return (
    <div className="per-neuron-panel">
      <h3>Per-Neuron FFN Attribution</h3>
      <div className="controls" style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center", marginBottom: 8 }}>
        <label>
          position:
          <select
            value={selectedPos}
            onChange={(e) => {
              const v = e.target.value;
              setSelectedPos(v === "all" ? "all" : Number(v));
            }}
          >
            <option value="all">all</option>
            {positions.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <label>
          layer:
          <select
            value={selectedLayer}
            onChange={(e) => {
              const v = e.target.value;
              setSelectedLayer(v === "all" ? "all" : Number(v));
            }}
          >
            <option value="all">all</option>
            {layers.map((L) => (
              <option key={L} value={L}>
                L{L}
              </option>
            ))}
          </select>
        </label>
        <label>
          neuron#:
          <input
            type="text"
            placeholder="e.g. 1234"
            value={neuronSearch}
            onChange={(e) => setNeuronSearch(e.target.value)}
            style={{ width: 80 }}
          />
        </label>
        <label>
          sort:
          <select value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}>
            <option value="ap_recovery">|ap_recovery|</option>
            <option value="layer">layer</option>
            <option value="neuron">neuron</option>
          </select>
        </label>
        <button onClick={() => setSortDesc((d) => !d)}>{sortDesc ? "↓" : "↑"}</button>
        <button onClick={exportTSV}>copy TSV</button>
      </div>

      <div className="stats" style={{ color: "#aaa", marginBottom: 8 }}>
        Showing <b>{visible.length}</b> of {neuronCells.length} cells. min={stats.min.toFixed(4)}, max={stats.max.toFixed(4)}, mean={stats.mean.toFixed(4)}.
      </div>

      <div style={{ maxHeight: 600, overflowY: "auto", border: "1px solid #333" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "monospace", fontSize: 12 }}>
          <thead style={{ position: "sticky", top: 0, background: "#1a1a1a" }}>
            <tr>
              <th style={{ textAlign: "left", padding: 4 }}>#</th>
              <th style={{ textAlign: "left", padding: 4 }}>layer</th>
              <th style={{ textAlign: "left", padding: 4 }}>neuron</th>
              <th style={{ textAlign: "left", padding: 4 }}>pos</th>
              <th style={{ textAlign: "right", padding: 4 }}>ap_recovery</th>
            </tr>
          </thead>
          <tbody>
            {visible.slice(0, 200).map((c, i) => (
              <tr key={`${c.layer}-${c.neuron}-${c.position}`} style={{ background: apColor(c.ap_recovery ?? 0) }}>
                <td style={{ padding: 4 }}>{i + 1}</td>
                <td style={{ padding: 4 }}>L{c.layer}</td>
                <td style={{ padding: 4 }}>n{c.neuron}</td>
                <td style={{ padding: 4 }}>{c.position}</td>
                <td style={{ padding: 4, textAlign: "right" }}>{(c.ap_recovery ?? 0).toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
