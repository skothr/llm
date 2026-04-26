import { useMemo, useState } from "react";
import * as d3 from "d3";
import type { PatchingCellData, PatchingCompleteData } from "../../types/api";
import { computeCircuit, CircuitEdge } from "../../utils/circuitBFS";
import { computeCausalStory, storyNodeId, type StoryNodeId } from "../../utils/causalStory";
import { useResidualGrid } from "../../utils/useResidualGrid";
import { CausalStoryPanel } from "./CausalStoryPanel";

interface Props {
  cells: PatchingCellData[];
  complete?: PatchingCompleteData;
  sessionName?: string;
  prompt?: string;
}

type Node = { id: string; layer: number; unit: string; position: number };

export function CircuitPanel({ cells, complete, sessionName, prompt }: Props) {
  const edges: CircuitEdge[] = useMemo(
    () =>
      cells
        .filter((c) => c.writer_unit !== undefined && c.reader_unit !== undefined)
        .map((c) => ({
          writer_layer: c.writer_layer ?? 0,
          writer_unit: c.writer_unit ?? "",
          reader_layer: c.reader_layer ?? 0,
          reader_unit: c.reader_unit ?? "",
          position: c.position ?? 0,
          ap_recovery: c.ap_recovery ?? 0,
        })),
    [cells],
  );

  const positions = useMemo(() => {
    const s = new Set<number>();
    for (const e of edges) s.add(e.position);
    return Array.from(s).sort((a, b) => a - b);
  }, [edges]);

  const initialPos =
    complete?.summary?.measurement_position ?? positions[positions.length - 1] ?? 0;
  const initialTau = complete?.summary?.tau ?? 0.02;

  const [selectedPos, setSelectedPos] = useState<number>(initialPos);
  const [tau, setTau] = useState<number>(initialTau);
  const [showAll, setShowAll] = useState<boolean>(false);
  const [selectedNodeId, setSelectedNodeId] = useState<StoryNodeId | null>(null);
  const [playStep, setPlayStep] = useState<number | null>(null);

  const lensGrid = useResidualGrid(sessionName, prompt, 3);

  const edgesAtPos = useMemo(
    () => edges.filter((e) => e.position === selectedPos),
    [edges, selectedPos],
  );
  const bfs = useMemo(() => computeCircuit(edgesAtPos, tau), [edgesAtPos, tau]);

  // Story is computed once, used by both the Sankey (for playback dimming)
  // and the CausalStoryPanel below. Recomputed when bfs, position, or grid change.
  const inCircuitCells = useMemo(() => edgesAtPos.map((e, i) => ({
    writer_layer: e.writer_layer,
    writer_unit: e.writer_unit,
    reader_layer: e.reader_layer,
    reader_unit: e.reader_unit,
    position: e.position,
    in_circuit: bfs.in_circuit[i],
  })), [edgesAtPos, bfs.in_circuit]);
  const story = useMemo(
    () => computeCausalStory(inCircuitCells, lensGrid.data, selectedPos, 3),
    [inCircuitCells, lensGrid.data, selectedPos],
  );
  const revealedWriterIds = useMemo(() => {
    if (playStep === null) return null;  // null sentinel = "all visible"
    const set = new Set<StoryNodeId>();
    for (let i = 0; i < Math.min(playStep, story.nodes.length); i++) {
      const n = story.nodes[i];
      set.add(storyNodeId(n.layer, n.unit));
    }
    return set;
  }, [playStep, story.nodes]);

  const maxMag = useMemo(() => {
    let m = 0;
    for (const e of edgesAtPos) m = Math.max(m, Math.abs(e.ap_recovery));
    return m || 1;
  }, [edgesAtPos]);

  const writerNodes = useMemo((): Node[] => {
    const seen = new Set<string>();
    const out: Node[] = [];
    for (const e of edgesAtPos) {
      const id = `W:${e.writer_layer}:${e.writer_unit}`;
      if (!seen.has(id)) {
        seen.add(id);
        out.push({ id, layer: e.writer_layer, unit: e.writer_unit, position: e.position });
      }
    }
    return out.sort((a, b) => a.layer - b.layer || a.unit.localeCompare(b.unit));
  }, [edgesAtPos]);

  const readerNodes = useMemo((): Node[] => {
    const seen = new Set<string>();
    const out: Node[] = [];
    for (const e of edgesAtPos) {
      const id = `R:${e.reader_layer}:${e.reader_unit}`;
      if (!seen.has(id)) {
        seen.add(id);
        out.push({ id, layer: e.reader_layer, unit: e.reader_unit, position: e.position });
      }
    }
    return out.sort((a, b) => a.layer - b.layer || a.unit.localeCompare(b.unit));
  }, [edgesAtPos]);

  const SVG_W = 800;
  const SVG_H = Math.max(400, Math.max(writerNodes.length, readerNodes.length) * 18);
  const W_X = 120;
  const R_X = SVG_W - 120;

  const nodeY = (arr: Node[], idx: number) => {
    const pad = 20;
    const step = (SVG_H - 2 * pad) / Math.max(arr.length - 1, 1);
    return pad + idx * step;
  };
  const writerY = (id: string) => {
    const i = writerNodes.findIndex((n) => n.id === id);
    return nodeY(writerNodes, i);
  };
  const readerY = (id: string) => {
    const i = readerNodes.findIndex((n) => n.id === id);
    return nodeY(readerNodes, i);
  };

  const exportJson = () => {
    const payload = edgesAtPos
      .map((e, i) => ({ ...e, in_circuit: bfs.in_circuit[i] }))
      .filter((e) => e.in_circuit || showAll);
    navigator.clipboard?.writeText(JSON.stringify(payload, null, 2)).catch(() => undefined);
  };

  return (
    <div className="circuit-panel">
      <h3 style={{ fontSize: 13, color: "#a0a0c0", marginBottom: 8 }}>
        Circuit (ACDC) — {complete?.summary?.n_edges_in_circuit ?? bfs.n_edges_in_circuit} edges in circuit
        {complete?.summary?.n_steps != null && complete.summary.n_steps > 1 && (
          <span style={{ color: "#a0a0c0", fontSize: 13, fontWeight: "normal", marginLeft: 8 }}>
            {` — IG ${complete.summary.n_steps} steps`}
          </span>
        )}
      </h3>
      <div className="controls" style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
        <label style={{ fontSize: 12 }}>
          position:
          <select value={selectedPos} onChange={(e) => setSelectedPos(Number(e.target.value))}
            style={{ marginLeft: 4 }}>
            {positions.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <label style={{ flex: 1, minWidth: 250, fontSize: 12 }}>
          {"τ"} = {tau.toFixed(3)}
          <input
            type="range"
            min={0}
            max={maxMag}
            step={Math.max(maxMag / 200, 0.001)}
            value={tau}
            onChange={(e) => setTau(Number(e.target.value))}
            style={{ width: "100%", display: "block" }}
          />
        </label>
        <label style={{ fontSize: 12 }}>
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
          {" "}show out-of-circuit (dimmed)
        </label>
        <button onClick={exportJson} style={{ fontSize: 12 }}>copy JSON</button>
      </div>

      <div className="stats" style={{ margin: "8px 0", color: "#aaa", fontSize: 12 }}>
        Edges in circuit: <b>{bfs.n_edges_in_circuit}</b> of {edgesAtPos.length} at{" "}
        {"τ"}={tau.toFixed(3)}. Nodes: <b>{bfs.n_nodes_in_circuit}</b>.
      </div>

      <svg width={SVG_W} height={SVG_H} style={{ background: "#0e0e12", borderRadius: 4 }}>
        {edgesAtPos.map((e, i) => {
          const isIn = bfs.in_circuit[i];
          if (!isIn && !showAll) return null;
          const wId = `W:${e.writer_layer}:${e.writer_unit}`;
          const rId = `R:${e.reader_layer}:${e.reader_unit}`;
          const y1 = writerY(wId);
          const y2 = readerY(rId);
          const p = d3.path();
          const cx1 = W_X + (R_X - W_X) * 0.5;
          const cx2 = W_X + (R_X - W_X) * 0.5;
          p.moveTo(W_X, y1);
          p.bezierCurveTo(cx1, y1, cx2, y2, R_X, y2);
          const color = e.ap_recovery >= 0 ? "#4caf50" : "#c62828";
          const stroke = Math.max(1, (Math.abs(e.ap_recovery) / maxMag) * 6);
          const writerStoryId = storyNodeId(e.writer_layer, e.writer_unit);
          const writerSelected = selectedNodeId === writerStoryId;
          const dimmed = selectedNodeId !== null && !writerSelected;
          const baseOpacity = isIn ? 0.7 : 0.15;
          const playHidden = revealedWriterIds !== null && !revealedWriterIds.has(writerStoryId);
          const finalOpacity = playHidden
            ? 0
            : (dimmed ? baseOpacity * 0.3 : baseOpacity);
          return (
            <path
              key={i}
              d={p.toString()}
              stroke={writerSelected ? "#fff" : color}
              strokeOpacity={finalOpacity}
              strokeWidth={writerSelected ? stroke + 2 : stroke}
              fill="none"
              style={{ cursor: "pointer", transition: "stroke-opacity 400ms ease-in" }}
              onClick={() => setSelectedNodeId(writerSelected ? null : writerStoryId)}
            />
          );
        })}
        {writerNodes.map((n) => {
          const y = writerY(n.id);
          const sId = storyNodeId(n.layer, n.unit);
          const selected = selectedNodeId === sId;
          const playHidden = revealedWriterIds !== null && !revealedWriterIds.has(sId);
          return (
            <g key={n.id}
               style={{ cursor: "pointer", opacity: playHidden ? 0 : 1, transition: "opacity 400ms ease-in" }}
               onClick={() => setSelectedNodeId(selected ? null : sId)}>
              <circle
                cx={W_X} cy={y}
                r={selected ? 6 : 4}
                fill="#8abaff"
                stroke={selected ? "#fff" : "none"}
                strokeWidth={selected ? 2 : 0}
                data-testid={`circuit-writer-${sId}`}
              />
              <text x={W_X - 8} y={y + 4}
                    fill={selected ? "#fff" : "#aaa"}
                    fontWeight={selected ? "bold" : "normal"}
                    fontSize={11} textAnchor="end">
                L{n.layer}.{n.unit}
              </text>
            </g>
          );
        })}
        {readerNodes.map((n) => {
          const y = readerY(n.id);
          return (
            <g key={n.id}>
              <circle cx={R_X} cy={y} r={4} fill="#ffca8a" />
              <text x={R_X + 8} y={y + 4} fill="#aaa" fontSize={11}>
                L{n.layer}.{n.unit}
              </text>
            </g>
          );
        })}
      </svg>
      {sessionName !== undefined && prompt !== undefined && (
        <CausalStoryPanel
          story={story}
          promptToken={lensGrid.data?.prompt_tokens?.[selectedPos]}
          selectedNodeId={selectedNodeId}
          onSelectNode={setSelectedNodeId}
          playStep={playStep}
          onPlayStepChange={setPlayStep}
        />
      )}
    </div>
  );
}
