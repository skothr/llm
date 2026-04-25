import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import * as d3 from "d3";
import { ExportButtons } from "../ExportButtons";
import type {
  ProbeResult, PatchingBaselinesData, PatchingCellData, PatchingCompleteData,
} from "../../types/api";

interface Props {
  result: ProbeResult;
}

interface PinnedCell {
  cell: PatchingCellData;
  x: number;
  y: number;
}

export function PerHeadPatchingHeatmap({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedPos, setSelectedPos] = useState<number>(0);
  const [pinned, setPinned] = useState<PinnedCell | null>(null);

  const [decodeState, setDecodeState] = useState<{
    loading: boolean;
    error: string | null;
    data: {
      top: Array<{ token: string; logit: number }>;
      bottom: Array<{ token: string; logit: number }>;
      sv_ratio: number;
    } | null;
  }>({ loading: false, error: null, data: null });

  useEffect(() => {
    if (pinned === null) {
      setDecodeState({ loading: false, error: null, data: null });
      return;
    }
    const unit = pinned.cell.unit;
    if (typeof unit !== "string" || !unit.startsWith("attn.h")) {
      setDecodeState({ loading: false, error: null, data: null });
      return;
    }
    const headIdx = parseInt(unit.slice("attn.h".length), 10);
    if (Number.isNaN(headIdx)) {
      setDecodeState({ loading: false, error: null, data: null });
      return;
    }
    const layer = pinned.cell.layer;
    if (typeof layer !== "number") return;

    const ctrl = new AbortController();
    setDecodeState({ loading: true, error: null, data: null });
    fetch(`/api/sessions/${result.sessionName}/decode-head`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ layer, head: headIdx, top_k: 10 }),
      signal: ctrl.signal,
    })
      .then((r) =>
        r.ok ? r.json() : Promise.reject(new Error(`decode-head ${r.status}`)),
      )
      .then((body: {
        top_tokens: Array<{ token: string; logit: number }>;
        bottom_tokens: Array<{ token: string; logit: number }>;
        singular_value_ratio: number;
      }) => {
        setDecodeState({
          loading: false,
          error: null,
          data: {
            top: body.top_tokens,
            bottom: body.bottom_tokens,
            sv_ratio: body.singular_value_ratio,
          },
        });
      })
      .catch((err: Error) => {
        if (err.name !== "AbortError") {
          setDecodeState({ loading: false, error: err.message, data: null });
        }
      });
    return () => ctrl.abort();
  }, [pinned, result.sessionName]);

  const completeFrame = useMemo(
    () => result.data.find((m): m is PatchingCompleteData => m.type === "complete"),
    [result.data]
  );
  const n_heads: number = completeFrame?.summary.n_heads ?? 0;
  const measPos: number = completeFrame?.summary.measurement_position ?? 0;

  // Initialize selectedPos to measurement_position on mount.
  const [posInit, setPosInit] = useState(false);
  if (!posInit && measPos !== 0) {
    setSelectedPos(measPos);
    setPosInit(true);
  }

  const baselines = useMemo(
    () => result.data.find((m): m is PatchingBaselinesData => m.type === "baselines"),
    [result.data]
  );
  const promptTokens: string[] = baselines?.prompt_tokens_clean ?? [];

  const allCells = useMemo(
    () => result.data.filter((m): m is PatchingCellData =>
      m.type === "data" && "unit" in m
    ),
    [result.data]
  );

  // Cells at the selected position.
  const positionCells = useMemo(
    () => allCells.filter((c) => c.position === selectedPos),
    [allCells, selectedPos]
  );

  // Unique layer indices (sorted ascending).
  const layerIds = useMemo(
    () => Array.from(new Set(allCells.map((c) => c.layer))).sort((a, b) => a - b),
    [allCells]
  );

  // Column order: "ffn" first, then "attn.h0" … "attn.h{N-1}".
  const columnUnits: string[] = useMemo(() => {
    const units = ["ffn", ...Array.from({ length: n_heads }, (_, h) => `attn.h${h}`)];
    return units;
  }, [n_heads]);

  const colorScale = useMemo(
    () => d3.scaleSequential(d3.interpolatePiYG).domain([-0.5, 1.0]),
    []
  );

  useEffect(() => {
    if (!svgRef.current || positionCells.length === 0 || columnUnits.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 24, right: 16, bottom: 32, left: 48 };
    const cellW = Math.max(12, Math.min(28, 700 / columnUnits.length));
    const cellH = 18;
    const width = margin.left + columnUnits.length * cellW + margin.right;
    const height = margin.top + layerIds.length * cellH + margin.bottom;
    svg.attr("width", width).attr("height", height);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Row labels (layer indices).
    layerIds.forEach((L, rowIdx) => {
      g.append("text")
        .attr("x", -4).attr("y", rowIdx * cellH + cellH / 2)
        .attr("text-anchor", "end").attr("dominant-baseline", "middle")
        .attr("font-size", 9).attr("fill", "#8888aa")
        .text(`L${L}`);
    });

    // Column labels ("ffn", "h0", "h1", …).
    columnUnits.forEach((unit, colIdx) => {
      const label = unit === "ffn" ? "ffn" : unit.replace("attn.h", "h");
      g.append("text")
        .attr("x", colIdx * cellW + cellW / 2).attr("y", -6)
        .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", "#666")
        .text(label);
    });

    // Build lookup: (layer, unit) → ap_recovery.
    const cellMap = new Map<string, number>();
    for (const c of positionCells) {
      const u = c.unit ?? "";
      if (u) cellMap.set(`${c.layer}.${u}`, c.ap_recovery ?? 0);
    }

    // Draw cells.
    layerIds.forEach((L, rowIdx) => {
      columnUnits.forEach((unit, colIdx) => {
        const v = cellMap.get(`${L}.${unit}`);
        const fill = v != null ? colorScale(v) : "#1a1a2e";
        g.append("rect")
          .attr("x", colIdx * cellW).attr("y", rowIdx * cellH)
          .attr("width", cellW - 1).attr("height", cellH - 1)
          .attr("fill", fill).attr("rx", 2)
          .attr("data-layer", String(L))
          .attr("data-unit", unit)
          .style("cursor", v != null ? "pointer" : "default")
          .on("click", (event) => {
            if (v == null) return;
            const syntheticCell: PatchingCellData = {
              type: "data", layer: L, unit, position: selectedPos, ap_recovery: v,
            };
            setPinned({ cell: syntheticCell, x: event.pageX + 10, y: event.pageY + 10 });
          });
      });
    });

    // Legend bar.
    const legendY = layerIds.length * cellH + 12;
    const legendW = Math.min(180, columnUnits.length * cellW);
    const gradId = `ph-grad-${result.id}`;
    const defs = svg.append("defs");
    const grad = defs.append("linearGradient").attr("id", gradId).attr("x1", "0%").attr("x2", "100%");
    for (let i = 0; i <= 16; i++) {
      grad.append("stop")
        .attr("offset", `${(i / 16) * 100}%`)
        .attr("stop-color", d3.interpolatePiYG(i / 16));
    }
    g.append("rect")
      .attr("x", 0).attr("y", legendY).attr("width", legendW).attr("height", 7)
      .attr("fill", `url(#${gradId})`).attr("rx", 1);
    g.append("text")
      .attr("x", 0).attr("y", legendY + 16)
      .attr("font-size", 8).attr("fill", "#888").text("-0.5");
    g.append("text")
      .attr("x", legendW).attr("y", legendY + 16)
      .attr("text-anchor", "end").attr("font-size", 8).attr("fill", "#888").text("1.0");
    g.append("text")
      .attr("x", legendW / 2).attr("y", legendY - 2)
      .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", "#aaa")
      .text("AP recovery (PiYG)");
  }, [positionCells, layerIds, columnUnits, colorScale, selectedPos, result.id]);

  const csvRows = useCallback((): (string | number)[][] => {
    const header = ["layer", "unit", "position", "ap_recovery"];
    const rows: (string | number)[][] = [header];
    for (const c of allCells) {
      rows.push([c.layer, c.unit ?? "", c.position, c.ap_recovery ?? ""]);
    }
    return rows;
  }, [allCells]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
          Per-head Attribution Patching
          {" — "}{result.sessionName}
          {" — \""}{result.prompt.slice(0, 40)}{"\""}
        </h3>
        {completeFrame?.summary?.n_steps != null && completeFrame.summary.n_steps > 1 && (
          <span style={{ color: "#a0a0c0", fontSize: 13, fontWeight: "normal" }}>
            {` — IG ${completeFrame.summary.n_steps} steps`}
          </span>
        )}
        <label style={{ fontSize: 12, color: "#8888aa", display: "flex", alignItems: "center", gap: 6 }}>
          position:
          <select
            value={selectedPos}
            onChange={(e) => setSelectedPos(Number(e.target.value))}
            style={{
              background: "#0f1626", color: "#e0e0f0", border: "1px solid #1a5276",
              borderRadius: 3, padding: "2px 6px", fontSize: 11,
            }}
          >
            {promptTokens.map((tok, i) => (
              <option key={i} value={i}>{i}: {tok}</option>
            ))}
          </select>
        </label>
        <div style={{ marginLeft: "auto" }}>
          <ExportButtons
            filenameBase={`per_head_ap_${result.sessionName}`}
            getSVG={() => svgRef.current}
            getCSVRows={csvRows}
          />
        </div>
      </div>
      <svg ref={svgRef} />
      {pinned && (
        <div
          style={{
            position: "fixed", top: pinned.y, left: pinned.x,
            background: "#0d1b2a", border: "1px solid #1a5276",
            borderRadius: 6, padding: "10px 14px", fontSize: 12,
            color: "#c0c8e0", zIndex: 1000, maxWidth: 280,
            boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <strong>
              L{pinned.cell.layer}, {pinned.cell.unit ?? pinned.cell.sublayer}
            </strong>
            <button
              onClick={() => setPinned(null)}
              style={{ background: "none", border: "none", color: "#888", cursor: "pointer", fontSize: 14 }}
            >
              &times;
            </button>
          </div>
          {typeof pinned.cell.ap_recovery === "number" && (
            <div>
              AP recovery:{" "}
              <strong style={{ color: "#a0e0a0" }}>
                {pinned.cell.ap_recovery.toFixed(4)}
              </strong>
            </div>
          )}
          <div style={{ fontSize: 10, color: "#666", marginTop: 6 }}>
            First-order approximation &mdash; run exact mode to confirm.
          </div>
          {typeof pinned.cell.unit === "string" && pinned.cell.unit.startsWith("attn.h") && (
            <div style={{ marginTop: 10, borderTop: "1px solid #234", paddingTop: 8 }}>
              {decodeState.loading && <div style={{ color: "#aaa" }}>loading decode…</div>}
              {decodeState.error && (
                <div style={{ color: "#c88" }}>decode error: {decodeState.error}</div>
              )}
              {decodeState.data && (
                <div>
                  <div style={{ fontSize: 10, color: "#888", marginBottom: 4 }}>
                    Dominant write direction (sv energy ratio:{" "}
                    {(decodeState.data.sv_ratio * 100).toFixed(0)}%)
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, fontFamily: "monospace", fontSize: 11 }}>
                    <div>
                      <div style={{ color: "#8abaff", marginBottom: 2 }}>promoted</div>
                      {decodeState.data.top.slice(0, 5).map((t, i) => (
                        <div key={`ht-${i}`} style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 100 }}>
                            {t.token}
                          </span>
                          <span style={{ color: "#4caf50" }}>+{t.logit.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                    <div>
                      <div style={{ color: "#ffca8a", marginBottom: 2 }}>suppressed</div>
                      {decodeState.data.bottom.slice(0, 5).map((t, i) => (
                        <div key={`hb-${i}`} style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 100 }}>
                            {t.token}
                          </span>
                          <span style={{ color: "#c62828" }}>{t.logit.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          {pinned.cell.unit === "ffn" && (
            <div style={{ fontSize: 10, color: "#888", marginTop: 8, fontStyle: "italic" }}>
              FFN blocks are decoded per-neuron &mdash; switch to the
              approx_neuron mode for interpretation.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
