import { useRef, useEffect, useState, useMemo } from "react";
import * as d3 from "d3";
import { displayToken } from "../../utils/displayToken";
import { useStore } from "../../state/store";
import type { LogitLensData, ProbeResult } from "../../types/api";

interface Props {
  resultA: ProbeResult;
  resultB: ProbeResult;
}

interface AlignedRow {
  originalLayer: number;
  sublayer: string;
  label: string;
  duplicate: boolean;
  dataA: LogitLensData | null;
  dataB: LogitLensData | null;
}

function buildOriginalLookup(data: LogitLensData[]) {
  const primary = new Map<string, LogitLensData>();
  const duplicates: { originalLayer: number; sublayer: string; insertAfter: number; data: LogitLensData }[] = [];

  for (const msg of data) {
    const orig = msg.original_layer ?? msg.layer;
    const key = `${orig}.${msg.sublayer}`;
    if (primary.has(key)) {
      const prevOrig = data[data.indexOf(msg) - 1]?.original_layer ?? data[data.indexOf(msg) - 1]?.layer ?? orig;
      duplicates.push({ originalLayer: orig, sublayer: msg.sublayer, insertAfter: prevOrig, data: msg });
    } else {
      primary.set(key, msg);
    }
  }
  return { primary, duplicates };
}

export function ABDiff({ resultA, resultB }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const sessionInfo = useStore((s) => s.sessionInfo);
  const [showDiff, setShowDiff] = useState(false);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    content: string;
  } | null>(null);

  const dataA = useMemo(
    () => resultA.data.filter((m): m is LogitLensData => m.type === "data" && "predictions" in m),
    [resultA.data]
  );
  const dataB = useMemo(
    () => resultB.data.filter((m): m is LogitLensData => m.type === "data" && "predictions" in m),
    [resultB.data]
  );

  const alignedRows = useMemo(() => {
    const infoA = sessionInfo[resultA.sessionName];
    const infoB = sessionInfo[resultB.sessionName];

    const lookupA = buildOriginalLookup(dataA);
    const lookupB = buildOriginalLookup(dataB);

    const numOrig = Math.max(
      infoA?.original_num_layers ?? Math.ceil(dataA.length / 2),
      infoB?.original_num_layers ?? Math.ceil(dataB.length / 2),
    );

    const rows: AlignedRow[] = [];
    for (let i = 0; i < numOrig; i++) {
      for (const sub of ["attn", "ffn"]) {
        const key = `${i}.${sub}`;
        rows.push({
          originalLayer: i,
          sublayer: sub,
          label: `L${i}.${sub}`,
          duplicate: false,
          dataA: lookupA.primary.get(key) ?? null,
          dataB: lookupB.primary.get(key) ?? null,
        });
      }

      const dupsA = lookupA.duplicates.filter((d) => d.insertAfter === i);
      const dupsB = lookupB.duplicates.filter((d) => d.insertAfter === i);
      const maxDups = Math.max(dupsA.length, dupsB.length);
      for (let d = 0; d < maxDups; d++) {
        const dupA = dupsA[d];
        const dupB = dupsB[d];
        const orig = dupA?.originalLayer ?? dupB?.originalLayer ?? i;
        const sub = dupA?.sublayer ?? dupB?.sublayer ?? "attn";
        rows.push({
          originalLayer: orig,
          sublayer: sub,
          label: `[L${orig}].${sub}`,
          duplicate: true,
          dataA: dupA?.data ?? null,
          dataB: dupB?.data ?? null,
        });
      }
    }

    return rows;
  }, [dataA, dataB, sessionInfo, resultA.sessionName, resultB.sessionName]);

  useEffect(() => {
    if (!svgRef.current || alignedRows.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const firstData = alignedRows.find((r) => r.dataA || r.dataB);
    const numPositions = (firstData?.dataA ?? firstData?.dataB)?.predictions.length ?? 1;

    const completeMsgA = resultA.data.find((m) => m.type === "complete") as
      | { type: "complete"; summary?: { prompt_tokens?: string[] } }
      | undefined;
    const promptTokens = completeMsgA?.summary?.prompt_tokens;

    const numRows = alignedRows.length;
    const cellW = Math.max(24, Math.min(40, 400 / numPositions));
    const cellH = 18;
    const margin = { top: promptTokens ? 50 : 30, right: 20, bottom: 30, left: 80 };
    const panelWidth = margin.left + numPositions * cellW + margin.right;
    const gap = 40;
    const totalWidth = showDiff ? panelWidth * 3 + gap * 2 : panelWidth * 2 + gap;
    const height = margin.top + numRows * cellH + margin.bottom;

    svg.attr("width", totalWidth).attr("height", height);

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
    const diffScale = d3.scaleDiverging(d3.interpolateRdBu).domain([-1, 0, 1]);

    function drawPanel(
      side: "A" | "B",
      offsetX: number,
      label: string,
    ) {
      const g = svg.append("g").attr("transform", `translate(${offsetX + margin.left},${margin.top})`);

      svg.append("text")
        .attr("x", offsetX + margin.left + (numPositions * cellW) / 2)
        .attr("y", promptTokens ? 14 : 16)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .attr("fill", "#a0a0c0")
        .text(label);

      if (promptTokens) {
        const tokG = svg.append("g").attr("transform", `translate(${offsetX + margin.left},${margin.top - 6})`);
        promptTokens.slice(0, numPositions).forEach((tok, i) => {
          tokG.append("text")
            .attr("x", i * cellW + cellW / 2)
            .attr("text-anchor", "middle")
            .attr("font-size", 9)
            .attr("font-family", "monospace")
            .attr("fill", "#6688aa")
            .text(displayToken(tok));
        });
      }

      alignedRows.forEach((row, rowIdx) => {
        const msg = side === "A" ? row.dataA : row.dataB;

        if (offsetX === 0 || side === "A") {
          g.append("text")
            .attr("x", -4)
            .attr("y", rowIdx * cellH + cellH / 2)
            .attr("text-anchor", "end")
            .attr("dominant-baseline", "middle")
            .attr("font-size", 9)
            .attr("fill", row.duplicate ? "#cc88ff" : "#8888aa")
            .text(row.label);
        }

        if (!msg) {
          g.append("rect")
            .attr("x", 0)
            .attr("y", rowIdx * cellH)
            .attr("width", numPositions * cellW - 1)
            .attr("height", cellH - 1)
            .attr("fill", "#0a0a15")
            .attr("rx", 2);
          return;
        }

        msg.predictions.forEach((posPreds, posIdx) => {
          const topPred = posPreds[0];
          if (!topPred) return;
          g.append("rect")
            .attr("x", posIdx * cellW)
            .attr("y", rowIdx * cellH)
            .attr("width", cellW - 1)
            .attr("height", cellH - 1)
            .attr("fill", colorScale(topPred.prob))
            .attr("rx", 2)
            .style("cursor", "pointer")
            .on("mouseenter", (event) => {
              const top = posPreds.slice(0, 5);
              const show = top.map((p) => displayToken(p.token));
              const maxLen = Math.max(...show.map((s) => s.length));
              const lines = top
                .map((p, i) => `${show[i].padEnd(maxLen)}  ${(p.prob * 100).toFixed(1).padStart(5)}%`)
                .join("\n");
              setTooltip({
                x: event.pageX + 10,
                y: event.pageY - 10,
                content: `${label} ${row.label} pos ${posIdx}\n${lines}`,
              });
            })
            .on("mouseleave", () => setTooltip(null));

          if (cellW >= 28) {
            const display = displayToken(topPred.token);
            g.append("text")
              .attr("x", posIdx * cellW + cellW / 2)
              .attr("y", rowIdx * cellH + cellH / 2)
              .attr("text-anchor", "middle")
              .attr("dominant-baseline", "middle")
              .attr("font-size", 8)
              .attr("fill", topPred.prob > 0.5 ? "#000" : "#fff")
              .style("pointer-events", "none")
              .text(display.length > 5 ? display.slice(0, 4) + "\u2026" : display);
          }
        });
      });
    }

    drawPanel("A", 0, resultA.sessionName);
    drawPanel("B", panelWidth + gap, resultB.sessionName);

    if (showDiff) {
      const diffOffset = (panelWidth + gap) * 2;
      const g = svg.append("g").attr("transform", `translate(${diffOffset + margin.left},${margin.top})`);

      svg.append("text")
        .attr("x", diffOffset + margin.left + (numPositions * cellW) / 2)
        .attr("y", promptTokens ? 14 : 16)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .attr("fill", "#a0a0c0")
        .text("Diff (A \u2212 B)");

      alignedRows.forEach((row, rowIdx) => {
        g.append("text")
          .attr("x", -4)
          .attr("y", rowIdx * cellH + cellH / 2)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 9)
          .attr("fill", row.duplicate ? "#cc88ff" : "#8888aa")
          .text(row.label);

        if (!row.dataA || !row.dataB) {
          g.append("rect")
            .attr("x", 0)
            .attr("y", rowIdx * cellH)
            .attr("width", numPositions * cellW - 1)
            .attr("height", cellH - 1)
            .attr("fill", "#0a0a15")
            .attr("rx", 2);
          return;
        }

        row.dataA.predictions.forEach((predsA, posIdx) => {
          const predsB = row.dataB!.predictions[posIdx];
          if (!predsA?.[0] || !predsB?.[0]) return;
          const diff = predsA[0].prob - predsB[0].prob;
          const tokensDiffer = predsA[0].token !== predsB[0].token;

          g.append("rect")
            .attr("x", posIdx * cellW)
            .attr("y", rowIdx * cellH)
            .attr("width", cellW - 1)
            .attr("height", cellH - 1)
            .attr("fill", diffScale(diff))
            .attr("opacity", Math.max(0.15, Math.abs(diff)))
            .attr("rx", 2)
            .attr("stroke", tokensDiffer ? "#ff6b6b" : "none")
            .attr("stroke-width", tokensDiffer ? 1.5 : 0)
            .style("cursor", "pointer")
            .on("mouseenter", (event) => {
              setTooltip({
                x: event.pageX + 10,
                y: event.pageY - 10,
                content: `${row.label} pos ${posIdx}\nA: ${predsA[0].token} (${(predsA[0].prob * 100).toFixed(1)}%)\nB: ${predsB[0].token} (${(predsB[0].prob * 100).toFixed(1)}%)\n\u0394: ${(diff * 100).toFixed(1)}%`,
              });
            })
            .on("mouseleave", () => setTooltip(null));
        });
      });
    }

  }, [alignedRows, showDiff, resultA.sessionName, resultB.sessionName, resultA.data]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0" }}>
          A/B: {resultA.sessionName} vs {resultB.sessionName}
        </h3>
        <button onClick={() => setShowDiff(!showDiff)} style={{ fontSize: 11 }}>
          {showDiff ? "Hide Diff" : "Show Diff"}
        </button>
      </div>
      <div style={{ overflowX: "auto" }}>
        <svg ref={svgRef} />
      </div>
      {tooltip && (
        <div
          style={{
            position: "fixed",
            left: tooltip.x,
            top: tooltip.y,
            background: "#16213e",
            border: "1px solid #1a5276",
            borderRadius: 4,
            padding: "6px 10px",
            fontFamily: "monospace",
            fontSize: 12,
            whiteSpace: "pre",
            pointerEvents: "none",
            zIndex: 100,
          }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
}
