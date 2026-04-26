import { useState } from "react";
import * as d3 from "d3";
import type { CausalStory } from "../../utils/causalStory";
import {
  computeDivergenceMatrix,
  rowSpecificityScores,
  type DivergenceMatrix,
} from "../../utils/divergenceMatrix";
import type { ResidualGridResponse } from "../../utils/useResidualGrid";

type Props = {
  story: CausalStory;
  referencePrompt: string;
  referenceGrid: ResidualGridResponse | null;
  comparisonPrompts: string[];
  comparisonGrids: (ResidualGridResponse | null)[];
  loading?: boolean;
};

type Mode = "top1" | "jaccard";

const CELL_W = 70;
const CELL_H = 22;
const ROW_LABEL_W = 110;
const COL_LABEL_H = 60;
const SPEC_W = 50;
const PAD = 8;

function cellFill(mode: Mode, cell: DivergenceMatrix["rows"][number][number]): string {
  if (mode === "top1") {
    if (cell.tokenTop === null || cell.tokenRefTop === null) return "#3a3a3a";
    return cell.matchTop1 ? "#2e7d32" : "#c62828";
  }
  // jaccard mode — linear interpolation 0=red → 1=green
  if (cell.jaccardTopK === null) return "#3a3a3a";
  return d3.interpolateRdYlGn(cell.jaccardTopK);
}

function truncateToken(t: string | null, maxLen: number = 7): string {
  if (t === null) return "";
  // Drop the leading SentencePiece space for axis-readability.
  const trimmed = t.startsWith(" ") ? t.slice(1) : t;
  if (trimmed.length <= maxLen) return trimmed;
  return trimmed.slice(0, maxLen - 1) + "…";
}

export function DivergenceHeatmap({
  story, referencePrompt, referenceGrid, comparisonPrompts, comparisonGrids, loading,
}: Props) {
  const [mode, setMode] = useState<Mode>("top1");

  if (story.nodes.length === 0) {
    return (
      <div data-testid="divergence-heatmap-empty"
        style={{ marginTop: 12, padding: 8, fontSize: 12, color: "#888" }}>
        no in-circuit writers — divergence heatmap needs a non-empty causal story.
      </div>
    );
  }

  const matrix = computeDivergenceMatrix(
    story,
    referencePrompt,
    referenceGrid,
    comparisonPrompts,
    comparisonGrids,
    3,
  );
  const specificity = rowSpecificityScores(matrix);

  const numRows = matrix.rows.length;
  const numCols = matrix.prompts.length;
  const totalW = ROW_LABEL_W + numCols * CELL_W + SPEC_W + 2 * PAD;
  const totalH = COL_LABEL_H + numRows * CELL_H + 2 * PAD;

  return (
    <div
      data-testid="divergence-heatmap"
      style={{
        marginTop: 16, paddingTop: 12,
        borderTop: "1px solid #2a2a3a",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: "#a0a0c0", fontWeight: "bold" }}>
          Divergence Heatmap — {numCols} prompt{numCols === 1 ? "" : "s"} × {numRows} writer{numRows === 1 ? "" : "s"}
        </div>
        {loading && (
          <span data-testid="divergence-heatmap-loading"
            style={{ fontSize: 11, color: "#888" }}>
            loading {comparisonGrids.filter((g) => g === null).length} of {comparisonGrids.length}…
          </span>
        )}
        <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
          <button
            onClick={() => setMode("top1")}
            aria-pressed={mode === "top1"}
            data-testid="divergence-mode-top1"
            style={{
              fontSize: 11, padding: "2px 8px",
              background: mode === "top1" ? "#2a3a55" : "#1a2438",
              color: "#cfd6e6", border: "1px solid #2a3a55",
              borderRadius: 3, cursor: "pointer",
            }}
          >
            top-1 match
          </button>
          <button
            onClick={() => setMode("jaccard")}
            aria-pressed={mode === "jaccard"}
            data-testid="divergence-mode-jaccard"
            style={{
              fontSize: 11, padding: "2px 8px",
              background: mode === "jaccard" ? "#2a3a55" : "#1a2438",
              color: "#cfd6e6", border: "1px solid #2a3a55",
              borderRadius: 3, cursor: "pointer",
            }}
          >
            Jaccard top-3
          </button>
        </div>
      </div>
      <svg width={totalW} height={totalH} style={{ background: "#0e0e12", borderRadius: 4 }}>
        {/* Column labels — rotated 30° for readable layout */}
        {matrix.promptLabels.map((label, j) => {
          const x = PAD + ROW_LABEL_W + j * CELL_W + CELL_W / 2;
          const y = COL_LABEL_H - 4;
          const isRef = j === 0;
          return (
            <text
              key={j}
              x={x} y={y}
              transform={`rotate(-30 ${x} ${y})`}
              fill={isRef ? "#8abaff" : "#aaa"}
              fontSize={10}
              fontWeight={isRef ? "bold" : "normal"}
              textAnchor="end"
            >
              <title>{matrix.prompts[j]}</title>
              {label}{isRef ? " (ref)" : ""}
            </text>
          );
        })}

        {/* Row labels + specificity strip + cells */}
        {matrix.rows.map((row, i) => {
          const node = story.nodes[i];
          const yTop = PAD + COL_LABEL_H + i * CELL_H;
          const labelY = yTop + CELL_H / 2 + 4;
          return (
            <g key={i}>
              <text
                x={PAD + ROW_LABEL_W - 6}
                y={labelY}
                fill="#aaa"
                fontSize={10}
                textAnchor="end"
                fontFamily="monospace"
              >
                L{node.layer} {node.unit}
              </text>
              {row.map((cell, j) => {
                const x = PAD + ROW_LABEL_W + j * CELL_W;
                const fill = cellFill(mode, cell);
                const matchAttr =
                  cell.tokenTop === null || cell.tokenRefTop === null
                    ? "null"
                    : cell.matchTop1 ? "true" : "false";
                return (
                  <g key={j} data-testid={`divergence-cell-${i}-${j}`}
                     data-match={matchAttr}>
                    <title>
                      {`prompt: ${matrix.prompts[j]}\n` +
                       `top-1: ${cell.tokenTop ?? "(no data)"}\n` +
                       `ref top-1: ${cell.tokenRefTop ?? "(no data)"}\n` +
                       `Jaccard top-3: ${cell.jaccardTopK?.toFixed(2) ?? "n/a"}`}
                    </title>
                    <rect
                      x={x} y={yTop}
                      width={CELL_W - 1} height={CELL_H - 1}
                      fill={fill}
                      stroke="#0e0e12"
                      strokeWidth={1}
                    />
                    <text
                      x={x + CELL_W / 2}
                      y={yTop + CELL_H / 2 + 3}
                      fill="#fff"
                      fontSize={9}
                      textAnchor="middle"
                      fontFamily="monospace"
                      pointerEvents="none"
                    >
                      {truncateToken(cell.tokenTop)}
                    </text>
                  </g>
                );
              })}
              {/* Specificity bar — at the right edge */}
              {(() => {
                const sx = PAD + ROW_LABEL_W + numCols * CELL_W + 4;
                const fillW = specificity[i] * (SPEC_W - 8);
                return (
                  <g data-testid={`divergence-spec-${i}`}
                     data-spec={specificity[i].toFixed(3)}>
                    <title>{`specificity: ${(specificity[i] * 100).toFixed(0)}% — fraction of comparison prompts diverging from ref`}</title>
                    <rect
                      x={sx} y={yTop + 4}
                      width={SPEC_W - 8} height={CELL_H - 8}
                      fill="#1a2438" stroke="#2a3a55" strokeWidth={1}
                    />
                    <rect
                      x={sx} y={yTop + 4}
                      width={fillW} height={CELL_H - 8}
                      fill="#ffa07a"
                    />
                  </g>
                );
              })()}
            </g>
          );
        })}
      </svg>
      <div style={{ fontSize: 11, color: "#888", marginTop: 6 }}>
        {mode === "top1"
          ? "green = writer's top-1 token matches reference; red = diverges; gray = no lens data."
          : "green→red = Jaccard overlap of top-3 lists with reference (1.0 = identical, 0.0 = disjoint)."}
        {" "}Right strip: per-writer specificity (fraction of comparison prompts that diverge).
      </div>
    </div>
  );
}
