import type { ResidualGridResponse } from "./useResidualGrid";

export type CausalStoryNode = {
  layer: number;
  unit: string;            // "embed" | "attn.hN" | "ffn"
  position: number;
  /** Lens point at which this writer's residual content is decoded. */
  lensSublayer: "attn" | "ffn" | "embed" | null;
  lensTokens: string[];    // top-K residual lens tokens; [] if grid not loaded
};

export type CausalStoryEdge = {
  writer: { layer: number; unit: string };
  reader: { layer: number; unit: string };
};

export type CausalStoryCell = {
  writer_layer?: number;
  writer_unit?: string;
  reader_layer?: number;
  reader_unit?: string;
  position?: number;
  in_circuit?: boolean;
};

export type CausalStory = {
  position: number;
  nodes: CausalStoryNode[];
  edges: CausalStoryEdge[];
  note: string | null;
};

function lensSublayerForWriter(unit: string): "attn" | "ffn" | "embed" | null {
  if (unit === "embed") return "embed";
  if (unit.startsWith("attn")) return "attn";
  if (unit === "ffn") return "ffn";
  return null;
}

function lensTokensFor(
  grid: ResidualGridResponse | null,
  layer: number,
  sublayer: "attn" | "ffn" | "embed" | null,
  position: number,
  topK: number,
): string[] {
  if (grid === null || sublayer === null) return [];
  // Embed cells live at layer=0 only; the writer's `layer` may be the
  // edge's writer_layer (from the AP cell), but lens lookup must use 0.
  const lookupLayer = sublayer === "embed" ? 0 : layer;
  const cell = grid.cells.find(
    (c) => c.layer === lookupLayer && c.sublayer === sublayer && c.position === position,
  );
  if (!cell) return [];
  return cell.tokens.slice(0, topK).map((t) => t.token);
}

/**
 * Build a chronologically-ordered narrative of the model's computation
 * at a single position from the in-circuit edges + lens grid.
 *
 * Nodes are unique writers (layer, unit, position) appearing in any
 * in-circuit edge at the given position. Output ordered by layer ASC,
 * then attn-before-ffn within layer.
 */
export function computeCausalStory(
  cells: CausalStoryCell[],
  grid: ResidualGridResponse | null,
  position: number,
  topK: number = 3,
): CausalStory {
  const inCircuit = cells.filter(
    (c) => c.in_circuit === true && c.position === position,
  );
  if (inCircuit.length === 0) {
    return { position, nodes: [], edges: [], note: "no in-circuit edges at this position" };
  }

  const nodeMap = new Map<string, CausalStoryNode>();
  const edges: CausalStoryEdge[] = [];

  for (const c of inCircuit) {
    if (c.writer_layer !== undefined && c.writer_unit !== undefined) {
      const wKey = `${c.writer_layer}|${c.writer_unit}`;
      if (!nodeMap.has(wKey)) {
        const sub = lensSublayerForWriter(c.writer_unit);
        nodeMap.set(wKey, {
          layer: c.writer_layer,
          unit: c.writer_unit,
          position,
          lensSublayer: sub,
          lensTokens: lensTokensFor(grid, c.writer_layer, sub, position, topK),
        });
      }
      if (c.reader_layer !== undefined && c.reader_unit !== undefined) {
        edges.push({
          writer: { layer: c.writer_layer, unit: c.writer_unit },
          reader: { layer: c.reader_layer, unit: c.reader_unit },
        });
      }
    }
  }

  const nodes = Array.from(nodeMap.values()).sort((a, b) => {
    if (a.layer !== b.layer) return a.layer - b.layer;
    // attn before ffn; embed first if same layer
    const rank = (u: string): number =>
      u === "embed" ? 0 : u.startsWith("attn") ? 1 : u === "ffn" ? 2 : 3;
    return rank(a.unit) - rank(b.unit);
  });

  return {
    position,
    nodes,
    edges,
    note: grid === null ? "lens grid loading…" : null,
  };
}

/**
 * Render a CausalStory as markdown for one-click research-note export.
 * Shape mirrors the panel's visual rendering: bulleted node list with
 * residual lens tokens; embed writers get an italic caption; trailing
 * line summarizes the edge count.
 */
export function storyToMarkdown(story: CausalStory, promptToken?: string): string {
  const header = promptToken
    ? `## Causal Story — pos ${story.position} ("${promptToken}")`
    : `## Causal Story — pos ${story.position}`;
  const lines: string[] = [header];
  if (story.note) lines.push("", `_${story.note}_`);
  if (story.nodes.length === 0) {
    lines.push("", "_no nodes to display_");
  } else {
    lines.push("");
    for (const n of story.nodes) {
      if (n.lensTokens.length === 0) {
        lines.push(`- **L${n.layer} ${n.unit}** — _no lens data_`);
      } else {
        const label = n.unit === "embed" ? "input" : "residual";
        lines.push(`- **L${n.layer} ${n.unit}** — ${label}: ${n.lensTokens.join(" · ")}`);
      }
    }
  }
  if (story.edges.length > 0) {
    lines.push("", `${story.edges.length} edge${story.edges.length === 1 ? "" : "s"} feeding through this circuit`);
  }
  return lines.join("\n");
}

export type StoryNodeId = string;
/** Stable ID for cross-view selection: `"L{layer}-{unit}"`. */
export function storyNodeId(layer: number, unit: string): StoryNodeId {
  return `L${layer}-${unit}`;
}
