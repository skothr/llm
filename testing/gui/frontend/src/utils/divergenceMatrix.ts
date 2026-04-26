import type { CausalStory } from "./causalStory";
import { lensTokensFor } from "./causalStory";
import type { ResidualGridResponse } from "./useResidualGrid";

/**
 * Per-cell readout for the divergence heatmap. One cell = one
 * (writer, prompt) pair. Carries enough state for the heatmap to
 * choose its color AND show a tooltip with the actual decoded tokens.
 */
export type DivergenceCell = {
  /** Top-1 token under this prompt at this writer's coordinates; null if no lens data. */
  tokenTop: string | null;
  /** Top-1 under the reference prompt at the SAME coordinates. */
  tokenRefTop: string | null;
  /** True iff `tokenTop` and `tokenRefTop` are both non-null and equal. */
  matchTop1: boolean;
  /** Jaccard overlap of the two top-K shortlists in [0, 1]; null if either side missing. */
  jaccardTopK: number | null;
};

export type DivergenceMatrix = {
  /** Per-prompt label-friendly tag (truncated for axis use). */
  promptLabels: string[];
  /** Full prompt strings for tooltips. */
  prompts: string[];
  /** rows[i][j] = cell for story.nodes[i] at prompts[j]. */
  rows: DivergenceCell[][];
};

function jaccard(a: string[], b: string[]): number {
  if (a.length === 0 && b.length === 0) return 1;
  const setA = new Set(a);
  const setB = new Set(b);
  let inter = 0;
  for (const x of setA) if (setB.has(x)) inter += 1;
  const uni = setA.size + setB.size - inter;
  return uni === 0 ? 1 : inter / uni;
}

/**
 * Truncate a prompt to a short axis label. Words preserved; cuts at the
 * last space ≤ maxLen so labels stay readable.
 */
export function truncatePromptLabel(prompt: string, maxLen: number = 24): string {
  if (prompt.length <= maxLen) return prompt;
  const cut = prompt.slice(0, maxLen);
  const lastSpace = cut.lastIndexOf(" ");
  if (lastSpace > maxLen / 2) return cut.slice(0, lastSpace) + "…";
  return cut + "…";
}

/**
 * Build the divergence matrix: rows = story writers (in order), cols =
 * [reference, ...comparisons]. Reference column is always 100% match
 * with itself (sanity check).
 *
 * Both `referenceGrid` and entries of `comparisonGrids` may be null
 * (still loading or fetch failed). Cells handle nulls by zeroing out
 * `matchTop1` and `jaccardTopK = null`.
 */
export function computeDivergenceMatrix(
  story: CausalStory,
  referencePrompt: string,
  referenceGrid: ResidualGridResponse | null,
  comparisonPrompts: string[],
  comparisonGrids: (ResidualGridResponse | null)[],
  topK: number = 3,
): DivergenceMatrix {
  const allPrompts = [referencePrompt, ...comparisonPrompts];
  const allGrids = [referenceGrid, ...comparisonGrids];

  const rows: DivergenceCell[][] = story.nodes.map((node) => {
    const refTokens = lensTokensFor(referenceGrid, node.layer, node.lensSublayer, node.position, topK);
    const tokenRefTop = refTokens.length > 0 ? refTokens[0] : null;
    return allGrids.map((grid) => {
      const tokens = lensTokensFor(grid, node.layer, node.lensSublayer, node.position, topK);
      const tokenTop = tokens.length > 0 ? tokens[0] : null;
      const matchTop1 = tokenTop !== null && tokenRefTop !== null && tokenTop === tokenRefTop;
      const jaccardTopK = tokens.length > 0 && refTokens.length > 0 ? jaccard(tokens, refTokens) : null;
      return { tokenTop, tokenRefTop, matchTop1, jaccardTopK };
    });
  });

  return {
    prompts: allPrompts,
    promptLabels: allPrompts.map((p) => truncatePromptLabel(p)),
    rows,
  };
}

/**
 * Per-row "specificity" score: fraction of comparison cells (excluding
 * reference) that diverge from reference at top-1. 0 = writer agrees
 * everywhere (structural); 1 = writer always disagrees (highly prompt-
 * specific).
 *
 * Useful for sorting/filtering writers by how "input-specific" they are.
 */
export function rowSpecificityScores(matrix: DivergenceMatrix): number[] {
  return matrix.rows.map((row) => {
    if (row.length <= 1) return 0;
    let diverge = 0;
    let counted = 0;
    for (let j = 1; j < row.length; j++) {
      const cell = row[j];
      if (cell.tokenTop !== null && cell.tokenRefTop !== null) {
        counted += 1;
        if (!cell.matchTop1) diverge += 1;
      }
    }
    return counted === 0 ? 0 : diverge / counted;
  });
}
