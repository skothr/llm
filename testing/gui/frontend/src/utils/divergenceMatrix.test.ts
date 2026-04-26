import { describe, it, expect } from "vitest";
import {
  computeDivergenceMatrix,
  rowSpecificityScores,
  truncatePromptLabel,
} from "./divergenceMatrix";
import { computeCausalStory } from "./causalStory";
import type { ResidualGridResponse } from "./useResidualGrid";

const ref = (): ResidualGridResponse => ({
  num_layers: 22,
  prompt_tokens: ["The", "capital", "of", "France"],
  cells: [
    { layer: 8,  sublayer: "attn", position: 3, tokens: [{ token: " Paris",  logit: 6.0 }, { token: " Lyon", logit: 5.0 }, { token: " France", logit: 4.0 }] },
    { layer: 21, sublayer: "ffn",  position: 3, tokens: [{ token: " Paris",  logit: 9.0 }, { token: " France", logit: 7.5 }, { token: " Lyon", logit: 6.0 }] },
  ],
});

const italyGrid = (): ResidualGridResponse => ({
  num_layers: 22,
  prompt_tokens: ["The", "capital", "of", "Italy"],
  cells: [
    { layer: 8,  sublayer: "attn", position: 3, tokens: [{ token: " Roma",  logit: 6.0 }, { token: " Milano", logit: 5.0 }, { token: " Italia", logit: 4.0 }] },
    { layer: 21, sublayer: "ffn",  position: 3, tokens: [{ token: " Roma",  logit: 9.0 }, { token: " Italia", logit: 7.5 }, { token: " Vaticano", logit: 6.0 }] },
  ],
});

const structuralGrid = (): ResidualGridResponse => ({
  // Same top-1 as reference at L8 (a "structural" writer that doesn't depend
  // on input identity), but different at L21 (an "answer" writer).
  num_layers: 22,
  prompt_tokens: ["The", "capital", "of", "Spain"],
  cells: [
    { layer: 8,  sublayer: "attn", position: 3, tokens: [{ token: " Paris",  logit: 6.0 }, { token: " Lyon", logit: 5.0 }, { token: " France", logit: 4.0 }] },
    { layer: 21, sublayer: "ffn",  position: 3, tokens: [{ token: " Madrid", logit: 9.0 }, { token: " Spain", logit: 7.5 }] },
  ],
});

const storyOf = (grid: ResidualGridResponse) =>
  computeCausalStory(
    [
      { writer_layer: 8,  writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
      { writer_layer: 21, writer_unit: "ffn",     reader_layer: 22, reader_unit: "logits",  position: 3, in_circuit: true },
    ],
    grid,
    3,
  );

describe("truncatePromptLabel", () => {
  it("returns input unchanged when within limit", () => {
    expect(truncatePromptLabel("short prompt")).toBe("short prompt");
  });

  it("cuts at last space and adds ellipsis when over limit", () => {
    expect(truncatePromptLabel("The capital of France is", 18)).toBe("The capital of…");
  });

  it("hard-cuts when no good break found", () => {
    expect(truncatePromptLabel("supercalifragilistic", 8)).toBe("supercal…");
  });
});

describe("computeDivergenceMatrix — shape", () => {
  it("rows match story.nodes; cols = 1 + comparisonPrompts.length", () => {
    const story = storyOf(ref());
    const m = computeDivergenceMatrix(
      story,
      "France",
      ref(),
      ["Italy", "Spain"],
      [italyGrid(), structuralGrid()],
    );
    expect(m.rows).toHaveLength(2);
    expect(m.rows[0]).toHaveLength(3);
    expect(m.prompts).toEqual(["France", "Italy", "Spain"]);
    expect(m.promptLabels).toHaveLength(3);
  });
});

describe("computeDivergenceMatrix — reference column always matches itself", () => {
  it("column 0 is matchTop1 = true for every row that has lens data", () => {
    const story = storyOf(ref());
    const m = computeDivergenceMatrix(story, "France", ref(), ["Italy"], [italyGrid()]);
    for (const row of m.rows) {
      expect(row[0].matchTop1).toBe(true);
      expect(row[0].tokenTop).toBe(row[0].tokenRefTop);
      expect(row[0].jaccardTopK).toBe(1);
    }
  });
});

describe("computeDivergenceMatrix — divergence detection", () => {
  it("Italy diverges at both writers (Paris vs Roma)", () => {
    const story = storyOf(ref());
    const m = computeDivergenceMatrix(story, "France", ref(), ["Italy"], [italyGrid()]);
    expect(m.rows[0][1].matchTop1).toBe(false);
    expect(m.rows[0][1].tokenTop).toBe(" Roma");
    expect(m.rows[0][1].tokenRefTop).toBe(" Paris");
    expect(m.rows[1][1].matchTop1).toBe(false);
  });

  it("structural writer matches at L8 but diverges at L21", () => {
    const story = storyOf(ref());
    const m = computeDivergenceMatrix(story, "France", ref(), ["Spain"], [structuralGrid()]);
    // L8 attn.h0 — Paris == Paris (structural)
    expect(m.rows[0][1].matchTop1).toBe(true);
    // L21 ffn — Paris vs Madrid (answer-specific)
    expect(m.rows[1][1].matchTop1).toBe(false);
  });
});

describe("computeDivergenceMatrix — null grid handling", () => {
  it("null comparisonGrid yields tokenTop=null and matchTop1=false", () => {
    const story = storyOf(ref());
    const m = computeDivergenceMatrix(story, "France", ref(), ["Italy"], [null]);
    expect(m.rows[0][1].tokenTop).toBeNull();
    expect(m.rows[0][1].matchTop1).toBe(false);
    expect(m.rows[0][1].jaccardTopK).toBeNull();
  });

  it("null referenceGrid yields tokenRefTop=null in all cells", () => {
    const story = storyOf(ref());
    const m = computeDivergenceMatrix(story, "France", null, ["Italy"], [italyGrid()]);
    for (const row of m.rows) {
      expect(row[0].tokenRefTop).toBeNull();
      expect(row[1].tokenRefTop).toBeNull();
    }
  });
});

describe("computeDivergenceMatrix — Jaccard scoring", () => {
  it("identical shortlists yield Jaccard=1; fully disjoint yield 0", () => {
    const story = storyOf(ref());
    const disjoint: ResidualGridResponse = {
      num_layers: 22,
      prompt_tokens: ["x"],
      cells: [
        { layer: 8,  sublayer: "attn", position: 3, tokens: [{ token: " A", logit: 1 }, { token: " B", logit: 1 }, { token: " C", logit: 1 }] },
        { layer: 21, sublayer: "ffn",  position: 3, tokens: [{ token: " X", logit: 1 }, { token: " Y", logit: 1 }, { token: " Z", logit: 1 }] },
      ],
    };
    const m = computeDivergenceMatrix(story, "France", ref(), ["disjoint"], [disjoint]);
    expect(m.rows[0][0].jaccardTopK).toBe(1);
    expect(m.rows[0][1].jaccardTopK).toBe(0);
  });

  it("partial overlap yields fractional Jaccard", () => {
    const story = storyOf(ref());
    const partial: ResidualGridResponse = {
      num_layers: 22,
      prompt_tokens: ["x"],
      cells: [
        // {Paris, Lyon, France} vs {Paris, X, Y} → intersection = {Paris}; union size 5 → 1/5
        { layer: 8,  sublayer: "attn", position: 3, tokens: [{ token: " Paris", logit: 1 }, { token: " X", logit: 1 }, { token: " Y", logit: 1 }] },
        { layer: 21, sublayer: "ffn",  position: 3, tokens: [{ token: " Paris", logit: 1 }, { token: " France", logit: 1 }, { token: " Z", logit: 1 }] },
      ],
    };
    const m = computeDivergenceMatrix(story, "France", ref(), ["partial"], [partial]);
    expect(m.rows[0][1].jaccardTopK).toBeCloseTo(1 / 5, 5);
    // {Paris, France, Lyon} vs {Paris, France, Z} → intersection={Paris, France}; union={Paris,France,Lyon,Z} → 2/4 = 0.5
    expect(m.rows[1][1].jaccardTopK).toBeCloseTo(0.5, 5);
  });
});

describe("rowSpecificityScores", () => {
  it("fully-divergent row scores 1.0", () => {
    const story = storyOf(ref());
    const m = computeDivergenceMatrix(story, "France", ref(), ["Italy"], [italyGrid()]);
    const scores = rowSpecificityScores(m);
    expect(scores[0]).toBe(1.0);
    expect(scores[1]).toBe(1.0);
  });

  it("structural-then-answer mix scores 0.0 for L8 and 1.0 for L21", () => {
    const story = storyOf(ref());
    const m = computeDivergenceMatrix(story, "France", ref(), ["Spain"], [structuralGrid()]);
    const scores = rowSpecificityScores(m);
    expect(scores[0]).toBe(0.0);  // L8 attn.h0 — structural
    expect(scores[1]).toBe(1.0);  // L21 ffn — answer-specific
  });

  it("mixed across multiple comparisons averages correctly", () => {
    const story = storyOf(ref());
    const m = computeDivergenceMatrix(
      story, "France", ref(),
      ["Italy", "Spain"],
      [italyGrid(), structuralGrid()],
    );
    const scores = rowSpecificityScores(m);
    // L8 attn.h0: Italy diverges, Spain matches → 1/2 = 0.5
    expect(scores[0]).toBe(0.5);
    // L21 ffn: Italy diverges, Spain diverges → 2/2 = 1.0
    expect(scores[1]).toBe(1.0);
  });

  it("ignores cells with missing lens data", () => {
    const story = storyOf(ref());
    const m = computeDivergenceMatrix(story, "France", ref(), ["x"], [null]);
    const scores = rowSpecificityScores(m);
    expect(scores).toEqual([0, 0]);
  });
});
