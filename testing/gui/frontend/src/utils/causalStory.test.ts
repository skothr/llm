import { describe, it, expect } from "vitest";
import {
  computeCausalStory,
  comparativeLensTokens,
  lensTokensFor,
  storyToMarkdown,
  storyNodeId,
} from "./causalStory";
import type { ResidualGridResponse } from "./useResidualGrid";

const grid = (): ResidualGridResponse => ({
  num_layers: 22,
  prompt_tokens: ["The", "capital", "of", "France"],
  cells: [
    { layer: 8,  sublayer: "attn", position: 3, tokens: [{ token: " Paris",  logit: 5.0 }, { token: " Lyon", logit: 4.5 }, { token: " France", logit: 4.0 }] },
    { layer: 8,  sublayer: "ffn",  position: 3, tokens: [{ token: " the",    logit: 3.0 }] },
    { layer: 12, sublayer: "attn", position: 3, tokens: [{ token: " Paris",  logit: 6.0 }, { token: " city", logit: 5.0 }, { token: " in", logit: 4.0 }] },
    { layer: 21, sublayer: "ffn",  position: 3, tokens: [{ token: " Paris",  logit: 9.0 }, { token: " France", logit: 7.5 }, { token: " Lyon", logit: 6.0 }] },
  ],
});

describe("computeCausalStory", () => {
  it("returns empty + note when no in-circuit edges", () => {
    const result = computeCausalStory(
      [{ writer_layer: 8, writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: false }],
      grid(),
      3,
    );
    expect(result.nodes).toEqual([]);
    expect(result.edges).toEqual([]);
    expect(result.note).toMatch(/no in-circuit edges/i);
  });

  it("filters by position", () => {
    const cells = [
      { writer_layer: 8, writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 0, in_circuit: true },
      { writer_layer: 8, writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
    ];
    const result = computeCausalStory(cells, grid(), 3);
    expect(result.nodes).toHaveLength(1);
    expect(result.nodes[0]).toMatchObject({ layer: 8, unit: "attn.h0", position: 3 });
  });

  it("orders nodes by layer ASC then attn-before-ffn", () => {
    const cells = [
      { writer_layer: 21, writer_unit: "ffn",     reader_layer: 22, reader_unit: "logits",  position: 3, in_circuit: true },
      { writer_layer: 8,  writer_unit: "ffn",     reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
      { writer_layer: 8,  writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
      { writer_layer: 12, writer_unit: "attn.h2", reader_layer: 21, reader_unit: "ffn_in",  position: 3, in_circuit: true },
    ];
    const result = computeCausalStory(cells, grid(), 3);
    expect(result.nodes.map((n) => `L${n.layer}.${n.unit}`)).toEqual([
      "L8.attn.h0",   // attn before ffn at L8
      "L8.ffn",
      "L12.attn.h2",
      "L21.ffn",
    ]);
  });

  it("attaches lens tokens via writer→sublayer mapping", () => {
    const cells = [
      { writer_layer: 8,  writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
      { writer_layer: 21, writer_unit: "ffn",     reader_layer: 22, reader_unit: "logits",  position: 3, in_circuit: true },
    ];
    const result = computeCausalStory(cells, grid(), 3);
    const l8 = result.nodes.find((n) => n.layer === 8);
    const l21 = result.nodes.find((n) => n.layer === 21);
    expect(l8?.lensSublayer).toBe("attn");
    expect(l8?.lensTokens).toEqual([" Paris", " Lyon", " France"]);
    expect(l21?.lensSublayer).toBe("ffn");
    expect(l21?.lensTokens).toEqual([" Paris", " France", " Lyon"]);
  });

  it("dedupes writer nodes that appear in multiple edges", () => {
    const cells = [
      { writer_layer: 8, writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
      { writer_layer: 8, writer_unit: "attn.h0", reader_layer: 17, reader_unit: "attn_in", position: 3, in_circuit: true },
    ];
    const result = computeCausalStory(cells, grid(), 3);
    expect(result.nodes).toHaveLength(1);
    expect(result.edges).toHaveLength(2);
  });

  it("emits note when grid is null but edges exist", () => {
    const cells = [
      { writer_layer: 8, writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
    ];
    const result = computeCausalStory(cells, null, 3);
    expect(result.nodes).toHaveLength(1);
    expect(result.nodes[0].lensTokens).toEqual([]);
    expect(result.note).toMatch(/lens grid loading/i);
  });

  it("embed writers get embed lensSublayer; tokens come from grid (L=0)", () => {
    const gridWithEmbed: ResidualGridResponse = {
      num_layers: 22,
      prompt_tokens: ["The", "capital", "of", "France"],
      cells: [
        { layer: 0, sublayer: "embed", position: 3, tokens: [{ token: "France", logit: 8.0 }] },
        { layer: 8, sublayer: "attn",  position: 3, tokens: [{ token: " Paris", logit: 5.0 }] },
      ],
    };
    const cells = [
      { writer_layer: 0, writer_unit: "embed",   reader_layer: 5, reader_unit: "attn_in", position: 3, in_circuit: true },
      { writer_layer: 8, writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
    ];
    const result = computeCausalStory(cells, gridWithEmbed, 3);
    const embed = result.nodes.find((n) => n.unit === "embed");
    expect(embed?.lensSublayer).toBe("embed");
    expect(embed?.lensTokens).toEqual(["France"]);
  });
});

describe("storyToMarkdown", () => {
  it("renders header with promptToken when provided", () => {
    const story = { position: 3, nodes: [], edges: [], note: null };
    const md = storyToMarkdown(story, "France");
    expect(md).toContain('## Causal Story — pos 3 ("France")');
  });

  it("renders bulleted node list with lens tokens", () => {
    const story = {
      position: 3,
      nodes: [
        { layer: 8, unit: "attn.h0", position: 3, lensSublayer: "attn" as const, lensTokens: [" Paris", " Lyon"] },
        { layer: 21, unit: "ffn", position: 3, lensSublayer: "ffn" as const, lensTokens: [" Paris"] },
      ],
      edges: [{ writer: { layer: 8, unit: "attn.h0" }, reader: { layer: 12, unit: "attn_in" } }],
      note: null,
    };
    const md = storyToMarkdown(story);
    expect(md).toContain("- **L8 attn.h0** — residual:  Paris ·  Lyon");
    expect(md).toContain("- **L21 ffn** — residual:  Paris");
    expect(md).toContain("1 edge feeding through this circuit");
  });

  it("renders embed writer with input-token lens content", () => {
    const story = {
      position: 0,
      nodes: [{ layer: 0, unit: "embed", position: 0, lensSublayer: "embed" as const, lensTokens: ["The"] }],
      edges: [],
      note: null,
    };
    const md = storyToMarkdown(story);
    expect(md).toContain("- **L0 embed** — input: The");
  });

  it("renders embed writer with no-lens-data fallback when grid missing", () => {
    const story = {
      position: 0,
      nodes: [{ layer: 0, unit: "embed", position: 0, lensSublayer: "embed" as const, lensTokens: [] }],
      edges: [],
      note: null,
    };
    const md = storyToMarkdown(story);
    expect(md).toContain("- **L0 embed** — _no lens data_");
  });

  it("renders empty-nodes case as italic placeholder", () => {
    const story = { position: 5, nodes: [], edges: [], note: "no in-circuit edges at this position" };
    const md = storyToMarkdown(story);
    expect(md).toContain("_no in-circuit edges at this position_");
    expect(md).toContain("_no nodes to display_");
  });
});

describe("storyNodeId", () => {
  it("formats layer + unit consistently", () => {
    expect(storyNodeId(8, "attn.h0")).toBe("L8-attn.h0");
    expect(storyNodeId(21, "ffn")).toBe("L21-ffn");
    expect(storyNodeId(0, "embed")).toBe("L0-embed");
  });
});

describe("lensTokensFor (public)", () => {
  it("returns empty for null grid", () => {
    expect(lensTokensFor(null, 8, "attn", 3, 3)).toEqual([]);
  });

  it("returns empty for null sublayer", () => {
    expect(lensTokensFor(grid(), 8, null, 3, 3)).toEqual([]);
  });

  it("normalizes embed lookup to layer=0", () => {
    const g: ResidualGridResponse = {
      num_layers: 22,
      prompt_tokens: ["The", "capital", "of", "France"],
      cells: [
        { layer: 0, sublayer: "embed", position: 3, tokens: [{ token: "France", logit: 8.0 }] },
      ],
    };
    // Caller asks for layer=5 (writer's edge layer); helper should still
    // hit the layer=0 embed cell.
    expect(lensTokensFor(g, 5, "embed", 3, 3)).toEqual(["France"]);
  });

  it("respects topK truncation", () => {
    expect(lensTokensFor(grid(), 8, "attn", 3, 1)).toEqual([" Paris"]);
    expect(lensTokensFor(grid(), 8, "attn", 3, 2)).toEqual([" Paris", " Lyon"]);
  });
});

describe("comparativeLensTokens", () => {
  it("returns matching length and aligned coordinates", () => {
    const cells = [
      { writer_layer: 8,  writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
      { writer_layer: 21, writer_unit: "ffn",     reader_layer: 22, reader_unit: "logits",  position: 3, in_circuit: true },
    ];
    const story = computeCausalStory(cells, grid(), 3);
    const gridB: ResidualGridResponse = {
      num_layers: 22,
      prompt_tokens: ["The", "capital", "of", "Italy"],
      cells: [
        { layer: 8,  sublayer: "attn", position: 3, tokens: [{ token: " Roma",  logit: 5.0 }, { token: " Milano", logit: 4.0 }, { token: " Italia", logit: 3.0 }] },
        { layer: 21, sublayer: "ffn",  position: 3, tokens: [{ token: " Roma",  logit: 9.0 }, { token: " Italia", logit: 7.0 }, { token: " Vaticano", logit: 6.0 }] },
      ],
    };
    const result = comparativeLensTokens(story, gridB, 3);
    expect(result).toHaveLength(2);
    expect(result[0]).toEqual([" Roma", " Milano", " Italia"]);
    expect(result[1]).toEqual([" Roma", " Italia", " Vaticano"]);
  });

  it("returns empty arrays when gridB has no matching cell", () => {
    const cells = [
      { writer_layer: 8, writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
    ];
    const story = computeCausalStory(cells, grid(), 3);
    const gridB: ResidualGridResponse = {
      num_layers: 22,
      prompt_tokens: ["other"],
      cells: [],
    };
    const result = comparativeLensTokens(story, gridB, 3);
    expect(result).toHaveLength(1);
    expect(result[0]).toEqual([]);
  });

  it("returns empty arrays when gridB is null", () => {
    const cells = [
      { writer_layer: 8, writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
    ];
    const story = computeCausalStory(cells, grid(), 3);
    const result = comparativeLensTokens(story, null, 3);
    expect(result).toEqual([[]]);
  });

  it("normalizes embed writer to layer=0 in gridB", () => {
    const gridA: ResidualGridResponse = {
      num_layers: 22,
      prompt_tokens: ["France"],
      cells: [{ layer: 0, sublayer: "embed", position: 0, tokens: [{ token: "France", logit: 8.0 }] }],
    };
    const cells = [
      { writer_layer: 0, writer_unit: "embed", reader_layer: 5, reader_unit: "attn_in", position: 0, in_circuit: true },
    ];
    const story = computeCausalStory(cells, gridA, 0);
    const gridB: ResidualGridResponse = {
      num_layers: 22,
      prompt_tokens: ["Italy"],
      cells: [{ layer: 0, sublayer: "embed", position: 0, tokens: [{ token: "Italy", logit: 8.0 }] }],
    };
    expect(comparativeLensTokens(story, gridB, 1)).toEqual([["Italy"]]);
  });
});

describe("storyToMarkdown — compare mode", () => {
  it("includes compare-prompt italic caption when present", () => {
    const story = {
      position: 3,
      nodes: [
        { layer: 8, unit: "attn.h0", position: 3, lensSublayer: "attn" as const, lensTokens: [" Paris"] },
      ],
      edges: [],
      note: null,
    };
    const md = storyToMarkdown(story, "France", {
      compareTokens: [[" Roma"]],
      comparePrompt: "The capital of Italy",
    });
    expect(md).toContain('_compare with: "The capital of Italy"_');
  });

  it("renders A and B sub-rows for each node with lens data", () => {
    const story = {
      position: 3,
      nodes: [
        { layer: 8, unit: "attn.h0", position: 3, lensSublayer: "attn" as const, lensTokens: [" Paris", " Lyon"] },
        { layer: 21, unit: "ffn", position: 3, lensSublayer: "ffn" as const, lensTokens: [" Paris"] },
      ],
      edges: [],
      note: null,
    };
    const md = storyToMarkdown(story, undefined, {
      compareTokens: [[" Roma", " Milano"], [" Roma"]],
      comparePrompt: "Italy",
    });
    expect(md).toContain("- **L8 attn.h0** — A residual:  Paris ·  Lyon");
    expect(md).toContain("  - B residual:  Roma ·  Milano");
    expect(md).toContain("- **L21 ffn** — A residual:  Paris");
    expect(md).toContain("  - B residual:  Roma");
  });

  it("uses 'input' label for embed in both A and B sub-rows", () => {
    const story = {
      position: 0,
      nodes: [
        { layer: 0, unit: "embed", position: 0, lensSublayer: "embed" as const, lensTokens: ["France"] },
      ],
      edges: [],
      note: null,
    };
    const md = storyToMarkdown(story, undefined, {
      compareTokens: [["Italy"]],
      comparePrompt: "Italy",
    });
    expect(md).toContain("- **L0 embed** — A input: France");
    expect(md).toContain("  - B input: Italy");
  });

  it("falls back to 'no lens data' for B when compareTokens entry is empty", () => {
    const story = {
      position: 3,
      nodes: [
        { layer: 8, unit: "attn.h0", position: 3, lensSublayer: "attn" as const, lensTokens: [" Paris"] },
      ],
      edges: [],
      note: null,
    };
    const md = storyToMarkdown(story, undefined, {
      compareTokens: [[]],
      comparePrompt: "Italy",
    });
    expect(md).toContain("  - B residual: _no lens data_");
  });

  it("non-compare invocation still works (backward compat)", () => {
    const story = {
      position: 3,
      nodes: [
        { layer: 8, unit: "attn.h0", position: 3, lensSublayer: "attn" as const, lensTokens: [" Paris"] },
      ],
      edges: [],
      note: null,
    };
    const md = storyToMarkdown(story);
    expect(md).toContain("- **L8 attn.h0** — residual:  Paris");
    expect(md).not.toContain("A residual");
    expect(md).not.toContain("B residual");
  });
});
