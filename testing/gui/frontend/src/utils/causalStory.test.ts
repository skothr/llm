import { describe, it, expect } from "vitest";
import { computeCausalStory, storyToMarkdown, storyNodeId } from "./causalStory";
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

  it("embed writers get null lensSublayer + empty tokens", () => {
    const cells = [
      { writer_layer: 0, writer_unit: "embed",   reader_layer: 5, reader_unit: "attn_in", position: 3, in_circuit: true },
      { writer_layer: 8, writer_unit: "attn.h0", reader_layer: 12, reader_unit: "attn_in", position: 3, in_circuit: true },
    ];
    const result = computeCausalStory(cells, grid(), 3);
    const embed = result.nodes.find((n) => n.unit === "embed");
    expect(embed?.lensSublayer).toBeNull();
    expect(embed?.lensTokens).toEqual([]);
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

  it("renders embed writer with italic caption", () => {
    const story = {
      position: 0,
      nodes: [{ layer: 0, unit: "embed", position: 0, lensSublayer: null, lensTokens: [] }],
      edges: [],
      note: null,
    };
    const md = storyToMarkdown(story);
    expect(md).toContain("- **L0 embed** — _input embedding (no lens in V1)_");
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
