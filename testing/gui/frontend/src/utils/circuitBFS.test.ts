import { describe, expect, it } from "vitest";
import { computeCircuit, CircuitEdge } from "./circuitBFS";

describe("computeCircuit", () => {
  it("throws on negative tau", () => {
    expect(() => computeCircuit([], -0.1)).toThrow("tau");
  });

  it("returns empty circuit when all edges below tau", () => {
    const edges: CircuitEdge[] = [
      { writer_layer: 0, writer_unit: "embed", reader_layer: 1, reader_unit: "logits", position: 0, ap_recovery: 0.01 },
    ];
    const r = computeCircuit(edges, 0.1);
    expect(r.in_circuit).toEqual([false]);
    expect(r.n_edges_in_circuit).toBe(0);
    expect(r.n_nodes_in_circuit).toBe(0);
  });

  it("keeps direct-to-logits edges at tau=0", () => {
    const edges: CircuitEdge[] = [
      { writer_layer: 0, writer_unit: "embed", reader_layer: 1, reader_unit: "logits", position: 0, ap_recovery: 0.5 },
      { writer_layer: 0, writer_unit: "embed", reader_layer: 1, reader_unit: "logits", position: 1, ap_recovery: 0.3 },
    ];
    const r = computeCircuit(edges, 0.0);
    expect(r.in_circuit).toEqual([true, true]);
    expect(r.n_edges_in_circuit).toBe(2);
    // visited = {logits@0, logits@1, embed@0, embed@1} - but embed@pos is a
    // writer-only node, so only logits@0 and logits@1 are seeded, then embed@0
    // and embed@1 are added during BFS. |visited| == 4.
    expect(r.n_nodes_in_circuit).toBe(4);
  });

  it("excludes disconnected components from circuit", () => {
    const edges: CircuitEdge[] = [
      { writer_layer: 0, writer_unit: "embed", reader_layer: 2, reader_unit: "logits", position: 0, ap_recovery: 0.5 },
      { writer_layer: 0, writer_unit: "ffn", reader_layer: 1, reader_unit: "ffn_in", position: 0, ap_recovery: 0.5 },
    ];
    const r = computeCircuit(edges, 0.1);
    expect(r.in_circuit).toEqual([true, false]);
    expect(r.n_edges_in_circuit).toBe(1);
  });

  it("is monotonic in tau (raising tau can only shrink circuit)", () => {
    const edges: CircuitEdge[] = [
      { writer_layer: 0, writer_unit: "embed", reader_layer: 1, reader_unit: "logits", position: 0, ap_recovery: 0.5 },
      { writer_layer: 0, writer_unit: "ffn", reader_layer: 1, reader_unit: "logits", position: 0, ap_recovery: 0.2 },
      { writer_layer: 0, writer_unit: "attn.h0", reader_layer: 1, reader_unit: "logits", position: 0, ap_recovery: 0.05 },
    ];
    const low = computeCircuit(edges, 0.0);
    const mid = computeCircuit(edges, 0.1);
    const high = computeCircuit(edges, 0.3);
    expect(low.n_edges_in_circuit).toBe(3);
    expect(mid.n_edges_in_circuit).toBe(2);
    expect(high.n_edges_in_circuit).toBe(1);
  });

  it("same position only — no cross-position edges", () => {
    // This mirrors the Phase 3.7 invariant: edges always connect same-position nodes.
    const edges: CircuitEdge[] = [
      { writer_layer: 0, writer_unit: "embed", reader_layer: 1, reader_unit: "logits", position: 0, ap_recovery: 0.5 },
    ];
    const r = computeCircuit(edges, 0.1);
    expect(r.in_circuit).toEqual([true]);
  });
});
