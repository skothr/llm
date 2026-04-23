/** Circuit extraction helpers (Phase 3.8).
 *
 * Client-side cheap-ACDC: filter edges by |ap_recovery| >= tau, then
 * reverse-BFS from the logits reader to mark edges whose reader is
 * reverse-reachable.
 *
 * Mirrors the Python implementation in probe.extract_circuit so the
 * τ slider can re-filter interactively without a backend round-trip.
 */

export interface CircuitEdge {
  writer_layer: number;
  writer_unit: string;
  reader_layer: number;
  reader_unit: string;
  position: number;
  ap_recovery: number;
}

export interface CircuitResult {
  /** For every input edge, whether it's in the circuit at this tau. Same order as input. */
  in_circuit: boolean[];
  /** Count of edges with in_circuit=true. */
  n_edges_in_circuit: number;
  /** |visited| from reverse-BFS, inclusive of the logits sink node. */
  n_nodes_in_circuit: number;
}

function nodeKey(layer: number, unit: string, position: number): string {
  return `${layer} ${unit} ${position}`;
}

export function computeCircuit<E extends CircuitEdge>(edges: E[], tau: number): CircuitResult {
  if (tau < 0) {
    throw new Error("tau must be >= 0");
  }

  // Build reverse adjacency for edges clearing tau: reader -> [writer, ...]
  const reverseAdj = new Map<string, string[]>();
  for (const e of edges) {
    if (Math.abs(e.ap_recovery) < tau) continue;
    const rKey = nodeKey(e.reader_layer, e.reader_unit, e.position);
    const wKey = nodeKey(e.writer_layer, e.writer_unit, e.position);
    const existing = reverseAdj.get(rKey);
    if (existing === undefined) {
      reverseAdj.set(rKey, [wKey]);
    } else {
      existing.push(wKey);
    }
  }

  // Seed BFS from every logits-reader node that appears in the adjacency.
  const visited = new Set<string>();
  const queue: string[] = [];
  for (const rKey of reverseAdj.keys()) {
    const parts = rKey.split(" ");
    if (parts[1] === "logits") {
      visited.add(rKey);
      queue.push(rKey);
    }
  }

  while (queue.length > 0) {
    const r = queue.pop() as string;
    const writers = reverseAdj.get(r);
    if (writers === undefined) continue;
    for (const w of writers) {
      if (!visited.has(w)) {
        visited.add(w);
        queue.push(w);
      }
    }
  }

  const inCircuit = edges.map((e) => {
    if (Math.abs(e.ap_recovery) < tau) return false;
    return visited.has(nodeKey(e.reader_layer, e.reader_unit, e.position));
  });

  return {
    in_circuit: inCircuit,
    n_edges_in_circuit: inCircuit.filter(Boolean).length,
    n_nodes_in_circuit: visited.size,
  };
}
