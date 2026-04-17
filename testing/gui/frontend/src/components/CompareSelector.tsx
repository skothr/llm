import { useMemo } from "react";
import { useStore } from "../state/store";
import type { ProbeResult, ResultOperation } from "../types/api";

interface Props {
  // Only candidates whose operation matches this are offered. Compares
  // across operations don't have a meaningful delta semantic.
  operation: ResultOperation;
  // The result currently displayed — excluded from the candidate list so
  // the user can't "compare X with itself".
  currentId: string;
  value: string | null;
  onChange: (id: string | null) => void;
}

// Shared compact selector rendered in viz headers. Emits a pickable list of
// historical results of the matching operation. Kept intentionally narrow
// (width-constrained) so it fits beside the export buttons without
// pushing the chart down.
export function CompareSelector({ operation, currentId, value, onChange }: Props) {
  const results = useStore((s) => s.results);
  const pendingResults = useStore((s) => s.pendingResults);

  const candidates = useMemo(() => {
    // Merge pending and completed; exclude self, exclude A/B side partners
    // (they'd double-count the same run), and match on operation.
    const all = [...Object.values(pendingResults), ...results];
    return all.filter((r) => r.id !== currentId && !r.isB && r.operation === operation);
  }, [results, pendingResults, currentId, operation]);

  if (candidates.length === 0) return null;

  return (
    <label style={{ fontSize: 11, color: "#8888aa", display: "flex", alignItems: "center", gap: 4 }}>
      vs
      <select
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value || null)}
        style={{
          fontSize: 11, padding: "1px 4px",
          background: "#0f1626", color: "#e0e0f0",
          border: "1px solid #1a2540", borderRadius: 3,
          maxWidth: 220,
        }}
      >
        <option value="">none</option>
        {candidates.map((r) => (
          <option key={r.id} value={r.id}>
            {r.sessionName}
            {r.pinned ? " \u2605" : ""}
            {` — "${r.prompt.slice(0, 24)}${r.prompt.length > 24 ? "\u2026" : ""}"`}
          </option>
        ))}
      </select>
    </label>
  );
}

// Shared helper: pull a typed result from the store by id. Returns null if
// the result vanished (e.g. was cleared mid-compare).
export function useResultById(id: string | null): ProbeResult | null {
  const results = useStore((s) => s.results);
  const pendingResults = useStore((s) => s.pendingResults);
  return useMemo(() => {
    if (!id) return null;
    return pendingResults[id] ?? results.find((r) => r.id === id) ?? null;
  }, [id, results, pendingResults]);
}
