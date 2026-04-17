import { useMemo } from "react";
import { useStore } from "../state/store";

// Chip-row filter bar above the VisualizationArea tab list. Query text,
// a pinned-only toggle, and one chip per known tag. Hides itself entirely
// when there's nothing to filter (no tags yet, no active filter) so the
// header stays clean for first-time users.
export function ResultFilterBar() {
  const results = useStore((s) => s.results);
  const filterTags = useStore((s) => s.filterTags);
  const filterPinnedOnly = useStore((s) => s.filterPinnedOnly);
  const filterQuery = useStore((s) => s.filterQuery);
  const toggleFilterTag = useStore((s) => s.toggleFilterTag);
  const setFilterPinnedOnly = useStore((s) => s.setFilterPinnedOnly);
  const setFilterQuery = useStore((s) => s.setFilterQuery);
  const clearFilters = useStore((s) => s.clearFilters);

  const allTags = useMemo(() => {
    const set = new Set<string>();
    for (const r of results) for (const t of r.tags ?? []) set.add(t);
    return [...set].sort();
  }, [results]);

  const hasAnyPinned = results.some((r) => r.pinned);
  const activeCount = filterTags.length + (filterPinnedOnly ? 1 : 0) + (filterQuery ? 1 : 0);

  if (allTags.length === 0 && !hasAnyPinned && activeCount === 0) return null;

  const chipStyle = (active: boolean): React.CSSProperties => ({
    fontSize: 10, padding: "1px 8px",
    background: active ? "#1a5276" : "#0d1b2a",
    border: `1px solid ${active ? "#2a7296" : "#1a2540"}`,
    color: active ? "#e0e0f0" : "#8888aa",
    cursor: "pointer", borderRadius: 10,
  });

  return (
    <div style={{
      display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap",
      marginBottom: 6, padding: "4px 0", fontSize: 11,
    }}>
      <input
        value={filterQuery}
        onChange={(e) => setFilterQuery(e.target.value)}
        placeholder="filter: session, prompt, notes"
        style={{
          fontSize: 11, padding: "1px 6px", width: 180,
          background: "#0d1b2a", border: "1px solid #1a2540",
          color: "#e0e0f0", borderRadius: 3,
        }}
      />
      {hasAnyPinned && (
        <button
          onClick={() => setFilterPinnedOnly(!filterPinnedOnly)}
          title="Show only pinned results"
          style={chipStyle(filterPinnedOnly)}
        >{filterPinnedOnly ? "\u2605 pinned" : "\u2606 pinned"}</button>
      )}
      {allTags.map((t) => (
        <button
          key={t}
          onClick={() => toggleFilterTag(t)}
          style={chipStyle(filterTags.includes(t))}
          title={filterTags.includes(t) ? `Stop filtering by "${t}"` : `Filter by tag "${t}"`}
        >{t}</button>
      ))}
      {activeCount > 0 && (
        <button
          onClick={clearFilters}
          title="Clear all filters"
          style={{
            fontSize: 10, padding: "1px 6px",
            background: "transparent", border: "1px solid #3a2020",
            color: "#c06060", cursor: "pointer", borderRadius: 10,
          }}
        >clear</button>
      )}
    </div>
  );
}

// Predicate consumed by VisualizationArea. Pulled out of the component
// so other views (e.g. a future history sidebar) can reuse the same rules.
export function makeResultPredicate(state: {
  filterTags: string[];
  filterPinnedOnly: boolean;
  filterQuery: string;
}): (r: { sessionName: string; prompt: string; tags?: string[]; notes?: string; pinned?: boolean }) => boolean {
  const q = state.filterQuery.trim().toLowerCase();
  const required = state.filterTags;
  return (r) => {
    if (state.filterPinnedOnly && !r.pinned) return false;
    if (required.length > 0) {
      const have = new Set(r.tags ?? []);
      for (const t of required) if (!have.has(t)) return false;
    }
    if (q) {
      const hay = `${r.sessionName} ${r.prompt} ${r.notes ?? ""} ${(r.tags ?? []).join(" ")}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  };
}
