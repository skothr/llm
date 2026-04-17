import { useStore } from "../state/store";

// Bulk-action toolbar for multi-selected results. Rendered inside
// VisualizationArea, just above the tab row. Hides itself entirely when
// nothing is selected so the header stays unchanged for single-result
// workflows.
export function BulkActionBar() {
  const selectedResultIds = useStore((s) => s.selectedResultIds);
  const pendingResults = useStore((s) => s.pendingResults);
  const results = useStore((s) => s.results);
  const bulkUpdateSelected = useStore((s) => s.bulkUpdateSelected);
  const bulkDeleteSelected = useStore((s) => s.bulkDeleteSelected);
  const clearResultSelection = useStore((s) => s.clearResultSelection);

  if (selectedResultIds.length === 0) return null;

  // Decide whether the batch action for pin should add or remove — if
  // ANY selected result is currently unpinned, we pin the whole batch;
  // otherwise we unpin. Mirrors the common file-manager convention.
  const all = [...Object.values(pendingResults), ...results];
  const selSet = new Set(selectedResultIds);
  const selResults = all.filter((r) => selSet.has(r.id));
  const anyUnpinned = selResults.some((r) => !r.pinned);

  const handlePinToggle = () => bulkUpdateSelected({ pinned: anyUnpinned });
  const handleAddTag = () => {
    const tag = window.prompt("Add tag to all selected:");
    if (!tag) return;
    const trimmed = tag.trim().toLowerCase();
    if (!trimmed) return;
    bulkUpdateSelected({ addTag: trimmed });
  };
  const handleDelete = () => {
    const pinnedCount = selResults.filter((r) => r.pinned).length;
    const msg = pinnedCount > 0
      ? `Delete ${selectedResultIds.length} result(s)? (${pinnedCount} pinned will also be deleted.)`
      : `Delete ${selectedResultIds.length} result(s)?`;
    if (window.confirm(msg)) bulkDeleteSelected();
  };

  const btnStyle: React.CSSProperties = {
    fontSize: 10, padding: "2px 8px", borderRadius: 3,
    background: "#0f1626", border: "1px solid #1a2540", color: "#a0c0e0",
    cursor: "pointer",
  };

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap",
      padding: "4px 8px", marginBottom: 6,
      background: "#0d1b2a", border: "1px solid #1a5276", borderRadius: 4,
      fontSize: 11, color: "#a0c0e0",
    }}>
      <strong style={{ color: "#e0e0f0" }}>
        {selectedResultIds.length} selected
      </strong>
      <button onClick={handlePinToggle} style={btnStyle} title={anyUnpinned ? "Pin all selected" : "Unpin all selected"}>
        {anyUnpinned ? "\u2606 pin all" : "\u2605 unpin all"}
      </button>
      <button onClick={handleAddTag} style={btnStyle} title="Apply a tag to every selected result">+ tag</button>
      <button
        onClick={handleDelete}
        style={{ ...btnStyle, background: "#3a1a1a", borderColor: "#6a2020", color: "#e0a0a0" }}
        title="Delete all selected (undo available for 8s afterwards)"
      >delete</button>
      <button onClick={clearResultSelection} style={btnStyle} title="Clear selection">clear</button>
      <span style={{ marginLeft: "auto", fontSize: 10, color: "#667" }}>
        Ctrl/Cmd+click to toggle · Shift+click to range
      </span>
    </div>
  );
}
