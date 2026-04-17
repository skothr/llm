import { useEffect, useState } from "react";
import { useStore } from "../state/store";

const TOAST_DURATION_MS = 8000;

// Single-slot undo toast surfaced bottom-left of the viewport whenever a
// result was just deleted. Shows a shrinking progress bar and an Undo
// button. Auto-clears after TOAST_DURATION_MS. Clicking × dismisses early
// (which also burns the undo capability).
export function UndoToast() {
  const lastDeleted = useStore((s) => s.lastDeleted);
  const undoDelete = useStore((s) => s.undoDelete);
  const clearUndoDelete = useStore((s) => s.clearUndoDelete);
  const [remaining, setRemaining] = useState<number>(TOAST_DURATION_MS);

  useEffect(() => {
    if (!lastDeleted) return;
    const start = lastDeleted.deletedAt;
    let rafId = 0;
    const tick = () => {
      const elapsed = Date.now() - start;
      const left = TOAST_DURATION_MS - elapsed;
      if (left <= 0) {
        clearUndoDelete();
        return;
      }
      setRemaining(left);
      rafId = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(rafId);
  }, [lastDeleted, clearUndoDelete]);

  if (!lastDeleted) return null;

  const pct = Math.max(0, Math.min(100, (remaining / TOAST_DURATION_MS) * 100));
  const { result } = lastDeleted;

  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        position: "fixed", left: 16, bottom: 16, zIndex: 400,
        background: "#0f1626", border: "1px solid #2a3a5a", borderRadius: 6,
        padding: "8px 12px 10px 12px", fontSize: 12, color: "#e0e0f0",
        boxShadow: "0 4px 18px rgba(0,0,0,0.55)", minWidth: 280, maxWidth: 440,
        display: "flex", flexDirection: "column", gap: 6,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ color: "#a0a0c0", fontSize: 11, marginBottom: 1 }}>
            Deleted {result.operation} result
          </div>
          <div style={{
            color: "#c0c0d0", fontSize: 11, fontFamily: "monospace",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {result.sessionName} {result.prompt && `— "${result.prompt.slice(0, 50)}${result.prompt.length > 50 ? "\u2026" : ""}"`}
          </div>
        </div>
        <button
          onClick={undoDelete}
          style={{
            fontSize: 11, padding: "3px 10px",
            background: "#1a5276", border: "1px solid #2a7296",
            color: "#e0e0f0", cursor: "pointer", borderRadius: 3,
          }}
        >undo</button>
        <button
          onClick={clearUndoDelete}
          title="Dismiss (can't undo after this)"
          style={{
            fontSize: 14, padding: 0, width: 20, height: 20,
            background: "transparent", border: "none", color: "#667",
            cursor: "pointer", lineHeight: 1,
          }}
        >{"\u00d7"}</button>
      </div>
      <div style={{ height: 2, background: "#1a2540", borderRadius: 1, overflow: "hidden" }}>
        <div
          style={{
            height: "100%", width: `${pct}%`,
            background: "#1a5276",
            // No CSS transition — we get smooth progress from rAF writing
            // the width on every frame.
          }}
        />
      </div>
    </div>
  );
}
