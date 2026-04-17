import { useEffect } from "react";
import { SHORTCUTS } from "../hooks/useKeyboardShortcuts";

interface Props {
  open: boolean;
  onClose: () => void;
}

// Simple centered modal listing available shortcuts. Triggered by '?'.
// Kept deliberately plain: this is a cheat sheet, not a settings dialog —
// no edit affordances, no search, just a reference card.
export function CheatSheet({ open, onClose }: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)",
        display: "flex", alignItems: "center", justifyContent: "center",
        zIndex: 500,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#0f1626", border: "1px solid #1a5276",
          borderRadius: 6, padding: "16px 20px", minWidth: 360,
          maxWidth: 560, color: "#e0e0f0", boxShadow: "0 12px 40px rgba(0,0,0,0.6)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
          <h3 style={{ margin: 0, fontSize: 14, color: "#a0a0c0" }}>Keyboard shortcuts</h3>
          <button
            onClick={onClose}
            style={{ background: "transparent", border: "none", color: "#888", cursor: "pointer", fontSize: 16, padding: 0 }}
          >{"\u00d7"}</button>
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <tbody>
            {SHORTCUTS.map((s) => (
              <tr key={s.keys}>
                <td style={{
                  padding: "4px 12px 4px 0", whiteSpace: "nowrap",
                  fontFamily: "monospace", color: "#a0c0e0", verticalAlign: "top",
                }}>{s.keys}</td>
                <td style={{ padding: "4px 0", color: "#c0c0d0" }}>{s.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ marginTop: 12, fontSize: 10, color: "#667" }}>
          Shortcuts are ignored while typing in an input or textarea.
        </div>
      </div>
    </div>
  );
}
