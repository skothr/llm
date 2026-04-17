import { useState } from "react";
import { useStore } from "../state/store";
import type { ProbeResult } from "../types/api";

interface Props { result: ProbeResult }

// Inline annotation bar for the currently-active result. Lives in the
// VisualizationArea header. Exposes three orthogonal affordances:
//   ★ pin   — shields the result from Clear All and from the recent-N cap
//             on persisted results
//   tags    — drive the filter bar's chip list
//   notes   — free text, collapsible to stay out of the way
export function ResultMetaEditor({ result }: Props) {
  const updateResultMeta = useStore((s) => s.updateResultMeta);
  const [tagDraft, setTagDraft] = useState("");
  const [showNotes, setShowNotes] = useState(!!result.notes);

  const tags = result.tags ?? [];
  const pinned = !!result.pinned;
  const notes = result.notes ?? "";

  const togglePin = () => updateResultMeta(result.id, { pinned: !pinned });
  const removeTag = (t: string) => updateResultMeta(result.id, { tags: tags.filter((x) => x !== t) });
  const addTag = (raw: string) => {
    // Lowercase + trim so "Foo" and "foo " don't duplicate the same concept.
    // Comma support lets the user type multiple tags at once ("a,b,c").
    const parts = raw.split(",").map((p) => p.trim().toLowerCase()).filter(Boolean);
    if (parts.length === 0) return;
    const merged = [...tags];
    for (const p of parts) if (!merged.includes(p)) merged.push(p);
    if (merged.length !== tags.length) updateResultMeta(result.id, { tags: merged });
  };

  const chipStyle: React.CSSProperties = {
    fontSize: 10, padding: "0 6px", background: "#16213e",
    border: "1px solid #1a5276", borderRadius: 10, color: "#a0c0e0",
    display: "inline-flex", alignItems: "center", gap: 3,
  };
  const chipXStyle: React.CSSProperties = {
    background: "transparent", border: "none", color: "#88a0c0",
    cursor: "pointer", padding: 0, lineHeight: 1, fontSize: 11,
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
        <button
          onClick={togglePin}
          title={pinned ? "Pinned — survives Clear All. Click to unpin." : "Pin this result (protects from Clear All)."}
          style={{
            fontSize: 14, width: 22, height: 22, padding: 0,
            background: pinned ? "#3a2a10" : "transparent",
            border: `1px solid ${pinned ? "#c08020" : "#2a3a4a"}`,
            color: pinned ? "#ffc040" : "#667788",
            cursor: "pointer", borderRadius: 3, lineHeight: 1,
          }}
        >{pinned ? "\u2605" : "\u2606"}</button>
        {tags.map((t) => (
          <span key={t} style={chipStyle}>
            {t}
            <button onClick={() => removeTag(t)} style={chipXStyle} title={`Remove tag "${t}"`}>{"\u00d7"}</button>
          </span>
        ))}
        <input
          value={tagDraft}
          onChange={(e) => setTagDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              addTag(tagDraft);
              setTagDraft("");
            } else if (e.key === "Backspace" && tagDraft === "" && tags.length > 0) {
              // Empty-field backspace removes the last tag — matches chip
              // input idiom from Gmail, Notion, etc.
              removeTag(tags[tags.length - 1]);
            }
          }}
          placeholder="+ tag"
          style={{
            fontSize: 11, padding: "1px 6px", width: 80,
            background: "#0d1b2a", border: "1px solid #1a2540",
            color: "#e0e0f0", borderRadius: 3,
          }}
        />
        <button
          onClick={() => setShowNotes((v) => !v)}
          title={notes ? "Toggle notes" : "Add a note"}
          style={{
            fontSize: 10, padding: "1px 6px",
            background: notes ? "#1a3a1a" : "#0d1b2a",
            border: `1px solid ${notes ? "#3a7a3a" : "#1a2540"}`,
            color: notes ? "#a0e0a0" : "#a0a0c0",
            cursor: "pointer", borderRadius: 3,
          }}
        >{notes ? "note \u25bc" : "+ note"}</button>
      </div>
      {showNotes && (
        <textarea
          value={notes}
          onChange={(e) => updateResultMeta(result.id, { notes: e.target.value })}
          placeholder="Notes on this result..."
          rows={2}
          style={{
            fontSize: 12, padding: 4, fontFamily: "inherit",
            background: "#0d1b2a", border: "1px solid #1a2540",
            color: "#e0e0f0", borderRadius: 3, resize: "vertical",
          }}
        />
      )}
    </div>
  );
}
