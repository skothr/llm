import { useState } from "react";
import { useStore } from "../state/store";

interface Props {
  // Current prompt text — passed in so the bar can save it, and also so
  // the "current name" can be highlighted when the library entry matches.
  value: string;
  onLoad: (text: string) => void;
}

// Compact library bar meant to sit directly above a prompt textarea in
// ProbePanel / IntervenePanel. Mirror-style UI: a select to load, a button
// to save, a button to delete the currently-loaded entry.
//
// Intentionally small: saving uses window.prompt() for the name. A richer
// dialog (tags, descriptions) is easy to retrofit because the storage model
// in the store already has optional `tags`.
export function PromptLibraryBar({ value, onLoad }: Props) {
  const library = useStore((s) => s.promptLibrary);
  const savePrompt = useStore((s) => s.savePrompt);
  const deletePrompt = useStore((s) => s.deletePrompt);
  const [selected, setSelected] = useState<string>("");

  const current = library.find((p) => p.id === selected);
  // "Dirty" = the current textarea differs from the loaded entry. Helpful
  // signal for the user to know if Save-as-existing would overwrite.
  const dirty = current != null && current.text !== value;

  const handleSave = () => {
    const suggested = current?.name ?? "";
    const name = window.prompt("Prompt name:", suggested);
    if (name == null) return;
    const trimmed = name.trim();
    if (!trimmed) return;
    savePrompt(trimmed, value);
    // If the user saved under the same name they loaded, the entry id stays
    // stable (store overwrites), so we can keep the selection.
    const existing = library.find((p) => p.name === trimmed);
    if (existing) setSelected(existing.id);
  };

  const handleDelete = () => {
    if (!current) return;
    if (!window.confirm(`Delete prompt "${current.name}"?`)) return;
    deletePrompt(current.id);
    setSelected("");
  };

  const handleLoad = (id: string) => {
    setSelected(id);
    if (!id) return;
    const entry = library.find((p) => p.id === id);
    if (entry) onLoad(entry.text);
  };

  return (
    <div style={{ display: "flex", gap: 4, alignItems: "center", fontSize: 11 }}>
      <select
        value={selected}
        onChange={(e) => handleLoad(e.target.value)}
        title={library.length ? "Load a saved prompt" : "No saved prompts yet"}
        style={{ flex: 1, fontSize: 11, padding: "1px 4px" }}
      >
        <option value="">
          {library.length ? `load… (${library.length})` : "load… (empty)"}
        </option>
        {library.map((p) => (
          <option key={p.id} value={p.id}>{p.name}</option>
        ))}
      </select>
      <button
        onClick={handleSave}
        disabled={!value.trim()}
        title={dirty ? "Save (will overwrite existing entry with same name)" : "Save as new prompt"}
        style={{
          fontSize: 11, padding: "1px 6px",
          background: dirty ? "#3a3a1a" : "#0d1b2a",
          cursor: value.trim() ? "pointer" : "not-allowed",
        }}
      >save{dirty ? "*" : ""}</button>
      <button
        onClick={handleDelete}
        disabled={!current}
        title={current ? `Delete "${current.name}"` : "Select a saved prompt first"}
        style={{
          fontSize: 11, padding: "1px 6px",
          background: current ? "#3a1a1a" : "#0d1b2a",
          color: current ? "#e0e0f0" : "#555",
          cursor: current ? "pointer" : "not-allowed",
        }}
      >×</button>
    </div>
  );
}
