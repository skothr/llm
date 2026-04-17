import { useEffect } from "react";
import { useStore } from "../state/store";

// Shortcut definitions live in one place so the cheat-sheet component can
// render them from the same source the hook listens for. `when` is a
// plain-English guard for the overlay; the handler is responsible for its
// own preconditions at runtime.
export interface ShortcutDef {
  keys: string;
  description: string;
}

export const SHORTCUTS: ShortcutDef[] = [
  { keys: "Ctrl/Cmd + Enter", description: "Run the active panel (in prompt textarea)" },
  { keys: "1 / 2 / 3", description: "Switch between Sessions / Probe / Intervene tabs" },
  { keys: "[  /  ]", description: "Previous / next result (among visible)" },
  { keys: "p", description: "Pin or unpin the active result" },
  { keys: "t", description: "Focus the result filter" },
  { keys: "Delete", description: "Delete the active result (confirms first)" },
  { keys: "Ctrl/Cmd + Z", description: "Undo the last result deletion" },
  { keys: "Ctrl/Cmd + S", description: "Save the current prompt to the library" },
  { keys: "?", description: "Toggle this cheat sheet" },
  { keys: "Esc", description: "Close cheat sheet or pinned overlays" },
];

// Returns true if the event should be ignored because the user is typing
// into a field. Without this, hitting 'p' while writing a prompt would
// pin the active result instead of inserting the letter.
function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (target.isContentEditable) return true;
  return false;
}

interface HookOptions {
  onToggleCheatSheet: () => void;
}

export function useKeyboardShortcuts({ onToggleCheatSheet }: HookOptions): void {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // '?' wants Shift+/. We accept either the composed '?' key or the
      // raw '/' + shift so different keyboard layouts work.
      const isQuestion = e.key === "?" || (e.key === "/" && e.shiftKey);
      if (isQuestion && !isEditableTarget(e.target)) {
        e.preventDefault();
        onToggleCheatSheet();
        return;
      }

      // Escape: close any overlay via its own listener. We don't duplicate
      // that here; individual consumers (cheat sheet, pinned cells) own
      // their own Esc handling so we don't compete for the event.

      // Undo fires even while typing in a field — users expect Ctrl+Z to
      // restore a just-deleted result regardless of where the focus is,
      // and text fields handle their own undo via the browser already
      // (distinct undo stacks, no conflict).
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && (e.key === "z" || e.key === "Z")) {
        const st = useStore.getState();
        if (st.lastDeleted) {
          e.preventDefault();
          st.undoDelete();
          return;
        }
      }

      // Ctrl/Cmd+S: save the current tab's prompt to the library. Always
      // preempts the browser's "Save Page" dialog — a full-page save from
      // a dev server is never what the user wants. Prompts for a name via
      // window.prompt so the shortcut stays self-contained.
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && (e.key === "s" || e.key === "S")) {
        e.preventDefault();
        const st = useStore.getState();
        const text = st.activeTab === "intervene" ? st.intervenePrompt : st.prompt;
        if (!text.trim()) return;
        // Suggest the first library entry with the same text as a name —
        // an update-under-familiar-name happens most when prompts evolve
        // slightly between sessions.
        const existing = st.promptLibrary.find((p) => p.text === text);
        const name = window.prompt("Save prompt as:", existing?.name ?? "");
        if (name && name.trim()) st.savePrompt(name.trim(), text);
        return;
      }

      if (isEditableTarget(e.target)) return;

      const s = useStore.getState();

      // Tab switching with plain number keys. Modifier guarded so
      // Ctrl+1 / Cmd+1 remain available for browser tab switching.
      if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        if (e.key === "1") { s.setActiveTab("sessions"); e.preventDefault(); return; }
        if (e.key === "2") { s.setActiveTab("probe"); e.preventDefault(); return; }
        if (e.key === "3") { s.setActiveTab("intervene"); e.preventDefault(); return; }
      }

      // Result navigation / annotation actions require a current result.
      const visibleIds = getVisibleResultIds();
      const activeIdx = visibleIds.indexOf(s.activeResultId ?? "");

      if (e.key === "[" && !e.ctrlKey && !e.metaKey) {
        if (visibleIds.length === 0) return;
        const next = activeIdx <= 0 ? visibleIds.length - 1 : activeIdx - 1;
        s.setActiveResult(visibleIds[next]);
        e.preventDefault();
        return;
      }
      if (e.key === "]" && !e.ctrlKey && !e.metaKey) {
        if (visibleIds.length === 0) return;
        const next = activeIdx < 0 || activeIdx >= visibleIds.length - 1 ? 0 : activeIdx + 1;
        s.setActiveResult(visibleIds[next]);
        e.preventDefault();
        return;
      }

      if (e.key === "p" && !e.ctrlKey && !e.metaKey) {
        const id = s.activeResultId;
        if (!id) return;
        // Look up current pinned state on the freshest snapshot so rapid
        // toggle doesn't flip based on stale state.
        const current = [...Object.values(s.pendingResults), ...s.results].find((r) => r.id === id);
        if (!current) return;
        s.updateResultMeta(id, { pinned: !current.pinned });
        e.preventDefault();
        return;
      }

      if (e.key === "t" && !e.ctrlKey && !e.metaKey) {
        // Focus the filter input by label text — keeps the hook decoupled
        // from a ref plumbing. Fallback: no-op if not mounted.
        const input = document.querySelector('input[placeholder^="filter:"]') as HTMLInputElement | null;
        if (input) {
          input.focus();
          input.select();
          e.preventDefault();
        }
        return;
      }

      if (e.key === "Delete" || e.key === "Backspace") {
        // Backspace included for mac keyboards that lack a forward-delete.
        // Guard Backspace tightly: we already excluded editable targets
        // above, but the bare-document backspace still triggers a
        // browser-back on some setups. Only act on Delete OR Shift+Backspace.
        const actDelete = e.key === "Delete" || (e.key === "Backspace" && e.shiftKey);
        if (!actDelete) return;
        const id = s.activeResultId;
        if (!id) return;
        const current = [...Object.values(s.pendingResults), ...s.results].find((r) => r.id === id);
        if (!current) return;
        if (current.pinned) {
          // Pinned results are the researcher's declared favorites —
          // require an extra unpin step rather than a Delete-swallows-all.
          window.alert("This result is pinned — unpin it first (press 'p') before deleting.");
          return;
        }
        if (window.confirm(`Delete this ${current.operation} result?`)) {
          s.deleteResult(id);
        }
        e.preventDefault();
      }
    };

    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onToggleCheatSheet]);
}

// Compute the same visible-id ordering the tab row uses so `[` / `]` match
// the visual list. Kept local to this file to keep the hook self-contained.
function getVisibleResultIds(): string[] {
  const s = useStore.getState();
  const all = [...Object.values(s.pendingResults), ...s.results];
  const head = all.filter((r) => r.operation !== "generate" && !r.isB);
  const q = s.filterQuery.trim().toLowerCase();
  return head.filter((r) => {
    if (r.id in s.pendingResults) return true;
    if (s.filterPinnedOnly && !r.pinned) return false;
    if (s.filterTags.length > 0) {
      const have = new Set(r.tags ?? []);
      for (const t of s.filterTags) if (!have.has(t)) return false;
    }
    if (q) {
      const hay = `${r.sessionName} ${r.prompt} ${r.notes ?? ""} ${(r.tags ?? []).join(" ")}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  }).map((r) => r.id);
}
