import { useEffect, useRef, useState, useCallback } from "react";
import { useStore } from "./state/store";
import { TabBar } from "./components/TabBar";
import { ExperimentIO } from "./components/ExperimentIO";
import { SessionsPanel } from "./components/SessionsPanel";
import { ProbePanel } from "./components/ProbePanel";
import { IntervenePanel } from "./components/IntervenePanel";
import { VisualizationArea } from "./components/VisualizationArea";
import { GenerationOutput } from "./components/GenerationOutput";
import { CheatSheet } from "./components/CheatSheet";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";

function useResize(initial: number, direction: "horizontal" | "vertical", min: number, max: number) {
  const [size, setSize] = useState(initial);
  const sizeRef = useRef(initial);

  const onMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    const startPos = direction === "horizontal" ? e.clientX : e.clientY;
    const startSize = sizeRef.current;

    const onMove = (ev: MouseEvent) => {
      const pos = direction === "horizontal" ? ev.clientX : ev.clientY;
      const delta = pos - startPos;
      const newSize = Math.max(min, Math.min(max, direction === "horizontal" ? startSize + delta : startSize - delta));
      sizeRef.current = newSize;
      setSize(newSize);
    };

    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    document.body.style.cursor = direction === "horizontal" ? "col-resize" : "row-resize";
    document.body.style.userSelect = "none";
  };

  return { size, onMouseDown };
}

export default function App() {
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSurgeryOps = useStore((s) => s.fetchSurgeryOps);
  const fetchAvailableModels = useStore((s) => s.fetchAvailableModels);
  const activeTab = useStore((s) => s.activeTab);
  const backendOnline = useStore((s) => s.backendOnline);

  const { size: panelWidth, onMouseDown: onHResize } = useResize(320, "horizontal", 200, 600);
  const { size: outputHeight, onMouseDown: onVResize } = useResize(280, "vertical", 60, 600);

  const [cheatSheetOpen, setCheatSheetOpen] = useState(false);
  const toggleCheatSheet = useCallback(() => setCheatSheetOpen((v) => !v), []);
  useKeyboardShortcuts({ onToggleCheatSheet: toggleCheatSheet });

  useEffect(() => {
    // All three initial fetches run concurrently. Model discovery is the
    // slowest (first-hit GGUF parsing), so starting it here — rather than
    // waiting for SessionsPanel to mount — removes visible dropdown lag.
    fetchSessions();
    fetchSurgeryOps();
    fetchAvailableModels();
  }, [fetchSessions, fetchSurgeryOps, fetchAvailableModels]);

  useEffect(() => {
    if (backendOnline) return;
    // Exponential backoff: 500 ms, then 1 s, 2 s, 4 s, capped at 10 s. A
    // brief restart closes in the first couple of polls; a longer outage
    // stops re-rendering every half second. The effect re-runs when
    // backendOnline flips true and the timeout is cleared via cleanup.
    let cancelled = false;
    let delay = 500;
    let timer: ReturnType<typeof setTimeout>;
    const poll = () => {
      if (cancelled) return;
      fetchSessions();
      delay = Math.min(delay * 2, 10_000);
      timer = setTimeout(poll, delay);
    };
    timer = setTimeout(poll, delay);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [backendOnline, fetchSessions]);

  useEffect(() => {
    // On transition offline → online (including first probe after backend
    // starts late), refresh data that may have failed silently during the
    // down period. fetchSurgeryOps is refetched by fetchSessions itself when
    // surgeryOps is empty; availableModels is independent so handle it here.
    if (!backendOnline) return;
    fetchAvailableModels();
  }, [backendOnline, fetchAvailableModels]);

  return (
    <div className="app-layout">
      <aside className="config-panel" style={{ width: panelWidth }}>
        <TabBar />
        <ExperimentIO />
        <div className="config-content">
          <div style={{ display: activeTab === "sessions" ? "contents" : "none" }}><SessionsPanel /></div>
          <div style={{ display: activeTab === "probe" ? "contents" : "none" }}><ProbePanel /></div>
          <div style={{ display: activeTab === "intervene" ? "contents" : "none" }}><IntervenePanel /></div>
        </div>
      </aside>
      <div className="resize-handle horizontal" onMouseDown={onHResize} />
      <main className="main-area">
        <div className="visualization-area">
          <VisualizationArea />
        </div>
        <div className="resize-handle vertical" onMouseDown={onVResize} />
        <div className="generation-output" style={{ height: outputHeight }}>
          <GenerationOutput />
        </div>
      </main>
      <CheatSheet open={cheatSheetOpen} onClose={() => setCheatSheetOpen(false)} />
      <button
        onClick={toggleCheatSheet}
        title="Keyboard shortcuts (?)"
        style={{
          position: "fixed", right: 8, bottom: 8, zIndex: 400,
          width: 24, height: 24, padding: 0, fontSize: 13,
          background: "#0f1626", border: "1px solid #1a2540",
          color: "#8888aa", borderRadius: 12, cursor: "pointer",
        }}
      >?</button>
    </div>
  );
}
