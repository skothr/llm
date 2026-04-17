import { useEffect, useRef, useState } from "react";
import { useStore } from "./state/store";
import { TabBar } from "./components/TabBar";
import { SessionsPanel } from "./components/SessionsPanel";
import { ProbePanel } from "./components/ProbePanel";
import { IntervenePanel } from "./components/IntervenePanel";
import { VisualizationArea } from "./components/VisualizationArea";
import { GenerationOutput } from "./components/GenerationOutput";

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
    // 500 ms keeps the banner alive for at most half a second after the
    // backend comes up. The /api/sessions handler is cheap (tens of μs on
    // an empty manager) so a higher poll rate is fine during development
    // when the backend is flapping. setInterval is cleared as soon as the
    // first success flips backendOnline true.
    const interval = setInterval(() => { fetchSessions(); }, 500);
    return () => clearInterval(interval);
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
    </div>
  );
}
