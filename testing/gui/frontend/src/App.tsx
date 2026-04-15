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
  const activeTab = useStore((s) => s.activeTab);
  const backendOnline = useStore((s) => s.backendOnline);

  const { size: panelWidth, onMouseDown: onHResize } = useResize(320, "horizontal", 200, 600);
  const { size: outputHeight, onMouseDown: onVResize } = useResize(280, "vertical", 60, 600);

  useEffect(() => {
    fetchSessions();
    fetchSurgeryOps();
  }, [fetchSessions, fetchSurgeryOps]);

  useEffect(() => {
    if (backendOnline) return;
    const interval = setInterval(() => { fetchSessions(); }, 5000);
    return () => clearInterval(interval);
  }, [backendOnline, fetchSessions]);

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
