import { useEffect } from "react";
import { useStore } from "./state/store";
import { SessionSidebar } from "./components/SessionSidebar";
import { ProbeConfig } from "./components/ProbeConfig";
import { VisualizationArea } from "./components/VisualizationArea";
import { GenerationOutput } from "./components/GenerationOutput";

export default function App() {
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSurgeryOps = useStore((s) => s.fetchSurgeryOps);

  useEffect(() => {
    fetchSessions();
    fetchSurgeryOps();
  }, [fetchSessions, fetchSurgeryOps]);

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <SessionSidebar />
        <div style={{ padding: 16, borderTop: "1px solid #0f3460" }}>
          <ProbeConfig />
        </div>
      </aside>
      <main className="main-area">
        <div className="visualization-area">
          <VisualizationArea />
        </div>
        <div className="generation-output">
          <GenerationOutput />
        </div>
      </main>
    </div>
  );
}
