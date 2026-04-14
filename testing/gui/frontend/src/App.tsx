import { useEffect } from "react";
import { useStore } from "./state/store";
import { TabBar } from "./components/TabBar";
import { SessionsPanel } from "./components/SessionsPanel";
import { ProbePanel } from "./components/ProbePanel";
import { IntervenePanel } from "./components/IntervenePanel";
import { VisualizationArea } from "./components/VisualizationArea";
import { GenerationOutput } from "./components/GenerationOutput";

export default function App() {
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSurgeryOps = useStore((s) => s.fetchSurgeryOps);
  const activeTab = useStore((s) => s.activeTab);
  const backendOnline = useStore((s) => s.backendOnline);

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
      <aside className="config-panel">
        <TabBar />
        <div className="config-content">
          {activeTab === "sessions" && <SessionsPanel />}
          {activeTab === "probe" && <ProbePanel />}
          {activeTab === "intervene" && <IntervenePanel />}
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
