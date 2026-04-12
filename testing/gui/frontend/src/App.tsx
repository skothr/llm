import { useEffect } from "react";
import { useStore } from "./state/store";
import { SessionSidebar } from "./components/SessionSidebar";
import { ProbeConfig } from "./components/ProbeConfig";

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
          <h2>Visualization</h2>
          <p style={{ color: "#666" }}>Run a probe to see results here</p>
        </div>
        <div className="generation-output">
          <h2>Generation Output</h2>
          <p style={{ color: "#666" }}>Run generate to see token stream here</p>
        </div>
      </main>
    </div>
  );
}
