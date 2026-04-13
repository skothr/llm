import { useState } from "react";
import { useSessions } from "../hooks/useSessions";
import type { SessionSummary, SurgeryOperation } from "../types/api";

export function SessionSidebar() {
  const {
    sessions, sessionInfo, surgeryOps,
    fetchSessions, fetchSessionInfo,
    deleteSession, applySurgery, undoSurgery, cloneSession,
  } = useSessions();

  const [loadModelId, setLoadModelId] = useState("");
  const [loadName, setLoadName] = useState("");
  const [loadMode, setLoadMode] = useState<"inspect" | "eval">("inspect");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [cloneName, setCloneName] = useState("");
  const [surgeryOp, setSurgeryOp] = useState("");
  const [surgeryParams, setSurgeryParams] = useState("{}");

  const handleLoad = async () => {
    setLoading(true);
    setError("");
    try {
      const resp = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: loadName, model_id: loadModelId, mode: loadMode }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ detail: "Failed to load model" }));
        const detail = body.detail;
        throw new Error(
          typeof detail === "string" ? detail :
          Array.isArray(detail) ? detail.map((d: { msg?: string }) => d.msg || "error").join("; ") :
          "Failed to load model"
        );
      }
      await fetchSessions();
      setLoadModelId("");
      setLoadName("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleClone = async (name: string) => {
    if (!cloneName) return;
    try {
      await cloneSession(name, cloneName);
      setCloneName("");
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleSurgery = async (name: string) => {
    try {
      const params = JSON.parse(surgeryParams);
      await applySurgery(name, surgeryOp, params);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <>
      <div className="sidebar-top">
        <h2>Sessions</h2>

        <div style={{ marginBottom: 12 }}>
          <input
            placeholder="Model ID (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
            value={loadModelId}
            onChange={(e) => setLoadModelId(e.target.value)}
            style={{ marginBottom: 4 }}
          />
          <input
            placeholder="Session name (slug)"
            value={loadName}
            onChange={(e) => setLoadName(e.target.value)}
            style={{ marginBottom: 4 }}
          />
          <div style={{ display: "flex", gap: 4 }}>
            <select value={loadMode} onChange={(e) => setLoadMode(e.target.value as "inspect" | "eval")}>
              <option value="inspect">4-bit (inspect)</option>
              <option value="eval">fp16 (eval)</option>
            </select>
            <button onClick={handleLoad} disabled={loading || !loadModelId || !loadName}>
              {loading ? "Loading..." : "Load"}
            </button>
          </div>
        </div>

        {error && <div style={{ color: "#ff6b6b", fontSize: 12, marginBottom: 8 }}>{error}</div>}

        {sessions.map((s: SessionSummary) => (
          <div
            key={s.name}
            style={{
              padding: 8,
              marginBottom: 4,
              background: selectedSession === s.name ? "#1a5276" : "#0d1b2a",
              borderRadius: 4,
              cursor: "pointer",
            }}
            onClick={() => {
              setSelectedSession(s.name);
              fetchSessionInfo(s.name);
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <strong>{s.name}</strong>
              <button
                onClick={(e) => { e.stopPropagation(); deleteSession(s.name); }}
                style={{ padding: "2px 6px", fontSize: 11 }}
              >
                x
              </button>
            </div>
            <div style={{ fontSize: 11, color: "#8888aa" }}>
              {s.model_id} | {s.mode} | {s.num_layers}L
              {s.has_snapshot && ` | undo: ${s.snapshot_size_mb.toFixed(0)}MB`}
            </div>
          </div>
        ))}
      </div>

      <div className="sidebar-bottom">
        {selectedSession && (
          <>
            <h2>Surgery - {selectedSession}</h2>

            <div style={{ marginBottom: 8 }}>
              <input
                placeholder="Clone name"
                value={cloneName}
                onChange={(e) => setCloneName(e.target.value)}
                style={{ marginBottom: 4 }}
              />
              <button onClick={() => handleClone(selectedSession)} disabled={!cloneName}>
                Clone
              </button>
            </div>

            <div style={{ marginBottom: 8 }}>
              <select value={surgeryOp} onChange={(e) => setSurgeryOp(e.target.value)} style={{ marginBottom: 4 }}>
                <option value="">Select operation...</option>
                {surgeryOps.map((op: SurgeryOperation) => (
                  <option key={op.name} value={op.name}>{op.name}</option>
                ))}
              </select>
              <textarea
                placeholder='Params JSON, e.g. {"layer_indices": [0]}'
                value={surgeryParams}
                onChange={(e) => setSurgeryParams(e.target.value)}
                rows={2}
                style={{ marginBottom: 4 }}
              />
              <div style={{ display: "flex", gap: 4 }}>
                <button onClick={() => handleSurgery(selectedSession)} disabled={!surgeryOp}>
                  Apply
                </button>
                <button
                  onClick={() => undoSurgery(selectedSession)}
                  disabled={!sessions.find((s) => s.name === selectedSession)?.has_snapshot}
                >
                  Undo
                </button>
              </div>
            </div>

            {sessionInfo[selectedSession] && (
              <div style={{ fontSize: 12, color: "#8888aa" }}>
                <div>Layers: {sessionInfo[selectedSession].num_layers}</div>
                <div>Heads: {sessionInfo[selectedSession].num_heads}</div>
                <div>Hidden: {sessionInfo[selectedSession].hidden_size}</div>
                <div>Params: {(sessionInfo[selectedSession].total_params / 1e6).toFixed(1)}M</div>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
