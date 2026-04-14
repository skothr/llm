import { useState, useEffect } from "react";
import { useStore } from "../state/store";
import { ModelCombobox } from "./ModelCombobox";
import { SurgeryParamForm } from "./SurgeryParamForm";
import type { SessionSummary, SurgeryOperation } from "../types/api";

export function SessionsPanel() {
  const sessions = useStore((s) => s.sessions);
  const sessionInfo = useStore((s) => s.sessionInfo);
  const surgeryOps = useStore((s) => s.surgeryOps);
  const isLoadingModel = useStore((s) => s.isLoadingModel);
  const loadingModelId = useStore((s) => s.loadingModelId);
  const availableModels = useStore((s) => s.availableModels);
  const backendOnline = useStore((s) => s.backendOnline);
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSessionInfo = useStore((s) => s.fetchSessionInfo);
  const fetchAvailableModels = useStore((s) => s.fetchAvailableModels);
  const deleteSession = useStore((s) => s.deleteSession);
  const applySurgery = useStore((s) => s.applySurgery);
  const undoSurgery = useStore((s) => s.undoSurgery);
  const cloneSession = useStore((s) => s.cloneSession);

  const [loadModelId, setLoadModelId] = useState("");
  const [loadName, setLoadName] = useState("");
  const [loadMode, setLoadMode] = useState<"inspect" | "eval">("inspect");
  const [error, setError] = useState("");

  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [cloneName, setCloneName] = useState("");
  const [surgeryOp, setSurgeryOp] = useState("");
  const [surgeryParams, setSurgeryParams] = useState<Record<string, unknown>>({});

  useEffect(() => { fetchAvailableModels(); }, [fetchAvailableModels]);

  const selectedOpDef = surgeryOps.find((op) => op.name === surgeryOp) || null;

  useEffect(() => {
    if (!selectedOpDef) { setSurgeryParams({}); return; }
    const defaults: Record<string, unknown> = {};
    for (const [key, schema] of Object.entries(selectedOpDef.params)) {
      const s = typeof schema === "string" ? { type: schema } : schema;
      if (s.type === "array") defaults[key] = [];
      else if (s.type === "float") defaults[key] = 0.0;
      else defaults[key] = 0;
    }
    setSurgeryParams(defaults);
  }, [surgeryOp, selectedOpDef]);

  const handleLoad = async () => {
    setError("");
    useStore.setState({ isLoadingModel: true, loadingModelId: loadModelId });
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
      await fetchAvailableModels();
      setLoadModelId("");
      setLoadName("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      useStore.setState({ isLoadingModel: false, loadingModelId: null });
    }
  };

  const handleClone = async (name: string) => {
    if (!cloneName) return;
    try { await cloneSession(name, cloneName); setCloneName(""); }
    catch (e) { setError((e as Error).message); }
  };

  const handleSurgery = async (name: string) => {
    try { await applySurgery(name, surgeryOp, surgeryParams); }
    catch (e) { setError((e as Error).message); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {!backendOnline && (
        <div style={{ padding: 8, background: "#4a2020", borderRadius: 4, fontSize: 12 }}>
          Backend offline — start with <code>./gui/run.sh</code>
        </div>
      )}

      <div>
        <ModelCombobox value={loadModelId} onChange={setLoadModelId} />
        <input
          placeholder="Session name (slug)"
          value={loadName}
          onChange={(e) => setLoadName(e.target.value)}
          style={{ marginTop: 4, marginBottom: 4 }}
        />
        <div style={{ display: "flex", gap: 4 }}>
          <select value={loadMode} onChange={(e) => setLoadMode(e.target.value as "inspect" | "eval")}>
            <option value="inspect">4-bit (inspect)</option>
            <option value="eval">fp16 (eval)</option>
          </select>
          <button onClick={handleLoad} disabled={isLoadingModel || !loadModelId || !loadName}>
            {isLoadingModel ? "Loading..." : "Load"}
          </button>
        </div>
        {isLoadingModel && (
          <div style={{ fontSize: 12, color: "#8888aa", marginTop: 4 }}>
            {availableModels.includes(loadModelId) ? `Loading ${loadingModelId} from cache...` : `Downloading ${loadingModelId}...`}
          </div>
        )}
      </div>

      {error && <div style={{ color: "#ff6b6b", fontSize: 12 }}>{error}</div>}

      {sessions.map((s: SessionSummary) => (
        <div
          key={s.name}
          style={{
            padding: 8,
            background: selectedSession === s.name ? "#1a5276" : "#0d1b2a",
            borderRadius: 4,
            cursor: "pointer",
          }}
          onClick={() => { setSelectedSession(s.name); fetchSessionInfo(s.name); }}
        >
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <strong>{s.name}</strong>
            <button
              onClick={(e) => { e.stopPropagation(); deleteSession(s.name); }}
              style={{ padding: "2px 6px", fontSize: 11 }}
            >x</button>
          </div>
          <div style={{ fontSize: 11, color: "#8888aa" }}>
            {s.model_id} | {s.mode} | {s.num_layers}L
            {s.has_snapshot && ` | undo: ${s.snapshot_size_mb.toFixed(0)}MB`}
          </div>
        </div>
      ))}

      {selectedSession && (
        <div style={{ borderTop: "1px solid #0f3460", paddingTop: 8 }}>
          <h3 style={{ fontSize: 12, color: "#a0a0c0", marginBottom: 8 }}>Surgery - {selectedSession}</h3>

          <div style={{ marginBottom: 8 }}>
            <input placeholder="Clone name" value={cloneName} onChange={(e) => setCloneName(e.target.value)} style={{ marginBottom: 4 }} />
            <button onClick={() => handleClone(selectedSession)} disabled={!cloneName}>Clone</button>
          </div>

          <div style={{ marginBottom: 8 }}>
            <select value={surgeryOp} onChange={(e) => setSurgeryOp(e.target.value)} style={{ marginBottom: 4 }}>
              <option value="">Select operation...</option>
              {surgeryOps.map((op: SurgeryOperation) => (
                <option key={op.name} value={op.name}>{op.name}</option>
              ))}
            </select>
            {selectedOpDef && (
              <SurgeryParamForm operation={selectedOpDef} params={surgeryParams} onChange={setSurgeryParams} />
            )}
            <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
              <button onClick={() => handleSurgery(selectedSession)} disabled={!surgeryOp}>Apply</button>
              <button
                onClick={() => undoSurgery(selectedSession)}
                disabled={!sessions.find((s) => s.name === selectedSession)?.has_snapshot}
              >Undo</button>
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
        </div>
      )}
    </div>
  );
}
