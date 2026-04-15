import { useState, useEffect } from "react";
import { useStore } from "../state/store";
import { ModelCombobox } from "./ModelCombobox";
import { SurgeryParamForm } from "./SurgeryParamForm";
import type { SessionSummary, SurgeryOperation, StagedOp } from "../types/api";

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
  const commitSurgery = useStore((s) => s.commitSurgery);
  const revertSurgery = useStore((s) => s.revertSurgery);
  const deleteStagedOp = useStore((s) => s.deleteStagedOp);
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

  const formatParams = (params: Record<string, unknown>) => {
    return Object.entries(params)
      .map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.join(",")}]` : v}`)
      .join(" ");
  };

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
    try {
      await applySurgery(name, surgeryOp, surgeryParams);
      setSurgeryOp("");
    } catch (e) { setError((e as Error).message); }
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
            {s.model_id} | {s.mode} | {s.num_layers}L | <span style={{ color: s.device.startsWith("cuda") ? "#4ecdc4" : "#aa8844" }}>{s.device}</span>
            {s.pending_count > 0 && <span style={{ color: "#f0ad4e" }}> | {s.pending_count} staged</span>}
            {s.applied_count > 0 && <span style={{ color: "#5cb85c" }}> | {s.applied_count} applied</span>}
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
            {selectedOpDef && sessionInfo[selectedSession] && (
              <div style={{ fontSize: 10, color: "#666", marginBottom: 2 }}>
                Original layers: 0–{sessionInfo[selectedSession].original_num_layers - 1}
              </div>
            )}
            {selectedOpDef && (
              <SurgeryParamForm operation={selectedOpDef} params={surgeryParams} onChange={setSurgeryParams} />
            )}
            <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
              <button onClick={() => handleSurgery(selectedSession)} disabled={!surgeryOp}>Stage</button>
            </div>
          </div>

          {sessionInfo[selectedSession] && (() => {
            const info = sessionInfo[selectedSession];
            const hasPending = info.pending_ops?.length > 0;
            const hasApplied = info.applied_ops?.length > 0;
            return (
              <>
                {hasPending && (
                  <div style={{ marginBottom: 8, padding: 6, background: "#1a1a2e", borderRadius: 4, border: "1px solid #f0ad4e44" }}>
                    <div style={{ fontSize: 11, color: "#f0ad4e", marginBottom: 4, fontWeight: 600 }}>
                      Pending ({info.pending_ops.length})
                    </div>
                    {info.pending_ops.map((op: StagedOp, i: number) => (
                      <div key={i} style={{ fontSize: 11, color: "#ccccdd", padding: "2px 0", display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: i < info.pending_ops.length - 1 ? "1px solid #ffffff0a" : "none" }}>
                        <span>
                          <span style={{ color: "#88aacc" }}>{i + 1}.</span>{" "}
                          <span style={{ color: "#ddd" }}>{op.operation}</span>{" "}
                          <span style={{ color: "#888" }}>{formatParams(op.params)}</span>
                        </span>
                        <button
                          style={{ padding: "1px 5px", fontSize: 10, background: "#4a2020", border: "1px solid #ff6b6b44", cursor: "pointer" }}
                          onClick={async (e) => {
                            e.stopPropagation();
                            try { await deleteStagedOp(selectedSession, i); }
                            catch (err) { setError((err as Error).message); }
                          }}
                        >x</button>
                      </div>
                    ))}
                    <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
                      <button
                        style={{ background: "#2e7d32", border: "1px solid #4caf50", fontSize: 11, padding: "3px 10px" }}
                        onClick={async () => {
                          try { await commitSurgery(selectedSession); }
                          catch (e) { setError((e as Error).message); }
                        }}
                      >Commit All</button>
                    </div>
                  </div>
                )}

                {hasApplied && (
                  <div style={{ marginBottom: 8, padding: 6, background: "#1a1a2e", borderRadius: 4, border: "1px solid #5cb85c44" }}>
                    <div style={{ fontSize: 11, color: "#5cb85c", marginBottom: 4, fontWeight: 600 }}>
                      Applied ({info.applied_ops.length})
                    </div>
                    {info.applied_ops.map((op: StagedOp, i: number) => (
                      <div key={i} style={{ fontSize: 11, color: "#99aa99", padding: "2px 0" }}>
                        <span style={{ color: "#6b8f6b" }}>{i + 1}.</span>{" "}
                        {op.operation} <span style={{ color: "#666" }}>{formatParams(op.params)}</span>
                      </div>
                    ))}
                    <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
                      <button
                        style={{ background: "#6d3a00", border: "1px solid #e67700", fontSize: 11, padding: "3px 10px" }}
                        onClick={async () => {
                          try { await revertSurgery(selectedSession); }
                          catch (e) { setError((e as Error).message); }
                        }}
                      >Revert to Clean</button>
                    </div>
                  </div>
                )}

                <div style={{ fontSize: 12, color: "#8888aa" }}>
                  <div>Params: {(info.total_params / 1e6).toFixed(1)}M | Layers: {info.num_layers} | Heads: {info.num_heads}{info.num_kv_heads ? `/${info.num_kv_heads}kv` : ""}</div>
                  <div>Hidden: {info.hidden_size}{info.intermediate_size ? ` | FFN: ${info.intermediate_size}` : ""}{info.vocab_size ? ` | Vocab: ${info.vocab_size.toLocaleString()}` : ""}</div>
                  {info.max_position_embeddings && <div>Max pos: {info.max_position_embeddings}{info.rope_theta ? ` | RoPE θ: ${info.rope_theta.toLocaleString()}` : ""}</div>}
                  {info.bos_token && <div>BOS: {info.bos_token} | EOS: {info.eos_token}</div>}
                  {info.chat_template && (
                    <details style={{ marginTop: 4 }}>
                      <summary style={{ cursor: "pointer", color: "#6688aa" }}>Chat template</summary>
                      <pre style={{ marginTop: 4, padding: 6, background: "#0d1b2a", borderRadius: 4, whiteSpace: "pre-wrap", fontSize: 11, maxHeight: 120, overflowY: "auto" }}>
                        {info.chat_template}
                      </pre>
                    </details>
                  )}
                </div>
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}
