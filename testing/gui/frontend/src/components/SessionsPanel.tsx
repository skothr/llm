import { useState, useEffect } from "react";
import { useStore } from "../state/store";
import { ModelCombobox } from "./ModelCombobox";
import { SurgeryParamForm } from "./SurgeryParamForm";
import type { SessionSummary, SurgeryOperation, StagedOp, AvailableModel } from "../types/api";

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
  const [loadMode, setLoadMode] = useState("auto");
  const [error, setError] = useState("");

  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [cloneName, setCloneName] = useState("");
  const [surgeryOp, setSurgeryOp] = useState("");
  const [surgeryParams, setSurgeryParams] = useState<Record<string, unknown>>({});

  useEffect(() => { fetchAvailableModels(); }, [fetchAvailableModels]);

  const selectedModel: AvailableModel | null =
    availableModels.find((m) => m.model_id === loadModelId) ?? null;

  const selectedOpDef = surgeryOps.find((op) => op.name === surgeryOp) || null;

  const fmtSize = (bytes: number) =>
    bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${(bytes / 1e6).toFixed(0)} MB`;

  const DTYPE_LABELS: Record<string, string> = {
    bfloat16: "BF16", float16: "FP16", float32: "FP32",
    "torch.bfloat16": "BF16", "torch.float16": "FP16", "torch.float32": "FP32",
  };
  const fmtDtype = (m: AvailableModel): string | null => {
    if (m.quantization) return m.quantization;
    if (m.dtype) return DTYPE_LABELS[m.dtype] ?? m.dtype;
    return null;
  };

  const resolveAutoMode = (m: AvailableModel | null): string => {
    if (!m) return "fp16";
    if (m.source === "ollama") {
      const q = m.quantization?.toUpperCase() ?? "";
      if (q === "F32") return "fp32";
      if (q === "F16") return "fp16";
      return "bf16";
    }
    const dt = m.dtype;
    if (dt === "bfloat16" || dt === "torch.bfloat16") return "bf16";
    if (dt === "float32" || dt === "torch.float32") return "fp32";
    return "fp16";
  };

  const effectiveMode = loadMode === "auto" ? resolveAutoMode(selectedModel) : loadMode;

  const getParams = (m: AvailableModel): number | null => {
    if (m.total_params) return m.total_params;
    const { num_layers, hidden_size, num_heads, num_kv_heads, intermediate_size, vocab_size } = m;
    if (num_layers == null || hidden_size == null || vocab_size == null) return null;
    const h = hidden_size;
    const kvDim = (num_kv_heads ?? num_heads ?? 1) * (h / (num_heads ?? 1));
    const ffn = intermediate_size ?? Math.round(h * 2.67);
    const perLayer = 2 * h * h + 2 * h * kvDim + 3 * h * ffn + 2 * h;
    return 2 * vocab_size * h + num_layers * perLayer + h;
  };

  const BYTES_PER_PARAM: Record<string, number> = {
    nf4: 0.55, int8: 1.1, bf16: 2, fp16: 2, fp32: 4, "fp32-cpu": 4,
  };

  const estimateLoadedBytes = (m: AvailableModel, mode: string): number | null => {
    const params = getParams(m);
    if (params == null) return null;
    return Math.round(params * (BYTES_PER_PARAM[mode] ?? 2));
  };

  const formatParams = (params: Record<string, unknown>) => {
    return Object.entries(params)
      .map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.join(",")}]` : v}`)
      .join(" ");
  };

  useEffect(() => {
    if (selectedSession && !sessions.some((s) => s.name === selectedSession)) {
      setSelectedSession(null);
    }
  }, [sessions, selectedSession]);

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
        body: JSON.stringify({ name: loadName, model_id: loadModelId, mode: effectiveMode }),
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
          <select value={loadMode} onChange={(e) => setLoadMode(e.target.value)}>
            <option value="auto">Default{selectedModel ? ` (${resolveAutoMode(selectedModel).toUpperCase()})` : ""}</option>
            <option value="nf4">NF4 (4-bit)</option>
            <option value="int8">INT8 (8-bit)</option>
            <option value="bf16">BF16</option>
            <option value="fp16">FP16</option>
            <option value="fp32">FP32</option>
          </select>
          <button onClick={handleLoad} disabled={isLoadingModel || !loadModelId || !loadName}>
            {isLoadingModel ? "Loading..." : "Load"}
          </button>
        </div>
        {isLoadingModel && (
          <div style={{ fontSize: 12, color: "#8888aa", marginTop: 4 }}>
            {availableModels.some((m) => m.model_id === loadModelId) ? `Loading ${loadingModelId} from cache...` : `Downloading ${loadingModelId}...`}
          </div>
        )}
      </div>

      {selectedModel && (
        <div style={{
          padding: 8,
          background: "#0d1b2a",
          borderRadius: 4,
          border: "1px solid #1a527644",
          fontSize: 11,
          color: "#8888aa",
          lineHeight: 1.6,
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 2 }}>
            <span style={{ color: "#a0a0c0", fontWeight: 600, fontSize: 12 }}>
              {selectedModel.architecture ?? "unknown"}{selectedModel.model_size_label ? ` (${selectedModel.model_size_label})` : ""}
            </span>
            <span style={{
              fontSize: 10, fontWeight: 700, padding: "1px 6px", borderRadius: 3,
              background: selectedModel.quantization ? "#7c6cf022" : "#ff9d0022",
              color: selectedModel.quantization ? "#b8aaff" : "#ffcc66",
              border: `1px solid ${selectedModel.quantization ? "#7c6cf044" : "#ff9d0044"}`,
            }}>
              {fmtDtype(selectedModel) ?? "unknown"}
            </span>
          </div>
          <div>
            {selectedModel.num_layers != null && <span>Layers: {selectedModel.num_layers}</span>}
            {selectedModel.hidden_size != null && <span> | Hidden: {selectedModel.hidden_size}</span>}
            {selectedModel.num_heads != null && (
              <span> | Heads: {selectedModel.num_heads}{selectedModel.num_kv_heads && selectedModel.num_kv_heads !== selectedModel.num_heads ? `/${selectedModel.num_kv_heads}kv` : ""}</span>
            )}
          </div>
          <div>
            {selectedModel.vocab_size != null && <span>Vocab: {selectedModel.vocab_size.toLocaleString()}</span>}
            {selectedModel.intermediate_size != null && <span> | FFN: {selectedModel.intermediate_size}</span>}
            {selectedModel.max_position_embeddings != null && <span> | Ctx: {selectedModel.max_position_embeddings.toLocaleString()}</span>}
          </div>
          {(() => {
            const params = getParams(selectedModel);
            const loaded = estimateLoadedBytes(selectedModel, effectiveMode);
            const exact = !!selectedModel.total_params;
            const bpw = selectedModel.bits_per_weight;
            return (
              <div>
                {selectedModel.file_size_bytes != null && <span>Disk: {fmtSize(selectedModel.file_size_bytes)}</span>}
                {selectedModel.total_bytes != null && selectedModel.total_bytes !== selectedModel.file_size_bytes && (
                  <span>{selectedModel.file_size_bytes != null ? " " : ""}({fmtSize(selectedModel.total_bytes)} weights)</span>
                )}
                {selectedModel.safetensors && <span> | safetensors</span>}
                {bpw != null && <span> | {bpw} bpw</span>}
                {params != null && (
                  <span>
                    {(selectedModel.file_size_bytes != null || bpw != null) ? " | " : ""}
                    {exact ? "" : "~"}{(params / 1e9).toFixed(2)}B params
                  </span>
                )}
                {loaded != null && (
                  <span> | <span style={{ color: loaded > 7.5e9 ? "#ff6b6b" : loaded > 4e9 ? "#f0ad4e" : "#5cb85c" }}>
                    ~{fmtSize(loaded)} loaded
                  </span>{" "}
                    <span style={{ color: "#666" }}>
                      ({(() => {
                        const src = fmtDtype(selectedModel);
                        const tgt = effectiveMode.toUpperCase();
                        if (!src) return tgt;
                        return src.toUpperCase() === tgt.toUpperCase() ? tgt : `${src}\u2192${tgt}`;
                      })()})
                    </span>
                  </span>
                )}
              </div>
            );
          })()}
        </div>
      )}

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
              onClick={async (e) => {
                e.stopPropagation();
                try { await deleteSession(s.name); }
                catch (err) { setError((err as Error).message); }
              }}
              style={{ padding: "2px 6px", fontSize: 11 }}
            >x</button>
          </div>
          <div style={{ fontSize: 11, color: "#8888aa" }}>
            {s.model_id} | {s.mode} | {s.num_layers}L | <span style={{ color: s.device.startsWith("cuda") ? "#4ecdc4" : "#aa8844" }}>{s.device}</span>
            {s.pending_count > 0 && <span style={{ color: "#f0ad4e" }}> | {s.pending_count} staged</span>}
            {s.applied_count > 0 && <span style={{ color: "#5cb85c" }}> | {s.applied_count} applied</span>}
            {s.engine_llama && (
              <span style={{ color: s.dirty ? "#f0ad4e" : "#4ecdc4", fontSize: 10, marginLeft: 4 }}>
                llama.cpp{s.dirty ? " [stale]" : ""}
              </span>
            )}
            {s.engine_pytorch && (
              <span style={{ color: "#8888cc", fontSize: 10, marginLeft: 4 }}>
                PyTorch{s.dirty ? " [modified]" : ""}
              </span>
            )}
            {s.engine_llama && !s.engine_pytorch && (
              <span style={{ color: "#666", fontSize: 10, marginLeft: 4 }}>
                PyTorch [deferred]
              </span>
            )}
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
