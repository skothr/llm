import { useRef, useState } from "react";
import { useStore } from "../state/store";
import { useWebSocket } from "../hooks/useWebSocket";
import { ArrayInput } from "./SurgeryParamForm";
import type { InterventionSpec, WsMessage } from "../types/api";

const num = (v: string, fallback: number): number => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};

type OpParam = { key: string; type: string; default?: number | string; step?: number };

const INTERVENTION_OPS: { name: string; params: OpParam[] }[] = [
  { name: "scale", params: [{ key: "factor", type: "float", default: 1.0, step: 0.1 }] },
  { name: "zero_dims", params: [{ key: "dims", type: "int[]", default: "" }] },
  { name: "clamp", params: [{ key: "min", type: "float", default: -1.0, step: 0.1 }, { key: "max", type: "float", default: 1.0, step: 0.1 }] },
  { name: "noise", params: [{ key: "std", type: "float", default: 0.1, step: 0.01 }] },
  { name: "replace", params: [{ key: "source", type: "source" }] },
  { name: "project_out", params: [{ key: "source", type: "source" }] },
];

function defaultParamsForOp(opName: string): Record<string, unknown> {
  const opDef = INTERVENTION_OPS.find((o) => o.name === opName) || INTERVENTION_OPS[0];
  const params: Record<string, unknown> = {};
  for (const p of opDef.params) {
    if (p.type === "source") params[p.key] = { session: "", prompt: "", layer: 0, sublayer: "ffn", position: 0 };
    else if (p.type === "int[]") params[p.key] = [];
    else params[p.key] = p.default;
  }
  return params;
}

function InterventionCard({
  spec, index, sessions, onUpdate, onRemove,
}: {
  spec: InterventionSpec;
  index: number;
  sessions: { name: string }[];
  onUpdate: (index: number, spec: InterventionSpec) => void;
  onRemove: (index: number) => void;
}) {
  const opDef = INTERVENTION_OPS.find((o) => o.name === spec.op) || INTERVENTION_OPS[0];

  const updateParam = (key: string, value: unknown) => {
    onUpdate(index, { ...spec, params: { ...spec.params, [key]: value } });
  };

  return (
    <div style={{ padding: 8, background: "#0d1b2a", borderRadius: 4, marginBottom: 4 }}>
      <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
        <label style={{ fontSize: 12 }}>
          L<input type="number" value={spec.layer} onChange={(e) => onUpdate(index, { ...spec, layer: num(e.target.value, spec.layer) })} style={{ width: 52 }} />
        </label>
        <select value={spec.sublayer} onChange={(e) => onUpdate(index, { ...spec, sublayer: e.target.value as "attn" | "ffn" })} style={{ width: 60 }}>
          <option value="ffn">FFN</option>
          <option value="attn">Attn</option>
        </select>
        <select value={spec.op} onChange={(e) => onUpdate(index, { ...spec, op: e.target.value, params: defaultParamsForOp(e.target.value) })} style={{ flex: 1 }}>
          {INTERVENTION_OPS.map((o) => <option key={o.name} value={o.name}>{o.name}</option>)}
        </select>
        <button onClick={() => onRemove(index)} style={{ padding: "2px 6px", fontSize: 11 }}>x</button>
      </div>

      {opDef.params.map((p) => {
        if (p.type === "float") {
          return (
            <label key={p.key} style={{ display: "block", fontSize: 12, marginBottom: 2 }}>
              {p.key}: <input type="number" step={p.step} value={Number(spec.params[p.key] ?? p.default)} onChange={(e) => updateParam(p.key, num(e.target.value, Number(spec.params[p.key] ?? p.default ?? 0)))} style={{ width: 80 }} />
            </label>
          );
        }
        if (p.type === "int[]") {
          return (
            <label key={p.key} style={{ display: "block", fontSize: 12, marginBottom: 2 }}>
              {p.key}: <ArrayInput
                value={Array.isArray(spec.params[p.key]) ? (spec.params[p.key] as number[]) : []}
                onChange={(next) => updateParam(p.key, next)}
                style={{ width: 120 }}
              />
            </label>
          );
        }
        if (p.type === "source") {
          const source = (spec.params[p.key] || { session: "", prompt: "", layer: 0, sublayer: "ffn", position: 0 }) as {
            session: string; prompt: string; layer: number; sublayer: string; position: number;
          };
          const updateSource = (field: string, value: unknown) => updateParam(p.key, { ...source, [field]: value });
          return (
            <div key={p.key} style={{ fontSize: 12, paddingLeft: 8, borderLeft: "2px solid #1a5276", marginTop: 4 }}>
              <label style={{ display: "block", marginBottom: 2 }}>
                session: <select value={source.session} onChange={(e) => updateSource("session", e.target.value)} style={{ width: "100%" }}>
                  <option value="">Select...</option>
                  {sessions.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
                </select>
              </label>
              <label style={{ display: "block", marginBottom: 2 }}>
                prompt: <input value={source.prompt} onChange={(e) => updateSource("prompt", e.target.value)} />
              </label>
              <div style={{ display: "flex", gap: 4 }}>
                <label>L<input type="number" value={source.layer} onChange={(e) => updateSource("layer", num(e.target.value, source.layer))} style={{ width: 52 }} /></label>
                <select value={source.sublayer} onChange={(e) => updateSource("sublayer", e.target.value)} style={{ width: 60 }}>
                  <option value="ffn">FFN</option>
                  <option value="attn">Attn</option>
                </select>
                <label>pos<input type="number" value={source.position} onChange={(e) => updateSource("position", num(e.target.value, source.position))} style={{ width: 52 }} /></label>
              </div>
            </div>
          );
        }
        return null;
      })}
    </div>
  );
}

export function IntervenePanel() {
  const sessions = useStore((s) => s.sessions);
  const interventionSpecs = useStore((s) => s.interventionSpecs);
  const captureLogitLens = useStore((s) => s.captureLogitLens);
  const intervenePrompt = useStore((s) => s.intervenePrompt);
  const interveneSession = useStore((s) => s.interveneSession);
  const isRunning = useStore((s) => s.isRunning);
  const addIntervention = useStore((s) => s.addIntervention);
  const removeIntervention = useStore((s) => s.removeIntervention);
  const updateIntervention = useStore((s) => s.updateIntervention);
  const setCaptureLogitLens = useStore((s) => s.setCaptureLogitLens);
  const setIntervenePrompt = useStore((s) => s.setIntervenePrompt);
  const setInterveneSession = useStore((s) => s.setInterveneSession);
  const setRunning = useStore((s) => s.setRunning);
  const pendingResults = useStore((s) => s.pendingResults);
  const setPendingResult = useStore((s) => s.setPendingResult);
  const updatePendingResult = useStore((s) => s.updatePendingResult);
  const finalizePendingResult = useStore((s) => s.finalizePendingResult);
  const removePendingResult = useStore((s) => s.removePendingResult);

  const { connect, cancelAll } = useWebSocket();
  const localPendingIdsRef = useRef<Set<string>>(new Set());
  const [error, setError] = useState("");

  const handleRun = () => {
    if (!interveneSession || interventionSpecs.length === 0) return;
    setError("");
    setRunning(true);

    const resultId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    localPendingIdsRef.current.add(resultId);

    setPendingResult(resultId, {
      id: resultId,
      operation: "intervene",
      sessionName: interveneSession,
      prompt: intervenePrompt,
      data: [],
      timestamp: Date.now(),
    });

    const clearLocal = () => localPendingIdsRef.current.delete(resultId);

    connect(resultId, `/ws/sessions/${interveneSession}/intervene`, {
      prompt: intervenePrompt,
      interventions: interventionSpecs,
      capture_logit_lens: captureLogitLens,
    }, {
      onMessage: (msg: WsMessage) => { updatePendingResult(resultId, msg); },
      onComplete: (msg: WsMessage) => { finalizePendingResult(resultId, msg); clearLocal(); setRunning(false); },
      onError: (message: string) => { finalizePendingResult(resultId); clearLocal(); setError(message); setRunning(false); },
      onDisconnect: () => { finalizePendingResult(resultId); clearLocal(); setError("Connection lost"); setRunning(false); },
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <textarea
        placeholder="Prompt text..."
        value={intervenePrompt}
        onChange={(e) => setIntervenePrompt(e.target.value)}
        rows={3}
      />

      <select value={interveneSession} onChange={(e) => setInterveneSession(e.target.value)}>
        <option value="">Select session...</option>
        {sessions.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
      </select>

      <div style={{ maxHeight: 300, overflowY: "auto" }}>
        {interventionSpecs.map((spec, i) => (
          <InterventionCard key={i} spec={spec} index={i} sessions={sessions} onUpdate={updateIntervention} onRemove={removeIntervention} />
        ))}
      </div>

      <button onClick={addIntervention} style={{ fontSize: 12 }}>+ Add Intervention</button>

      <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}>
        <input type="checkbox" checked={captureLogitLens} onChange={(e) => setCaptureLogitLens(e.target.checked)} />
        Capture logit lens
      </label>

      <div style={{ display: "flex", gap: 4 }}>
        {!isRunning ? (
          <button onClick={handleRun} disabled={!interveneSession || interventionSpecs.length === 0}>Run</button>
        ) : (
          <button onClick={() => {
            cancelAll();
            for (const id of localPendingIdsRef.current) {
              if (pendingResults[id]) removePendingResult(id);
            }
            localPendingIdsRef.current.clear();
            setRunning(false);
          }} style={{ background: "#6b2020" }}>Cancel</button>
        )}
      </div>

      {error && <div style={{ color: "#ff6b6b", fontSize: 12 }}>{error}</div>}
    </div>
  );
}
