import { useState } from "react";
import { useStore } from "../state/store";
import { useWebSocket } from "../hooks/useWebSocket";
import type { WsMessage, ProbeOperation } from "../types/api";

const WS_OPS = new Set<ProbeOperation>(["logit-lens", "generate"]);

export function ProbePanel() {
  const prompt = useStore((s) => s.prompt);
  const operation = useStore((s) => s.operation);
  const targetSession = useStore((s) => s.targetSession);
  const targetSessionB = useStore((s) => s.targetSessionB);
  const isRunning = useStore((s) => s.isRunning);
  const sessions = useStore((s) => s.sessions);
  const setPrompt = useStore((s) => s.setPrompt);
  const setOperation = useStore((s) => s.setOperation);
  const setTargetSession = useStore((s) => s.setTargetSession);
  const setTargetSessionB = useStore((s) => s.setTargetSessionB);
  const setRunning = useStore((s) => s.setRunning);
  const addResult = useStore((s) => s.addResult);
  const setPendingResult = useStore((s) => s.setPendingResult);
  const updatePendingResult = useStore((s) => s.updatePendingResult);
  const finalizePendingResult = useStore((s) => s.finalizePendingResult);
  const removePendingResult = useStore((s) => s.removePendingResult);

  const { connect, cancelAll } = useWebSocket();
  const [topK, setTopK] = useState(10);
  const [maxTokens, setMaxTokens] = useState(64);
  const [temperature, setTemperature] = useState(0.0);
  const [repPenalty, setRepPenalty] = useState(1.0);
  const [stopSeqs, setStopSeqs] = useState("\\n\\n");
  const [error, setError] = useState("");

  const isWs = WS_OPS.has(operation);

  const makeWsHandlers = (resultId: string, isFinalConnection: boolean) => ({
    onMessage: (msg: WsMessage) => { updatePendingResult(resultId, msg); },
    onComplete: (msg: WsMessage) => {
      finalizePendingResult(resultId, msg);
      if (isFinalConnection) setRunning(false);
    },
    onError: (message: string) => {
      finalizePendingResult(resultId);
      setError(message);
      if (isFinalConnection) setRunning(false);
    },
    onDisconnect: () => {
      finalizePendingResult(resultId);
      setError("Connection lost");
      if (isFinalConnection) setRunning(false);
    },
  });

  const getWsConfig = () => {
    if (operation === "logit-lens") return { prompt, top_k: topK };
    return {
      prompt, max_tokens: maxTokens, temperature, prob_top_k: topK,
      repetition_penalty: repPenalty,
      stop_sequences: stopSeqs.split(",").map((s) => s.replace(/\\n/g, "\n").trim()).filter(Boolean),
    };
  };

  const getWsPath = (session: string) => {
    if (operation === "logit-lens") return `/ws/sessions/${session}/logit-lens`;
    return `/ws/sessions/${session}/generate`;
  };

  const handleRun = () => {
    if (!targetSession || !prompt) return;
    setError("");
    setRunning(true);

    const resultId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

    if (isWs) {
      const hasB = !!targetSessionB;

      setPendingResult(resultId, {
        id: resultId, operation, sessionName: targetSession, prompt, data: [], timestamp: Date.now(),
      });
      connect(resultId, getWsPath(targetSession), getWsConfig(), makeWsHandlers(resultId, !hasB));

      if (hasB) {
        const idB = `${resultId}-B`;
        setPendingResult(idB, {
          id: idB, operation, sessionName: targetSessionB!, prompt, data: [], timestamp: Date.now(),
        });
        connect(idB, getWsPath(targetSessionB!), getWsConfig(), makeWsHandlers(idB, true));
      }
    } else {
      let url = "";
      let body: unknown = {};

      if (operation === "influence") {
        url = `/api/sessions/${targetSession}/inspect/influence`;
        body = { prompts: [prompt] };
      } else if (operation === "attention") {
        url = `/api/sessions/${targetSession}/inspect/attention`;
        body = { prompt };
      } else if (operation === "residual-norms") {
        url = `/api/sessions/${targetSession}/inspect/residual-norms`;
        body = { prompt };
      }

      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
        .then((r) => r.json())
        .then((data) => {
          addResult({
            id: resultId, operation, sessionName: targetSession, prompt,
            data: [{ type: "complete" as const, ...data }],
            timestamp: Date.now(),
          });
          setRunning(false);
        })
        .catch((e) => { setError((e as Error).message); setRunning(false); });
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <textarea
        placeholder="Prompt text..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={3}
      />

      <select value={operation} onChange={(e) => setOperation(e.target.value as ProbeOperation)}>
        <option value="logit-lens">Logit Lens</option>
        <option value="generate">Generate</option>
        <option value="influence">Layer Influence</option>
        <option value="attention">Attention Entropy</option>
        <option value="residual-norms">Residual Norms</option>
      </select>

      <div style={{ display: "flex", gap: 4 }}>
        <select value={targetSession} onChange={(e) => setTargetSession(e.target.value)}>
          <option value="">Session A...</option>
          {sessions.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
        </select>
        <select
          value={targetSessionB ?? ""}
          onChange={(e) => setTargetSessionB(e.target.value || null)}
          disabled={!isWs}
          title={!isWs ? "A/B not available for REST operations" : ""}
        >
          <option value="">Session B (A/B)...</option>
          {sessions.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
        </select>
      </div>

      {(operation === "logit-lens" || operation === "generate") && (
        <div style={{ display: "flex", gap: 8, fontSize: 12 }}>
          <label>top_k: <input type="number" value={topK} onChange={(e) => setTopK(+e.target.value)} style={{ width: 50 }} /></label>
          {operation === "generate" && (
            <>
              <label>max: <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)} style={{ width: 50 }} /></label>
              <label>temp: <input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(+e.target.value)} style={{ width: 50 }} /></label>
              <label>rep: <input type="number" step="0.1" value={repPenalty} onChange={(e) => setRepPenalty(+e.target.value)} style={{ width: 50 }} /></label>
            </>
          )}
        </div>
      )}

      {operation === "generate" && (
        <div style={{ fontSize: 12 }}>
          <label>stop: <input value={stopSeqs} onChange={(e) => setStopSeqs(e.target.value)} placeholder="comma-separated, use \n for newline" style={{ width: "100%" }} /></label>
        </div>
      )}

      <div style={{ display: "flex", gap: 4 }}>
        {!isRunning ? (
          <button onClick={handleRun} disabled={!targetSession || !prompt}>Run</button>
        ) : (
          <button onClick={() => {
            cancelAll();
            for (const id of Object.keys(useStore.getState().pendingResults)) {
              removePendingResult(id);
            }
            setRunning(false);
          }} style={{ background: "#6b2020" }}>Cancel</button>
        )}
      </div>

      {error && <div style={{ color: "#ff6b6b", fontSize: 12 }}>{error}</div>}
    </div>
  );
}
