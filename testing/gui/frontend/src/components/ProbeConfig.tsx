import { useState } from "react";
import { useStore } from "../state/store";
import { useWebSocket } from "../hooks/useWebSocket";
import type { WsMessage, ProbeOperation } from "../types/api";

export function ProbeConfig() {
  const {
    prompt, operation, targetSession, targetSessionB, isRunning,
    sessions,
    setPrompt, setOperation, setTargetSession, setTargetSessionB,
    setRunning, addResult,
  } = useStore();

  const { connect, cancel } = useWebSocket();
  const [topK, setTopK] = useState(10);
  const [maxTokens, setMaxTokens] = useState(64);
  const [temperature, setTemperature] = useState(0.0);
  const [repPenalty, setRepPenalty] = useState(1.0);
  const [stopSeqs, setStopSeqs] = useState("\\n\\n");
  const [error, setError] = useState("");

  const handleRun = () => {
    if (!targetSession || !prompt) return;
    setError("");
    setRunning(true);

    const resultData: WsMessage[] = [];
    const resultId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

    const handlers = {
      onMessage: (msg: WsMessage) => {
        resultData.push(msg);
      },
      onComplete: (msg: WsMessage) => {
        resultData.push(msg);
        addResult({
          id: resultId,
          operation,
          sessionName: targetSession,
          prompt,
          data: resultData,
          timestamp: Date.now(),
        });
        setRunning(false);
      },
      onError: (message: string) => {
        setError(message);
        setRunning(false);
      },
    };

    if (operation === "logit-lens") {
      connect(`/ws/sessions/${targetSession}/logit-lens`, { prompt, top_k: topK }, handlers);
    } else if (operation === "generate") {
      connect(`/ws/sessions/${targetSession}/generate`, {
        prompt, max_tokens: maxTokens, temperature, prob_top_k: topK,
        repetition_penalty: repPenalty,
        stop_sequences: stopSeqs.split(",").map((s) => s.replace(/\\n/g, "\n").trim()).filter(Boolean),
      }, handlers);
    } else if (operation === "intervene") {
      setError("Configure interventions in the surgery panel first");
      setRunning(false);
    } else if (operation === "influence") {
      fetch(`/api/sessions/${targetSession}/inspect/influence`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompts: [prompt] }),
      })
        .then((r) => r.json())
        .then((data) => {
          addResult({
            id: resultId,
            operation: "influence",
            sessionName: targetSession,
            prompt,
            data: [{ type: "complete" as const, scores: data.scores }],
            timestamp: Date.now(),
          });
          setRunning(false);
        })
        .catch((e) => {
          setError((e as Error).message);
          setRunning(false);
        });
    }

    if (targetSessionB && (operation === "logit-lens" || operation === "generate")) {
      const resultIdB = `${resultId}-B`;
      const resultDataB: WsMessage[] = [];
      const handlersB = {
        onMessage: (msg: WsMessage) => { resultDataB.push(msg); },
        onComplete: (msg: WsMessage) => {
          resultDataB.push(msg);
          addResult({
            id: resultIdB,
            operation,
            sessionName: targetSessionB,
            prompt,
            data: resultDataB,
            timestamp: Date.now(),
          });
        },
        onError: (message: string) => { setError(message); },
      };

      if (operation === "logit-lens") {
        connect(`/ws/sessions/${targetSessionB}/logit-lens`, { prompt, top_k: topK }, handlersB);
      } else if (operation === "generate") {
        connect(`/ws/sessions/${targetSessionB}/generate`, {
          prompt, max_tokens: maxTokens, temperature, prob_top_k: topK,
        repetition_penalty: repPenalty,
        stop_sequences: stopSeqs.split(",").map((s) => s.replace(/\\n/g, "\n").trim()).filter(Boolean),
        }, handlersB);
      }
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <h2>Probe Config</h2>

      <textarea
        placeholder="Prompt text..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={3}
      />

      <select value={operation} onChange={(e) => setOperation(e.target.value as ProbeOperation)}>
        <option value="logit-lens">Logit Lens</option>
        <option value="influence">Layer Influence</option>
        <option value="generate">Generate</option>
        <option value="intervene">Intervene</option>
      </select>

      <div style={{ display: "flex", gap: 4 }}>
        <select value={targetSession} onChange={(e) => setTargetSession(e.target.value)}>
          <option value="">Session A...</option>
          {sessions.map((s) => (
            <option key={s.name} value={s.name}>{s.name}</option>
          ))}
        </select>
        <select value={targetSessionB ?? ""} onChange={(e) => setTargetSessionB(e.target.value || null)}>
          <option value="">Session B (A/B)...</option>
          {sessions.map((s) => (
            <option key={s.name} value={s.name}>{s.name}</option>
          ))}
        </select>
      </div>

      {(operation === "logit-lens" || operation === "generate") && (
        <div style={{ display: "flex", gap: 8, fontSize: 12 }}>
          <label>
            top_k: <input type="number" value={topK} onChange={(e) => setTopK(+e.target.value)} style={{ width: 50 }} />
          </label>
          {operation === "generate" && (
            <>
              <label>
                max: <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)} style={{ width: 50 }} />
              </label>
              <label>
                temp: <input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(+e.target.value)} style={{ width: 50 }} />
              </label>
              <label>
                rep: <input type="number" step="0.1" value={repPenalty} onChange={(e) => setRepPenalty(+e.target.value)} style={{ width: 50 }} />
              </label>
            </>
          )}
        </div>
      )}

      {operation === "generate" && (
        <div style={{ fontSize: 12 }}>
          <label>
            stop: <input
              value={stopSeqs}
              onChange={(e) => setStopSeqs(e.target.value)}
              placeholder="comma-separated, use \n for newline"
              style={{ width: "100%" }}
            />
          </label>
        </div>
      )}

      <div style={{ display: "flex", gap: 4 }}>
        {!isRunning ? (
          <button onClick={handleRun} disabled={!targetSession || !prompt}>
            Run
          </button>
        ) : (
          <button onClick={cancel} style={{ background: "#6b2020" }}>
            Cancel
          </button>
        )}
      </div>

      {error && <div style={{ color: "#ff6b6b", fontSize: 12 }}>{error}</div>}
    </div>
  );
}
