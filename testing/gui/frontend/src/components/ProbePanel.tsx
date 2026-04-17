import { useRef, useState } from "react";
import { useStore } from "../state/store";
import { useWebSocket } from "../hooks/useWebSocket";
import type { WsMessage, ProbeOperation } from "../types/api";

const num = (v: string, fallback: number): number => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};

// Shared styles for the param grids below. A 4-column grid
// (label | input | label | input) packs two name/value pairs per row,
// which fits the sidebar width without forcing inputs to wrap and keeps
// all labels in two vertical lanes — much easier to scan than the prior
// flex-wrap soup that let labels and inputs interleave unpredictably.
const paramGridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "auto 1fr auto 1fr",
  gap: "4px 8px",
  alignItems: "center",
  fontSize: 12,
};

const sectionHeaderStyle: React.CSSProperties = {
  fontSize: 10,
  color: "#8888aa",
  textTransform: "uppercase",
  letterSpacing: 1,
  marginTop: 4,
  marginBottom: -2,
};

const labelStyle: React.CSSProperties = {
  fontFamily: "monospace",
  color: "#a0a0c0",
  justifySelf: "end",
};

const numInputStyle: React.CSSProperties = {
  width: "100%",
  minWidth: 0,
  textAlign: "right",
  fontFamily: "monospace",
};

const textInputStyle: React.CSSProperties = {
  width: "100%",
  minWidth: 0,
  fontFamily: "monospace",
};

// Streaming operations use WebSocket (live per-layer or per-token frames).
// One-shot inspection ops (influence, attention, residual-norms) go via REST
// because the backend returns a single complete result — a WS would buy
// nothing but a handshake cost. Keep in sync with backend route types.
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
  const pendingResults = useStore((s) => s.pendingResults);
  const setPendingResult = useStore((s) => s.setPendingResult);
  const updatePendingResult = useStore((s) => s.updatePendingResult);
  const finalizePendingResult = useStore((s) => s.finalizePendingResult);
  const removePendingResult = useStore((s) => s.removePendingResult);

  const { connect, cancelAll } = useWebSocket();
  const localPendingIdsRef = useRef<Set<string>>(new Set());
  // displayTopK = how many candidate tokens to stream alongside each step
  // (UI display only, e.g. the top-k popover in GenerationOutput).
  // Real sampling cutoffs live in samplingTopK / topP / minP below.
  const [displayTopK, setDisplayTopK] = useState(10);
  const [maxTokens, setMaxTokens] = useState(64);
  const [temperature, setTemperature] = useState(0.0);
  const [samplingTopK, setSamplingTopK] = useState(0);
  const [topP, setTopP] = useState(1.0);
  const [minP, setMinP] = useState(0.0);
  const [seed, setSeed] = useState("");
  const [repPenalty, setRepPenalty] = useState(1.0);
  const [stopSeqs, setStopSeqs] = useState("\\n\\n");
  const [error, setError] = useState("");

  const isWs = WS_OPS.has(operation);

  const makeWsHandlers = (resultId: string) => ({
    onMessage: (msg: WsMessage) => { updatePendingResult(resultId, msg); },
    onComplete: (msg: WsMessage) => {
      finalizePendingResult(resultId, msg);
      localPendingIdsRef.current.delete(resultId);
      if (localPendingIdsRef.current.size === 0) setRunning(false);
    },
    onError: (message: string) => {
      removePendingResult(resultId);
      localPendingIdsRef.current.delete(resultId);
      setError(message);
      if (localPendingIdsRef.current.size === 0) setRunning(false);
    },
    onDisconnect: () => {
      removePendingResult(resultId);
      localPendingIdsRef.current.delete(resultId);
      setError("Connection lost");
      if (localPendingIdsRef.current.size === 0) setRunning(false);
    },
  });

  const getWsConfig = () => {
    if (operation === "logit-lens") return { prompt, top_k: displayTopK };
    const seedNum = seed.trim() === "" ? null : Number(seed);
    return {
      prompt,
      max_tokens: maxTokens,
      temperature,
      display_top_k: displayTopK,
      sampling_top_k: samplingTopK,
      top_p: topP,
      min_p: minP,
      seed: Number.isFinite(seedNum) ? seedNum : null,
      repetition_penalty: repPenalty,
      stop_sequences: stopSeqs.split(",").map((s) => s.replace(/\\n/g, "\n").trim()).filter(Boolean),
    };
  };

  const getWsPath = (session: string) => {
    if (operation === "logit-lens") return `/ws/sessions/${session}/logit-lens`;
    return `/ws/sessions/${session}/generate`;
  };

  const handleRun = () => {
    if (!targetSession) return;
    setError("");
    setRunning(true);

    const resultId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

    if (isWs) {
      const hasB = !!targetSessionB;

      localPendingIdsRef.current.add(resultId);
      if (hasB) {
        localPendingIdsRef.current.add(`${resultId}-B`);
      }

      setPendingResult(resultId, {
        id: resultId, operation, sessionName: targetSession, prompt, data: [], timestamp: Date.now(),
      });
      connect(resultId, getWsPath(targetSession), getWsConfig(), makeWsHandlers(resultId));

      if (hasB) {
        const idB = `${resultId}-B`;
        setPendingResult(idB, {
          id: idB, operation, sessionName: targetSessionB!, prompt, data: [], timestamp: Date.now(),
        });
        connect(idB, getWsPath(targetSessionB!), getWsConfig(), makeWsHandlers(idB));
      }
    } else {
      const fetchInspect = (session: string, id: string) => {
        let url = "";
        let body: unknown = {};
        if (operation === "influence") {
          url = `/api/sessions/${session}/inspect/influence`;
          body = { prompts: [prompt] };
        } else if (operation === "attention") {
          url = `/api/sessions/${session}/inspect/attention`;
          body = { prompt };
        } else if (operation === "residual-norms") {
          url = `/api/sessions/${session}/inspect/residual-norms`;
          body = { prompt };
        } else {
          return Promise.reject(new Error(`Unsupported REST operation: ${operation}`));
        }
        return fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        })
          .then(async (r) => {
            if (!r.ok) {
              const text = await r.text().catch(() => r.statusText);
              throw new Error(`${r.status} ${text || r.statusText}`);
            }
            return r.json();
          })
          .then((data) => {
            addResult({
              id, operation, sessionName: session, prompt,
              data: [{ type: "complete" as const, ...data }],
              timestamp: Date.now(),
            });
          });
      };

      const fetches = [fetchInspect(targetSession, resultId)];
      if (targetSessionB) {
        fetches.push(fetchInspect(targetSessionB, `${resultId}-B`));
      }

      Promise.all(fetches)
        .then(() => setRunning(false))
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
        >
          <option value="">Session B (A/B)...</option>
          {sessions.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
        </select>
      </div>

      {operation === "logit-lens" && (
        <div style={paramGridStyle}>
          <label style={labelStyle} title="Number of candidate tokens streamed per layer (shown in the heatmap).">top_k</label>
          <input type="number" value={displayTopK}
            onChange={(e) => setDisplayTopK(num(e.target.value, displayTopK))}
            style={numInputStyle} />
        </div>
      )}

      {operation === "generate" && (
        <>
          <div style={sectionHeaderStyle}>Output</div>
          <div style={paramGridStyle}>
            <label style={labelStyle} title="Maximum new tokens to generate before stopping.">max</label>
            <input type="number" value={maxTokens}
              onChange={(e) => setMaxTokens(num(e.target.value, maxTokens))}
              style={numInputStyle} />
            <label style={labelStyle} title="Repetition penalty (1.0 = no penalty).">rep</label>
            <input type="number" step="0.1" value={repPenalty}
              onChange={(e) => setRepPenalty(num(e.target.value, repPenalty))}
              style={numInputStyle} />

            <label style={labelStyle} title="Comma-separated stop strings. When any substring matches, generation halts and the matched text is truncated from the output (same as <eos>).">stop</label>
            <input type="text" value={stopSeqs}
              onChange={(e) => setStopSeqs(e.target.value)}
              placeholder="comma-separated, \n for newline"
              style={{ ...textInputStyle, gridColumn: "2 / -1" }} />
          </div>

          <div style={sectionHeaderStyle}>Sampling</div>
          <div style={paramGridStyle}>
            <label style={labelStyle} title="Softmax sharpness. 0 = greedy argmax, 1 = untouched, >1 = flatter distribution.">temp</label>
            <input type="number" step="0.1" value={temperature}
              onChange={(e) => setTemperature(num(e.target.value, temperature))}
              style={numInputStyle} />
            <label style={labelStyle} title="Integer seed for reproducible sampling (temp > 0 only). Blank = random each run.">seed</label>
            <input type="text" value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="random"
              style={textInputStyle} />

            <label style={labelStyle} title="Truncate sampling to the top-K logits before softmax. 0 disables.">top_k</label>
            <input type="number" value={samplingTopK}
              onChange={(e) => setSamplingTopK(num(e.target.value, samplingTopK))}
              style={numInputStyle} />
            <label style={labelStyle} title="Nucleus sampling: keep smallest set of tokens whose cumulative prob ≥ top_p. 1.0 disables.">top_p</label>
            <input type="number" step="0.05" min="0" max="1" value={topP}
              onChange={(e) => setTopP(num(e.target.value, topP))}
              style={numInputStyle} />

            <label style={labelStyle} title="Drop tokens whose prob < min_p × max(prob). Relative-floor filter robust to long tails. 0 disables.">min_p</label>
            <input type="number" step="0.01" min="0" max="1" value={minP}
              onChange={(e) => setMinP(num(e.target.value, minP))}
              style={numInputStyle} />
            <label style={labelStyle} title="How many candidate tokens to stream per step for the display popover. Does not affect sampling.">show</label>
            <input type="number" value={displayTopK}
              onChange={(e) => setDisplayTopK(num(e.target.value, displayTopK))}
              style={numInputStyle} />
          </div>
        </>
      )}

      <div style={{ display: "flex", gap: 4 }}>
        {!isRunning ? (
          <button onClick={handleRun} disabled={!targetSession}>Run</button>
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
