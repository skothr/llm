import { useEffect, useRef, useState } from "react";
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

// Native number-input spinner arrows track the padding-box edge, so adding
// paddingRight would pull the spinners leftward instead of pushing the
// value rightward — exactly the collision we're trying to avoid. Hide the
// browser chrome entirely; keyboard ↑/↓ still step values when the input
// is focused, so we only lose the click target, which was tiny anyway.
const hideSpinnerCSS = `
  .num-input::-webkit-inner-spin-button,
  .num-input::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  .num-input { -moz-appearance: textfield; }
`;

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

// Every parameter has a well-defined default. When a knob is at its default
// we dim the label + input so tweaked settings visibly pop. The defaults
// double as initial state below — single source of truth.
const DEFAULTS = {
  displayTopK: 10,
  maxTokens: 64,
  temperature: 0.0,
  samplingTopK: 0,
  topP: 1.0,
  minP: 0.0,
  seed: "",
  repPenalty: 1.0,
  stopSeqs: "\\n\\n",
} as const;

const offColor = "#55556a";
const dim = (off: boolean, base: React.CSSProperties): React.CSSProperties =>
  off ? { ...base, color: offColor } : base;

// Streaming operations use WebSocket (live per-layer or per-token frames).
// One-shot inspection ops (influence, attention, residual-norms) go via REST
// because the backend returns a single complete result — a WS would buy
// nothing but a handshake cost. Keep in sync with backend route types.
const WS_OPS = new Set<ProbeOperation>(["logit-lens", "generate"]);

export function ProbePanel() {
  const prompt = useStore((s) => s.prompt);
  const operation = useStore((s) => s.operation);
  const backendOnline = useStore((s) => s.backendOnline);
  const backendProbed = useStore((s) => s.backendProbed);
  const targetSession = useStore((s) => s.targetSession);
  const targetSessionB = useStore((s) => s.targetSessionB);
  const isRunning = useStore((s) => s.isRunning);
  const sessions = useStore((s) => s.sessions);
  const sessionInfo = useStore((s) => s.sessionInfo);
  const fetchSessionInfo = useStore((s) => s.fetchSessionInfo);
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
  const [displayTopK, setDisplayTopK] = useState<number>(DEFAULTS.displayTopK);
  const [maxTokens, setMaxTokens] = useState<number>(DEFAULTS.maxTokens);
  const [temperature, setTemperature] = useState<number>(DEFAULTS.temperature);
  const [samplingTopK, setSamplingTopK] = useState<number>(DEFAULTS.samplingTopK);
  const [topP, setTopP] = useState<number>(DEFAULTS.topP);
  const [minP, setMinP] = useState<number>(DEFAULTS.minP);
  const [seed, setSeed] = useState<string>(DEFAULTS.seed);
  const [repPenalty, setRepPenalty] = useState<number>(DEFAULTS.repPenalty);
  const [stopSeqs, setStopSeqs] = useState<string>(DEFAULTS.stopSeqs);
  const [error, setError] = useState("");
  // Authoritative token count from the session's tokenizer; null until the
  // first successful fetch (or while re-fetching). Falls back to a char/3.5
  // estimate if the backend hasn't answered yet, so the "Context" line is
  // never blank.
  const [promptTokens, setPromptTokens] = useState<number | null>(null);
  const [clampNote, setClampNote] = useState("");

  const isWs = WS_OPS.has(operation);

  // Ensure we have info (including max_position_embeddings) for the current
  // target session. SessionsPanel only fetches on click, so a user who picked
  // a session from the ProbePanel dropdown without visiting SessionsPanel
  // first would otherwise have no ctx info to display.
  useEffect(() => {
    if (!targetSession) return;
    if (sessionInfo[targetSession]) return;
    fetchSessionInfo(targetSession).catch(() => { /* backend-offline is handled elsewhere */ });
  }, [targetSession, sessionInfo, fetchSessionInfo]);

  // Debounce tokenizer hits so typing doesn't spam the backend. 250 ms
  // means the meter lags user input by a quarter second — imperceptible
  // given token counts change at roughly word-level granularity anyway.
  useEffect(() => {
    if (!targetSession || !prompt) {
      setPromptTokens(prompt ? null : 0);
      return;
    }
    const t = setTimeout(() => {
      fetch(`/api/sessions/${targetSession}/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: prompt }),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (d && typeof d.count === "number") setPromptTokens(d.count); })
        .catch(() => { /* leave previous count; estimate fallback will kick in */ });
    }, 250);
    return () => clearTimeout(t);
  }, [prompt, targetSession]);

  const ctxInfo = sessionInfo[targetSession];
  const ctxWindow = ctxInfo?.max_position_embeddings ?? null;
  // Rough char-based estimate as a fallback — good to ±20 % for English.
  const estimatedPromptTokens = Math.ceil(prompt.length / 3.5);
  const promptTokenCount = promptTokens ?? estimatedPromptTokens;
  const budget = ctxWindow != null ? Math.max(0, ctxWindow - promptTokenCount) : null;

  // "At default" flags, used purely for UI dimming: true when the control
  // still holds its initial default value. Bright = user has touched it.
  // The backend receives the raw values regardless.
  const off = {
    max: maxTokens === DEFAULTS.maxTokens,
    temp: temperature === DEFAULTS.temperature,
    seed: seed === DEFAULTS.seed,
    top_k: samplingTopK === DEFAULTS.samplingTopK,
    top_p: topP === DEFAULTS.topP,
    min_p: minP === DEFAULTS.minP,
    rep: repPenalty === DEFAULTS.repPenalty,
    show: displayTopK === DEFAULTS.displayTopK,
    stop: stopSeqs === DEFAULTS.stopSeqs,
  };

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

  const getWsConfig = (maxTokensOverride?: number) => {
    if (operation === "logit-lens") return { prompt, top_k: displayTopK };
    const seedNum = seed.trim() === "" ? null : Number(seed);
    return {
      prompt,
      max_tokens: maxTokensOverride ?? maxTokens,
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
    setClampNote("");

    // Safety clamp: if the user asked for more new tokens than the context
    // window can hold, reduce max_tokens to fit and surface a one-line note.
    // A 4-token cushion covers any BOS/EOS bookkeeping the backend adds.
    // React state updates are async so we also thread the clamped value
    // directly into the WS config rather than relying on the re-render.
    let effectiveMax = maxTokens;
    if (ctxWindow != null && budget != null && maxTokens > budget) {
      effectiveMax = Math.max(1, budget - 4);
      setMaxTokens(effectiveMax);
      setClampNote(`max capped at ${effectiveMax} to fit ${ctxWindow}-token context`);
    }

    setRunning(true);

    const resultId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

    if (isWs) {
      const hasB = !!targetSessionB;

      localPendingIdsRef.current.add(resultId);
      if (hasB) {
        localPendingIdsRef.current.add(`${resultId}-B`);
      }

      setPendingResult(resultId, {
        id: resultId, operation, sessionName: targetSession, prompt, data: [], timestamp: Date.now(), isB: false,
      });
      connect(resultId, getWsPath(targetSession), getWsConfig(effectiveMax), makeWsHandlers(resultId));

      if (hasB) {
        const idB = `${resultId}-B`;
        setPendingResult(idB, {
          id: idB, operation, sessionName: targetSessionB!, prompt, data: [], timestamp: Date.now(), isB: true,
        });
        connect(idB, getWsPath(targetSessionB!), getWsConfig(effectiveMax), makeWsHandlers(idB));
      }
    } else {
      const fetchInspect = async (session: string, isB: boolean) => {
        const id = isB ? `${resultId}-B` : resultId;
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
          throw new Error(`Unsupported REST operation: ${operation}`);
        }
        const r = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          const text = await r.text().catch(() => r.statusText);
          throw new Error(`${r.status} ${text || r.statusText}`);
        }
        const data = await r.json();
        addResult({
          id, operation, sessionName: session, prompt,
          data: [{ type: "complete" as const, ...data }],
          timestamp: Date.now(),
          isB,
        });
      };

      const fetches = [fetchInspect(targetSession, false)];
      if (targetSessionB) {
        fetches.push(fetchInspect(targetSessionB, true));
      }

      Promise.all(fetches)
        .then(() => setRunning(false))
        .catch((e) => { setError((e as Error).message); setRunning(false); });
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <style>{hideSpinnerCSS}</style>
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
          <label style={dim(off.show, labelStyle)} title="Number of candidate tokens streamed per layer (shown in the heatmap).">top_k</label>
          <input className="num-input" type="number" value={displayTopK}
            onChange={(e) => setDisplayTopK(num(e.target.value, displayTopK))}
            style={dim(off.show, numInputStyle)} />
        </div>
      )}

      {operation === "generate" && (
        <>
          <div style={sectionHeaderStyle}>Output</div>
          <div style={paramGridStyle}>
            <label style={dim(off.max, labelStyle)} title="Maximum new tokens to generate before stopping.">max</label>
            <input className="num-input" type="number" value={maxTokens}
              onChange={(e) => setMaxTokens(num(e.target.value, maxTokens))}
              style={dim(off.max, numInputStyle)} />
            <label style={dim(off.rep, labelStyle)} title="Repetition penalty (1.0 = no penalty).">rep</label>
            <input className="num-input" type="number" step="0.1" value={repPenalty}
              onChange={(e) => setRepPenalty(num(e.target.value, repPenalty))}
              style={dim(off.rep, numInputStyle)} />

            <label style={dim(off.stop, labelStyle)} title="Comma-separated stop strings. When any substring matches, generation halts and the matched text is truncated from the output (same as <eos>).">stop</label>
            <input type="text" value={stopSeqs}
              onChange={(e) => setStopSeqs(e.target.value)}
              placeholder="comma-separated, \n for newline"
              style={dim(off.stop, { ...textInputStyle, gridColumn: "2 / -1" })} />
          </div>

          {ctxWindow != null && (
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 6,
              fontSize: 11,
              fontFamily: "monospace",
              color: "#8888aa",
              padding: "2px 0",
            }}>
              <span>
                ctx {ctxWindow.toLocaleString()}
                {" · "}prompt {promptTokens == null ? `~${estimatedPromptTokens}` : promptTokens}
                {budget != null && <> · budget <span style={{
                  color: maxTokens > budget ? "#ff6b6b" : budget < 32 ? "#f0ad4e" : "#8888aa",
                }}>{budget}</span></>}
              </span>
              <button
                title="Set max to the remaining budget (context window minus prompt)."
                onClick={() => {
                  if (budget == null) return;
                  setMaxTokens(Math.max(1, budget - 4));
                }}
                disabled={budget == null || budget < 2}
                style={{
                  fontSize: 10, padding: "1px 6px",
                  background: "#1a2540", border: "1px solid #2a4a7a",
                  cursor: budget == null || budget < 2 ? "not-allowed" : "pointer",
                }}
              >fill</button>
            </div>
          )}

          {clampNote && (
            <div style={{ fontSize: 11, color: "#f0ad4e" }}>{clampNote}</div>
          )}

          <div style={sectionHeaderStyle}>Sampling</div>
          <div style={paramGridStyle}>
            <label style={dim(off.temp, labelStyle)} title="Softmax sharpness. 0 = greedy argmax, 1 = untouched, >1 = flatter distribution.">temp</label>
            <input className="num-input" type="number" step="0.1" value={temperature}
              onChange={(e) => setTemperature(num(e.target.value, temperature))}
              style={dim(off.temp, numInputStyle)} />
            <label style={dim(off.seed, labelStyle)} title="Integer seed for reproducible sampling (temp > 0 only). Blank = random each run.">seed</label>
            <input type="text" value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="random"
              style={dim(off.seed, textInputStyle)} />

            <label style={dim(off.top_k, labelStyle)} title="Truncate sampling to the top-K logits before softmax. 0 disables.">top_k</label>
            <input className="num-input" type="number" value={samplingTopK}
              onChange={(e) => setSamplingTopK(num(e.target.value, samplingTopK))}
              style={dim(off.top_k, numInputStyle)} />
            <label style={dim(off.top_p, labelStyle)} title="Nucleus sampling: keep smallest set of tokens whose cumulative prob ≥ top_p. 1.0 disables.">top_p</label>
            <input className="num-input" type="number" step="0.05" min="0" max="1" value={topP}
              onChange={(e) => setTopP(num(e.target.value, topP))}
              style={dim(off.top_p, numInputStyle)} />

            <label style={dim(off.min_p, labelStyle)} title="Drop tokens whose prob < min_p × max(prob). Relative-floor filter robust to long tails. 0 disables.">min_p</label>
            <input className="num-input" type="number" step="0.01" min="0" max="1" value={minP}
              onChange={(e) => setMinP(num(e.target.value, minP))}
              style={dim(off.min_p, numInputStyle)} />
            <label style={dim(off.show, labelStyle)} title="How many candidate tokens to stream per step for the display popover. Does not affect sampling.">show</label>
            <input className="num-input" type="number" value={displayTopK}
              onChange={(e) => setDisplayTopK(num(e.target.value, displayTopK))}
              style={dim(off.show, numInputStyle)} />
          </div>
        </>
      )}

      <div style={{ display: "flex", gap: 4 }}>
        {!isRunning ? (
          <button
            onClick={handleRun}
            disabled={!targetSession || (backendProbed && !backendOnline)}
            title={backendProbed && !backendOnline ? "Backend offline — wait for reconnect" : undefined}
          >Run</button>
        ) : (
          <>
            {/* Stop: halt generation but keep whatever the model has streamed
                so far — finalize each pending result into the results list
                instead of throwing it away. Useful when an early token
                sequence is already what you wanted. */}
            <button onClick={() => {
              cancelAll();
              for (const id of localPendingIdsRef.current) {
                if (pendingResults[id]) finalizePendingResult(id);
              }
              localPendingIdsRef.current.clear();
              setRunning(false);
            }} style={{ background: "#6b5020" }} title="Halt generation; keep what's been streamed so far.">Stop</button>
            {/* Cancel: halt AND discard. For when the partial output is
                garbage and you just want a clean slate. */}
            <button onClick={() => {
              cancelAll();
              for (const id of localPendingIdsRef.current) {
                if (pendingResults[id]) removePendingResult(id);
              }
              localPendingIdsRef.current.clear();
              setRunning(false);
            }} style={{ background: "#6b2020" }} title="Halt generation and discard the partial output.">Cancel</button>
          </>
        )}
      </div>

      {error && <div style={{ color: "#ff6b6b", fontSize: 12 }}>{error}</div>}
    </div>
  );
}
