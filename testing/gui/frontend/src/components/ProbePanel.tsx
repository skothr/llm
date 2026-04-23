import { useEffect, useRef, useState } from "react";
import { useStore, SAMPLING_DEFAULTS } from "../state/store";
import { useWebSocket } from "../hooks/useWebSocket";
import { useStopCancel } from "../hooks/useStopCancel";
import { num } from "../utils/num";
import { PromptLibraryBar } from "./PromptLibraryBar";
import type { WsMessage, ProbeOperation } from "../types/api";
import { PatchingControls, DEFAULT_PATCHING_STATE, type PatchingState } from "./PatchingControls";

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

// Knob defaults live in the store (SAMPLING_DEFAULTS) so persistence and
// reset actions share one source of truth. When a knob is at its default we
// dim the label + input so tweaked settings visibly pop.
const DEFAULTS = SAMPLING_DEFAULTS;

const offColor = "#55556a";
const dim = (off: boolean, base: React.CSSProperties): React.CSSProperties =>
  off ? { ...base, color: offColor } : base;

// Streaming operations use WebSocket (live per-layer or per-token frames).
// One-shot inspection ops (influence, attention, residual-norms) go via REST
// because the backend returns a single complete result — a WS would buy
// nothing but a handshake cost. Keep in sync with backend route types.
const WS_OPS = new Set<ProbeOperation>(["logit-lens", "generate", "activation-patching"]);

// Sensible default ranges per sweep axis. Used when the user flips axis
// away from seed and the range is still at the 0..1 seed-mode default.
type SweepAxis = "seed" | "temperature" | "top_p" | "top_k" | "min_p";
const AXIS_DEFAULTS: Record<Exclude<SweepAxis, "seed">, { from: number; to: number; step: number; label: string; backendKey: string; format: (v: number) => string }> = {
  temperature: { from: 0.1, to: 1.5, step: 0.1, label: "temp",  backendKey: "temperature", format: (v) => v.toFixed(2) },
  top_p:       { from: 0.3, to: 1.0, step: 0.05, label: "top_p", backendKey: "top_p",       format: (v) => v.toFixed(2) },
  top_k:       { from: 1,   to: 100, step: 1,   label: "top_k", backendKey: "sampling_top_k", format: (v) => `${Math.round(v)}` },
  min_p:       { from: 0.0, to: 0.3, step: 0.02, label: "min_p", backendKey: "min_p",       format: (v) => v.toFixed(2) },
};

// Produce N evenly-spaced values from [from, to] inclusive. For integer
// axes (top_k) we round; for others we keep full precision.
function linspace(from: number, to: number, n: number, integer = false): number[] {
  if (n <= 1) return [from];
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const v = from + (to - from) * t;
    out.push(integer ? Math.round(v) : v);
  }
  return out;
}

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
  const setPendingResult = useStore((s) => s.setPendingResult);
  const updatePendingResult = useStore((s) => s.updatePendingResult);
  const finalizePendingResult = useStore((s) => s.finalizePendingResult);
  const removePendingResult = useStore((s) => s.removePendingResult);

  const { connect, cancelAll } = useWebSocket();
  const localPendingIdsRef = useRef<Set<string>>(new Set());
  const { makeHandler } = useStopCancel(localPendingIdsRef);
  // Sampling params live in the persisted store so they survive reloads and
  // can be shared with the seed fan-out flow. Destructured here into locals
  // so the render body reads like the old useState version.
  const samplingParams = useStore((s) => s.samplingParams);
  const setSamplingParams = useStore((s) => s.setSamplingParams);
  const resetSamplingParams = useStore((s) => s.resetSamplingParams);
  const {
    displayTopK, maxTokens, temperature, samplingTopK, topP, minP, seed,
    repPenalty, stopSeqs, numSeeds, sweepAxis, sweepFrom, sweepTo,
    sweepAxis2, sweepFrom2, sweepTo2, numSeeds2,
  } = samplingParams;
  const setDisplayTopK = (v: number) => setSamplingParams({ displayTopK: v });
  const setMaxTokens = (v: number) => setSamplingParams({ maxTokens: v });
  const setTemperature = (v: number) => setSamplingParams({ temperature: v });
  const setSamplingTopK = (v: number) => setSamplingParams({ samplingTopK: v });
  const setTopP = (v: number) => setSamplingParams({ topP: v });
  const setMinP = (v: number) => setSamplingParams({ minP: v });
  const setSeed = (v: string) => setSamplingParams({ seed: v });
  const setRepPenalty = (v: number) => setSamplingParams({ repPenalty: v });
  const setStopSeqs = (v: string) => setSamplingParams({ stopSeqs: v });
  const setNumSeeds = (v: number) => setSamplingParams({ numSeeds: Math.max(1, Math.min(32, v)) });
  const setSweepAxis = (a: typeof sweepAxis) => {
    // When switching to a knob sweep and the range is still at the
    // generic 0..1, adopt that axis's sensible defaults.
    if (a !== "seed" && sweepFrom === 0 && sweepTo === 1) {
      const defaults = AXIS_DEFAULTS[a];
      setSamplingParams({ sweepAxis: a, sweepFrom: defaults.from, sweepTo: defaults.to });
    } else {
      setSamplingParams({ sweepAxis: a });
    }
  };
  const setSweepFrom = (v: number) => setSamplingParams({ sweepFrom: v });
  const setSweepTo = (v: number) => setSamplingParams({ sweepTo: v });
  const setSweepAxis2 = (a: typeof sweepAxis2) => {
    // Bumping into a new secondary axis adopts that axis's default range
    // the first time. Picking "none" collapses the grid back to 1D.
    if (a !== "none" && sweepFrom2 === 0 && sweepTo2 === 1) {
      const d = AXIS_DEFAULTS[a];
      setSamplingParams({ sweepAxis2: a, sweepFrom2: d.from, sweepTo2: d.to });
    } else {
      setSamplingParams({ sweepAxis2: a });
    }
  };
  const setSweepFrom2 = (v: number) => setSamplingParams({ sweepFrom2: v });
  const setSweepTo2 = (v: number) => setSamplingParams({ sweepTo2: v });
  const setNumSeeds2 = (v: number) => setSamplingParams({ numSeeds2: Math.max(1, Math.min(8, v)) });
  const [error, setError] = useState("");
  const [patchingState, setPatchingStateLocal] = useState<PatchingState>(DEFAULT_PATCHING_STATE);
  const [patchingLengthsMatch, setPatchingLengthsMatch] = useState(false);
  const updatePatchingState = (patch: Partial<PatchingState>) =>
    setPatchingStateLocal((prev) => ({ ...prev, ...patch }));
  // Authoritative token count from the session's tokenizer; null until the
  // first successful fetch (or while re-fetching). Falls back to a char/3.5
  // estimate if the backend hasn't answered yet, so the "Context" line is
  // never blank.
  const [promptTokens, setPromptTokens] = useState<number | null>(null);
  const [clampNote, setClampNote] = useState("");

  // Clear the clamp note once the user changes either of the inputs that
  // caused it. Lingering until the next Run made it easy to miss that the
  // original reason (e.g. prompt too long) was already resolved.
  useEffect(() => { if (clampNote) setClampNote(""); }, [prompt, maxTokens]); // eslint-disable-line react-hooks/exhaustive-deps

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
  // AbortController cancels the previous request so fast typing can't
  // resolve stale counts out of order over the current one.
  useEffect(() => {
    if (!targetSession || !prompt) {
      setPromptTokens(prompt ? null : 0);
      return;
    }
    const ac = new AbortController();
    const t = setTimeout(() => {
      fetch(`/api/sessions/${targetSession}/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: prompt }),
        signal: ac.signal,
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (d && typeof d.count === "number") setPromptTokens(d.count); })
        .catch(() => { /* abort / offline — leave prior count; estimate fills in */ });
    }, 250);
    return () => { clearTimeout(t); ac.abort(); };
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
    n: numSeeds === DEFAULTS.numSeeds,
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
    if (operation === "activation-patching") {
      // AP runs go through the dedicated branch in handleRun(); this fallback
      // is for fan-out / "Run on all" paths that shouldn't route here anyway.
      return { prompt };
    }
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
    if (operation === "activation-patching") return `/ws/sessions/${session}/activation-patching`;
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
      if (operation === "activation-patching") {
        if (!patchingLengthsMatch) {
          setError("Clean and corrupted prompts must tokenize to the same length.");
          setRunning(false);
          return;
        }
        localPendingIdsRef.current.add(resultId);
        const runParamsSnapshot = { ...samplingParams };
        setPendingResult(resultId, {
          id: resultId,
          operation,
          sessionName: targetSession,
          prompt: patchingState.cleanPrompt,
          data: [],
          timestamp: Date.now(),
          isB: false,
          runParams: runParamsSnapshot,
        });
        const cfg: Record<string, unknown> = {
          clean_prompt: patchingState.cleanPrompt,
          corrupted_prompt: patchingState.corruptedPrompt,
          direction: patchingState.direction,
          measurement_position: patchingState.measurementPos,
          mode: patchingState.mode,
        };
        if (patchingState.mode === "edge") {
          cfg.top_k_edges = patchingState.top_k_edges;
        } else if (patchingState.mode === "circuit") {
          cfg.top_k_candidates = patchingState.top_k_candidates;
          cfg.tau = patchingState.tau;
        } else if (patchingState.mode === "approx_neuron") {
          cfg.top_k_neurons = patchingState.top_k_neurons;
        }
        if (patchingState.tokenPairMode === "manual") {
          cfg.correct_token = patchingState.manualCorrect;
          cfg.incorrect_token = patchingState.manualIncorrect;
        }
        const path = `/ws/sessions/${targetSession}/activation-patching`;
        connect(resultId, path, cfg, makeWsHandlers(resultId), targetSession);
        return;
      }

      // Fan-out: N>1 generate runs on the same session. Mutually exclusive
      // with A/B — UI disables numSeeds when B is selected, so this branch
      // can assume !targetSessionB. Optional 2nd axis produces N×M grid.
      const has2D = sweepAxis2 !== "none" && numSeeds2 > 1;
      const rows = operation === "generate" && numSeeds > 1 && !targetSessionB ? numSeeds : 0;
      const cols = has2D ? numSeeds2 : 1;
      const total = rows * cols;

      if (rows > 0) {
        const batchId = resultId;
        const seedTrimmed = seed.trim();
        const seedBase = seedTrimmed === "" ? null : Number(seedTrimmed);
        const hasBaseSeed = seedBase !== null && Number.isFinite(seedBase);

        // Primary axis knob spec.
        const isKnobSweep = sweepAxis !== "seed";
        const axisDef = isKnobSweep ? AXIS_DEFAULTS[sweepAxis] : null;
        const primaryValues = isKnobSweep && axisDef
          ? linspace(sweepFrom, sweepTo, rows, sweepAxis === "top_k")
          : null;

        // Secondary axis (always a knob when present).
        const axis2Def = has2D ? AXIS_DEFAULTS[sweepAxis2 as Exclude<typeof sweepAxis2, "none">] : null;
        const secondaryValues = axis2Def
          ? linspace(sweepFrom2, sweepTo2, cols, sweepAxis2 === "top_k")
          : null;

        // Shared seed across the grid when a knob axis is active so
        // sampling noise isn't confounded with the knob(s).
        const sharedSeed = isKnobSweep || has2D
          ? (hasBaseSeed ? (seedBase as number) : Math.floor(Math.random() * 2_147_483_647))
          : null;

        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            const linearIdx = i * cols + j;
            const childId = cols > 1 ? `${batchId}-g${i}-${j}` : `${batchId}-s${i}`;
            let childSeed: number;
            const overrides: Record<string, unknown> = {};
            const labelParts: string[] = [];

            if (isKnobSweep && axisDef && primaryValues) {
              overrides[axisDef.backendKey] = primaryValues[i];
              labelParts.push(`${axisDef.label}=${axisDef.format(primaryValues[i])}`);
              childSeed = sharedSeed!;
            } else {
              // Seed sweep on primary axis.
              childSeed = hasBaseSeed
                ? (seedBase as number) + i
                : (cols === 1 ? Math.floor(Math.random() * 2_147_483_647) : sharedSeed!);
              labelParts.push(`seed=${childSeed}`);
            }

            if (axis2Def && secondaryValues) {
              overrides[axis2Def.backendKey] = secondaryValues[j];
              labelParts.push(`${axis2Def.label}=${axis2Def.format(secondaryValues[j])}`);
            }

            localPendingIdsRef.current.add(childId);
            setPendingResult(childId, {
              id: childId,
              operation,
              sessionName: targetSession,
              prompt,
              data: [],
              timestamp: Date.now(),
              isB: false,
              batchId,
              batchIndex: linearIdx,
              batchSize: total,
              seed: childSeed,
              sweepLabel: labelParts.join(", "),
              runParams: { ...samplingParams },
              // Only set gridCols for 2D grids so 1D fan-out keeps its
              // existing horizontal-scroll layout.
              ...(cols > 1 ? { gridCols: cols } : {}),
            });
            const cfgBase = getWsConfig(effectiveMax) as Record<string, unknown>;
            const cfg = { ...cfgBase, seed: childSeed, ...overrides };
            connect(childId, getWsPath(targetSession), cfg, makeWsHandlers(childId), targetSession);
          }
        }
        return;
      }

      const hasB = !!targetSessionB;

      localPendingIdsRef.current.add(resultId);
      if (hasB) {
        localPendingIdsRef.current.add(`${resultId}-B`);
      }

      const runParamsSnapshot = { ...samplingParams };
      setPendingResult(resultId, {
        id: resultId, operation, sessionName: targetSession, prompt, data: [], timestamp: Date.now(), isB: false,
        runParams: runParamsSnapshot,
      });
      connect(resultId, getWsPath(targetSession), getWsConfig(effectiveMax), makeWsHandlers(resultId), targetSession);

      if (hasB) {
        const idB = `${resultId}-B`;
        setPendingResult(idB, {
          id: idB, operation, sessionName: targetSessionB!, prompt, data: [], timestamp: Date.now(), isB: true,
          runParams: runParamsSnapshot,
        });
        connect(idB, getWsPath(targetSessionB!), getWsConfig(effectiveMax), makeWsHandlers(idB), targetSessionB!);
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
          runParams: { ...samplingParams },
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

  // Fan out across every loaded session with the current prompt+params.
  // Intentionally does NOT combine with seed/knob sweeps or A/B — those
  // already fan out along a different axis. Seed fan-out with the
  // all-sessions fan-out would multiply run counts combinatorially and
  // is rarely what a researcher wants at the keystroke of a single button.
  const handleRunAll = () => {
    if (sessions.length === 0) return;
    setError("");
    setClampNote("");
    setRunning(true);

    const baseId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const runParamsSnapshot = { ...samplingParams };
    // Keep the clamp logic modest here: each session may have a
    // different ctx window, and re-clamping per session would need
    // awaited fetchSessionInfo calls. Callers doing serious batch work
    // should set max_tokens conservatively or run individually per
    // session. Surface a note if the *current* session's window is
    // tighter than the requested max.
    let effectiveMax = maxTokens;
    if (ctxWindow != null && budget != null && maxTokens > budget) {
      effectiveMax = Math.max(1, budget - 4);
      setMaxTokens(effectiveMax);
      setClampNote(`max capped at ${effectiveMax} (tightest known ctx=${ctxWindow})`);
    }

    if (isWs) {
      sessions.forEach((s, i) => {
        const childId = `${baseId}-all-${i}`;
        localPendingIdsRef.current.add(childId);
        setPendingResult(childId, {
          id: childId,
          operation,
          sessionName: s.name,
          prompt,
          data: [],
          timestamp: Date.now(),
          isB: false,
          runParams: runParamsSnapshot,
        });
        connect(childId, getWsPath(s.name), getWsConfig(effectiveMax), makeWsHandlers(childId), s.name);
      });
    } else {
      // REST path: one POST per session; aggregate errors.
      const fetches = sessions.map(async (s, i) => {
        const childId = `${baseId}-all-${i}`;
        let url = "";
        let body: unknown = {};
        if (operation === "influence") {
          url = `/api/sessions/${s.name}/inspect/influence`;
          body = { prompts: [prompt] };
        } else if (operation === "attention") {
          url = `/api/sessions/${s.name}/inspect/attention`;
          body = { prompt };
        } else if (operation === "residual-norms") {
          url = `/api/sessions/${s.name}/inspect/residual-norms`;
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
          throw new Error(`${s.name}: ${r.status} ${text || r.statusText}`);
        }
        const data = await r.json();
        addResult({
          id: childId, operation, sessionName: s.name, prompt,
          data: [{ type: "complete" as const, ...data }],
          timestamp: Date.now(),
          isB: false,
          runParams: runParamsSnapshot,
        });
      });
      Promise.all(fetches)
        .then(() => setRunning(false))
        .catch((e) => { setError((e as Error).message); setRunning(false); });
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <style>{hideSpinnerCSS}</style>
      <PromptLibraryBar value={prompt} onLoad={setPrompt} />
      <textarea
        placeholder="Prompt text... (Ctrl/Cmd+Enter to run)"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          // Ctrl/Cmd+Enter runs the probe. Plain Enter still inserts a
          // newline — prompts often span multiple lines, so we don't steal
          // that behavior.
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            if (!isRunning && targetSession && !(backendProbed && !backendOnline)) {
              handleRun();
            }
          }
        }}
        rows={3}
      />

      <select value={operation} onChange={(e) => setOperation(e.target.value as ProbeOperation)}>
        <option value="logit-lens">Logit Lens</option>
        <option value="generate">Generate</option>
        <option value="influence">Layer Influence</option>
        <option value="attention">Attention Entropy</option>
        <option value="residual-norms">Residual Norms</option>
        <option value="activation-patching">Activation Patching</option>
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

      {operation === "activation-patching" && targetSession && (
        <PatchingControls
          targetSession={targetSession}
          state={patchingState}
          onChange={updatePatchingState}
          onLengthMatchChange={setPatchingLengthsMatch}
        />
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

          <div style={{ ...sectionHeaderStyle, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <span>Sampling</span>
            {/* Subtle reset — only visible when at least one knob has been
                touched, to keep the panel uncluttered in its common state. */}
            {Object.values(off).some((v) => !v) && (
              <button
                onClick={resetSamplingParams}
                title="Restore all sampling defaults"
                style={{
                  fontSize: 9, padding: "0 4px", background: "transparent",
                  border: "1px solid #2a2a3a", color: "#6666aa", cursor: "pointer",
                  textTransform: "none", letterSpacing: 0,
                }}
              >reset</button>
            )}
          </div>
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

            <label
              style={dim(off.n || !!targetSessionB, labelStyle)}
              title={targetSessionB
                ? "Fan-out is disabled while Session B is selected (A/B and fan-out are exclusive)."
                : "Fan out one Run into N variants. Axis below controls what varies across the N runs."}
            >N</label>
            <input className="num-input" type="number" min="1" max="32" value={numSeeds}
              onChange={(e) => setNumSeeds(num(e.target.value, numSeeds))}
              disabled={!!targetSessionB}
              style={dim(off.n || !!targetSessionB, numInputStyle)} />

            <label
              style={dim(numSeeds < 2 || !!targetSessionB, labelStyle)}
              title="Which knob varies across the N fan-out runs. 'seed' draws independent seeds (explores sampling noise); the others linspace the selected knob from → to (explores sensitivity to one parameter)."
            >sweep</label>
            <select
              value={sweepAxis}
              onChange={(e) => setSweepAxis(e.target.value as SweepAxis)}
              disabled={numSeeds < 2 || !!targetSessionB}
              style={{ gridColumn: "2 / -1", fontSize: 11, padding: "1px 4px", background: "#0f1626", color: "#e0e0f0", border: "1px solid #1a2540", borderRadius: 3 }}
            >
              <option value="seed">seed (sampling noise)</option>
              <option value="temperature">temperature</option>
              <option value="top_p">top_p (nucleus)</option>
              <option value="top_k">top_k</option>
              <option value="min_p">min_p</option>
            </select>

            {sweepAxis !== "seed" && numSeeds >= 2 && !targetSessionB && (
              <>
                <label style={labelStyle} title="Starting value for the sweep axis (inclusive).">from</label>
                <input className="num-input" type="number" step={AXIS_DEFAULTS[sweepAxis].step} value={sweepFrom}
                  onChange={(e) => setSweepFrom(num(e.target.value, sweepFrom))}
                  style={numInputStyle} />
                <label style={labelStyle} title="Ending value for the sweep axis (inclusive). N runs evenly spaced.">to</label>
                <input className="num-input" type="number" step={AXIS_DEFAULTS[sweepAxis].step} value={sweepTo}
                  onChange={(e) => setSweepTo(num(e.target.value, sweepTo))}
                  style={numInputStyle} />
              </>
            )}

            {/* Secondary axis — turns the fan-out into an N×M grid. */}
            {numSeeds >= 2 && !targetSessionB && (
              <>
                <label
                  style={dim(sweepAxis2 === "none", labelStyle)}
                  title="Optional secondary axis for a 2D parameter grid. When set, total runs = N × M. Keep M small (≤4) — run count multiplies."
                >axis{"\u00b2"}</label>
                <select
                  value={sweepAxis2}
                  onChange={(e) => setSweepAxis2(e.target.value as typeof sweepAxis2)}
                  style={{ gridColumn: "2 / -1", fontSize: 11, padding: "1px 4px", background: "#0f1626", color: "#e0e0f0", border: "1px solid #1a2540", borderRadius: 3 }}
                >
                  <option value="none">none (1D sweep)</option>
                  <option value="temperature" disabled={sweepAxis === "temperature"}>temperature</option>
                  <option value="top_p" disabled={sweepAxis === "top_p"}>top_p</option>
                  <option value="top_k" disabled={sweepAxis === "top_k"}>top_k</option>
                  <option value="min_p" disabled={sweepAxis === "min_p"}>min_p</option>
                </select>

                {sweepAxis2 !== "none" && (
                  <>
                    <label style={labelStyle} title="Number of points along the secondary axis (capped at 8 — N × M multiplies).">M</label>
                    <input className="num-input" type="number" min="2" max="8" value={numSeeds2}
                      onChange={(e) => setNumSeeds2(num(e.target.value, numSeeds2))}
                      style={numInputStyle} />
                    <label style={labelStyle} title="Secondary axis range: total runs = N × M with the grand total shown in the button.">from{"\u00b2"}</label>
                    <input className="num-input" type="number" step={AXIS_DEFAULTS[sweepAxis2].step} value={sweepFrom2}
                      onChange={(e) => setSweepFrom2(num(e.target.value, sweepFrom2))}
                      style={numInputStyle} />
                    <label style={labelStyle}>to{"\u00b2"}</label>
                    <input className="num-input" type="number" step={AXIS_DEFAULTS[sweepAxis2].step} value={sweepTo2}
                      onChange={(e) => setSweepTo2(num(e.target.value, sweepTo2))}
                      style={numInputStyle} />
                  </>
                )}
              </>
            )}
          </div>
        </>
      )}

      <div style={{ display: "flex", gap: 4 }}>
        {!isRunning ? (
          <>
            <button
              onClick={handleRun}
              disabled={!targetSession || (backendProbed && !backendOnline)}
              title={backendProbed && !backendOnline ? "Backend offline — wait for reconnect" : undefined}
            >Run</button>
            {sessions.length >= 2 && (
              <button
                onClick={handleRunAll}
                disabled={backendProbed && !backendOnline}
                title={`Fan this prompt+params out to every loaded session (${sessions.length} total). Does not combine with seed/knob sweeps or A/B — each session gets one run.`}
                style={{ background: "#16213e", border: "1px solid #1a5276", color: "#a0c0e0" }}
              >Run on all ({sessions.length})</button>
            )}
          </>
        ) : (
          <>
            {/* Stop keeps partial output; Cancel discards it. Both halt the
                stream and flip isRunning false via useStopCancel. */}
            <button onClick={makeHandler("stop", cancelAll)}
                    style={{ background: "#6b5020" }}
                    title="Halt generation; keep what's been streamed so far.">Stop</button>
            <button onClick={makeHandler("cancel", cancelAll)}
                    style={{ background: "#6b2020" }}
                    title="Halt generation and discard the partial output.">Cancel</button>
          </>
        )}
      </div>

      {error && <div style={{ color: "#ff6b6b", fontSize: 12 }}>{error}</div>}
    </div>
  );
}
