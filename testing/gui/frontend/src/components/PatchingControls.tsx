/**
 * Conditional subpanel rendered by ProbePanel when operation is
 * "activation-patching". Owns patching-only local state (debounced
 * tokenize counts); does NOT own the Run button (ProbePanel does).
 *
 * Layout:
 *   [clean prompt textarea]              [N tokens]
 *   [corrupted prompt textarea]          [M tokens ✓/✗]
 *   direction: (●) denoise  ( ) noise
 *   measure @ [-1]
 *   target:   (●) auto-pick  ( ) manual
 *     correct: [ ]   incorrect: [ ]
 */
import { useEffect, useState } from "react";

export type PatchingMode = "exact" | "approx" | "approx_head" | "edge" | "circuit" | "approx_neuron";

export interface PatchingState {
  cleanPrompt: string;
  corruptedPrompt: string;
  direction: "denoise" | "noise";
  measurementPos: number;
  tokenPairMode: "auto" | "manual";
  manualCorrect: string;
  manualIncorrect: string;
  mode: PatchingMode;
  top_k_edges: number;
  top_k_candidates: number;
  tau: number;
  top_k_neurons: number;
  n_steps: number;
}

export const DEFAULT_PATCHING_STATE: PatchingState = {
  cleanPrompt: "",
  corruptedPrompt: "",
  direction: "denoise",
  measurementPos: -1,
  tokenPairMode: "auto",
  manualCorrect: "",
  manualIncorrect: "",
  // Default to gradient AP — single forward+backward, fits in <2 GB extra
  // VRAM even for 3B fp16. Exact AP is fine on TinyLlama but OOMs 8 GB
  // GPUs on 3B+, so don't expose it as the path of least resistance.
  mode: "approx",
  top_k_edges: 200,
  top_k_candidates: 2000,
  tau: 0.02,
  top_k_neurons: 200,
  n_steps: 1,
};

interface Props {
  targetSession: string;
  state: PatchingState;
  onChange: (patch: Partial<PatchingState>) => void;
  /** True when clean and corrupted tokenize to same length. Bound to Run enable. */
  onLengthMatchChange: (match: boolean) => void;
}

const labelStyle: React.CSSProperties = {
  fontFamily: "monospace", color: "#a0a0c0", fontSize: 12,
};

const gridStyle: React.CSSProperties = {
  display: "grid", gridTemplateColumns: "auto 1fr", gap: "4px 8px",
  alignItems: "center", fontSize: 12,
};

export function PatchingControls({ targetSession, state, onChange, onLengthMatchChange }: Props) {
  const [cleanTokens, setCleanTokens] = useState<number | null>(null);
  const [corrTokens, setCorrTokens] = useState<number | null>(null);

  // Debounced tokenize probe for the clean prompt — mirrors the pattern
  // ProbePanel uses for its context-budget display. 250 ms debounce keeps
  // typing snappy without flooding the backend.
  useEffect(() => {
    if (!targetSession || !state.cleanPrompt) { setCleanTokens(null); return; }
    const ac = new AbortController();
    const t = setTimeout(() => {
      fetch(`/api/sessions/${targetSession}/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: state.cleanPrompt }),
        signal: ac.signal,
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (d && typeof d.count === "number") setCleanTokens(d.count); })
        .catch(() => { /* abort / offline — leave prior count */ });
    }, 250);
    return () => { clearTimeout(t); ac.abort(); };
  }, [state.cleanPrompt, targetSession]);

  useEffect(() => {
    if (!targetSession || !state.corruptedPrompt) { setCorrTokens(null); return; }
    const ac = new AbortController();
    const t = setTimeout(() => {
      fetch(`/api/sessions/${targetSession}/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: state.corruptedPrompt }),
        signal: ac.signal,
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (d && typeof d.count === "number") setCorrTokens(d.count); })
        .catch(() => { /* abort / offline */ });
    }, 250);
    return () => { clearTimeout(t); ac.abort(); };
  }, [state.corruptedPrompt, targetSession]);

  const lengthsMatch =
    cleanTokens != null && corrTokens != null && cleanTokens === corrTokens && cleanTokens > 0;
  useEffect(() => { onLengthMatchChange(lengthsMatch); }, [lengthsMatch, onLengthMatchChange]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div>
        <textarea
          placeholder="Clean prompt"
          value={state.cleanPrompt}
          onChange={(e) => onChange({ cleanPrompt: e.target.value })}
          rows={2}
          style={{ width: "100%", fontFamily: "monospace", fontSize: 12 }}
        />
        <div style={{ fontSize: 10, color: "#8888aa", textAlign: "right" }}>
          {cleanTokens == null ? "\u2014" : `${cleanTokens} tokens`}
        </div>
      </div>

      <div>
        <textarea
          placeholder="Corrupted prompt"
          value={state.corruptedPrompt}
          onChange={(e) => onChange({ corruptedPrompt: e.target.value })}
          rows={2}
          style={{ width: "100%", fontFamily: "monospace", fontSize: 12 }}
        />
        <div style={{
          fontSize: 10,
          textAlign: "right",
          color: cleanTokens == null || corrTokens == null
            ? "#8888aa"
            : lengthsMatch ? "#6bc06b" : "#c06060",
        }}>
          {corrTokens == null
            ? "\u2014"
            : lengthsMatch
              ? `${corrTokens} tokens \u2713`
              : cleanTokens != null
                ? `${corrTokens} tokens \u2717 (lengths differ: ${cleanTokens} vs ${corrTokens})`
                : `${corrTokens} tokens`}
        </div>
      </div>

      <div style={gridStyle}>
        <label style={labelStyle}>direction</label>
        <div style={{ display: "flex", gap: 10 }}>
          <label style={{ fontSize: 12 }}>
            <input type="radio" name="direction" value="denoise"
              checked={state.direction === "denoise"}
              onChange={() => onChange({ direction: "denoise" })} />
            {" "}denoise
          </label>
          <label style={{ fontSize: 12 }}>
            <input type="radio" name="direction" value="noise"
              checked={state.direction === "noise"}
              onChange={() => onChange({ direction: "noise" })} />
            {" "}noise
          </label>
        </div>

        <label style={labelStyle}>mode</label>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {([
              ["exact", "exact (slow)", "Activation patching: re-run the model with one residual cell swapped from the corrupted prompt's clean cache. Exact but expensive — N_layers × N_tokens forward passes. OOM-risk on small (≤8 GB) GPUs for 3B+ models; prefer 'approx' there."],
              ["approx", "approx", "Attribution patching (gradient AP). Linear approximation of activation patching from a single forward+backward pass — much faster, similar shape, fits in <2 GB extra VRAM. Default."],
              ["approx_head", "per-head", "Per-head attribution patching. Resolves attribution down to individual attention heads (rows = heads, cols = positions)."],
              ["edge", "edge AP", "Edge attribution patching (EAP). Scores writer→reader pairs across the residual stream — yields a graph instead of a heatmap."],
              ["circuit", "circuit (ACDC)", "Cheap-ACDC circuit extraction. BFS over edge attribution above τ to return the connected subgraph that carries the answer."],
              ["approx_neuron", "per-neuron FFN", "Per-neuron FFN attribution (approx). Attribution per individual MLP neuron — too many to plot, rendered as a ranked list."],
            ] as const).map(([value, label, desc]) => (
              <label
                key={value}
                title={desc}
                style={{ fontSize: 12, display: "inline-flex", alignItems: "center", gap: 4, cursor: "help" }}
              >
                <input
                  type="radio"
                  name="mode"
                  value={value}
                  checked={state.mode === value}
                  onChange={() => onChange({ mode: value })}
                />
                {label}
              </label>
            ))}
          </div>
          {state.mode === "edge" && (
            <label style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              top-k edges:
              <input
                type="number"
                min={1}
                max={10000}
                value={state.top_k_edges}
                onChange={(e) =>
                  onChange({ top_k_edges: Math.max(1, Number(e.target.value)) })
                }
                style={{ width: 70 }}
              />
            </label>
          )}
          {state.mode === "circuit" && (
            <div className="row" style={{ marginTop: 6, display: "flex", flexWrap: "wrap", gap: 12 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                top_k_candidates:
                <input
                  type="number"
                  min={1}
                  value={state.top_k_candidates}
                  onChange={(e) =>
                    onChange({ top_k_candidates: Math.max(1, Number(e.target.value)) })
                  }
                  style={{ width: 70 }}
                />
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                {"τ"} (threshold):
                <input
                  type="number"
                  min={0}
                  step={0.005}
                  value={state.tau}
                  onChange={(e) =>
                    onChange({ tau: Math.max(0, Number(e.target.value)) })
                  }
                  style={{ width: 70 }}
                />
              </label>
            </div>
          )}
          {state.mode === "approx_neuron" && (
            <div className="row" style={{ marginTop: 6 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                top_k_neurons:
                <input
                  type="number"
                  min={1}
                  value={state.top_k_neurons}
                  onChange={(e) =>
                    onChange({ top_k_neurons: Math.max(1, Number(e.target.value)) })
                  }
                  style={{ width: 70 }}
                />
              </label>
            </div>
          )}
          {["approx", "approx_head", "approx_neuron", "edge", "circuit"].includes(state.mode) && (
            <label style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
              IG steps:
              <input
                type="number"
                min={1}
                max={50}
                value={state.n_steps}
                onChange={(e) =>
                  onChange({ n_steps: Math.max(1, Math.min(50, Number(e.target.value) || 1)) })
                }
                style={{ width: 60 }}
                title="n_steps=1: first-order AP (fast). n_steps>=2: Integrated Gradients (N× cost, more accurate)."
              />
              <span style={{ color: "#888", fontSize: 11 }}>
                {state.n_steps === 1 ? "(first-order)" : `(IG, ${state.n_steps}×)`}
              </span>
            </label>
          )}
          {(state.mode === "approx" || state.mode === "approx_head" || state.mode === "edge" || state.mode === "circuit" || state.mode === "approx_neuron") && state.tokenPairMode === "auto" && (
            <div style={{ color: "#7f7", fontSize: 11 }}>
              auto-pick uses clean argmax; switch to manual for a specific target
            </div>
          )}
        </div>

        <label style={labelStyle} title="Absolute or negative index. -1 = last token.">measure @</label>
        <input
          type="number"
          value={state.measurementPos}
          onChange={(e) => onChange({ measurementPos: Number(e.target.value) })}
          style={{ width: 60, fontFamily: "monospace", fontSize: 12 }}
        />

        <label style={labelStyle}>target</label>
        <div style={{ display: "flex", gap: 10 }}>
          <label style={{ fontSize: 12 }}>
            <input type="radio" name="tokenPairMode" value="auto"
              checked={state.tokenPairMode === "auto"}
              onChange={() => onChange({ tokenPairMode: "auto" })} />
            {" "}auto-pick
          </label>
          <label style={{ fontSize: 12 }}>
            <input type="radio" name="tokenPairMode" value="manual"
              checked={state.tokenPairMode === "manual"}
              onChange={() => onChange({ tokenPairMode: "manual" })} />
            {" "}manual
          </label>
        </div>

        <label style={labelStyle}>correct</label>
        <input type="text"
          value={state.manualCorrect}
          disabled={state.tokenPairMode !== "manual"}
          onChange={(e) => onChange({ manualCorrect: e.target.value })}
          placeholder="e.g. Paris"
          style={{ width: "100%", fontFamily: "monospace", fontSize: 12 }}
        />

        <label style={labelStyle}>incorrect</label>
        <input type="text"
          value={state.manualIncorrect}
          disabled={state.tokenPairMode !== "manual"}
          onChange={(e) => onChange({ manualIncorrect: e.target.value })}
          placeholder="e.g. Rome"
          style={{ width: "100%", fontFamily: "monospace", fontSize: 12 }}
        />
      </div>
    </div>
  );
}
