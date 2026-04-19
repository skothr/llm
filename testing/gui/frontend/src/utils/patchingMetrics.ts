/**
 * Pure-function metric helpers for the activation-patching heatmap.
 * All functions operate on Float32Array logit vectors (one per cell).
 * Metric computation happens client-side so the metric dropdown switches
 * without a backend round-trip — same design as LogitLensHeatmap.
 */

import type { EncodedTensor } from "../types/api";

/** Decode a base64-float32 EncodedTensor to a Float32Array. */
export function decodeLogits(enc: EncodedTensor): Float32Array {
  const binary = atob(enc.b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

/**
 * logit_diff_recovery = (Δ_patched − Δ_corrupted) / (Δ_clean − Δ_corrupted)
 * where Δ = logit(correct) − logit(incorrect).
 *
 * Returns 1.0 when patched == clean, 0.0 when patched == corrupted, signed.
 * Returns 0 when the denominator collapses — caller should ensure the
 * prompt pair actually has a clean/corrupt logit-diff signal.
 */
export function logitDiffRecovery(
  patched: Float32Array,
  clean: Float32Array,
  corrupted: Float32Array,
  correctId: number,
  incorrectId: number,
): number {
  const deltaPatched = patched[correctId] - patched[incorrectId];
  const deltaClean = clean[correctId] - clean[incorrectId];
  const deltaCorr = corrupted[correctId] - corrupted[incorrectId];
  const denom = deltaClean - deltaCorr;
  if (denom === 0) return 0;
  return (deltaPatched - deltaCorr) / denom;
}

/**
 * KL(softmax(patched) ‖ softmax(clean)) in nats.
 *
 * Clamps q to avoid log(0) when the reference distribution has zero
 * probability under machine epsilon. p=0 bins don't contribute (xlogy).
 */
export function klFromClean(patched: Float32Array, clean: Float32Array): number {
  const p = softmax(patched);
  const q = softmax(clean);
  const floor = 1e-45;
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] <= 0) continue;
    const qi = Math.max(q[i], floor);
    kl += p[i] * Math.log(p[i] / qi);
  }
  return kl;
}

/** True when argmax(patched) == argmax(clean). */
export function top1Match(patched: Float32Array, clean: Float32Array): boolean {
  return argmax(patched) === argmax(clean);
}

/**
 * p_patched(cleanTopId) − p_corrupted(cleanTopId).
 * Signed probability delta on the clean-top-1 token.
 */
export function probDelta(
  patched: Float32Array,
  corrupted: Float32Array,
  cleanTopId: number,
): number {
  const pp = softmax(patched);
  const pc = softmax(corrupted);
  return pp[cleanTopId] - pc[cleanTopId];
}

function argmax(arr: Float32Array): number {
  let best = 0;
  let bestVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > bestVal) { bestVal = arr[i]; best = i; }
  }
  return best;
}

export interface TopKEntry {
  id: number;
  prob: number;
}

/**
 * Top-k indices by probability after softmax. Used by the AP pin card to show
 * top tokens for the clicked cell. O(n log k) via partial sort; cheap for
 * vocab sizes up to ~100k and k ≤ 10.
 */
export function topK(logits: Float32Array, k: number): TopKEntry[] {
  const probs = softmax(logits);
  const result: TopKEntry[] = [];
  for (let i = 0; i < probs.length; i++) {
    if (result.length < k) {
      result.push({ id: i, prob: probs[i] });
      if (result.length === k) result.sort((a, b) => b.prob - a.prob);
    } else if (probs[i] > result[k - 1].prob) {
      result[k - 1] = { id: i, prob: probs[i] };
      for (let j = k - 1; j > 0 && result[j].prob > result[j - 1].prob; j--) {
        [result[j], result[j - 1]] = [result[j - 1], result[j]];
      }
    }
  }
  return result;
}
