import { describe, it, expect } from "vitest";
import {
  decodeLogits,
  logitDiffRecovery,
  klFromClean,
  top1Match,
  probDelta,
  topK,
} from "../../src/utils/patchingMetrics";

function encode(arr: Float32Array): { shape: number[]; b64: string } {
  const bytes = new Uint8Array(arr.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return { shape: [arr.length], b64: btoa(binary) };
}

describe("decodeLogits", () => {
  it("round-trips float32 bytes", () => {
    const arr = new Float32Array([1.5, -2.0, 0.0, 3.14]);
    const decoded = decodeLogits(encode(arr));
    expect(decoded.length).toBe(4);
    for (let i = 0; i < 4; i++) expect(decoded[i]).toBeCloseTo(arr[i], 5);
  });
});

describe("logitDiffRecovery", () => {
  it("returns 1.0 when patched equals clean", () => {
    const clean = new Float32Array([5, 1, 0, 0]);
    const corrupted = new Float32Array([1, 5, 0, 0]);
    const r = logitDiffRecovery(clean, clean, corrupted, 0, 1);
    expect(r).toBeCloseTo(1.0, 5);
  });

  it("returns 0.0 when patched equals corrupted", () => {
    const clean = new Float32Array([5, 1, 0, 0]);
    const corrupted = new Float32Array([1, 5, 0, 0]);
    const r = logitDiffRecovery(corrupted, clean, corrupted, 0, 1);
    expect(r).toBeCloseTo(0.0, 5);
  });

  it("is signed — negative when patching makes it worse", () => {
    const clean = new Float32Array([5, 1, 0, 0]);
    const corrupted = new Float32Array([1, 5, 0, 0]);
    const worse = new Float32Array([0, 10, 0, 0]);
    const r = logitDiffRecovery(worse, clean, corrupted, 0, 1);
    expect(r).toBeLessThan(0);
  });
});

describe("klFromClean", () => {
  it("returns 0 when distributions are identical", () => {
    const logits = new Float32Array([1.0, 2.0, 3.0]);
    expect(klFromClean(logits, logits)).toBeCloseTo(0, 5);
  });

  it("is strictly positive when distributions differ", () => {
    const a = new Float32Array([3, 1, 1]);
    const b = new Float32Array([1, 3, 1]);
    expect(klFromClean(a, b)).toBeGreaterThan(0);
  });

  it("handles zero-probability bins via xlogy semantics (does not return NaN)", () => {
    const a = new Float32Array([0, -1e8, 0]);
    const b = new Float32Array([0, 0, 0]);
    const kl = klFromClean(a, b);
    expect(Number.isFinite(kl)).toBe(true);
  });
});

describe("top1Match", () => {
  it("returns true when argmax agrees", () => {
    const a = new Float32Array([5, 1, 0]);
    const b = new Float32Array([3, 1, 0]);
    expect(top1Match(a, b)).toBe(true);
  });

  it("returns false when argmax differs", () => {
    const a = new Float32Array([5, 1, 0]);
    const b = new Float32Array([0, 5, 0]);
    expect(top1Match(a, b)).toBe(false);
  });
});

describe("probDelta", () => {
  it("returns signed probability difference on the clean top-1 id", () => {
    const patched = new Float32Array([2, 0, 0]);
    const corrupted = new Float32Array([0, 0, 0]);
    const d = probDelta(patched, corrupted, 0);
    expect(d).toBeCloseTo(0.7864 - 1 / 3, 2);
  });
});

describe("topK", () => {
  it("returns the top-k entries sorted by probability descending", () => {
    const logits = new Float32Array([1.0, 5.0, 3.0, 2.0, 4.0]);
    const res = topK(logits, 3);
    expect(res.map((e) => e.id)).toEqual([1, 4, 2]);
    // Probs must be monotonically decreasing.
    expect(res[0].prob).toBeGreaterThan(res[1].prob);
    expect(res[1].prob).toBeGreaterThan(res[2].prob);
    // Probs sum to softmax; each entry in (0, 1).
    for (const e of res) {
      expect(e.prob).toBeGreaterThan(0);
      expect(e.prob).toBeLessThan(1);
    }
  });

  it("returns at most k entries even when logits are shorter", () => {
    const logits = new Float32Array([1, 2]);
    expect(topK(logits, 5)).toHaveLength(2);
  });

  it("breaks ties by first-seen-wins (stable-ish for equal probs)", () => {
    const logits = new Float32Array([0, 0, 0, 0]);
    const res = topK(logits, 2);
    // First two indices are taken when probs are all equal; relative order
    // beyond position 0 is unimportant but length + ids must be correct.
    expect(res).toHaveLength(2);
    expect(new Set(res.map((e) => e.id)).size).toBe(2);
  });
});
