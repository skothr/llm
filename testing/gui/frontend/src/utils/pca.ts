// Small, dependency-free PCA implementation tuned for hidden-state
// visualization. The inputs we expect: a few hundred to a few thousand
// vectors, each of dimension 512–8192 (i.e. an LLM hidden size). Shipping
// a full ml-pca dependency for this is overkill.
//
// Strategy: top-k via power iteration on the implicit covariance. For
// large D (hidden size) and modest N (samples), we never materialize the
// D×D covariance matrix — instead we apply M = X^T X through two
// matrix-vector products (X, then X^T), keeping per-iteration work O(N·D)
// instead of O(D²).
//
// We deflate after finding each component: subtract the projection onto
// v_k from every sample, then repeat. For k=2 this is enough accuracy for
// visualization; we're not claiming publication-grade eigendecomp here.

export interface PCAResult {
  // The 2D (or k-D) projections of each input sample, [N × k] row-major.
  projections: Float32Array[];
  // Principal component vectors (k of them), each length D.
  components: Float32Array[];
  // Fraction of total variance each component explains (sum ≤ 1).
  explainedVariance: number[];
  // Sample mean subtracted before PCA — useful if the caller wants to
  // project additional samples later.
  mean: Float32Array;
}

function norm(v: Float32Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

function scaleInto(out: Float32Array, v: Float32Array, s: number): void {
  for (let i = 0; i < v.length; i++) out[i] = v[i] * s;
}

function normalizeInto(out: Float32Array, v: Float32Array): void {
  const n = norm(v);
  if (n === 0) return;
  scaleInto(out, v, 1 / n);
}

// Compute y = X^T (X v), where X is a centered N×D matrix given as an
// array of rows. Each matrix-vector product is O(N·D) — much cheaper
// than forming the D×D covariance explicitly.
function covMul(X: Float32Array[], v: Float32Array, tmp: Float32Array): Float32Array {
  const N = X.length;
  const D = v.length;
  // tmp = X v  (N-vector)
  for (let i = 0; i < N; i++) {
    let s = 0;
    const row = X[i];
    for (let j = 0; j < D; j++) s += row[j] * v[j];
    tmp[i] = s;
  }
  // out = X^T tmp  (D-vector)
  const out = new Float32Array(D);
  for (let i = 0; i < N; i++) {
    const row = X[i];
    const t = tmp[i];
    for (let j = 0; j < D; j++) out[j] += row[j] * t;
  }
  return out;
}

function powerIter(
  X: Float32Array[],
  D: number,
  iterations: number,
  seed: () => number,
): Float32Array {
  const tmp = new Float32Array(X.length);
  let v = new Float32Array(D);
  for (let i = 0; i < D; i++) v[i] = seed() - 0.5;
  normalizeInto(v, v);
  for (let it = 0; it < iterations; it++) {
    const next = covMul(X, v, tmp);
    const n = norm(next);
    if (n === 0) break;
    for (let i = 0; i < D; i++) v[i] = next[i] / n;
  }
  return v;
}

// Deflate: subtract the projection onto v from each row of X (in place).
function deflate(X: Float32Array[], v: Float32Array): void {
  const N = X.length;
  const D = v.length;
  for (let i = 0; i < N; i++) {
    const row = X[i];
    let dot = 0;
    for (let j = 0; j < D; j++) dot += row[j] * v[j];
    for (let j = 0; j < D; j++) row[j] -= dot * v[j];
  }
}

// Returns an RNG that's stable across calls (seeded from the sample count
// so repeat runs on the same data converge to the same orientation,
// modulo sign flip). Non-cryptographic; xorshift32.
function seededRng(seed: number): () => number {
  let s = seed || 0x12345678;
  return () => {
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
    // Map to [0,1).
    return ((s >>> 0) / 4294967296);
  };
}

export function pca(
  samples: Float32Array[],
  k = 2,
  iterations = 40,
): PCAResult | null {
  const N = samples.length;
  if (N < 2) return null;
  const D = samples[0].length;
  if (D < 1) return null;

  // Center: subtract the mean vector from every sample. We copy into a
  // freshly-allocated buffer so the caller's Float32Arrays aren't
  // mutated, since Float32Array views often alias a shared ArrayBuffer.
  const mean = new Float32Array(D);
  for (let i = 0; i < N; i++) {
    const row = samples[i];
    for (let j = 0; j < D; j++) mean[j] += row[j];
  }
  for (let j = 0; j < D; j++) mean[j] /= N;

  const X: Float32Array[] = new Array(N);
  let totalVar = 0;
  for (let i = 0; i < N; i++) {
    const row = new Float32Array(D);
    const src = samples[i];
    for (let j = 0; j < D; j++) {
      const v = src[j] - mean[j];
      row[j] = v;
      totalVar += v * v;
    }
    X[i] = row;
  }
  totalVar /= N;
  if (totalVar === 0) return null;

  const components: Float32Array[] = [];
  const eigenvalues: number[] = [];
  const rng = seededRng(N);

  for (let c = 0; c < k; c++) {
    const v = powerIter(X, D, iterations, rng);
    components.push(v);
    // Variance along v: (1/N) Σ (row · v)²
    let ev = 0;
    for (let i = 0; i < N; i++) {
      const row = X[i];
      let dot = 0;
      for (let j = 0; j < D; j++) dot += row[j] * v[j];
      ev += dot * dot;
    }
    ev /= N;
    eigenvalues.push(ev);
    deflate(X, v);
  }

  // Now project ORIGINAL centered samples onto each component. Note we
  // must NOT use the deflated X here — deflation was in-place for power
  // iter economy. Reconstruct centered vectors from samples - mean.
  const projections: Float32Array[] = new Array(N);
  for (let i = 0; i < N; i++) {
    const proj = new Float32Array(k);
    const src = samples[i];
    for (let c = 0; c < k; c++) {
      const vec = components[c];
      let dot = 0;
      for (let j = 0; j < D; j++) dot += (src[j] - mean[j]) * vec[j];
      proj[c] = dot;
    }
    projections[i] = proj;
  }

  const explainedVariance = eigenvalues.map((ev) => ev / totalVar);
  return { projections, components, explainedVariance, mean };
}
