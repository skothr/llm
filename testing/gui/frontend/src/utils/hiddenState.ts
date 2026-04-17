export interface EncodedHiddenState {
  shape: [number, number];
  b64: string;
}

export function decodeHiddenState(encoded: EncodedHiddenState): Float32Array {
  const bin = atob(encoded.b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function sliceHiddenStatePosition(
  encoded: EncodedHiddenState,
  position: number,
): Float32Array {
  const [seqLen, hiddenSize] = encoded.shape;
  if (position < 0 || position >= seqLen) {
    throw new Error(`position ${position} out of range [0, ${seqLen})`);
  }
  const full = decodeHiddenState(encoded);
  return full.subarray(position * hiddenSize, (position + 1) * hiddenSize);
}
