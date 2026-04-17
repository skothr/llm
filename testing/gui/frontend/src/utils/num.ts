// Shared numeric input parser. `Number("")` returns 0, `Number("abc")` returns
// NaN — both of which silently stomp user intent. Callers pass the current
// value as the fallback so mid-edit transient states (e.g. a lone minus sign)
// don't snap the controlled input to zero every keystroke.
export const num = (v: string, fallback: number): number => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};
