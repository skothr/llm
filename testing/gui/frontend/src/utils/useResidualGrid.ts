import { useEffect, useState } from "react";

export type ResidualGridToken = { token: string; logit: number };
export type ResidualGridCell = {
  layer: number;
  sublayer: "attn" | "ffn" | "embed";
  position: number;
  tokens: ResidualGridToken[];
};
export type ResidualGridResponse = {
  cells: ResidualGridCell[];
  prompt_tokens: string[];
  num_layers: number;
};

export function useResidualGrid(
  sessionName: string | undefined,
  prompt: string | undefined,
  topK: number = 1,
): { data: ResidualGridResponse | null; error: string | null; loading: boolean } {
  const [data, setData] = useState<ResidualGridResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (sessionName === undefined || prompt === undefined) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    const ctl = new AbortController();
    setLoading(true);
    setError(null);
    setData(null);
    fetch(`/api/sessions/${encodeURIComponent(sessionName)}/decode-residual-grid`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt, top_k: topK }),
      signal: ctl.signal,
    })
      .then(async (r) => {
        if (!r.ok) {
          const text = await r.text();
          throw new Error(text || `HTTP ${r.status}`);
        }
        return r.json();
      })
      .then((j: ResidualGridResponse) => {
        if (!ctl.signal.aborted) {
          setData(j);
          setLoading(false);
        }
      })
      .catch((e: unknown) => {
        if (ctl.signal.aborted) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
    return () => ctl.abort();
  }, [sessionName, prompt, topK]);

  return { data, error, loading };
}
