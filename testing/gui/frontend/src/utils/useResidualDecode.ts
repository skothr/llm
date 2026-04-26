import { useEffect, useState } from "react";

export type ResidualDecodeToken = { token: string; logit: number };
export type ResidualDecodeResponse = {
  top_tokens: ResidualDecodeToken[];
  bottom_tokens: ResidualDecodeToken[];
  prompt_tokens: string[];
};

export function useResidualDecode(
  sessionName: string | undefined,
  prompt: string | undefined,
  layer: number | undefined,
  sublayer: "attn" | "ffn" | "embed" | undefined,
  position: number | undefined,
  topK: number = 10,
): { data: ResidualDecodeResponse | null; error: string | null; loading: boolean } {
  const [data, setData] = useState<ResidualDecodeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (
      sessionName === undefined
      || prompt === undefined
      || layer === undefined
      || sublayer === undefined
      || position === undefined
    ) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    const ctl = new AbortController();
    setLoading(true);
    setError(null);
    setData(null);
    fetch(`/api/sessions/${encodeURIComponent(sessionName)}/decode-residual`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt, layer, sublayer, position, top_k: topK }),
      signal: ctl.signal,
    })
      .then(async (r) => {
        if (!r.ok) {
          const text = await r.text();
          throw new Error(text || `HTTP ${r.status}`);
        }
        return r.json();
      })
      .then((j: ResidualDecodeResponse) => {
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
  }, [sessionName, prompt, layer, sublayer, position, topK]);

  return { data, error, loading };
}
