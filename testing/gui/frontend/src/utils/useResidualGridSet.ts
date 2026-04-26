import { useEffect, useState } from "react";
import type { ResidualGridResponse } from "./useResidualGrid";

export type ResidualGridSetEntry = {
  prompt: string;
  data: ResidualGridResponse | null;
  error: string | null;
};

export type ResidualGridSetState = {
  results: ResidualGridSetEntry[];
  loading: boolean;
};

/**
 * Fan-out variant of `useResidualGrid` for "ONE circuit, N grids" patterns
 * (Phase 3.18 divergence heatmap). Fires N parallel fetches against
 * `decode-residual-grid`; returns a stable array of results aligned 1-1
 * with the input prompts.
 *
 * Re-fetches when sessionName, prompts, or topK change. Uses
 * `JSON.stringify(prompts)` for dependency comparison since arrays don't
 * have value-equal default semantics in React's dep-array.
 */
export function useResidualGridSet(
  sessionName: string | undefined,
  prompts: string[] | undefined,
  topK: number = 1,
): ResidualGridSetState {
  const [state, setState] = useState<ResidualGridSetState>({
    results: [],
    loading: false,
  });

  const promptsKey = prompts ? JSON.stringify(prompts) : "";

  useEffect(() => {
    if (sessionName === undefined || prompts === undefined || prompts.length === 0) {
      setState({ results: [], loading: false });
      return;
    }
    const ctl = new AbortController();
    setState({
      results: prompts.map((p) => ({ prompt: p, data: null, error: null })),
      loading: true,
    });

    Promise.all(
      prompts.map(async (prompt, idx): Promise<ResidualGridSetEntry> => {
        try {
          const r = await fetch(
            `/api/sessions/${encodeURIComponent(sessionName)}/decode-residual-grid`,
            {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({ prompt, top_k: topK }),
              signal: ctl.signal,
            },
          );
          if (!r.ok) {
            const text = await r.text();
            throw new Error(text || `HTTP ${r.status}`);
          }
          const j = (await r.json()) as ResidualGridResponse;
          return { prompt, data: j, error: null };
        } catch (e: unknown) {
          if (ctl.signal.aborted) {
            return { prompt, data: null, error: null };
          }
          // Per-prompt error isolation: one failed fetch doesn't kill the rest.
          // The caller can inspect entry.error to render a per-cell "fetch failed" state.
          return {
            prompt,
            data: null,
            error: e instanceof Error ? e.message : String(e),
          };
        }
      }),
    ).then((results) => {
      if (!ctl.signal.aborted) {
        setState({ results, loading: false });
      }
    });

    return () => ctl.abort();
    // promptsKey is the stable dep for `prompts`; React's dep array can't
    // value-compare arrays.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionName, promptsKey, topK]);

  return state;
}
