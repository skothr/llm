import { useMemo, useState } from "react";
import { useResidualGrid } from "../../utils/useResidualGrid";

type Props = {
  sessionName: string;
  prompt: string;
  promptTokens?: string[];
  initialPosition?: number;
};

export function LensTraceStrip({
  sessionName, prompt, promptTokens, initialPosition,
}: Props) {
  const { data, error, loading } = useResidualGrid(sessionName, prompt, 1);

  const tokens = data?.prompt_tokens ?? promptTokens ?? [];
  const fallbackInitial = tokens.length > 0 ? tokens.length - 1 : 0;
  const [position, setPosition] = useState<number>(initialPosition ?? fallbackInitial);

  // Effective position is clamped once tokens are known.
  const effectivePos = tokens.length > 0
    ? Math.min(Math.max(0, position), tokens.length - 1)
    : position;

  const rows = useMemo(() => {
    if (!data) return [];
    return data.cells
      .filter((c) => c.position === effectivePos)
      .sort((a, b) => {
        if (a.layer !== b.layer) return a.layer - b.layer;
        return a.sublayer === "attn" ? -1 : 1;
      });
  }, [data, effectivePos]);

  return (
    <div
      data-testid="lens-trace-strip"
      style={{
        width: 200, flexShrink: 0,
        borderLeft: "1px solid #2a2a3a", paddingLeft: 12,
        maxHeight: 600, overflowY: "auto",
      }}
    >
      <div style={{ fontSize: 12, color: "#a0a0c0", marginBottom: 6, fontWeight: "bold" }}>
        Lens trace
      </div>
      {tokens.length > 0 && (
        <label style={{ fontSize: 11, color: "#8888aa", display: "block", marginBottom: 8 }}>
          Position:{" "}
          <select
            value={effectivePos}
            onChange={(e) => setPosition(Number(e.target.value))}
            style={{
              background: "#0f1626", color: "#e0e0f0", border: "1px solid #1a5276",
              borderRadius: 3, padding: "1px 4px", fontSize: 11, marginLeft: 4,
            }}
          >
            {tokens.map((tok, i) => (
              <option key={i} value={i}>
                {i}: {tok}
              </option>
            ))}
          </select>
        </label>
      )}
      {loading && (
        <div style={{ fontSize: 11, color: "#888" }}>decoding lens grid…</div>
      )}
      {error && (
        <div style={{ fontSize: 11, color: "#e88" }}>error: {error}</div>
      )}
      {data && rows.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {rows.map((c) => (
            <div
              key={`${c.layer}-${c.sublayer}`}
              style={{
                display: "flex", justifyContent: "space-between",
                fontSize: 11, fontFamily: "monospace",
                color: "#cfc", padding: "1px 4px",
                background: c.sublayer === "ffn" ? "#1a1f2a" : "transparent",
              }}
            >
              <span style={{ color: "#888" }}>L{c.layer}.{c.sublayer}</span>
              <span>{c.tokens[0]?.token ?? ""}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
