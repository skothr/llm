import { useResidualDecode } from "../../utils/useResidualDecode";

type Props = {
  sessionName: string;
  prompt: string;
  layer: number;
  sublayer: "attn" | "ffn" | "embed";
  position: number;
};

export function ResidualDecodeBlock({
  sessionName, prompt, layer, sublayer, position,
}: Props) {
  const { data, error, loading } = useResidualDecode(
    sessionName, prompt, layer, sublayer, position, 10,
  );

  return (
    <div style={{ marginTop: 12, paddingTop: 8, borderTop: "1px solid #2a2a3a" }}>
      <div style={{ fontSize: 12, color: "#a0a0c0", marginBottom: 6 }}>
        — Logit lens at (L{layer}, {sublayer}, pos {position})
      </div>
      {loading && (
        <div style={{ fontSize: 12, color: "#888" }}>decoding…</div>
      )}
      {error && (
        <div style={{ fontSize: 12, color: "#e88" }}>error: {error}</div>
      )}
      {data && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div>
            <div style={{ fontSize: 11, color: "#7c7", fontWeight: "bold" }}>Promoted</div>
            {data.top_tokens.map((t, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, fontFamily: "monospace", color: "#cfc" }}>
                <span>{t.token}</span>
                <span>+{t.logit.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div>
            <div style={{ fontSize: 11, color: "#c77", fontWeight: "bold" }}>Suppressed</div>
            {data.bottom_tokens.map((t, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, fontFamily: "monospace", color: "#fcc" }}>
                <span>{t.token}</span>
                <span>{t.logit.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
