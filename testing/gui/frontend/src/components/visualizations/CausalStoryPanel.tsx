import type { CausalStory } from "../../utils/causalStory";

type Props = {
  story: CausalStory;
  promptToken?: string;   // the token at story.position; used for header
};

export function CausalStoryPanel({ story, promptToken }: Props) {
  const headerPos = promptToken ? `pos ${story.position} ("${promptToken}")` : `pos ${story.position}`;

  return (
    <div
      data-testid="causal-story-panel"
      style={{
        marginTop: 16, paddingTop: 12,
        borderTop: "1px solid #2a2a3a",
      }}
    >
      <div style={{ fontSize: 12, color: "#a0a0c0", fontWeight: "bold", marginBottom: 8 }}>
        Causal Story — {headerPos}
      </div>
      {story.note && (
        <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>
          {story.note}
        </div>
      )}
      {story.nodes.length === 0 ? null : (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {story.nodes.map((n, i) => {
            const isEmbed = n.unit === "embed";
            return (
              <div
                key={`${n.layer}-${n.unit}`}
                style={{
                  display: "flex", gap: 12, alignItems: "baseline",
                  fontSize: 12, fontFamily: "monospace",
                  padding: "2px 6px",
                  background: i % 2 === 0 ? "transparent" : "#15161e",
                  borderRadius: 2,
                }}
              >
                <span style={{ color: "#888", minWidth: 110 }}>
                  L{n.layer} {n.unit}
                </span>
                {isEmbed ? (
                  <span style={{ color: "#888", fontStyle: "italic" }}>
                    input embedding (no lens in V1)
                  </span>
                ) : n.lensTokens.length > 0 ? (
                  <span>
                    <span style={{ color: "#888" }}>residual: </span>
                    <span style={{ color: "#cfc" }}>
                      {n.lensTokens.map((t, j) => (
                        <span key={j}>
                          {j > 0 && <span style={{ color: "#666" }}>{" · "}</span>}
                          {t}
                        </span>
                      ))}
                    </span>
                  </span>
                ) : (
                  <span style={{ color: "#888", fontStyle: "italic" }}>(no lens data)</span>
                )}
              </div>
            );
          })}
        </div>
      )}
      {story.edges.length > 0 && (
        <div style={{ fontSize: 11, color: "#888", marginTop: 8 }}>
          {story.edges.length} edge{story.edges.length === 1 ? "" : "s"} feeding through this circuit
        </div>
      )}
    </div>
  );
}
