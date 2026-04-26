import type { CausalStory } from "../../utils/causalStory";
import { storyToMarkdown, storyNodeId, type StoryNodeId } from "../../utils/causalStory";

type Props = {
  story: CausalStory;
  promptToken?: string;
  selectedNodeId?: StoryNodeId | null;
  onSelectNode?: (id: StoryNodeId | null) => void;
};

export function CausalStoryPanel({ story, promptToken, selectedNodeId, onSelectNode }: Props) {
  const headerPos = promptToken ? `pos ${story.position} ("${promptToken}")` : `pos ${story.position}`;
  const interactive = onSelectNode !== undefined;

  const handleCopyMarkdown = () => {
    const md = storyToMarkdown(story, promptToken);
    navigator.clipboard?.writeText(md).catch(() => undefined);
  };

  return (
    <div
      data-testid="causal-story-panel"
      style={{
        marginTop: 16, paddingTop: 12,
        borderTop: "1px solid #2a2a3a",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: "#a0a0c0", fontWeight: "bold" }}>
          Causal Story — {headerPos}
        </div>
        <button
          onClick={handleCopyMarkdown}
          data-testid="causal-story-copy-md"
          style={{
            fontSize: 11, padding: "2px 8px", marginLeft: "auto",
            background: "#1a2438", color: "#cfd6e6", border: "1px solid #2a3a55",
            borderRadius: 3, cursor: "pointer",
          }}
        >
          copy as markdown
        </button>
      </div>
      {story.note && (
        <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>
          {story.note}
        </div>
      )}
      {story.nodes.length === 0 ? null : (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {story.nodes.map((n, i) => {
            const id = storyNodeId(n.layer, n.unit);
            const isSelected = selectedNodeId === id;
            const isEmbed = n.unit === "embed";
            const handleClick = interactive
              ? () => onSelectNode!(isSelected ? null : id)
              : undefined;
            return (
              <div
                key={id}
                data-testid={`causal-story-row-${id}`}
                onClick={handleClick}
                style={{
                  display: "flex", gap: 12, alignItems: "baseline",
                  fontSize: 12, fontFamily: "monospace",
                  padding: "2px 6px",
                  background: isSelected
                    ? "#2a3a55"
                    : (i % 2 === 0 ? "transparent" : "#15161e"),
                  borderLeft: isSelected ? "3px solid #8abaff" : "3px solid transparent",
                  borderRadius: 2,
                  cursor: interactive ? "pointer" : "default",
                }}
              >
                <span style={{ color: isSelected ? "#cfd6e6" : "#888", minWidth: 110 }}>
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
