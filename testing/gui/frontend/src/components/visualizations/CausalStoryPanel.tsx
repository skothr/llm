import { useEffect, useRef, useState } from "react";
import type { CausalStory } from "../../utils/causalStory";
import { storyToMarkdown, storyNodeId, type StoryNodeId } from "../../utils/causalStory";

type Props = {
  story: CausalStory;
  promptToken?: string;
  selectedNodeId?: StoryNodeId | null;
  onSelectNode?: (id: StoryNodeId | null) => void;
  /** When set, only nodes with index < playStep are revealed; when null, all nodes are visible. */
  playStep?: number | null;
  onPlayStepChange?: (step: number | null) => void;
};

const PLAY_STEP_INTERVAL_MS = 600;

export function CausalStoryPanel({
  story, promptToken, selectedNodeId, onSelectNode, playStep, onPlayStepChange,
}: Props) {
  const headerPos = promptToken ? `pos ${story.position} ("${promptToken}")` : `pos ${story.position}`;
  const interactive = onSelectNode !== undefined;
  const playable = onPlayStepChange !== undefined;
  const playing = playStep !== null && playStep !== undefined;

  const totalNodes = story.nodes.length;
  // Local state when no parent control is provided (component remains usable standalone).
  const [localStep, setLocalStep] = useState<number | null>(null);
  const effectiveStep = playable ? (playStep ?? null) : localStep;
  const setStep = playable ? onPlayStepChange! : setLocalStep;

  // Auto-advance when in playback. Cleared on unmount or when step reaches total.
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (effectiveStep === null) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }
    if (effectiveStep >= totalNodes) {
      // Last frame held briefly, then clear.
      const timeout = setTimeout(() => setStep(null), PLAY_STEP_INTERVAL_MS);
      return () => clearTimeout(timeout);
    }
    if (intervalRef.current === null) {
      intervalRef.current = setInterval(() => {
        setStep((effectiveStep ?? 0) + 1);
      }, PLAY_STEP_INTERVAL_MS);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
    // setStep / setStep changes per render; effectiveStep + totalNodes drive the schedule.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveStep, totalNodes]);

  const handleCopyMarkdown = () => {
    const md = storyToMarkdown(story, promptToken);
    navigator.clipboard?.writeText(md).catch(() => undefined);
  };

  const handlePlay = () => {
    if (totalNodes === 0) return;
    setStep(playing ? null : 0);
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
          onClick={handlePlay}
          aria-pressed={playing}
          data-testid="causal-story-play"
          style={{
            fontSize: 11, padding: "2px 8px", marginLeft: "auto",
            background: playing ? "#2a3a55" : "#1a2438",
            color: "#cfd6e6", border: "1px solid #2a3a55",
            borderRadius: 3, cursor: totalNodes === 0 ? "default" : "pointer",
            opacity: totalNodes === 0 ? 0.4 : 1,
          }}
          disabled={totalNodes === 0}
        >
          {playing ? "⏸ pause" : "▶ play"}
        </button>
        <button
          onClick={handleCopyMarkdown}
          data-testid="causal-story-copy-md"
          style={{
            fontSize: 11, padding: "2px 8px",
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
            // During playback, only nodes with index < step are shown.
            const isRevealed = effectiveStep === null || i < effectiveStep;
            return (
              <div
                key={id}
                data-testid={`causal-story-row-${id}`}
                data-revealed={isRevealed ? "true" : "false"}
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
                  opacity: isRevealed ? 1 : 0,
                  transition: "opacity 400ms ease-in",
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
