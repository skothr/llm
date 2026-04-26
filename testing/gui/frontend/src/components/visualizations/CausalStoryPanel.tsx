import { useEffect, useMemo, useRef, useState } from "react";
import type { CausalStory } from "../../utils/causalStory";
import {
  comparativeLensTokens,
  storyToMarkdown,
  storyNodeId,
  type StoryNodeId,
} from "../../utils/causalStory";
import type { ResidualGridResponse } from "../../utils/useResidualGrid";
import { PROMPT_SET_PRESETS, getPresetById } from "../../utils/promptSetLibrary";

type Props = {
  story: CausalStory;
  promptToken?: string;
  selectedNodeId?: StoryNodeId | null;
  onSelectNode?: (id: StoryNodeId | null) => void;
  /** When set, only nodes with index < playStep are revealed; when null, all nodes are visible. */
  playStep?: number | null;
  onPlayStepChange?: (step: number | null) => void;
  /** Phase 3.17 — comparative-story support. Parent owns prompt B + grid B. */
  comparePrompt?: string | null;
  onComparePromptChange?: (next: string | null) => void;
  compareGrid?: ResidualGridResponse | null;
  compareLoading?: boolean;
  compareError?: string | null;
  /** Phase 3.18 — prompt-set / divergence-heatmap support. Parent owns the set + grids. */
  compareSet?: string[];
  onCompareSetChange?: (next: string[]) => void;
};

const PLAY_STEP_INTERVAL_MS = 600;

export function CausalStoryPanel({
  story, promptToken, selectedNodeId, onSelectNode, playStep, onPlayStepChange,
  comparePrompt, onComparePromptChange, compareGrid, compareLoading, compareError,
  compareSet, onCompareSetChange,
}: Props) {
  const headerPos = promptToken ? `pos ${story.position} ("${promptToken}")` : `pos ${story.position}`;
  const interactive = onSelectNode !== undefined;
  const playable = onPlayStepChange !== undefined;
  const playing = playStep !== null && playStep !== undefined;
  const compareEnabled = onComparePromptChange !== undefined;
  const compareActive = compareEnabled && comparePrompt !== null && comparePrompt !== undefined && comparePrompt.length > 0;
  const compareTokens = useMemo(
    () => (compareActive ? comparativeLensTokens(story, compareGrid ?? null, 3) : null),
    [compareActive, story, compareGrid],
  );
  // Local draft state so the input can hold partial text without firing
  // a fetch on every keystroke. Parent only sees the value on commit.
  const [compareDraft, setCompareDraft] = useState<string>(comparePrompt ?? "");
  useEffect(() => {
    setCompareDraft(comparePrompt ?? "");
  }, [comparePrompt]);
  const [compareInputOpen, setCompareInputOpen] = useState<boolean>(compareActive);
  // Phase 3.18 — prompt-set draft (textarea, one prompt per line).
  const compareSetEnabled = onCompareSetChange !== undefined;
  const compareSetActive = compareSetEnabled && compareSet !== undefined && compareSet.length > 0;
  const [setDraft, setSetDraft] = useState<string>((compareSet ?? []).join("\n"));
  useEffect(() => {
    setSetDraft((compareSet ?? []).join("\n"));
  }, [compareSet]);
  const [setInputOpen, setSetInputOpen] = useState<boolean>(compareSetActive);

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
    const md =
      compareActive && compareTokens !== null
        ? storyToMarkdown(story, promptToken, {
            compareTokens,
            comparePrompt: comparePrompt as string,
          })
        : storyToMarkdown(story, promptToken);
    navigator.clipboard?.writeText(md).catch(() => undefined);
  };

  const handleCompareCommit = () => {
    if (!onComparePromptChange) return;
    const trimmed = compareDraft.trim();
    onComparePromptChange(trimmed.length === 0 ? null : trimmed);
  };
  const handleCompareClear = () => {
    if (!onComparePromptChange) return;
    setCompareDraft("");
    onComparePromptChange(null);
  };

  const handleSetCommit = () => {
    if (!onCompareSetChange) return;
    const lines = setDraft
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    onCompareSetChange(lines);
  };
  const handleSetClear = () => {
    if (!onCompareSetChange) return;
    setSetDraft("");
    onCompareSetChange([]);
  };

  const [presetSelection, setPresetSelection] = useState<string>("");
  const [presetCaption, setPresetCaption] = useState<string | null>(null);
  const handlePresetLoad = (id: string) => {
    if (!onCompareSetChange) return;
    setPresetSelection(id);
    if (id === "") {
      setPresetCaption(null);
      return;
    }
    const preset = getPresetById(id);
    if (!preset) return;
    setSetDraft(preset.prompts.join("\n"));
    onCompareSetChange(preset.prompts);
    setSetInputOpen(true);
    setPresetCaption(
      `${preset.label} — recommended main prompt: "${preset.recommendedMainPrompt}". ${preset.description}`,
    );
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
        <span style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          {compareEnabled && (
            <button
              onClick={() => setCompareInputOpen((v) => !v)}
              aria-pressed={compareInputOpen}
              data-testid="causal-story-compare-toggle"
              style={{
                fontSize: 11, padding: "2px 8px",
                background: compareActive ? "#3a2a55" : (compareInputOpen ? "#2a3a55" : "#1a2438"),
                color: "#cfd6e6", border: "1px solid #2a3a55",
                borderRadius: 3, cursor: "pointer",
              }}
            >
              {compareActive ? "✓ compare on" : "compare…"}
            </button>
          )}
          {compareSetEnabled && (
            <button
              onClick={() => setSetInputOpen((v) => !v)}
              aria-pressed={setInputOpen}
              data-testid="causal-story-set-toggle"
              style={{
                fontSize: 11, padding: "2px 8px",
                background: compareSetActive ? "#3a2a55" : (setInputOpen ? "#2a3a55" : "#1a2438"),
                color: "#cfd6e6", border: "1px solid #2a3a55",
                borderRadius: 3, cursor: "pointer",
              }}
            >
              {compareSetActive ? `✓ set (${compareSet?.length ?? 0})` : "prompt set…"}
            </button>
          )}
          <button
            onClick={handlePlay}
            aria-pressed={playing}
            data-testid="causal-story-play"
            style={{
              fontSize: 11, padding: "2px 8px",
              background: playing ? "#2a3a55" : "#1a2438",
              color: "#cfd6e6", border: "1px solid #2a3a55",
              borderRadius: 3, cursor: totalNodes === 0 ? "default" : "pointer",
              opacity: totalNodes === 0 ? 0.4 : 1,
            }}
            disabled={totalNodes === 0}
          >
            {playing ? "⏸ pause" : "▶ play"}
          </button>
        </span>
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
      {compareEnabled && compareInputOpen && (
        <div
          data-testid="causal-story-compare-input-row"
          style={{ display: "flex", gap: 6, marginBottom: 8, alignItems: "center" }}
        >
          <input
            data-testid="causal-story-compare-input"
            type="text"
            value={compareDraft}
            onChange={(e) => setCompareDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") handleCompareCommit(); }}
            placeholder="compare with another prompt (e.g. 'The capital of Italy')"
            style={{
              flex: 1, fontSize: 12, padding: "3px 6px",
              background: "#0e0e12", color: "#cfd6e6",
              border: "1px solid #2a3a55", borderRadius: 3,
            }}
          />
          <button
            onClick={handleCompareCommit}
            data-testid="causal-story-compare-apply"
            style={{
              fontSize: 11, padding: "2px 8px",
              background: "#1a2438", color: "#cfd6e6",
              border: "1px solid #2a3a55", borderRadius: 3, cursor: "pointer",
            }}
          >
            apply
          </button>
          {compareActive && (
            <button
              onClick={handleCompareClear}
              data-testid="causal-story-compare-clear"
              style={{
                fontSize: 11, padding: "2px 8px",
                background: "#1a2438", color: "#cfd6e6",
                border: "1px solid #2a3a55", borderRadius: 3, cursor: "pointer",
              }}
            >
              clear
            </button>
          )}
        </div>
      )}
      {compareActive && compareLoading && (
        <div style={{ fontSize: 11, color: "#888", marginBottom: 6 }}>loading prompt-B lens…</div>
      )}
      {compareActive && compareError && (
        <div style={{ fontSize: 11, color: "#ff8a8a", marginBottom: 6 }}>
          prompt-B lens failed: {compareError}
        </div>
      )}
      {compareSetEnabled && setInputOpen && (
        <div
          data-testid="causal-story-set-input-row"
          style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 8 }}
        >
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <label style={{ fontSize: 11, color: "#888" }}>load preset:</label>
            <select
              data-testid="causal-story-set-preset"
              value={presetSelection}
              onChange={(e) => handlePresetLoad(e.target.value)}
              style={{
                fontSize: 11, padding: "2px 6px",
                background: "#0e0e12", color: "#cfd6e6",
                border: "1px solid #2a3a55", borderRadius: 3,
              }}
            >
              <option value="">(custom — type below)</option>
              {PROMPT_SET_PRESETS.map((p) => (
                <option key={p.id} value={p.id} title={p.description}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>
          {presetCaption && (
            <div
              data-testid="causal-story-set-preset-caption"
              style={{ fontSize: 11, color: "#a0a0c0", fontStyle: "italic", lineHeight: 1.4 }}
            >
              {presetCaption}
            </div>
          )}
          <textarea
            data-testid="causal-story-set-input"
            value={setDraft}
            onChange={(e) => {
              setSetDraft(e.target.value);
              // Manual edit clears the preset selection — the textarea no
              // longer reflects the named preset's state.
              if (presetSelection !== "") {
                setPresetSelection("");
                setPresetCaption(null);
              }
            }}
            placeholder="one prompt per line — divergence heatmap will compare each against the panel's main prompt"
            rows={4}
            style={{
              fontSize: 12, padding: "4px 6px", fontFamily: "monospace",
              background: "#0e0e12", color: "#cfd6e6",
              border: "1px solid #2a3a55", borderRadius: 3, resize: "vertical",
            }}
          />
          <div style={{ display: "flex", gap: 6 }}>
            <button
              onClick={handleSetCommit}
              data-testid="causal-story-set-apply"
              style={{
                fontSize: 11, padding: "2px 8px",
                background: "#1a2438", color: "#cfd6e6",
                border: "1px solid #2a3a55", borderRadius: 3, cursor: "pointer",
              }}
            >
              apply
            </button>
            {compareSetActive && (
              <button
                onClick={handleSetClear}
                data-testid="causal-story-set-clear"
                style={{
                  fontSize: 11, padding: "2px 8px",
                  background: "#1a2438", color: "#cfd6e6",
                  border: "1px solid #2a3a55", borderRadius: 3, cursor: "pointer",
                }}
              >
                clear
              </button>
            )}
            <span style={{ fontSize: 11, color: "#888", alignSelf: "center" }}>
              {compareSetActive
                ? `${compareSet?.length ?? 0} prompt${(compareSet?.length ?? 0) === 1 ? "" : "s"} active`
                : "(empty — heatmap hidden)"}
            </span>
          </div>
        </div>
      )}
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
            const labelWord = isEmbed ? "input" : "residual";
            const handleClick = interactive
              ? () => onSelectNode!(isSelected ? null : id)
              : undefined;
            // During playback, only nodes with index < step are shown.
            const isRevealed = effectiveStep === null || i < effectiveStep;
            const aTokens = n.lensTokens;
            const bTokens = compareTokens?.[i] ?? null;
            const showCompareRow = compareActive && bTokens !== null;
            return (
              <div
                key={id}
                data-testid={`causal-story-row-${id}`}
                data-revealed={isRevealed ? "true" : "false"}
                data-compare-active={compareActive ? "true" : "false"}
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
                <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  {aTokens.length > 0 ? (
                    <span>
                      <span style={{ color: "#888" }}>
                        {showCompareRow ? `A ${labelWord}: ` : `${labelWord}: `}
                      </span>
                      {aTokens.map((t, j) => {
                        const matchesB = showCompareRow && bTokens !== null && bTokens[j] === t;
                        const aColor = showCompareRow && matchesB ? "#888" : "#cfc";
                        return (
                          <span key={j}>
                            {j > 0 && <span style={{ color: "#666" }}>{" · "}</span>}
                            <span style={{ color: aColor }}>{t}</span>
                          </span>
                        );
                      })}
                    </span>
                  ) : (
                    <span style={{ color: "#888", fontStyle: "italic" }}>(no lens data)</span>
                  )}
                  {showCompareRow && (
                    <span data-testid={`causal-story-row-${id}-compare`}>
                      <span style={{ color: "#888" }}>{`B ${labelWord}: `}</span>
                      {bTokens !== null && bTokens.length === 0 ? (
                        <span style={{ color: "#888", fontStyle: "italic" }}>(no lens data)</span>
                      ) : (
                        bTokens!.map((t, j) => {
                          const matchesA = aTokens[j] === t;
                          const bColor = matchesA ? "#888" : "#ffa07a";
                          return (
                            <span key={j}>
                              {j > 0 && <span style={{ color: "#666" }}>{" · "}</span>}
                              <span style={{ color: bColor }}>{t}</span>
                            </span>
                          );
                        })
                      )}
                    </span>
                  )}
                </div>
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
