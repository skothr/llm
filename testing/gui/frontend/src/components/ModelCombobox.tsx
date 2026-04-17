import { useState } from "react";
import { useStore } from "../state/store";
import type { AvailableModel } from "../types/api";

const SOURCE_BADGE: Record<string, { label: string; color: string }> = {
  huggingface: { label: "HF", color: "#ff9d00" },
  ollama: { label: "OL", color: "#7c6cf0" },
};

export function ModelCombobox({
  value,
  onChange,
}: {
  value: string;
  onChange: (modelId: string, source: AvailableModel["source"] | null) => void;
}) {
  const availableModels = useStore((s) => s.availableModels);
  const [isOpen, setIsOpen] = useState(false);
  const [filter, setFilter] = useState("");

  const displayValue = isOpen ? filter : value;

  const filtered = availableModels.filter((m: AvailableModel) =>
    m.model_id.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div style={{ position: "relative" }}>
      <input
        value={displayValue}
        onChange={(e) => {
          setFilter(e.target.value);
          onChange(e.target.value, null);
          setIsOpen(true);
        }}
        onFocus={() => { setFilter(""); setIsOpen(true); }}
        onBlur={() => setTimeout(() => setIsOpen(false), 200)}
        placeholder="Model ID (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
      />
      {isOpen && filtered.length > 0 && (
        <div style={{
          position: "absolute",
          top: "100%",
          left: 0,
          right: 0,
          background: "#0d1b2a",
          border: "1px solid #1a5276",
          borderRadius: "0 0 4px 4px",
          maxHeight: 150,
          overflowY: "auto",
          zIndex: 10,
        }}>
          {filtered.map((m: AvailableModel) => {
            const badge = SOURCE_BADGE[m.source];
            return (
              <div
                key={`${m.source}:${m.model_id}`}
                style={{ padding: "4px 8px", cursor: "pointer", fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}
                onMouseDown={() => {
                  onChange(m.model_id, m.source);
                  setFilter("");
                  setIsOpen(false);
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "#1a5276"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
              >
                {badge && (
                  <span style={{
                    fontSize: 9,
                    fontWeight: 700,
                    padding: "1px 4px",
                    borderRadius: 3,
                    background: badge.color + "22",
                    color: badge.color,
                    border: `1px solid ${badge.color}44`,
                    flexShrink: 0,
                  }}>{badge.label}</span>
                )}
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.model_id}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
