import { useState, useEffect } from "react";
import { useStore } from "../state/store";

export function ModelCombobox({
  value,
  onChange,
}: {
  value: string;
  onChange: (modelId: string) => void;
}) {
  const availableModels = useStore((s) => s.availableModels);
  const [isOpen, setIsOpen] = useState(false);
  const [filter, setFilter] = useState(value);

  useEffect(() => { setFilter(value); }, [value]);

  const filtered = availableModels.filter((m) =>
    m.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div style={{ position: "relative" }}>
      <input
        value={filter}
        onChange={(e) => {
          setFilter(e.target.value);
          onChange(e.target.value);
          setIsOpen(true);
        }}
        onFocus={() => setIsOpen(true)}
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
          {filtered.map((m) => (
            <div
              key={m}
              style={{ padding: "4px 8px", cursor: "pointer", fontSize: 12 }}
              onMouseDown={() => {
                onChange(m);
                setFilter(m);
                setIsOpen(false);
              }}
              onMouseEnter={(e) => { (e.target as HTMLElement).style.background = "#1a5276"; }}
              onMouseLeave={(e) => { (e.target as HTMLElement).style.background = "transparent"; }}
            >
              {m}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
