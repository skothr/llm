import { useEffect, useRef, useState } from "react";
import type { SurgeryOperation } from "../types/api";

interface Props {
  operation: SurgeryOperation;
  params: Record<string, unknown>;
  onChange: (params: Record<string, unknown>) => void;
}

function parseIntList(s: string): number[] {
  return s.split(",")
    .map((t) => t.trim())
    .filter((t) => /^-?\d+$/.test(t))
    .map((t) => parseInt(t, 10));
}

export function ArrayInput({ value, onChange, style }: {
  value: number[];
  onChange: (next: number[]) => void;
  style?: React.CSSProperties;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [raw, setRaw] = useState(value.join(", "));

  useEffect(() => {
    const next = value.join(", ");
    if (document.activeElement !== inputRef.current && next !== raw) {
      setRaw(next);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <input
      ref={inputRef}
      value={raw}
      onChange={(e) => {
        setRaw(e.target.value);
        onChange(parseIntList(e.target.value));
      }}
      onBlur={() => {
        const parsed = parseIntList(raw);
        setRaw(parsed.join(", "));
        onChange(parsed);
      }}
      placeholder="0, 1, 2"
      style={style}
    />
  );
}

export function SurgeryParamForm({ operation, params, onChange }: Props) {
  const [showJson, setShowJson] = useState(false);
  const [jsonText, setJsonText] = useState(JSON.stringify(params, null, 2));

  useEffect(() => {
    setJsonText(JSON.stringify(params, null, 2));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [operation.name]);

  if (showJson) {
    return (
      <div>
        <textarea
          value={jsonText}
          onChange={(e) => {
            setJsonText(e.target.value);
            try { onChange(JSON.parse(e.target.value)); } catch { /* invalid JSON */ }
          }}
          rows={3}
          style={{ marginBottom: 4 }}
        />
        <button onClick={() => setShowJson(false)} style={{ fontSize: 11 }}>Form view</button>
      </div>
    );
  }

  return (
    <div>
      {Object.entries(operation.params).map(([key, schema]) => {
        const schemaObj = typeof schema === "string" ? { type: schema } : schema;
        const type = schemaObj.type || "string";

        if (type === "int") {
          return (
            <label key={key} style={{ display: "block", fontSize: 12, marginBottom: 4 }}>
              {key}: <input
                type="number"
                step={1}
                value={Number(params[key] ?? 0)}
                onChange={(e) => {
                  const n = parseInt(e.target.value);
                  onChange({ ...params, [key]: Number.isNaN(n) ? 0 : n });
                }}
                style={{ width: 72 }}
              />
            </label>
          );
        }
        if (type === "float") {
          return (
            <label key={key} style={{ display: "block", fontSize: 12, marginBottom: 4 }}>
              {key}: <input
                type="number"
                step={0.1}
                value={Number(params[key] ?? 0)}
                onChange={(e) => {
                  const n = parseFloat(e.target.value);
                  onChange({ ...params, [key]: Number.isNaN(n) ? 0 : n });
                }}
                style={{ width: 72 }}
              />
            </label>
          );
        }
        if (type === "array") {
          return (
            <label key={key} style={{ display: "block", fontSize: 12, marginBottom: 4 }}>
              {key}: <ArrayInput
                value={Array.isArray(params[key]) ? (params[key] as number[]) : []}
                onChange={(next) => onChange({ ...params, [key]: next })}
                style={{ width: "100%" }}
              />
            </label>
          );
        }
        return null;
      })}
      <button onClick={() => { setJsonText(JSON.stringify(params, null, 2)); setShowJson(true); }} style={{ fontSize: 11 }}>
        JSON
      </button>
    </div>
  );
}
