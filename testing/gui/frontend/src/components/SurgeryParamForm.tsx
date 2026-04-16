import { useState } from "react";
import type { SurgeryOperation } from "../types/api";

interface Props {
  operation: SurgeryOperation;
  params: Record<string, unknown>;
  onChange: (params: Record<string, unknown>) => void;
}

export function SurgeryParamForm({ operation, params, onChange }: Props) {
  const [showJson, setShowJson] = useState(false);
  const [jsonText, setJsonText] = useState(JSON.stringify(params, null, 2));

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
              {key}: <input
                value={Array.isArray(params[key]) ? (params[key] as number[]).join(", ") : ""}
                onChange={(e) => onChange({
                  ...params,
                  [key]: e.target.value.split(",").map((s) => parseInt(s.trim())).filter((n) => !isNaN(n)),
                })}
                placeholder="0, 1, 2"
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
