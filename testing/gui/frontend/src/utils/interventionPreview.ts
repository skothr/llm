import type { InterventionSpec, SessionInfo } from "../types/api";

export interface InterventionPreview {
  summary: string;
  // Hidden dims directly manipulated. For scale / noise that's all of
  // hidden_size; for zero_dims it's the explicit list length.
  dimsAffected: number | null;
  severity: "info" | "warn" | "danger";
}

function asNum(v: unknown, d = 0): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : d;
}

function asIntArr(v: unknown): number[] {
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is number => typeof x === "number" && Number.isFinite(x));
}

// Source-description for replace / project_out ops. These reference
// another session/prompt/layer/sublayer/position; the visible "where"
// of an intervention depends on that.
function sourceSummary(src: unknown): string {
  if (!src || typeof src !== "object") return "unspecified source";
  const s = src as Record<string, unknown>;
  const session = typeof s.session === "string" ? s.session : "?";
  const layer = asNum(s.layer, 0);
  const sub = typeof s.sublayer === "string" ? s.sublayer : "?";
  const pos = asNum(s.position, 0);
  return `${session}/L${layer}.${sub}[pos ${pos}]`;
}

export function previewIntervention(spec: InterventionSpec, info: SessionInfo | null): InterventionPreview {
  const h = info?.hidden_size ?? 0;
  const loc = `L${spec.layer}.${spec.sublayer}`;

  switch (spec.op) {
    case "scale": {
      const factor = asNum(spec.params.factor, 1);
      if (factor === 1) {
        return { summary: `${loc}: no-op (factor=1)`, dimsAffected: 0, severity: "info" };
      }
      if (factor === 0) {
        return { summary: `${loc}: full zero (factor=0)`, dimsAffected: h, severity: "danger" };
      }
      // Large scale factors push activations far outside training
      // distribution — flag as warn/danger depending on magnitude.
      const absDev = Math.abs(factor - 1);
      const sev: "info" | "warn" | "danger" = absDev > 4 ? "danger" : absDev > 0.5 ? "warn" : "info";
      return { summary: `${loc}: scale by ${factor}\u00d7`, dimsAffected: h, severity: sev };
    }
    case "zero_dims": {
      const dims = asIntArr(spec.params.dims);
      if (dims.length === 0) return { summary: `${loc}: no-op (no dims)`, dimsAffected: 0, severity: "info" };
      if (h > 0) {
        const pct = (dims.length / h) * 100;
        const sev: "info" | "warn" | "danger" = pct > 25 ? "danger" : pct > 5 ? "warn" : "info";
        return { summary: `${loc}: zero ${dims.length} of ${h} dims (${pct.toFixed(1)}%)`, dimsAffected: dims.length, severity: sev };
      }
      return { summary: `${loc}: zero ${dims.length} dims`, dimsAffected: dims.length, severity: "warn" };
    }
    case "clamp": {
      const min = asNum(spec.params.min, -Infinity);
      const max = asNum(spec.params.max, Infinity);
      if (!Number.isFinite(min) && !Number.isFinite(max)) {
        return { summary: `${loc}: no-op clamp`, dimsAffected: 0, severity: "info" };
      }
      // Typical LLaMA-style hidden activations live in ~[-3, 3] at most
      // layers; a clamp tighter than that will cut into the distribution.
      const tight = (Number.isFinite(min) && min > -1) || (Number.isFinite(max) && max < 1);
      return {
        summary: `${loc}: clamp [${Number.isFinite(min) ? min : "\u2212\u221e"}, ${Number.isFinite(max) ? max : "\u221e"}]`,
        dimsAffected: h,
        severity: tight ? "warn" : "info",
      };
    }
    case "noise": {
      const std = asNum(spec.params.std, 0);
      if (std === 0) return { summary: `${loc}: no-op (std=0)`, dimsAffected: 0, severity: "info" };
      // 0.1 is well inside the activation scale; 1.0 drowns the signal.
      const sev: "info" | "warn" | "danger" = std > 0.5 ? "danger" : std > 0.1 ? "warn" : "info";
      return { summary: `${loc}: +\u03b5 (std=${std})`, dimsAffected: h, severity: sev };
    }
    case "replace": {
      return {
        summary: `${loc}: replace with ${sourceSummary(spec.params.source)}`,
        dimsAffected: h,
        severity: "danger",
      };
    }
    case "project_out": {
      return {
        summary: `${loc}: project out direction from ${sourceSummary(spec.params.source)}`,
        dimsAffected: h,
        severity: "warn",
      };
    }
    default:
      return { summary: `${loc}: ${spec.op} (no preview)`, dimsAffected: null, severity: "info" };
  }
}
