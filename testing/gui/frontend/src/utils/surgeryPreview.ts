import type { SessionInfo, StagedOp } from "../types/api";

export interface OpPreview {
  // Human-readable one-line summary ("removes 3 layers", "zeros 2 heads in L5").
  summary: string;
  // Number of layers the op touches (useful for red-amber warnings).
  layersAffected: number;
  // Estimated number of parameters affected. Null when we can't tell
  // (e.g. unknown op type, or hidden_size missing). Zero means "no
  // numeric change, only reordering" (swap, reorder).
  paramsAffected: number | null;
  // Severity tier — drives the badge color.
  // info: identity / reorder-only; warn: localized change;
  // danger: multiple layers or large param volume.
  severity: "info" | "warn" | "danger";
}

// Per-layer param count. Derived from the standard transformer-block shape:
//   attn: Q,K,V,O each (hidden × hidden) with GQA scaling kv to
//         (hidden × kv_dim), plus the small kv/o biases we ignore here;
//   ffn:  3 × hidden × intermediate  (LLaMA-style gated MLP);
//   norms: 2 × hidden (rmsnorm pre-attn, pre-ffn).
// Good within ~5% for mainstream decoder-only models; exact numbers
// are not the point — the user wants a gut-check, not a verification.
function perLayerParams(info: SessionInfo): number | null {
  const h = info.hidden_size;
  if (h == null) return null;
  const heads = info.num_heads;
  const kvHeads = info.num_kv_heads ?? heads;
  if (heads == null) return null;
  const headDim = Math.max(1, Math.floor(h / heads));
  const kvDim = kvHeads * headDim;
  const ffn = info.intermediate_size ?? Math.round(h * 2.67);
  const attn = 2 * h * h + 2 * h * kvDim; // Q, O on hidden²; K, V on hidden × kv_dim
  const mlp = 3 * h * ffn;                // gate, up, down
  const norms = 2 * h;
  return attn + mlp + norms;
}

function perHeadAttnParams(info: SessionInfo): number | null {
  // One attention head's share of Q/K/V/O. This under-counts the O-projection
  // share because O is applied to concatenated heads, but the practical
  // "zero this head's contribution" affects roughly 4 × hidden × head_dim
  // weights worth of "head contribution". Good enough for a preview.
  const h = info.hidden_size;
  const heads = info.num_heads;
  if (h == null || heads == null) return null;
  const headDim = Math.max(1, Math.floor(h / heads));
  return 4 * h * headDim;
}

function mlpParams(info: SessionInfo): number | null {
  const h = info.hidden_size;
  if (h == null) return null;
  const ffn = info.intermediate_size ?? Math.round(h * 2.67);
  return 3 * h * ffn;
}

function attnParams(info: SessionInfo): number | null {
  const h = info.hidden_size;
  if (h == null) return null;
  const heads = info.num_heads;
  const kvHeads = info.num_kv_heads ?? heads;
  if (heads == null) return null;
  const headDim = Math.max(1, Math.floor(h / heads));
  return 2 * h * h + 2 * h * (kvHeads * headDim);
}

function asIntArray(v: unknown): number[] {
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is number => typeof x === "number" && Number.isFinite(x));
}

function num(v: unknown, fallback = 0): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

// Compute a layer-count severity tier so callers don't re-derive it.
function severityFromLayerFraction(fraction: number, paramsAffected: number | null, total: number | null): "info" | "warn" | "danger" {
  if (fraction >= 0.25) return "danger";
  if (paramsAffected != null && total != null && paramsAffected / total > 0.05) return "danger";
  if (fraction >= 0.05) return "warn";
  return "info";
}

export function previewStagedOp(op: StagedOp, info: SessionInfo): OpPreview {
  const total = info.total_params || null;
  const numLayers = info.num_layers;

  switch (op.operation) {
    case "remove_layers": {
      const idx = asIntArray(op.params.layer_indices);
      const count = idx.length;
      const per = perLayerParams(info);
      const params = per != null ? count * per : null;
      return {
        summary: `removes ${count} layer${count === 1 ? "" : "s"} (${idx.join(",") || "—"})`,
        layersAffected: count,
        paramsAffected: params,
        severity: severityFromLayerFraction(count / (numLayers || 1), params, total),
      };
    }
    case "keep_layers": {
      const keep = asIntArray(op.params.layer_indices);
      const removed = Math.max(0, numLayers - keep.length);
      const per = perLayerParams(info);
      const params = per != null ? removed * per : null;
      return {
        summary: `keeps ${keep.length} layers; removes ${removed}`,
        layersAffected: removed,
        paramsAffected: params,
        severity: severityFromLayerFraction(removed / (numLayers || 1), params, total),
      };
    }
    case "zero_heads":
    case "scale_heads": {
      const heads = asIntArray(op.params.heads);
      const perHead = perHeadAttnParams(info);
      const params = perHead != null ? heads.length * perHead : null;
      const factor = op.operation === "scale_heads" ? ` by ${num(op.params.factor, 1)}\u00d7` : "";
      return {
        summary: `${op.operation === "zero_heads" ? "zeros" : "scales"} ${heads.length} head${heads.length === 1 ? "" : "s"} in L${num(op.params.layer)}${factor}`,
        layersAffected: 1,
        paramsAffected: params,
        severity: heads.length > (info.num_heads ?? 16) / 2 ? "warn" : "info",
      };
    }
    case "zero_mlp": {
      const params = mlpParams(info);
      return {
        summary: `zeros the MLP in L${num(op.params.layer)}`,
        layersAffected: 1,
        paramsAffected: params,
        severity: "warn",
      };
    }
    case "zero_attention": {
      const params = attnParams(info);
      return {
        summary: `zeros the attention block in L${num(op.params.layer)}`,
        layersAffected: 1,
        paramsAffected: params,
        severity: "warn",
      };
    }
    case "swap_layers":
      return {
        summary: `swaps L${num(op.params.i)} \u2194 L${num(op.params.j)}`,
        layersAffected: 2,
        paramsAffected: 0,
        severity: "info",
      };
    case "duplicate_layer": {
      const per = perLayerParams(info);
      return {
        summary: `duplicates L${num(op.params.src)} into position ${num(op.params.dst)}`,
        layersAffected: 1,
        paramsAffected: per,
        severity: "warn",
      };
    }
    case "swap_heads":
      return {
        summary: `swaps heads H${num(op.params.h1)} \u2194 H${num(op.params.h2)} in L${num(op.params.layer)}`,
        layersAffected: 1,
        paramsAffected: 0,
        severity: "info",
      };
    case "reorder_layers": {
      const order = asIntArray(op.params.new_order);
      // Conservative: reorder touches any layer whose index actually moves.
      let moved = 0;
      for (let i = 0; i < order.length; i++) if (order[i] !== i) moved++;
      return {
        summary: `reorders layers (${moved} moved)`,
        layersAffected: moved,
        paramsAffected: 0,
        severity: moved > (numLayers || 1) / 2 ? "warn" : "info",
      };
    }
    default:
      return {
        summary: `${op.operation} (no preview available)`,
        layersAffected: 0,
        paramsAffected: null,
        severity: "info",
      };
  }
}

export function formatParams(n: number): string {
  if (n === 0) return "0";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return `${n}`;
}
