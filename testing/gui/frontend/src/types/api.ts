export interface SessionSummary {
  name: string;
  model_id: string;
  mode: string;
  num_layers: number;
  pending_count: number;
  applied_count: number;
  device: string;
  engine_llama: boolean;
  engine_pytorch: boolean;
  dirty: boolean;
}

export interface StagedOp {
  operation: string;
  params: Record<string, unknown>;
}

export interface SessionInfo {
  num_layers: number;
  num_heads: number;
  num_kv_heads: number | null;
  hidden_size: number;
  intermediate_size: number | null;
  vocab_size: number | null;
  max_position_embeddings: number | null;
  rope_theta: number | null;
  total_params: number;
  chat_template: string | null;
  bos_token: string | null;
  eos_token: string | null;
  layer_map: number[];
  original_num_layers: number;
  pending_ops: StagedOp[];
  applied_ops: StagedOp[];
}

export interface SurgeryOperation {
  name: string;
  params: Record<string, { type?: string; items?: string } | string>;
  description: string;
}

export interface SurgeryRequest {
  operation: string;
  params: Record<string, unknown>;
}

export interface CellMetrics {
  entropy: number;
  top1_prob: number;
  top1_margin: number;
}

export interface PairMetrics {
  kl_ab: number;
  js: number;
  cosine: number;
  top1_delta_prob: number;
  top1_match: boolean;
}

export interface CompareCell {
  original_layer: number;
  sublayer: string;
  position: number;
  top_k_a: Array<{ token: string; prob: number }>;
  top_k_b: Array<{ token: string; prob: number }>;
  metrics_a: CellMetrics;
  metrics_b: CellMetrics;
  compare: PairMetrics;
}

export interface CompareLogitLensData {
  type: "data";
  original_layer: number;
  sublayer: string;
  cells: CompareCell[];
  hidden_state_a?: EncodedHiddenStateMsg;
  hidden_state_b?: EncodedHiddenStateMsg;
}

export interface EncodedHiddenStateMsg {
  shape: [number, number];
  b64: string;
}

export interface EncodedTensor {
  shape: number[];
  b64: string;
}

export interface LogitLensData {
  type: "data";
  layer: number;
  original_layer?: number;
  sublayer: string;
  predictions: Array<Array<{ token: string; prob: number }>>;
  metrics?: Array<CellMetrics>;
  hidden_state?: EncodedHiddenStateMsg;
}

export interface PatchingBaselinesData {
  type: "baselines";
  clean_logits: EncodedTensor;
  corrupted_logits: EncodedTensor;
  prompt_tokens_clean: string[];
  prompt_tokens_corrupted: string[];
  measurement_position: number;
  correct_token_id?: number;
  incorrect_token_id?: number;
}

export interface PatchingCellData {
  type: "data";
  layer: number;
  original_layer?: number;
  sublayer?: "attn" | "ffn";     // present in exact/approx modes
  unit?: string;                  // present in approx_head mode: "attn.hN" or "ffn"
  head?: number | null;           // derived client-side from unit; not sent by backend
  position: number;
  patched_logits?: EncodedTensor;
  ap_recovery?: number;
}

export interface PatchingCompleteData {
  type: "complete";
  summary: {
    num_cells: number;
    direction: "denoise" | "noise";
    measurement_position: number;
    mode?: "exact" | "approx" | "approx_head";
    n_heads?: number;
  };
}

export interface GenerateData {
  type: "data";
  step: number;
  token: string;
  token_id: number;
  top_k: Array<{ token: string; prob: number }>;
}

export interface InterveneData {
  type: "data";
  layer: number;
  sublayer: string;
  modified: boolean;
  predictions?: Array<Array<{ token: string; prob: number }>>;
}

export interface WsComplete {
  type: "complete";
  [key: string]: unknown;
}

export interface WsError {
  type: "error";
  message: string;
}

export type WsMessage =
  | LogitLensData
  | GenerateData
  | InterveneData
  | CompareLogitLensData
  | WsComplete
  | WsError
  | PatchingBaselinesData
  | PatchingCellData
  | PatchingCompleteData;

export interface InterventionSpec {
  layer: number;
  sublayer: "attn" | "ffn";
  op: string;
  params: Record<string, unknown>;
}

export type ProbeOperation =
  | "logit-lens"
  | "influence"
  | "attention"
  | "residual-norms"
  | "generate"
  | "activation-patching";
export type ResultOperation = ProbeOperation | "intervene";
export interface AvailableModel {
  model_id: string;
  source: "huggingface" | "ollama";
  architecture?: string;
  model_name?: string;
  dtype?: string;
  quantization?: string;
  model_size_label?: string;
  num_layers?: number;
  hidden_size?: number;
  num_heads?: number;
  num_kv_heads?: number;
  vocab_size?: number;
  intermediate_size?: number;
  max_position_embeddings?: number;
  rope_theta?: number;
  file_size_bytes?: number;
  num_tensors?: number;
  tensor_type_counts?: Record<string, number>;
  total_params?: number;
  total_bytes?: number;
  bits_per_weight?: number;
}

export type ConfigTab = "sessions" | "probe" | "intervene";

export interface ProbeResult {
  id: string;
  operation: ResultOperation;
  sessionName: string;
  prompt: string;
  data: WsMessage[];
  timestamp: number;
  // Explicit A/B flag. The `${A.id}-B` id-suffix convention still holds so
  // VisualizationArea can locate the pair, but classification (A vs B)
  // reads this flag rather than string-sniffing the id.
  isB?: boolean;
  // Seed fan-out / parameter-sweep grouping. All N runs in a single batch
  // share batchId; batchIndex is 0..N-1. batchSize is duplicated on every
  // member so a single result is enough to know the full batch size
  // without consulting the others.
  //
  // `seed` is the per-run numeric seed when axis=seed. `sweepLabel` is
  // the human-readable variant label across *any* sweep axis (e.g.
  // "temp=0.3", "top_p=0.85"), used by panels to label each column.
  batchId?: string;
  batchIndex?: number;
  batchSize?: number;
  seed?: number;
  sweepLabel?: string;
  // For 2D parameter grids, `gridCols` is set on every sibling so the
  // output panel can render them as a rows×cols grid instead of a long
  // scrolling row. Absent on 1D batches (renders as row, existing).
  gridCols?: number;

  // Researcher annotations. `pinned` results survive Clear All and are
  // always kept in the persisted result slice regardless of the 50-cap.
  // `tags` drive the filter bar's chip list. `notes` is free text.
  pinned?: boolean;
  tags?: string[];
  notes?: string;

  // Snapshot of the sampling knobs as they were at the moment this result
  // was launched. Populated by ProbePanel. Used by the "↺ recall" button
  // to load the exact configuration back into the probe panel for
  // reproduction. Untyped here (passed through as an opaque record) to
  // avoid a circular import with state/store.
  runParams?: Record<string, unknown>;
}

export interface InfluenceResult {
  scores: Record<string, number>;
}

export interface AttentionResult {
  entropy: Record<string, number[]>;
}

export interface ResidualNormsResult {
  norms: number[];
}
