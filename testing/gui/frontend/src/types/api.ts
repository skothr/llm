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

export interface LogitLensData {
  type: "data";
  layer: number;
  original_layer?: number;
  sublayer: string;
  predictions: Array<Array<{ token: string; prob: number }>>;
  metrics?: Array<CellMetrics>;
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

export type WsMessage = LogitLensData | GenerateData | InterveneData | WsComplete | WsError;

export interface InterventionSpec {
  layer: number;
  sublayer: "attn" | "ffn";
  op: string;
  params: Record<string, unknown>;
}

export type ProbeOperation = "logit-lens" | "influence" | "attention" | "residual-norms" | "generate";
export type ResultOperation = ProbeOperation | "intervene";
export interface AvailableModel {
  model_id: string;
  source: "huggingface" | "ollama";
  safetensors: boolean;
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
