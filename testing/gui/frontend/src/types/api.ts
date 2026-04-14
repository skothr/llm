export interface SessionSummary {
  name: string;
  model_id: string;
  mode: string;
  num_layers: number;
  has_snapshot: boolean;
  snapshot_size_mb: number;
}

export interface SessionInfo {
  num_layers: number;
  num_heads: number;
  hidden_size: number;
  total_params: number;
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

export interface LogitLensData {
  type: "data";
  layer: number;
  sublayer: string;
  predictions: Array<Array<{ token: string; prob: number }>>;
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
