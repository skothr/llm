import { create } from "zustand";
import type {
  SessionSummary,
  SessionInfo,
  SurgeryOperation,
  ProbeOperation,
  ProbeResult,
  ConfigTab,
  InterventionSpec,
  WsMessage,
} from "../types/api";

async function apiError(resp: Response): Promise<Error> {
  const body = await resp.json().catch(() => ({ detail: resp.statusText }));
  const detail = body.detail;
  if (typeof detail === "string") return new Error(detail);
  if (Array.isArray(detail)) {
    return new Error(detail.map((d: { msg?: string; loc?: string[] }) =>
      `${(d.loc || []).join(".")}: ${d.msg || "validation error"}`
    ).join("; "));
  }
  return new Error(JSON.stringify(detail));
}

interface StoreState {
  activeTab: ConfigTab;
  backendOnline: boolean;

  sessions: SessionSummary[];
  sessionInfo: Record<string, SessionInfo>;
  surgeryOps: SurgeryOperation[];

  isLoadingModel: boolean;
  loadingModelId: string | null;
  availableModels: string[];

  prompt: string;
  operation: ProbeOperation;
  targetSession: string;
  targetSessionB: string | null;
  isRunning: boolean;

  results: ProbeResult[];
  activeResultId: string | null;
  pendingResults: Record<string, ProbeResult>;

  interventionSpecs: InterventionSpec[];
  captureLogitLens: boolean;
  intervenePrompt: string;
  interveneSession: string;

  setActiveTab: (tab: ConfigTab) => void;
  fetchSessions: () => Promise<void>;
  fetchSessionInfo: (name: string) => Promise<void>;
  fetchSurgeryOps: () => Promise<void>;
  fetchAvailableModels: () => Promise<void>;
  deleteSession: (name: string) => Promise<void>;
  applySurgery: (name: string, operation: string, params: Record<string, unknown>) => Promise<void>;
  undoSurgery: (name: string) => Promise<void>;
  cloneSession: (name: string, targetName: string) => Promise<void>;

  setPrompt: (prompt: string) => void;
  setOperation: (op: ProbeOperation) => void;
  setTargetSession: (name: string) => void;
  setTargetSessionB: (name: string | null) => void;
  setRunning: (running: boolean) => void;

  addResult: (result: ProbeResult) => void;
  clearResults: () => void;
  setActiveResult: (id: string | null) => void;
  setPendingResult: (id: string, result: ProbeResult) => void;
  updatePendingResult: (id: string, msg: WsMessage) => void;
  finalizePendingResult: (id: string, extraData?: WsMessage) => void;
  removePendingResult: (id: string) => void;

  addIntervention: () => void;
  removeIntervention: (index: number) => void;
  updateIntervention: (index: number, spec: InterventionSpec) => void;
  clearInterventions: () => void;
  setCaptureLogitLens: (v: boolean) => void;
  setIntervenePrompt: (prompt: string) => void;
  setInterveneSession: (name: string) => void;
}

export const useStore = create<StoreState>((set, get) => ({
  activeTab: "sessions",
  backendOnline: false,

  sessions: [],
  sessionInfo: {},
  surgeryOps: [],

  isLoadingModel: false,
  loadingModelId: null,
  availableModels: [],

  prompt: "",
  operation: "logit-lens",
  targetSession: "",
  targetSessionB: null,
  isRunning: false,

  results: [],
  activeResultId: null,
  pendingResults: {},

  interventionSpecs: [],
  captureLogitLens: false,
  intervenePrompt: "",
  interveneSession: "",

  setActiveTab: (tab) => set({ activeTab: tab }),

  fetchSessions: async () => {
    try {
      const resp = await fetch("/api/sessions");
      const data = await resp.json();
      set({ sessions: data, backendOnline: true });
      if (get().surgeryOps.length === 0) {
        get().fetchSurgeryOps();
      }
    } catch {
      set({ backendOnline: false });
    }
  },

  fetchSessionInfo: async (name: string) => {
    const resp = await fetch(`/api/sessions/${name}/info`);
    const data = await resp.json();
    set((s) => ({ sessionInfo: { ...s.sessionInfo, [name]: data } }));
  },

  fetchSurgeryOps: async () => {
    try {
      const resp = await fetch("/api/surgery/operations");
      const data = await resp.json();
      set({ surgeryOps: data });
    } catch { /* backend not ready */ }
  },

  fetchAvailableModels: async () => {
    try {
      const resp = await fetch("/api/models/available");
      const data = await resp.json();
      set({ availableModels: data.map((m: { model_id: string }) => m.model_id) });
    } catch { /* backend not ready */ }
  },

  deleteSession: async (name: string) => {
    await fetch(`/api/sessions/${name}`, { method: "DELETE" });
    await get().fetchSessions();
  },

  applySurgery: async (name: string, operation: string, params: Record<string, unknown>) => {
    const resp = await fetch(`/api/sessions/${name}/surgery`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operation, params }),
    });
    if (!resp.ok) throw await apiError(resp);
    await get().fetchSessions();
    await get().fetchSessionInfo(name);
  },

  undoSurgery: async (name: string) => {
    const resp = await fetch(`/api/sessions/${name}/surgery/undo`, { method: "POST" });
    if (!resp.ok) throw await apiError(resp);
    await get().fetchSessions();
    await get().fetchSessionInfo(name);
  },

  cloneSession: async (name: string, targetName: string) => {
    const resp = await fetch(`/api/sessions/${name}/clone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_name: targetName }),
    });
    if (!resp.ok) throw await apiError(resp);
    await get().fetchSessions();
  },

  setPrompt: (prompt) => set({ prompt }),
  setOperation: (operation) => set({ operation }),
  setTargetSession: (name) => set({ targetSession: name }),
  setTargetSessionB: (name) => set({ targetSessionB: name }),
  setRunning: (running) => set({ isRunning: running }),

  addResult: (result) => set((s) => ({
    results: [result, ...s.results],
    activeResultId: result.id,
  })),
  clearResults: () => set({ results: [], activeResultId: null }),
  setActiveResult: (id) => set({ activeResultId: id }),

  setPendingResult: (id, result) => set((s) => ({
    pendingResults: { ...s.pendingResults, [id]: result },
    activeResultId: id,
  })),

  updatePendingResult: (id, msg) => set((s) => {
    const pending = s.pendingResults[id];
    if (!pending) return s;
    return {
      pendingResults: {
        ...s.pendingResults,
        [id]: { ...pending, data: [...pending.data, msg] },
      },
    };
  }),

  finalizePendingResult: (id, extraData) => set((s) => {
    const pending = s.pendingResults[id];
    if (!pending) return s;
    const finalData = extraData ? [...pending.data, extraData] : pending.data;
    const { [id]: _, ...remaining } = s.pendingResults;
    return {
      pendingResults: remaining,
      results: [{ ...pending, data: finalData }, ...s.results],
      activeResultId: pending.id,
    };
  }),

  removePendingResult: (id) => set((s) => {
    const { [id]: _, ...remaining } = s.pendingResults;
    return { pendingResults: remaining };
  }),

  addIntervention: () => set((s) => ({
    interventionSpecs: [...s.interventionSpecs, { layer: 0, sublayer: "ffn", op: "scale", params: { factor: 1.0 } }],
  })),
  removeIntervention: (index) => set((s) => ({
    interventionSpecs: s.interventionSpecs.filter((_, i) => i !== index),
  })),
  updateIntervention: (index, spec) => set((s) => ({
    interventionSpecs: s.interventionSpecs.map((existing, i) => i === index ? spec : existing),
  })),
  clearInterventions: () => set({ interventionSpecs: [] }),
  setCaptureLogitLens: (v) => set({ captureLogitLens: v }),
  setIntervenePrompt: (prompt) => set({ intervenePrompt: prompt }),
  setInterveneSession: (name) => set({ interveneSession: name }),
}));
