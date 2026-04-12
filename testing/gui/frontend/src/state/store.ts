import { create } from "zustand";
import type {
  SessionSummary,
  SessionInfo,
  SurgeryOperation,
  ProbeOperation,
  ProbeResult,
} from "../types/api";

interface StoreState {
  sessions: SessionSummary[];
  sessionInfo: Record<string, SessionInfo>;
  surgeryOps: SurgeryOperation[];
  prompt: string;
  operation: ProbeOperation;
  targetSession: string;
  targetSessionB: string | null;
  isRunning: boolean;
  results: ProbeResult[];
  activeResultId: string | null;

  fetchSessions: () => Promise<void>;
  fetchSessionInfo: (name: string) => Promise<void>;
  fetchSurgeryOps: () => Promise<void>;
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
}

export const useStore = create<StoreState>((set, get) => ({
  sessions: [],
  sessionInfo: {},
  surgeryOps: [],
  prompt: "",
  operation: "logit-lens",
  targetSession: "",
  targetSessionB: null,
  isRunning: false,
  results: [],
  activeResultId: null,

  fetchSessions: async () => {
    const resp = await fetch("/api/sessions");
    const data = await resp.json();
    set({ sessions: data });
  },

  fetchSessionInfo: async (name: string) => {
    const resp = await fetch(`/api/sessions/${name}/info`);
    const data = await resp.json();
    set((s) => ({ sessionInfo: { ...s.sessionInfo, [name]: data } }));
  },

  fetchSurgeryOps: async () => {
    const resp = await fetch("/api/surgery/operations");
    const data = await resp.json();
    set({ surgeryOps: data });
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
    if (!resp.ok) throw new Error((await resp.json()).detail);
    await get().fetchSessions();
    await get().fetchSessionInfo(name);
  },

  undoSurgery: async (name: string) => {
    const resp = await fetch(`/api/sessions/${name}/surgery/undo`, { method: "POST" });
    if (!resp.ok) throw new Error((await resp.json()).detail);
    await get().fetchSessions();
    await get().fetchSessionInfo(name);
  },

  cloneSession: async (name: string, targetName: string) => {
    const resp = await fetch(`/api/sessions/${name}/clone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_name: targetName }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail);
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
}));
