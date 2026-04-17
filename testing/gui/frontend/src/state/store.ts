import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { cancelSessionSockets } from "../hooks/useWebSocket";
import { idbStorage } from "../utils/idbStorage";
import type {
  SessionSummary,
  SessionInfo,
  SurgeryOperation,
  AvailableModel,
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

async function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (e) {
    if (e instanceof TypeError) {
      useStore.setState({ backendOnline: false });
    }
    throw e;
  }
}

export interface SamplingParams {
  displayTopK: number;
  maxTokens: number;
  temperature: number;
  samplingTopK: number;
  topP: number;
  minP: number;
  seed: string;
  repPenalty: number;
  stopSeqs: string;
  // Seed fan-out: number of parallel runs per Run click. 1 = normal single
  // run. N>1 triggers N WS connections each with a distinct random seed,
  // grouped by a shared batchId on the resulting ProbeResults.
  numSeeds: number;
}

export const SAMPLING_DEFAULTS: SamplingParams = {
  displayTopK: 10,
  maxTokens: 64,
  temperature: 0.0,
  samplingTopK: 0,
  topP: 1.0,
  minP: 0.0,
  seed: "",
  repPenalty: 1.0,
  stopSeqs: "\\n\\n",
  numSeeds: 1,
};

export interface NamedPrompt {
  id: string;
  name: string;
  text: string;
  tags?: string[];
  createdAt: number;
}

interface StoreState {
  activeTab: ConfigTab;
  backendOnline: boolean;
  // False until the first /api/sessions probe has completed (success or
  // failure). Used to suppress the "Backend offline" banner on initial page
  // load so we don't flash it while the probe is still in flight.
  backendProbed: boolean;

  sessions: SessionSummary[];
  sessionInfo: Record<string, SessionInfo>;
  surgeryOps: SurgeryOperation[];

  isLoadingModel: boolean;
  loadingModelId: string | null;
  availableModels: AvailableModel[];

  prompt: string;
  operation: ProbeOperation;
  targetSession: string;
  targetSessionB: string | null;
  isRunning: boolean;

  samplingParams: SamplingParams;

  promptLibrary: NamedPrompt[];

  results: ProbeResult[];
  activeResultId: string | null;
  pendingResults: Record<string, ProbeResult>;

  // Filter state — runtime only (not persisted). An empty selectedTags set
  // means "don't filter by tag"; pinnedOnly narrows to favorites; query
  // substring-matches prompt + sessionName + notes.
  filterTags: string[];
  filterPinnedOnly: boolean;
  filterQuery: string;

  // User-declared baseline session name. Used as the default "B side" for
  // compare flows — CompareSelector offers a one-click "vs baseline"
  // whenever one is set. Persisted so it survives reloads; auto-cleared if
  // the baseline session is deleted.
  baselineSession: string | null;

  // Single-slot undo buffer for result deletion. Captures the full result
  // + its prior index so Undo restores to the same position. Overwritten
  // by subsequent deletes (one-step undo only — matches most editor conventions).
  // Runtime-only, not persisted: undo across a full page refresh would be
  // more confusing than useful.
  lastDeleted: { result: ProbeResult; index: number; deletedAt: number } | null;

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
  deleteStagedOp: (name: string, index: number) => Promise<void>;
  commitSurgery: (name: string) => Promise<void>;
  revertSurgery: (name: string) => Promise<void>;
  cloneSession: (name: string, targetName: string) => Promise<void>;

  setPrompt: (prompt: string) => void;
  setOperation: (op: ProbeOperation) => void;
  setTargetSession: (name: string) => void;
  setTargetSessionB: (name: string | null) => void;
  setRunning: (running: boolean) => void;

  setSamplingParams: (patch: Partial<SamplingParams>) => void;
  resetSamplingParams: () => void;

  savePrompt: (name: string, text: string, tags?: string[]) => void;
  deletePrompt: (id: string) => void;

  addResult: (result: ProbeResult) => void;
  clearResults: () => void;
  setActiveResult: (id: string | null) => void;
  setPendingResult: (id: string, result: ProbeResult) => void;
  updatePendingResult: (id: string, msg: WsMessage) => void;
  finalizePendingResult: (id: string, extraData?: WsMessage) => void;
  removePendingResult: (id: string) => void;

  // Result annotation actions.
  updateResultMeta: (id: string, patch: { pinned?: boolean; tags?: string[]; notes?: string }) => void;
  deleteResult: (id: string) => void;

  // Filter actions.
  toggleFilterTag: (tag: string) => void;
  setFilterPinnedOnly: (v: boolean) => void;
  setFilterQuery: (q: string) => void;
  clearFilters: () => void;

  setBaselineSession: (name: string | null) => void;

  undoDelete: () => void;
  clearUndoDelete: () => void;

  addIntervention: () => void;
  removeIntervention: (index: number) => void;
  updateIntervention: (index: number, spec: InterventionSpec) => void;
  clearInterventions: () => void;
  setCaptureLogitLens: (v: boolean) => void;
  setIntervenePrompt: (prompt: string) => void;
  setInterveneSession: (name: string) => void;
}

// Results can carry base64 hidden-state blobs that balloon a single frame
// from <10KB to >1MB. Those are not worth persisting — reloading should
// restore the user's prompts/params/metrics, and re-running recomputes the
// hidden state cheaply. Strip `b64` on write; keep shape+other fields so the
// UI can render "hidden state unavailable (reload)" if ever needed.
function stripHeavyBlobs(r: ProbeResult): ProbeResult {
  return {
    ...r,
    data: r.data.map((msg) => {
      if (msg.type !== "data") return msg;
      // LogitLensData + CompareLogitLensData both expose hidden_state.b64.
      const m = msg as unknown as Record<string, unknown>;
      const out: Record<string, unknown> = { ...m };
      for (const key of ["hidden_state", "hidden_state_a", "hidden_state_b"] as const) {
        const hs = m[key] as { shape?: [number, number]; b64?: string } | undefined;
        if (hs && typeof hs === "object" && "b64" in hs) {
          out[key] = { shape: hs.shape };
        }
      }
      return out as unknown as WsMessage;
    }),
  };
}

const PERSIST_RESULT_CAP = 50;

export const useStore = create<StoreState>()(
  persist(
    (set, get) => ({
      activeTab: "sessions",
      backendOnline: false,
      backendProbed: false,

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

      samplingParams: { ...SAMPLING_DEFAULTS },
      promptLibrary: [],

      results: [],
      activeResultId: null,
      pendingResults: {},

      filterTags: [],
      filterPinnedOnly: false,
      filterQuery: "",

      baselineSession: null,

      lastDeleted: null,

      interventionSpecs: [],
      captureLogitLens: false,
      intervenePrompt: "",
      interveneSession: "",

      setActiveTab: (tab) => set({ activeTab: tab }),

      fetchSessions: async () => {
        try {
          const resp = await apiFetch("/api/sessions");
          if (!resp.ok) {
            set({ backendOnline: false, backendProbed: true });
            return;
          }
          const data = await resp.json();
          if (!Array.isArray(data)) {
            set({ backendOnline: false, backendProbed: true });
            return;
          }
          // Auto-clear a stale baseline if the backend no longer has that
          // session (restart, manual backend-side deletion, etc).
          const st = get();
          if (st.baselineSession && !data.some((x: SessionSummary) => x.name === st.baselineSession)) {
            set({ sessions: data, backendOnline: true, backendProbed: true, baselineSession: null });
          } else {
            set({ sessions: data, backendOnline: true, backendProbed: true });
          }
          if (get().surgeryOps.length === 0) {
            get().fetchSurgeryOps();
          }
        } catch {
          set({ backendOnline: false, backendProbed: true });
        }
      },

      fetchSessionInfo: async (name: string) => {
        const resp = await apiFetch(`/api/sessions/${name}/info`);
        if (!resp.ok) throw await apiError(resp);
        const data = await resp.json();
        set((s) => ({ sessionInfo: { ...s.sessionInfo, [name]: data } }));
      },

      fetchSurgeryOps: async () => {
        try {
          const resp = await apiFetch("/api/surgery/operations");
          const data = await resp.json();
          set({ surgeryOps: data });
        } catch { /* backend not ready */ }
      },

      fetchAvailableModels: async () => {
        try {
          const resp = await apiFetch("/api/models/available");
          const data: AvailableModel[] = await resp.json();
          set({ availableModels: data });
        } catch { /* backend not ready */ }
      },

      deleteSession: async (name: string) => {
        const resp = await apiFetch(`/api/sessions/${name}`, { method: "DELETE" });
        if (!resp.ok) throw await apiError(resp);
        // Close any live WebSockets tied to this session *before* we forget the
        // pending results — otherwise the socket keeps streaming frames into a
        // pending entry that no longer exists, and only closes when the backend
        // eventually drops it.
        cancelSessionSockets(name);
        set((s) => {
          const { [name]: _removed, ...remainingInfo } = s.sessionInfo;
          const filteredPending: Record<string, ProbeResult> = {};
          for (const [id, r] of Object.entries(s.pendingResults)) {
            if (r.sessionName !== name) filteredPending[id] = r;
          }
          return {
            sessionInfo: remainingInfo,
            pendingResults: filteredPending,
            targetSession: s.targetSession === name ? "" : s.targetSession,
            targetSessionB: s.targetSessionB === name ? null : s.targetSessionB,
            interveneSession: s.interveneSession === name ? "" : s.interveneSession,
            baselineSession: s.baselineSession === name ? null : s.baselineSession,
          };
        });
        await get().fetchSessions();
      },

      applySurgery: async (name: string, operation: string, params: Record<string, unknown>) => {
        const resp = await apiFetch(`/api/sessions/${name}/surgery`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ operation, params }),
        });
        if (!resp.ok) throw await apiError(resp);
        await get().fetchSessions();
        await get().fetchSessionInfo(name);
      },

      undoSurgery: async (name: string) => {
        const resp = await apiFetch(`/api/sessions/${name}/surgery/last`, { method: "DELETE" });
        if (!resp.ok) throw await apiError(resp);
        await get().fetchSessions();
        await get().fetchSessionInfo(name);
      },

      deleteStagedOp: async (name: string, index: number) => {
        const resp = await apiFetch(`/api/sessions/${name}/surgery/${index}`, { method: "DELETE" });
        if (!resp.ok) throw await apiError(resp);
        await get().fetchSessions();
        await get().fetchSessionInfo(name);
      },

      commitSurgery: async (name: string) => {
        const resp = await apiFetch(`/api/sessions/${name}/surgery/commit`, { method: "POST" });
        if (!resp.ok) throw await apiError(resp);
        await get().fetchSessions();
        await get().fetchSessionInfo(name);
      },

      revertSurgery: async (name: string) => {
        const resp = await apiFetch(`/api/sessions/${name}/surgery/revert`, { method: "POST" });
        if (!resp.ok) throw await apiError(resp);
        await get().fetchSessions();
        await get().fetchSessionInfo(name);
      },

      cloneSession: async (name: string, targetName: string) => {
        const resp = await apiFetch(`/api/sessions/${name}/clone`, {
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

      setSamplingParams: (patch) => set((s) => ({ samplingParams: { ...s.samplingParams, ...patch } })),
      resetSamplingParams: () => set({ samplingParams: { ...SAMPLING_DEFAULTS } }),

      savePrompt: (name, text, tags) => set((s) => {
        // Overwrite if a prompt with the same name already exists — the user
        // almost always means "update" when re-saving under a familiar name,
        // and silently duplicating would clutter the dropdown fast.
        const trimmed = name.trim() || "(untitled)";
        const existing = s.promptLibrary.find((p) => p.name === trimmed);
        const entry: NamedPrompt = {
          id: existing?.id ?? `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          name: trimmed,
          text,
          tags,
          createdAt: existing?.createdAt ?? Date.now(),
        };
        const others = s.promptLibrary.filter((p) => p.id !== entry.id);
        return { promptLibrary: [entry, ...others] };
      }),

      deletePrompt: (id) => set((s) => ({ promptLibrary: s.promptLibrary.filter((p) => p.id !== id) })),

      addResult: (result) => set((s) => ({
        // Dedupe by id. Defensive: addResult and finalizePendingResult should
        // never fire for the same id today, but filtering here keeps the state
        // machine correct if a future caller slips up.
        results: [result, ...s.results.filter((r) => r.id !== result.id)],
        activeResultId: result.operation !== "generate" && !result.isB ? result.id : s.activeResultId,
      })),
      clearResults: () => set((s) => {
        // Pinned results are the researcher's labeled favorites; Clear All
        // preserves them so one accidental click doesn't wipe annotated
        // work. Reset activeResultId only if it pointed at something we
        // just removed.
        const kept = s.results.filter((r) => r.pinned);
        const keptIds = new Set(kept.map((r) => r.id));
        return {
          results: kept,
          activeResultId: s.activeResultId && keptIds.has(s.activeResultId) ? s.activeResultId : null,
        };
      }),
      setActiveResult: (id) => set({ activeResultId: id }),

      setPendingResult: (id, result) => set((s) => ({
        pendingResults: { ...s.pendingResults, [id]: result },
        activeResultId: result.operation !== "generate" && !result.isB ? id : s.activeResultId,
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
          results: [{ ...pending, data: finalData }, ...s.results.filter((r) => r.id !== id)],
          activeResultId: pending.operation !== "generate" && !pending.isB ? pending.id : s.activeResultId,
        };
      }),

      removePendingResult: (id) => set((s) => {
        const { [id]: _, ...remaining } = s.pendingResults;
        return { pendingResults: remaining };
      }),

      updateResultMeta: (id, patch) => set((s) => {
        // Meta can apply either while a result is still pending (so pin
        // survives the pending→finalized handoff) or after it's landed in
        // `results`. Check both; no-op if id is unknown.
        if (s.pendingResults[id]) {
          return {
            pendingResults: {
              ...s.pendingResults,
              [id]: { ...s.pendingResults[id], ...patch },
            },
          };
        }
        const idx = s.results.findIndex((r) => r.id === id);
        if (idx < 0) return s;
        const next = [...s.results];
        next[idx] = { ...next[idx], ...patch };
        return { results: next };
      }),

      deleteResult: (id) => set((s) => {
        const { [id]: _, ...remainingPending } = s.pendingResults;
        const idx = s.results.findIndex((r) => r.id === id);
        const snapshot = idx >= 0 ? s.results[idx] : null;
        return {
          pendingResults: remainingPending,
          results: idx >= 0
            ? [...s.results.slice(0, idx), ...s.results.slice(idx + 1)]
            : s.results,
          activeResultId: s.activeResultId === id ? null : s.activeResultId,
          // Pending-only deletions don't populate undo — the result has no
          // finalized content yet, restoring a half-streamed pending entry
          // would tangle with the WebSocket state machine.
          lastDeleted: snapshot
            ? { result: snapshot, index: idx, deletedAt: Date.now() }
            : s.lastDeleted,
        };
      }),

      toggleFilterTag: (tag) => set((s) => ({
        filterTags: s.filterTags.includes(tag)
          ? s.filterTags.filter((t) => t !== tag)
          : [...s.filterTags, tag],
      })),
      setFilterPinnedOnly: (v) => set({ filterPinnedOnly: v }),
      setFilterQuery: (q) => set({ filterQuery: q }),
      clearFilters: () => set({ filterTags: [], filterPinnedOnly: false, filterQuery: "" }),

      setBaselineSession: (name) => set({ baselineSession: name }),

      undoDelete: () => set((s) => {
        if (!s.lastDeleted) return s;
        const { result, index } = s.lastDeleted;
        // Insert at the original index; clamp if the list got shorter due
        // to other operations between delete and undo.
        const cap = s.results.length;
        const where = Math.min(Math.max(0, index), cap);
        const next = [...s.results.slice(0, where), result, ...s.results.slice(where)];
        return {
          results: next,
          lastDeleted: null,
          activeResultId: result.id,
        };
      }),

      clearUndoDelete: () => set({ lastDeleted: null }),

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
    }),
    {
      name: "llm-surgeon-gui/v1",
      version: 1,
      storage: createJSONStorage(() => idbStorage),
      // Whitelist-style: only the fields worth restoring survive a reload.
      // Backend-refreshed lists (sessions, models, surgeryOps, sessionInfo)
      // are intentionally re-fetched on mount. `pendingResults` are in-flight
      // WebSocket streams and would dangle if persisted. `isRunning` /
      // `isLoadingModel` describe *this* tab's transient UI state.
      partialize: (s) => ({
        activeTab: s.activeTab,
        prompt: s.prompt,
        operation: s.operation,
        targetSession: s.targetSession,
        targetSessionB: s.targetSessionB,
        samplingParams: s.samplingParams,
        promptLibrary: s.promptLibrary,
        // Pinned results always persist; unpinned follow the recent-N cap.
        // Merged ordering matches the in-memory list so tab order is stable.
        results: (() => {
          const keepIds = new Set<string>();
          let unpinnedBudget = PERSIST_RESULT_CAP;
          for (const r of s.results) {
            if (r.pinned) { keepIds.add(r.id); continue; }
            if (unpinnedBudget > 0) { keepIds.add(r.id); unpinnedBudget--; }
          }
          return s.results.filter((r) => keepIds.has(r.id)).map(stripHeavyBlobs);
        })(),
        activeResultId: s.activeResultId,
        interventionSpecs: s.interventionSpecs,
        captureLogitLens: s.captureLogitLens,
        intervenePrompt: s.intervenePrompt,
        interveneSession: s.interveneSession,
        baselineSession: s.baselineSession,
      }),
    },
  ),
);
