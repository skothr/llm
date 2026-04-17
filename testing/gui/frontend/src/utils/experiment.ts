import type { SamplingParams, NamedPrompt } from "../state/store";
import type {
  ProbeResult,
  InterventionSpec,
  ProbeOperation,
  SessionSummary,
  ConfigTab,
} from "../types/api";

export const EXPERIMENT_SCHEMA = "llm-surgeon-gui-experiment/v1";

// Stable on-disk shape for a saved experiment. Deliberately plain JSON; the
// importer validates `schema` and tolerates missing optional fields, so
// future shape bumps can add without breaking older files.
export interface ExperimentFile {
  schema: typeof EXPERIMENT_SCHEMA;
  exportedAt: string;
  // User-written note, optional — useful to annotate a saved experiment
  // ("baseline for layer-15 ablation", etc.) without touching result data.
  note?: string;
  activeTab?: ConfigTab;
  prompt: string;
  operation: ProbeOperation;
  targetSession: string;
  targetSessionB: string | null;
  samplingParams: SamplingParams;
  interventionSpecs: InterventionSpec[];
  captureLogitLens: boolean;
  intervenePrompt: string;
  interveneSession: string;
  promptLibrary: NamedPrompt[];
  results: ProbeResult[];
  activeResultId: string | null;
  // Snapshot of session summaries at export time. Purely informational —
  // the importer does not try to rehydrate sessions (that requires a live
  // backend with the same model cache). Used by the UI to warn when an
  // imported result references a session the current backend doesn't have.
  sessionSnapshots: SessionSummary[];
}

export interface ExperimentSnapshotInput {
  activeTab?: ConfigTab;
  prompt: string;
  operation: ProbeOperation;
  targetSession: string;
  targetSessionB: string | null;
  samplingParams: SamplingParams;
  interventionSpecs: InterventionSpec[];
  captureLogitLens: boolean;
  intervenePrompt: string;
  interveneSession: string;
  promptLibrary: NamedPrompt[];
  results: ProbeResult[];
  activeResultId: string | null;
  sessions: SessionSummary[];
}

export function buildExperimentFile(
  input: ExperimentSnapshotInput,
  note?: string,
): ExperimentFile {
  return {
    schema: EXPERIMENT_SCHEMA,
    exportedAt: new Date().toISOString(),
    ...(note ? { note } : {}),
    activeTab: input.activeTab,
    prompt: input.prompt,
    operation: input.operation,
    targetSession: input.targetSession,
    targetSessionB: input.targetSessionB,
    samplingParams: input.samplingParams,
    interventionSpecs: input.interventionSpecs,
    captureLogitLens: input.captureLogitLens,
    intervenePrompt: input.intervenePrompt,
    interveneSession: input.interveneSession,
    promptLibrary: input.promptLibrary,
    results: input.results,
    activeResultId: input.activeResultId,
    sessionSnapshots: input.sessions,
  };
}

export interface ImportIssue {
  severity: "warn" | "error";
  message: string;
}

export interface ImportOutcome {
  file: ExperimentFile | null;
  issues: ImportIssue[];
}

// Defensive parse. Schema mismatches downgrade to warnings (still try to
// load) unless the shape is unrecognizable, in which case we return null.
export function parseExperimentFile(text: string): ImportOutcome {
  const issues: ImportIssue[] = [];
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { file: null, issues: [{ severity: "error", message: `JSON parse failed: ${(e as Error).message}` }] };
  }
  if (!raw || typeof raw !== "object") {
    return { file: null, issues: [{ severity: "error", message: "Top-level value is not an object" }] };
  }
  const obj = raw as Record<string, unknown>;
  if (obj.schema !== EXPERIMENT_SCHEMA) {
    issues.push({
      severity: "warn",
      message: `Schema mismatch: expected ${EXPERIMENT_SCHEMA}, got ${String(obj.schema)}. Attempting to load anyway.`,
    });
  }

  // Required-ish string fields. Missing ones default to empty; most panels
  // render gracefully with empties, so partial imports are still useful.
  const asStr = (v: unknown, d = ""): string => typeof v === "string" ? v : d;
  const asBool = (v: unknown, d = false): boolean => typeof v === "boolean" ? v : d;
  const asArr = <T>(v: unknown): T[] => Array.isArray(v) ? (v as T[]) : [];

  const samplingParams = (obj.samplingParams as SamplingParams | undefined) ?? null;
  if (!samplingParams || typeof samplingParams !== "object") {
    issues.push({ severity: "warn", message: "samplingParams missing; will keep current values" });
  }

  const file: ExperimentFile = {
    schema: EXPERIMENT_SCHEMA,
    exportedAt: asStr(obj.exportedAt, new Date().toISOString()),
    note: typeof obj.note === "string" ? obj.note : undefined,
    activeTab: (obj.activeTab as ConfigTab | undefined),
    prompt: asStr(obj.prompt),
    operation: (asStr(obj.operation, "logit-lens") as ProbeOperation),
    targetSession: asStr(obj.targetSession),
    targetSessionB: typeof obj.targetSessionB === "string" ? obj.targetSessionB : null,
    samplingParams: samplingParams as SamplingParams,
    interventionSpecs: asArr<InterventionSpec>(obj.interventionSpecs),
    captureLogitLens: asBool(obj.captureLogitLens),
    intervenePrompt: asStr(obj.intervenePrompt),
    interveneSession: asStr(obj.interveneSession),
    promptLibrary: asArr<NamedPrompt>(obj.promptLibrary),
    results: asArr<ProbeResult>(obj.results),
    activeResultId: typeof obj.activeResultId === "string" ? obj.activeResultId : null,
    sessionSnapshots: asArr<SessionSummary>(obj.sessionSnapshots),
  };

  return { file, issues };
}

// Convenience: find sessions referenced by imported results/fields that the
// current backend doesn't know about — so the UI can flag them to the user.
export function missingSessionsFromImport(
  file: ExperimentFile,
  currentSessions: SessionSummary[],
): string[] {
  const have = new Set(currentSessions.map((s) => s.name));
  const referenced = new Set<string>();
  if (file.targetSession) referenced.add(file.targetSession);
  if (file.targetSessionB) referenced.add(file.targetSessionB);
  if (file.interveneSession) referenced.add(file.interveneSession);
  for (const r of file.results) referenced.add(r.sessionName);
  return [...referenced].filter((s) => s && !have.has(s));
}
