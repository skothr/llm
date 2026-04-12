import { useStore } from "../state/store";

export function useSessions() {
  const sessions = useStore((s) => s.sessions);
  const sessionInfo = useStore((s) => s.sessionInfo);
  const surgeryOps = useStore((s) => s.surgeryOps);
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSessionInfo = useStore((s) => s.fetchSessionInfo);
  const deleteSession = useStore((s) => s.deleteSession);
  const applySurgery = useStore((s) => s.applySurgery);
  const undoSurgery = useStore((s) => s.undoSurgery);
  const cloneSession = useStore((s) => s.cloneSession);

  return {
    sessions,
    sessionInfo,
    surgeryOps,
    fetchSessions,
    fetchSessionInfo,
    deleteSession,
    applySurgery,
    undoSurgery,
    cloneSession,
  };
}
