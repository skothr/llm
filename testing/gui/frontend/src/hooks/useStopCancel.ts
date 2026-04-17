import { useCallback } from "react";
import { useStore } from "../state/store";

// The Stop / Cancel pair: both halt in-flight WebSocket runs, but Stop
// preserves whatever the model has streamed (finalize → move to results)
// while Cancel discards it (remove). Both clear the local pending-id set
// and flip isRunning false.
//
// Each panel owns its own localPendingIdsRef (the set of ids that *this*
// panel started) so Stop in Probe doesn't touch Intervene's pending, and
// vice versa — the ref is passed in rather than kept in the store.
export function useStopCancel(localPendingIdsRef: React.RefObject<Set<string>>) {
  const pendingResults = useStore((s) => s.pendingResults);
  const finalizePendingResult = useStore((s) => s.finalizePendingResult);
  const removePendingResult = useStore((s) => s.removePendingResult);
  const setRunning = useStore((s) => s.setRunning);

  // useWebSocket keeps its connections map inside a ref; one hook instance
  // per panel means callers must pass in their cancelAll function. Rather
  // than threading that through, we accept the same cancel signature.
  type CancelAll = () => void;

  const makeHandler = useCallback(
    (mode: "stop" | "cancel", cancelAll: CancelAll) => () => {
      cancelAll();
      const ids = localPendingIdsRef.current;
      if (ids) {
        for (const id of ids) {
          if (!pendingResults[id]) continue;
          if (mode === "stop") finalizePendingResult(id);
          else removePendingResult(id);
        }
        ids.clear();
      }
      setRunning(false);
    },
    [localPendingIdsRef, pendingResults, finalizePendingResult, removePendingResult, setRunning],
  );

  return { makeHandler };
}
