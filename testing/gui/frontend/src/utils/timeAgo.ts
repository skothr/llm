// Compact relative-time formatter: "just now", "3m", "2h", "5d", else ISO
// date. Tuned for result-tab labels where vertical space is tight and the
// exact precision doesn't matter.
//
// Called with `now` injected so callers can memoize over a shared tick
// (most useful in lists where re-rendering per-tab on every second would
// be wasteful).
export function timeAgo(timestamp: number, now = Date.now()): string {
  const secs = Math.max(0, Math.floor((now - timestamp) / 1000));
  if (secs < 10) return "now";
  if (secs < 60) return `${secs}s`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d`;
  // Older than a week: show the date (no time) so researchers can tell
  // which day's experiment this was.
  const d = new Date(timestamp);
  return d.toISOString().slice(0, 10);
}

// Hook that triggers a re-render on an interval so components using
// timeAgo() re-compute their labels without manually managing timers.
// Default tick is 30 seconds — matches the granularity of the "Xm"
// labels, so anything shorter would be wasted renders.
import { useEffect, useState } from "react";
export function useTimeAgoTick(intervalMs = 30_000): number {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}
