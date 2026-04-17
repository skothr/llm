import { useRef, useCallback } from "react";
import type { WsMessage } from "../types/api";

interface WsHandlers {
  onMessage: (msg: WsMessage) => void;
  onComplete: (msg: WsMessage) => void;
  onError: (message: string) => void;
  onDisconnect?: () => void;
}

// Module-level registry of live sockets so callers outside any single hook
// instance (notably the store's deleteSession) can close sockets owned by a
// session that's about to disappear. Without this, a delete followed by a
// still-streaming generate leaks the socket until the backend drops it.
type RegEntry = { sessionName: string; ws: WebSocket };
const _registry: Map<string, RegEntry> = new Map();

/**
 * Close every live WebSocket associated with `sessionName`.
 * Called by the store when the session is deleted.
 */
export function cancelSessionSockets(sessionName: string): void {
  for (const [key, entry] of _registry) {
    if (entry.sessionName !== sessionName) continue;
    try {
      if (entry.ws.readyState !== WebSocket.CLOSED) entry.ws.close();
    } catch { /* already closing */ }
    _registry.delete(key);
  }
}

export function useWebSocket() {
  const connectionsRef = useRef<Map<string, WebSocket>>(new Map());
  const resolvedRef = useRef<Set<string>>(new Set());
  const cancelledRef = useRef<Set<string>>(new Set());

  const connect = useCallback(
    (
      key: string,
      path: string,
      config: Record<string, unknown>,
      handlers: WsHandlers,
      sessionName: string,
    ) => {
      const existing = connectionsRef.current.get(key);
      if (existing && existing.readyState !== WebSocket.CLOSED) {
        existing.close();
      }
      resolvedRef.current.delete(key);
      cancelledRef.current.delete(key);

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${window.location.host}${path}`);
      connectionsRef.current.set(key, ws);
      _registry.set(key, { sessionName, ws });

      ws.onopen = () => {
        ws.send(JSON.stringify(config));
      };

      ws.onmessage = (event) => {
        let msg: WsMessage;
        try {
          msg = JSON.parse(event.data) as WsMessage;
        } catch {
          resolvedRef.current.add(key);
          handlers.onError(`Malformed frame from server: ${String(event.data).slice(0, 120)}`);
          return;
        }
        if (msg.type === "data") {
          handlers.onMessage(msg);
        } else if (msg.type === "complete") {
          resolvedRef.current.add(key);
          handlers.onComplete(msg);
        } else if (msg.type === "error") {
          resolvedRef.current.add(key);
          const raw = (msg as { message?: unknown }).message;
          handlers.onError(typeof raw === "string" ? raw : "Unknown server error");
        }
      };

      ws.onerror = () => {
        if (resolvedRef.current.has(key)) return;
        resolvedRef.current.add(key);
        handlers.onError("WebSocket connection error");
      };

      ws.onclose = () => {
        if (connectionsRef.current.get(key) !== ws) return;
        connectionsRef.current.delete(key);
        _registry.delete(key);
        if (!resolvedRef.current.has(key) && !cancelledRef.current.has(key)) {
          handlers.onDisconnect?.();
        }
        resolvedRef.current.delete(key);
        cancelledRef.current.delete(key);
      };

      return ws;
    },
    []
  );

  const cancel = useCallback((key: string) => {
    cancelledRef.current.add(key);
    const ws = connectionsRef.current.get(key);
    if (ws && ws.readyState !== WebSocket.CLOSED) {
      ws.close();
    }
    connectionsRef.current.delete(key);
    _registry.delete(key);
  }, []);

  const cancelAll = useCallback(() => {
    connectionsRef.current.forEach((_, key) => {
      cancelledRef.current.add(key);
    });
    connectionsRef.current.forEach((ws, key) => {
      if (ws.readyState !== WebSocket.CLOSED) {
        ws.close();
      }
      _registry.delete(key);
    });
    connectionsRef.current.clear();
  }, []);

  const disconnect = useCallback((key: string) => {
    const ws = connectionsRef.current.get(key);
    if (ws) {
      cancelledRef.current.add(key);
      ws.close();
      connectionsRef.current.delete(key);
      _registry.delete(key);
    }
  }, []);

  return { connect, cancel, cancelAll, disconnect };
}
