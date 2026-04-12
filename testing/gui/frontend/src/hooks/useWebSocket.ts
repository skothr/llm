import { useRef, useCallback } from "react";
import type { WsMessage } from "../types/api";

interface UseWebSocketOptions {
  onMessage: (msg: WsMessage) => void;
  onComplete: (msg: WsMessage) => void;
  onError: (msg: string) => void;
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(
    (path: string, config: Record<string, unknown>, handlers: UseWebSocketOptions) => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${window.location.host}${path}`);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify(config));
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as WsMessage;
        if (msg.type === "data") {
          handlers.onMessage(msg);
        } else if (msg.type === "complete") {
          handlers.onComplete(msg);
        } else if (msg.type === "error") {
          handlers.onError((msg as { message: string }).message);
        }
      };

      ws.onerror = () => {
        handlers.onError("WebSocket connection error");
      };

      ws.onclose = () => {
        wsRef.current = null;
      };

      return ws;
    },
    []
  );

  const cancel = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "cancel" }));
    }
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  return { connect, cancel, disconnect };
}
