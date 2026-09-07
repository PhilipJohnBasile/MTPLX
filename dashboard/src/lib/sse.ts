import { probeServer } from "./api";
import type { BusEvent, ConnectionState, DashboardSnapshot } from "./types";

const BACKOFF_SEQ_MS = [1000, 2000, 4000, 8000, 16000, 30000];

export type MetricsStreamHandlers = {
  onSnapshot: (snapshot: DashboardSnapshot) => void;
  onEvent: (event: BusEvent) => void;
  onConnectionChange?: (state: ConnectionState) => void;
  // The server refused the stream with 401: the browser needs to sign in.
  // Reconnecting would only repeat the refusal, so the stream stops until
  // the page is reloaded after sign-in.
  onUnauthorized?: () => void;
};

export type MetricsStreamHandle = {
  close: () => void;
  state: () => ConnectionState;
};

export function startMetricsStream(handlers: MetricsStreamHandlers): MetricsStreamHandle {
  let state: ConnectionState = "idle";
  let source: EventSource | null = null;
  let closed = false;
  let attempt = 0;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  function setState(next: ConnectionState) {
    state = next;
    handlers.onConnectionChange?.(next);
  }

  function clearReconnect() {
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  }

  function scheduleReconnect() {
    if (closed) return;
    setState("reconnecting");
    const delay = BACKOFF_SEQ_MS[Math.min(attempt, BACKOFF_SEQ_MS.length - 1)];
    attempt += 1;
    reconnectTimer = setTimeout(connect, delay);
  }

  function connect() {
    if (closed) return;
    clearReconnect();
    setState("connecting");
    try {
      source = new EventSource("/v1/mtplx/metrics/stream");
    } catch (err) {
      console.error("EventSource construction failed", err);
      scheduleReconnect();
      return;
    }

    source.addEventListener("open", () => {
      attempt = 0;
      setState("open");
    });

    source.addEventListener("snapshot", (e) => {
      try {
        const snapshot = JSON.parse((e as MessageEvent).data) as DashboardSnapshot;
        handlers.onSnapshot(snapshot);
      } catch (err) {
        console.warn("failed to parse snapshot event", err);
      }
    });

    const forward = (kind: BusEvent["kind"]) => (e: Event) => {
      try {
        const payload = JSON.parse((e as MessageEvent).data) as BusEvent;
        handlers.onEvent({ ...payload, kind } as BusEvent);
      } catch (err) {
        console.warn(`failed to parse ${kind} event`, err);
      }
    };

    source.addEventListener("progress", forward("progress"));
    source.addEventListener("completed", forward("completed"));
    source.addEventListener("new_max_tps", forward("new_max_tps"));
    source.addEventListener("thermal", forward("thermal"));
    source.addEventListener("prefill", forward("prefill"));

    source.addEventListener("error", () => {
      if (closed) return;
      // EventSource auto-reconnects, but a long absence (server restart,
      // network drop) deserves an explicit user signal.
      if (source && source.readyState === EventSource.CLOSED) {
        try {
          source.close();
        } catch {
          // ignore
        }
        source = null;
        // EventSource hides the status code; a closed stream from a server
        // that still answers is a refusal, not an outage. Ask /health which.
        void probeServer().then((verdict) => {
          if (closed) return;
          if (verdict === "unauthorized") {
            setState("unauthorized");
            handlers.onUnauthorized?.();
            return;
          }
          if (attempt >= BACKOFF_SEQ_MS.length) {
            setState("failed");
          }
          scheduleReconnect();
        });
      } else {
        setState("reconnecting");
      }
    });
  }

  connect();

  return {
    close: () => {
      closed = true;
      clearReconnect();
      if (source) {
        try {
          source.close();
        } catch {
          // ignore
        }
        source = null;
      }
      setState("idle");
    },
    state: () => state,
  };
}
