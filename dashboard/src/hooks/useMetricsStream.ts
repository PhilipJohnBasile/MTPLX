import { useEffect, useRef } from "react";
import { startMetricsStream, type MetricsStreamHandle } from "../lib/sse";
import { useDashboardStore } from "../state/store";

export function useMetricsStream(): void {
  const handleRef = useRef<MetricsStreamHandle | null>(null);
  const applySnapshot = useDashboardStore((s) => s.applySnapshot);
  const applyEvent = useDashboardStore((s) => s.applyEvent);
  const setConnection = useDashboardStore((s) => s.setConnection);
  const setAuthRequired = useDashboardStore((s) => s.setAuthRequired);

  useEffect(() => {
    setConnection("connecting");
    const handle = startMetricsStream({
      onSnapshot: applySnapshot,
      onEvent: applyEvent,
      onConnectionChange: setConnection,
      onUnauthorized: () => setAuthRequired(true),
    });
    handleRef.current = handle;
    return () => {
      handle.close();
      handleRef.current = null;
    };
  }, [applySnapshot, applyEvent, setConnection, setAuthRequired]);
}
