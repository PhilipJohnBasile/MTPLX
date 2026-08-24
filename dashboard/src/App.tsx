import { useEffect, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BottomStatsBar } from "./components/BottomStatsBar";
import { CacheTab } from "./components/CacheTab";
import { ControlsSidebar } from "./components/ControlsSidebar";
import { KeyboardShortcutsOverlay } from "./components/KeyboardShortcutsOverlay";
import { MemoryTab } from "./components/MemoryTab";
import { NativeTab } from "./components/NativeTab";
import { NewMaxTPSToast } from "./components/NewMaxTPSToast";
import { OverviewTab } from "./components/OverviewTab";
import { RequestsTab } from "./components/RequestsTab";
import { Shell, type TabId } from "./components/Shell";
import { SpeculativeTab } from "./components/SpeculativeTab";
import { SystemsTab } from "./components/SystemsTab";
import { ThermalTab } from "./components/ThermalTab";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { useMetricsStream } from "./hooks/useMetricsStream";
import { useDashboardStore } from "./state/store";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DashboardRoot />
      <NewMaxTPSToast />
      <KeyboardShortcutsOverlay />
    </QueryClientProvider>
  );
}

function DashboardRoot() {
  const [active, setActive] = useState<TabId>("overview");
  useMetricsStream();
  useKeyboardShortcuts(setActive);
  const pauseStream = useDashboardStore((s) => s.pauseStream);

  useEffect(() => {
    document.body.dataset.activeTab = active;
  }, [active]);

  useEffect(() => {
    document.body.dataset.streamPaused = String(pauseStream);
  }, [pauseStream]);

  return (
    <Shell active={active} onSelect={setActive} bottomBar={<BottomStatsBar />}>
      {active === "overview" ? (
        <OverviewTab />
      ) : active === "speculative" ? (
        <SpeculativeTab />
      ) : active === "cache" ? (
        <CacheTab />
      ) : active === "memory" ? (
        <MemoryTab />
      ) : active === "thermal" ? (
        <ThermalTab />
      ) : active === "requests" ? (
        <RequestsTab />
      ) : active === "systems" ? (
        <SystemsTab />
      ) : active === "native" ? (
        <NativeTab />
      ) : active === "settings" ? (
        <ControlsSidebar />
      ) : null}
    </Shell>
  );
}
