import { useDashboardStore } from "../state/store";
import { cn } from "../lib/utils";

const COLORS: Record<string, string> = {
  open: "bg-emerald-400 shadow-[0_0_12px_rgb(74,222,128,0.6)]",
  connecting: "bg-amber-400 animate-pulse",
  reconnecting: "bg-amber-500 animate-pulse",
  failed: "bg-rose-500",
  unauthorized: "bg-rose-500",
  idle: "bg-slate-500",
};

const LABELS: Record<string, string> = {
  open: "live",
  connecting: "connecting",
  reconnecting: "reconnecting",
  failed: "offline",
  unauthorized: "sign-in required",
  idle: "idle",
};

export function ConnectionDot() {
  const state = useDashboardStore((s) => s.connection);
  return (
    <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
      <span className={cn("w-2 h-2 rounded-full", COLORS[state] ?? COLORS.idle)} />
      <span className="hidden sm:inline">{LABELS[state] ?? state}</span>
    </div>
  );
}
