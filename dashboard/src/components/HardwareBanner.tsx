import { Cpu, MemoryStick, Sparkles } from "lucide-react";
import { Card } from "./Card";
import { fmtBytes } from "../lib/utils";
import { useDashboardStore } from "../state/store";

// Fallback only — the server reports the real chip string (sysctl
// machdep.cpu.brand_string) and that always wins. Identifier prefixes map to
// chip families, not tiers: Mac13,x is the M1-era Mac Studio, Mac14,x is the
// M2 family (incl. Mac Studio M2 Max/Ultra, #329), Mac15/16/17 are M3/M4/M5.
function chipFromModel(machineModel: string | null | undefined): string {
  if (!machineModel) return "Apple Silicon";
  const id = machineModel.toLowerCase();
  if (id.includes("mac17")) return "M5";
  if (id.includes("mac16")) return "M4";
  if (id.includes("mac15")) return "M3";
  if (id.includes("mac14")) return "M2";
  if (id.includes("mac13")) return "M1";
  return "Apple Silicon";
}

function chipBadgeFor(
  chip: string | null | undefined,
  machineModel: string | null | undefined,
): string {
  const reported = (chip ?? "").replace(/^Apple\s+/i, "").trim();
  return reported || chipFromModel(machineModel);
}

export function HardwareBanner() {
  const machine = useDashboardStore((s) => s.machine);
  const profileName = useDashboardStore((s) => s.profileName);
  const modelId = useDashboardStore((s) => s.modelId);
  const contextWindow = useDashboardStore((s) => s.contextWindow);
  const chipBadge = chipBadgeFor(machine?.chip, machine?.machine_model);

  return (
    <Card title="Hardware" subtitle={machine?.machine_model ?? "unknown machine model"}>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <Tile
          icon={<Cpu className="size-4 text-[var(--accent)]" />}
          label="chip"
          value={chipBadge}
        />
        <Tile
          icon={<MemoryStick className="size-4 text-[var(--accent-cool)]" />}
          label="unified memory"
          value={fmtBytes(machine?.unified_memory_bytes ?? null)}
        />
        <Tile
          icon={<Sparkles className="size-4 text-[var(--accent-warm)]" />}
          label="profile"
          value={profileName ?? "—"}
        />
        <Tile
          icon={<Cpu className="size-4 text-[var(--text-muted)]" />}
          label="context window"
          value={contextWindow ? `${contextWindow.toLocaleString()} tok` : "—"}
        />
      </div>
      <div className="mt-3 text-xs text-[var(--text-muted)] truncate">
        loaded model: <span className="text-[var(--text-primary)]">{modelId ?? "—"}</span>
      </div>
    </Card>
  );
}

function Tile({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-md border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-3">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        {icon}
        {label}
      </div>
      <div className="text-base font-semibold text-[var(--text-primary)] mt-1 truncate">
        {value}
      </div>
    </div>
  );
}
