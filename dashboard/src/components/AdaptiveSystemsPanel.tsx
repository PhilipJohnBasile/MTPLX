import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BrainCircuit,
  DatabaseZap,
  Network,
  ShieldCheck,
} from "lucide-react";
import { Card } from "./Card";

type SystemState = Record<string, unknown>;
type SystemsPayload = Record<string, SystemState | unknown>;

type AdaptiveRow = {
  key: string;
  label: string;
  description: string;
  icon: typeof Activity;
};

const ROWS: AdaptiveRow[] = [
  {
    key: "expert_residency",
    label: "Expert residency",
    description: "Locality-guided warm-set planning without router mutation.",
    icon: BrainCircuit,
  },
  {
    key: "unified_memory",
    label: "Unified memory",
    description: "Atomic budgets for SessionBank, expert state, and KV headroom.",
    icon: DatabaseZap,
  },
  {
    key: "otlp_export",
    label: "OTLP export",
    description: "Bounded, privacy-first OTLP/HTTP telemetry with no SDK dependency.",
    icon: Network,
  },
  {
    key: "policy_hooks",
    label: "Policy hooks",
    description: "Timeout-bounded request, stream, response, and error policies.",
    icon: ShieldCheck,
  },
  {
    key: "replay_orchestration",
    label: "Replay orchestration",
    description: "Capture selection, stale-plan checks, and non-automatic promotion receipts.",
    icon: Activity,
  },
];

function asRecord(value: unknown): SystemState {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as SystemState)
    : {};
}

function bool(value: unknown): boolean {
  return value === true;
}

function stateLabel(state: SystemState): string {
  if (!bool(state.available)) return "unavailable";
  if (!bool(state.enabled)) return "inactive";
  const lastApply = asRecord(state.last_receipt ?? state.last_apply);
  if (lastApply.applied === false) return "blocked";
  if (lastApply.applied === true) return "active";
  if (state.sampled === true || state.exported_spans) return "observed";
  return "enabled";
}

function metricRows(state: SystemState): [string, unknown][] {
  const preferred = [
    "backend_mode",
    "budget_bytes",
    "tracked_experts",
    "resident_experts",
    "registered_hooks",
    "exported_spans",
    "dropped_spans",
    "receipt_count",
    "promotion_is_automatic",
    "failure_is_request_fatal",
  ];
  return preferred
    .filter((key) => key in state)
    .slice(0, 4)
    .map((key) => [key, state[key]]);
}

function formatValue(value: unknown): string {
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return value.toLocaleString();
  if (typeof value === "string") return value;
  if (value == null) return "—";
  return JSON.stringify(value);
}

export function AdaptiveSystemsPanel() {
  const [payload, setPayload] = useState<SystemsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;

    async function refresh() {
      try {
        const response = await fetch("/v1/mtplx/systems", {
          headers: { Accept: "application/json" },
          cache: "no-store",
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const next = (await response.json()) as SystemsPayload;
        if (!cancelled) {
          setPayload(next);
          setError(null);
        }
      } catch (reason) {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : "request failed");
        }
      } finally {
        if (!cancelled) timer = window.setTimeout(refresh, 2000);
      }
    }

    void refresh();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, []);

  const installed = useMemo(
    () => ROWS.filter((row) => payload && row.key in payload),
    [payload],
  );

  return (
    <Card
      title="Adaptive native systems"
      subtitle="Independent MTPLX implementations; no FreeToken or Future AGI runtime dependency"
    >
      {error ? (
        <div className="mb-3 rounded-lg border border-[var(--accent-warm)]/30 bg-[var(--accent-warm)]/5 px-3 py-2 text-xs text-[var(--accent-warm)]">
          Systems refresh failed: {error}
        </div>
      ) : null}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
        {ROWS.map((row) => {
          const state = asRecord(payload?.[row.key]);
          const label = stateLabel(state);
          const Icon = row.icon;
          const metrics = metricRows(state);
          return (
            <div
              key={row.key}
              className="rounded-xl border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-4"
            >
              <div className="flex items-start justify-between gap-2">
                <span className="inline-flex size-8 items-center justify-center rounded-lg border border-[var(--border-soft)] bg-[var(--bg-card)] text-[var(--accent-cool)]">
                  <Icon className="size-4" />
                </span>
                <span className="rounded-full border border-[var(--border-soft)] px-2 py-0.5 text-[9px] uppercase tracking-widest text-[var(--text-muted)]">
                  {label}
                </span>
              </div>
              <div className="mt-3 text-sm font-semibold text-[var(--text-primary)]">
                {row.label}
              </div>
              <p className="mt-1 min-h-12 text-xs leading-relaxed text-[var(--text-muted)]">
                {row.description}
              </p>
              <dl className="mt-3 space-y-1.5 text-[10px]">
                {metrics.length ? (
                  metrics.map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between gap-2">
                      <dt className="truncate text-[var(--text-muted)]">
                        {key.replaceAll("_", " ")}
                      </dt>
                      <dd className="max-w-28 truncate font-mono text-[var(--text-primary)]">
                        {formatValue(value)}
                      </dd>
                    </div>
                  ))
                ) : (
                  <div className="text-[var(--text-muted)]">
                    {payload ? "not reported by this server" : "loading…"}
                  </div>
                )}
              </dl>
            </div>
          );
        })}
      </div>
      <div className="mt-3 text-[10px] text-[var(--text-muted)]">
        {installed.length}/{ROWS.length} phase-two contracts reported. “Enabled” is not
        treated as “active” until a backend, sample, export, hook, or receipt proves work
        occurred.
      </div>
    </Card>
  );
}
