import type { ReactNode } from "react";
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  CircleOff,
  Gauge,
  Network,
  ShieldCheck,
} from "lucide-react";
import { useRuntimeSystems } from "../hooks/usePolling";
import type {
  DeterministicReplaySystem,
  ExpertLocalityLayer,
  ExpertLocalitySystem,
  MemoryGovernorSystem,
  SemanticMemorySystem,
} from "../lib/types";
import { fmtBytes, fmtNumber, relativeTime } from "../lib/utils";
import { Card } from "./Card";
import { AdaptiveSystemsPanel } from "./AdaptiveSystemsPanel";

export function SystemsTab() {
  const systems = useRuntimeSystems();

  if (systems.isLoading) {
    return (
      <Card title="Runtime systems" subtitle="loading live system contracts...">
        <div className="text-sm text-[var(--text-muted)]">
          Waiting for <code>/v1/mtplx/systems</code>.
        </div>
      </Card>
    );
  }

  if (systems.isError || !systems.data) {
    return (
      <Card title="Runtime systems" subtitle="the systems endpoint is unavailable">
        <div className="flex items-start gap-2 text-sm text-[var(--accent-hot)]">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <span>{String((systems.error as Error | undefined)?.message ?? "No payload")}</span>
        </div>
      </Card>
    );
  }

  const payload = systems.data;
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12">
        <Card bodyClassName="pt-5">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="text-base font-semibold text-[var(--text-primary)]">
                Integrated runtime systems
              </div>
              <p className="mt-1 max-w-3xl text-sm text-[var(--text-muted)]">
                Live state for exact semantic checkpointing, sparse-expert locality,
                safe-point memory governance, and deterministic replay. Disabled,
                offline, and unsampled states are shown explicitly rather than inferred.
              </p>
            </div>
            <div className="text-xs text-[var(--text-muted)]">
              polled {relativeTime(payload.ts)}
            </div>
          </div>
          {payload.error ? (
            <div className="mt-4 rounded-lg border border-[var(--accent-hot)]/30 bg-[var(--accent-hot)]/5 px-3 py-2 text-xs text-[var(--accent-hot)]">
              {payload.error}
            </div>
          ) : null}
        </Card>
      </div>

      <div className="col-span-12 xl:col-span-6">
        <SemanticMemoryCard system={payload.semantic_memory} />
      </div>
      <div className="col-span-12 xl:col-span-6">
        <MemoryGovernorCard system={payload.memory_governor} />
      </div>
      <div className="col-span-12">
        <DeterministicReplayCard system={payload.deterministic_replay} />
      </div>
      <div className="col-span-12">
        <ExpertLocalityCard system={payload.expert_locality} />
      </div>
    </div>
  );
}

function SemanticMemoryCard({ system }: { system: SemanticMemorySystem }) {
  const latest = system.latest;
  const edges = latest?.semantic_anchor_edges ?? [];
  const anchors = latest?.anchors ?? [];

  return (
    <Card
      title={
        <span className="inline-flex items-center gap-2">
          <BrainCircuit className="size-4 text-[var(--accent)]" />
          Semantic memory
        </span>
      }
      subtitle="Exact complete-message prefixes become mandatory recurrent-cache edges"
      action={<SystemStatus enabled={system.enabled} wired={system.wired} />}
    >
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric label="planned requests" value={fmtNumber(system.planned_requests)} />
        <Metric label="skipped requests" value={fmtNumber(system.skipped_requests)} />
        <Metric label="latest anchors" value={fmtNumber(latest?.semantic_anchor_count)} />
        <Metric label="rejected" value={fmtNumber(latest?.semantic_anchor_rejected)} />
      </div>

      {!system.enabled ? (
        <InactiveNotice>
          Restart with <code>MTPLX_SEMANTIC_ANCHORS=1</code>. Planning remains
          fail-closed: only byte-exact token prefixes are admitted.
        </InactiveNotice>
      ) : !latest ? (
        <EmptyNotice>No request has reached the semantic planner yet.</EmptyNotice>
      ) : latest.status !== "planned" ? (
        <InactiveNotice>
          Latest request: <code>{latest.status}</code>
          {latest.reason ? ` · ${latest.reason}` : ""}
        </InactiveNotice>
      ) : (
        <>
          <div className="mt-4 rounded-xl border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="text-xs font-semibold uppercase tracking-widest text-[var(--text-muted)]">
                prefill edges
              </div>
              <div className="text-xs text-[var(--text-muted)]">
                {fmtBytes(latest.semantic_anchor_estimated_bytes)} estimated · {" "}
                {fmtNumber(latest.candidate_count)} candidates
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {edges.length ? (
                edges.map((edge) => (
                  <code
                    key={edge}
                    className="rounded-md border border-[var(--border-soft)] bg-[var(--bg-card)] px-2 py-1 text-xs text-[var(--accent)]"
                  >
                    {fmtNumber(edge)}
                  </code>
                ))
              ) : (
                <span className="text-xs text-[var(--text-muted)]">
                  No interior edge survived exact-prefix and budget checks.
                </span>
              )}
            </div>
          </div>

          {anchors.length ? (
            <div className="mt-4 overflow-x-auto rounded-xl border border-[var(--border-soft)]">
              <table className="w-full min-w-[620px] text-left text-xs">
                <thead className="bg-[var(--bg-elevated)] text-[var(--text-muted)]">
                  <tr>
                    <HeaderCell>kind</HeaderCell>
                    <HeaderCell>message</HeaderCell>
                    <HeaderCell>token edge</HeaderCell>
                    <HeaderCell>priority</HeaderCell>
                    <HeaderCell>prefix hash</HeaderCell>
                  </tr>
                </thead>
                <tbody>
                  {anchors.slice(0, 12).map((anchor) => (
                    <tr
                      key={`${anchor.message_index}:${anchor.token_offset}:${anchor.kind}`}
                      className="border-t border-[var(--border-soft)]"
                    >
                      <BodyCell>{anchor.kind}</BodyCell>
                      <BodyCell>{fmtNumber(anchor.message_index)}</BodyCell>
                      <BodyCell>{fmtNumber(anchor.token_offset)}</BodyCell>
                      <BodyCell>{fmtNumber(anchor.priority)}</BodyCell>
                      <BodyCell>
                        <code>{anchor.token_prefix_hash?.slice(0, 12) ?? "—"}</code>
                      </BodyCell>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </>
      )}
    </Card>
  );
}

function DeterministicReplayCard({
  system,
}: {
  system: DeterministicReplaySystem;
}) {
  const capture = system.request_capture;
  const latest = capture.latest;

  return (
    <Card
      title={
        <span className="inline-flex items-center gap-2">
          <ShieldCheck className="size-4 text-[var(--accent-cool)]" />
          Deterministic replay and promotion gates
        </span>
      }
      subtitle="Privacy-first request capture plus explicit offline replay; never an automatic serving mutation"
      action={<SystemStatus enabled={capture.enabled} wired={capture.wired} />}
    >
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Metric label="captured" value={fmtNumber(capture.dispatched)} />
        <Metric label="completed" value={fmtNumber(capture.completed)} />
        <Metric label="capture failures" value={fmtNumber(capture.failures)} />
        <Metric
          label="last capture"
          value={latest?.updated_at_s ? relativeTime(latest.updated_at_s) : "—"}
        />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-3">
        <SystemContract
          label="Request capture"
          available={capture.available}
          wired={capture.wired}
          detail={
            capture.enabled
              ? "Active. Content remains off unless separately opted in."
              : "Off. Set MTPLX_REQUEST_CAPTURE_DIR to retain redacted envelopes."
          }
        />
        <SystemContract
          label="Counterfactual replay"
          available={system.replay.available}
          wired={system.replay.wired}
          detail="Offline callable core with stable ordering, credential-isolated dedupe, and explicit regression policy."
        />
        <SystemContract
          label="Trace parity"
          available={system.trace_parity.available}
          wired={system.trace_parity.wired}
          detail="Offline ordered-boundary comparison with explicit numerical tolerances."
        />
      </div>

      {!capture.enabled ? (
        <InactiveNotice>
          Request capture is disabled by default. Replay remains an explicit offline
          operation, and <code>promotion_is_automatic=false</code> is enforced by the
          capability contract.
        </InactiveNotice>
      ) : latest ? (
        <div className="mt-4 rounded-xl border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-2 text-xs text-[var(--text-muted)]">
          Latest: <code>{latest.phase}</code> · <code>{latest.request_id ?? "unknown"}</code>
          {latest.session_id ? <> · session <code>{latest.session_id}</code></> : null}
          {latest.persisted ? " · persisted" : " · not persisted"}
        </div>
      ) : (
        <EmptyNotice>Capture is enabled, but no request has reached dispatch yet.</EmptyNotice>
      )}
    </Card>
  );
}

function SystemContract({
  label,
  available,
  wired,
  detail,
}: {
  label: string;
  available: boolean;
  wired: boolean;
  detail: string;
}) {
  return (
    <div className="rounded-xl border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-sm font-semibold text-[var(--text-primary)]">{label}</div>
        <StatusPill
          label={!available ? "unavailable" : wired ? "wired" : "offline"}
          tone={!available ? "warn" : wired ? "ok" : "muted"}
        />
      </div>
      <p className="mt-2 text-xs leading-relaxed text-[var(--text-muted)]">{detail}</p>
    </div>
  );
}

function ExpertLocalityCard({ system }: { system: ExpertLocalitySystem }) {
  const metrics = system.metrics ?? {};
  const layers = metrics.layers ?? [];
  const install = system.install ?? {};

  return (
    <Card
      title={
        <span className="inline-flex items-center gap-2">
          <Network className="size-4 text-[var(--accent-cool)]" />
          Expert locality matrix
        </span>
      }
      subtitle="Sampled router assignments; measurement only, never an expert-placement policy"
      action={<SystemStatus enabled={system.enabled} wired={system.wired} />}
    >
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-7">
        <Metric label="instrumented" value={fmtNumber(install.instrumented_modules)} />
        <Metric label="router calls" value={fmtNumber(metrics.calls)} />
        <Metric label="sampled" value={fmtNumber(metrics.accepted_calls)} />
        <Metric label="dropped" value={fmtNumber(metrics.dropped_calls)} />
        <Metric label="layer lanes" value={fmtNumber(layers.length)} />
        <Metric label="60% LRU" value={capacityLabel(metrics.recommended_capacity_60)} />
        <Metric label="80% LRU" value={capacityLabel(metrics.recommended_capacity_80)} />
      </div>

      {!system.enabled ? (
        <InactiveNotice>
          Restart with <code>MTPLX_EXPERT_LOCALITY=1</code>. The normal path
          returns before router-index arrays are materialized.
        </InactiveNotice>
      ) : !system.wired ? (
        <InactiveNotice>
          Instrumentation did not attach: <code>{install.reason ?? "unknown"}</code>
          {install.error ? ` · ${install.error}` : ""}
        </InactiveNotice>
      ) : layers.length === 0 ? (
        <EmptyNotice>
          Router taps are installed, but no sampled MoE forward has completed.
        </EmptyNotice>
      ) : (
        <div className="mt-4 overflow-x-auto rounded-xl border border-[var(--border-soft)]">
          <table className="w-full min-w-[920px] text-left text-xs">
            <thead className="bg-[var(--bg-elevated)] text-[var(--text-muted)]">
              <tr>
                <HeaderCell>layer</HeaderCell>
                <HeaderCell>lane</HeaderCell>
                <HeaderCell>events</HeaderCell>
                <HeaderCell>assignments</HeaderCell>
                <HeaderCell>unique</HeaderCell>
                <HeaderCell>WS50</HeaderCell>
                <HeaderCell>WS90</HeaderCell>
                <HeaderCell>WS99</HeaderCell>
                <HeaderCell>consecutive Jaccard</HeaderCell>
                <HeaderCell>invalid</HeaderCell>
              </tr>
            </thead>
            <tbody>
              {sortExpertRows(layers).slice(0, 64).map((row) => (
                <ExpertLayerRow key={`${row.layer_id}:${row.lane}`} row={row} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function MemoryGovernorCard({ system }: { system: MemoryGovernorSystem }) {
  const metrics = system.metrics ?? {};
  const decision = metrics.memory_governor_last_decision;
  const apply = metrics.memory_governor_last_apply;
  const utilization = decision?.utilization;

  return (
    <Card
      title={
        <span className="inline-flex items-center gap-2">
          <Gauge className="size-4 text-[var(--accent-warm)]" />
          Safe-point memory governor
        </span>
      }
      subtitle="Rebalances SessionBank budgets only when model and session work are idle"
      action={<SystemStatus enabled={system.enabled} wired={system.wired} />}
    >
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric
          label="bank budget"
          value={fmtBytes(metrics.memory_governor_bank_max_bytes)}
        />
        <Metric
          label="per-session"
          value={fmtBytes(metrics.memory_governor_per_session_max_bytes)}
        />
        <Metric
          label="utilization"
          value={
            utilization === null || utilization === undefined
              ? "—"
              : `${(utilization * 100).toFixed(1)}%`
          }
        />
        <Metric label="pressure" value={decision?.pressure ?? "—"} />
      </div>

      {!system.enabled ? (
        <InactiveNotice>
          Restart with <code>MTPLX_MEMORY_GOVERNOR=1</code>. The existing macOS
          pressure guard remains independent.
        </InactiveNotice>
      ) : !system.wired ? (
        <InactiveNotice>The SessionBank governor could not initialize.</InactiveNotice>
      ) : !decision ? (
        <EmptyNotice>No memory observation has completed yet.</EmptyNotice>
      ) : (
        <div className="mt-4 space-y-3">
          <div className="rounded-xl border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="text-sm font-semibold text-[var(--text-primary)]">
                {decision.action} · {decision.reason}
              </div>
              <StatusPill
                label={decision.safe ? "safe point" : "blocked"}
                tone={decision.safe ? "ok" : "warn"}
              />
            </div>
            <dl className="mt-3 grid grid-cols-2 gap-3 text-xs">
              <Definition label="target bank" value={fmtBytes(decision.target_bank_max_bytes)} />
              <Definition
                label="target per-session"
                value={fmtBytes(decision.target_per_session_max_bytes)}
              />
              <Definition
                label="prefill chunk"
                value={fmtNumber(decision.prefill_chunk_tokens)}
              />
              <Definition
                label="speculative allowed"
                value={
                  decision.speculative_allowed === null ||
                  decision.speculative_allowed === undefined
                    ? "unchanged"
                    : decision.speculative_allowed
                      ? "yes"
                      : "no"
                }
              />
            </dl>
          </div>
          {apply ? (
            <div className="rounded-xl border border-[var(--border-soft)] px-3 py-2 text-xs text-[var(--text-muted)]">
              Last apply: {apply.applied ? "applied" : "held"} · {apply.reason} · {" "}
              {fmtNumber(apply.evicted_entries)} entries / {fmtBytes(apply.evicted_bytes)}
              {" "}evicted
            </div>
          ) : null}
        </div>
      )}
    </Card>
  );
}

function ExpertLayerRow({ row }: { row: ExpertLocalityLayer }) {
  return (
    <tr className="border-t border-[var(--border-soft)]">
      <BodyCell>
        <code>{row.layer_id}</code>
      </BodyCell>
      <BodyCell>{row.lane}</BodyCell>
      <BodyCell>{fmtNumber(row.events)}</BodyCell>
      <BodyCell>{fmtNumber(row.assignments)}</BodyCell>
      <BodyCell>{fmtNumber(row.unique_experts)}</BodyCell>
      <BodyCell>{fmtNumber(row.working_set_50)}</BodyCell>
      <BodyCell>{fmtNumber(row.working_set_90)}</BodyCell>
      <BodyCell>{fmtNumber(row.working_set_99)}</BodyCell>
      <BodyCell>{(row.consecutive_jaccard * 100).toFixed(1)}%</BodyCell>
      <BodyCell>{fmtNumber(row.invalid_assignments)}</BodyCell>
    </tr>
  );
}

function SystemStatus({ enabled, wired }: { enabled: boolean; wired: boolean }) {
  if (!enabled) return <StatusPill label="off" tone="muted" />;
  if (!wired) return <StatusPill label="not attached" tone="warn" />;
  return <StatusPill label="active" tone="ok" />;
}

function StatusPill({
  label,
  tone,
}: {
  label: string;
  tone: "ok" | "warn" | "muted";
}) {
  const Icon = tone === "ok" ? CheckCircle2 : tone === "warn" ? AlertTriangle : CircleOff;
  const className =
    tone === "ok"
      ? "border-[var(--accent-cool)]/30 bg-[var(--accent-cool)]/10 text-[var(--accent-cool)]"
      : tone === "warn"
        ? "border-[var(--accent-warm)]/30 bg-[var(--accent-warm)]/10 text-[var(--accent-warm)]"
        : "border-[var(--border-soft)] bg-[var(--bg-elevated)] text-[var(--text-muted)]";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-wider ${className}`}
    >
      <Icon className="size-3" />
      {label}
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-2">
      <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        {label}
      </div>
      <div className="mt-1 truncate text-sm font-semibold tabular-nums text-[var(--text-primary)]">
        {value}
      </div>
    </div>
  );
}

function Definition({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        {label}
      </dt>
      <dd className="mt-0.5 font-medium text-[var(--text-primary)]">{value}</dd>
    </div>
  );
}

function InactiveNotice({ children }: { children: ReactNode }) {
  return (
    <div className="mt-4 rounded-xl border border-[var(--accent-warm)]/25 bg-[var(--accent-warm)]/5 px-3 py-2 text-xs leading-relaxed text-[var(--text-muted)]">
      {children}
    </div>
  );
}

function EmptyNotice({ children }: { children: ReactNode }) {
  return (
    <div className="mt-4 rounded-xl border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-3 text-xs text-[var(--text-muted)]">
      {children}
      {/* MTPLX_NATIVE_ADAPTIVE_PANEL */}
      <div className="col-span-12">
        <AdaptiveSystemsPanel />
      </div>
    </div>
  );
}

function HeaderCell({ children }: { children: ReactNode }) {
  return <th className="px-3 py-2 font-medium">{children}</th>;
}

function BodyCell({ children }: { children: ReactNode }) {
  return <td className="px-3 py-2 text-[var(--text-primary)]">{children}</td>;
}

function capacityLabel(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${fmtNumber(value)} experts`;
}

function sortExpertRows(rows: ExpertLocalityLayer[]): ExpertLocalityLayer[] {
  return [...rows].sort((left, right) => {
    if (left.lane !== right.lane) return left.lane.localeCompare(right.lane);
    return left.layer_id.localeCompare(right.layer_id, undefined, { numeric: true });
  });
}
