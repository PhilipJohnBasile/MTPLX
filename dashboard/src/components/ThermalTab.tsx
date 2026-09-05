import { Card } from "./Card";
import { FanRingPanel } from "./FanRing";
import { useDashboardStore } from "../state/store";
import { relativeTime } from "../lib/utils";

export function ThermalTab() {
  const thermal = useDashboardStore((s) => s.thermal);
  const whenS = useDashboardStore((s) => s.thermalWhenS);
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 lg:col-span-7">
        <FanRingPanel />
      </div>
      <div className="col-span-12 lg:col-span-5">
        <Card title="Thermal snapshot" subtitle={whenS ? relativeTime(whenS) : "no poll yet"}>
          {thermal ? (
            <dl className="text-sm space-y-1">
              <Field label="ok" value={String(thermal.ok)} />
              <Field label="min RPM" value={String(thermal.min_rpm ?? "—")} />
              <Field label="max RPM" value={String(thermal.max_rpm ?? "—")} />
              <Field label="fans" value={String(thermal.fans?.length ?? 0)} />
            </dl>
          ) : (
            <p className="text-sm text-[var(--text-muted)]">
              Thermal polling is off by default. Pass{" "}
              <code>--enable-thermal-poll</code> when starting MTPLX.
            </p>
          )}
        </Card>
      </div>
      <div className="col-span-12">
        <Card
          title="GPU MHz · coming in v2"
          subtitle="ThermalForge does not expose GPU clock; powermetrics integration lands later"
        >
          <p className="text-sm text-[var(--text-muted)] leading-relaxed">
            ThermalForge's <code>status</code> JSON shape (verified May 2026) covers fan
            RPMs and modes but not GPU MHz or thermal pressure. The dashboard plan
            documents GPU MHz as a v2 add via <code>powermetrics</code>; until then
            this slot is intentionally empty so we don't render a fake number.
          </p>
        </Card>
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-[var(--text-muted)]">{label}</dt>
      <dd className="text-[var(--text-primary)] tabular-nums">{value}</dd>
    </div>
  );
}
