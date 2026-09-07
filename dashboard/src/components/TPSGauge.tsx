import { useEffect, useRef } from "react";
import { motion, useMotionValue, useSpring, useTransform } from "motion/react";
import { Loader2 } from "lucide-react";
import { Card } from "./Card";
import { fmtTokS } from "../lib/utils";
import { useDashboardStore, usePrimaryActivePrefill } from "../state/store";

const BAND_THRESHOLDS = [20, 40, 60]; // bands at 20 / 40 / 60 tok/s
const SCALE_MAX = 80;

function bandColor(value: number): string {
  if (value >= 60) return "var(--accent)";
  if (value >= 40) return "var(--accent-cool)";
  if (value >= 20) return "var(--accent-warm)";
  return "var(--accent-hot)";
}

export function TPSGauge() {
  const liveTokS = useDashboardStore((s) => s.liveTokS);
  const lastTokS = useDashboardStore((s) => s.latest?.decode_tok_s ?? null);
  const rolling = useDashboardStore((s) => s.rolling);
  const prefill = usePrimaryActivePrefill();

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Live while a request is decoding; otherwise the last finished request's
  // speed is shown held, in a muted arc, and labelled as such. The needle
  // never pretends an idle server is generating.
  const isLive = typeof liveTokS === "number";
  const heldTokS = isLive ? liveTokS : lastTokS;
  const targetValue = Math.max(0, heldTokS ?? 0);
  const motionValue = useMotionValue(targetValue);
  const springValue = useSpring(motionValue, { stiffness: 140, damping: 22, mass: 0.6 });
  const displayValue = useTransform(springValue, (v) => v.toFixed(1));
  const isLiveRef = useRef(isLive);
  isLiveRef.current = isLive;

  useEffect(() => {
    motionValue.set(targetValue);
  }, [targetValue, motionValue]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const size = 220;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    function draw(value: number) {
      if (!ctx) return;
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, size, size);
      const cx = size / 2;
      const cy = size / 2 + 10;
      const radius = 84;
      const startAngle = Math.PI * 0.75; // 135°
      const endAngle = Math.PI * 2.25; // 405° => 270° sweep
      const sweep = endAngle - startAngle;

      // Background arc
      ctx.beginPath();
      ctx.arc(cx, cy, radius, startAngle, endAngle);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 14;
      ctx.lineCap = "round";
      ctx.stroke();

      // Band ticks
      BAND_THRESHOLDS.forEach((threshold) => {
        const ratio = Math.min(1, threshold / SCALE_MAX);
        const angle = startAngle + sweep * ratio;
        ctx.beginPath();
        const inner = radius - 18;
        const outer = radius + 8;
        ctx.moveTo(cx + Math.cos(angle) * inner, cy + Math.sin(angle) * inner);
        ctx.lineTo(cx + Math.cos(angle) * outer, cy + Math.sin(angle) * outer);
        ctx.strokeStyle = "rgba(255,255,255,0.18)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = "rgba(200,210,220,0.45)";
        ctx.font = "10px ui-sans-serif, system-ui";
        ctx.textAlign = "center";
        ctx.fillText(
          String(threshold),
          cx + Math.cos(angle) * (radius - 30),
          cy + Math.sin(angle) * (radius - 30) + 3,
        );
      });

      // Value arc: full colour and glow only while decoding; a held value
      // draws dimmed with no glow so an idle page reads as idle.
      const live = isLiveRef.current;
      const ratio = Math.min(1, value / SCALE_MAX);
      const valueAngle = startAngle + sweep * ratio;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, startAngle, valueAngle);
      ctx.strokeStyle = bandColor(value);
      ctx.globalAlpha = live ? 1 : 0.35;
      ctx.shadowColor = bandColor(value);
      ctx.shadowBlur = live ? 16 : 0;
      ctx.lineWidth = 14;
      ctx.lineCap = "round";
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;

      // Center reading
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.font = "10px ui-sans-serif, system-ui";
      ctx.textAlign = "center";
      ctx.fillText("tok/s", cx, cy + 38);

      ctx.restore();
    }

    function loop() {
      draw(springValue.get());
      raf = requestAnimationFrame(loop);
    }
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [springValue]);

  // Window min/max are null when nothing ran in the window; show a dash
  // rather than borrowing the all-time figure under a different label.
  const max = rolling?.max ?? null;
  const min = rolling?.min ?? null;
  const peak = rolling?.sticky_all_time_max ?? null;

  return (
    <Card
      title="Live decode TPS"
      subtitle={
        prefill.active
          ? `prefilling ${prefill.pct.toFixed(0)}% — decode not started`
          : isLive
            ? `current ${fmtTokS(liveTokS)} tok/s`
            : heldTokS !== null
              ? `idle · last request ${fmtTokS(heldTokS)} tok/s`
              : "waiting for generation"
      }
    >
      <div className="relative grid place-items-center min-h-[220px]">
        <canvas ref={canvasRef} aria-hidden="true" />
        <div className="absolute inset-0 grid place-items-center pointer-events-none">
          <div className="text-center -mt-2">
            {prefill.active ? (
              <>
                <span className="inline-flex items-center gap-2 text-[20px] font-semibold tracking-wide text-[var(--accent-warm)] leading-none">
                  <Loader2 className="size-5 animate-spin" />
                  PREFILLING
                </span>
                <span className="text-xs text-[var(--text-muted)] mt-2 block tabular-nums">
                  {prefill.pct.toFixed(1)}% · decode hasn't started yet
                </span>
              </>
            ) : (
              <>
                <motion.span
                  className={`block text-[44px] font-semibold tabular-nums leading-none ${
                    isLive ? "text-[var(--text-primary)]" : "text-[var(--text-muted)]"
                  }`}
                >
                  {displayValue}
                </motion.span>
                <span className="text-xs text-[var(--text-muted)] mt-1 block">
                  {isLive ? "live" : heldTokS !== null ? "last request" : "idle"}
                </span>
              </>
            )}
          </div>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 mt-3 text-xs">
        <Stat label="window min" value={fmtTokS(min)} />
        <Stat label="window max" value={fmtTokS(max)} tone="warm" />
        <Stat label="all-time" value={fmtTokS(peak)} tone="accent" />
      </div>
    </Card>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "warm" | "accent";
}) {
  const color =
    tone === "warm"
      ? "text-[var(--accent-warm)]"
      : tone === "accent"
        ? "text-[var(--accent)]"
        : "text-[var(--text-primary)]";
  return (
    <div className="bg-[var(--bg-elevated)] rounded-md px-2 py-1.5 border border-[var(--border-soft)]">
      <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        {label}
      </div>
      <div className={`text-sm font-semibold tabular-nums ${color}`}>{value}</div>
    </div>
  );
}
