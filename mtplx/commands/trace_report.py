"""MTPLX trace report: one-file HTML diagnosis for an agent session.

Renders the joined trace (serve receipts + flight samples + OpenCode history)
as one self-contained HTML file — inline SVG, zero external resources, opens
from file://. Sections: summary cards, pathology flags, wall-clock timeline
(TTFT vs decode), cache waterfall, per-request TPS (flight samples when
present, else the receipt sliding-window sketch marked approximate) with
draft acceptance-by-depth, the context-vs-decode-speed scatter across every
receipt on the port, and a per-turn digest. Charts follow the dataviz method:
validated light palette (blue #2a78d6 / orange #eb6834 — adjacent CVD dE
24.7, both >=3:1 on the surface), thin rounded marks, hairline solid grid,
legends on multi-series charts, tooltips that enhance but never gate (the
digest table carries every per-turn number).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from .trace import (
    METRICS_DIR,
    _detect_pathologies,
    _detect_port,
    _fmt_dur,
    _fmt_tok,
    _join_session,
    _load_flight,
    _load_receipts,
    _opencode_connect,
    _opencode_parts,
    _resolve_session_arg,
    _turn_row,
)

# dataviz reference palette, light mode (validated with validate_palette.js)
_MUT, _AXIS, _SURF = "#898781", "#c3c2b7", "#fcfcfb"
_BLUE, _ORANGE, _CRIT = "#2a78d6", "#eb6834", "#d03b3b"
_W = 1112  # shared chart width
_SLIDING = [("first 32", "first_32"), ("first 64", "first_64"),
            ("last 64", "last_64"), ("last 32", "last_32")]


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _hms(ts: float | None) -> str:
    if not ts:
        return "-"
    return _dt.datetime.fromtimestamp(ts, tz=_dt.UTC).astimezone().strftime("%H:%M:%S")


def _num_ticks(hi: float, target: int = 5) -> list[float]:
    """Round ticks for a 0..hi axis (1/2/2.5/5 steps); last tick covers hi."""
    if hi <= 0:
        return [0.0, 1.0]
    raw = hi / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    return [i * step for i in range(math.ceil(hi / step - 1e-9) + 1)]


def _time_ticks(t0: float, t1: float) -> list[float]:
    span = max(t1 - t0, 1.0)
    step = next((s for s in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200)
                 if span / s <= 7), 14400)
    first = math.ceil(t0 / step) * step
    return [first + i * step for i in range(8) if first + i * step <= t1]


def _rbar(x: float, y: float, w: float, h: float, fill: str, extra: str = "") -> str:
    """Horizontal bar: 4px-rounded data end (right), square at the baseline."""
    w, r = max(w, 0.5), min(4.0, max(w, 0.5) / 2, h / 2)
    if r < 1.5:
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"{extra}/>'
    return (f'<path d="M{x:.1f},{y:.1f} h{w - r:.1f} q{r:.1f},0 {r:.1f},{r:.1f} v{h - 2 * r:.1f}'
            f' q0,{r:.1f} -{r:.1f},{r:.1f} h-{w - r:.1f} z" fill="{fill}"{extra}/>')


def _vbar(x: float, ytop: float, w: float, h: float, fill: str, tip: str) -> str:
    """Vertical bar: 4px-rounded cap, square at the baseline; carries a tooltip."""
    h, r = max(h, 0.5), min(4.0, max(h, 0.5) / 2, w / 2)
    attrs = f' data-tip="{_esc(tip)}" tabindex="0"'
    if r < 1.5:
        return f'<rect x="{x:.1f}" y="{ytop:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"{attrs}/>'
    return (f'<path d="M{x:.1f},{ytop + h:.1f} v-{h - r:.1f} q0,-{r:.1f} {r:.1f},-{r:.1f} h{w - 2 * r:.1f}'
            f' q{r:.1f},0 {r:.1f},{r:.1f} v{h - r:.1f} z" fill="{fill}"{attrs}/>')


def _chips(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><span class="sw" style="background:{c}"></span>{_esc(t)}</span>' for c, t in items) + "</div>"


def _card(title: str, body: str) -> str:
    return f"<section><h2>{_esc(title)}</h2>{body}</section>"


_QUIET = '<p class="quiet">{}</p>'


# ---------------------------------------------------------------------------
# per-turn record (receipt extras layered over trace._turn_row)


def _enrich(turn: dict) -> dict:
    row = _turn_row(turn)
    receipt = turn.get("receipt") or {}
    row["decode_elapsed_s"] = receipt.get("decode_elapsed_s")
    row["sliding"] = [(lbl, receipt.get(f"sliding_decode_tok_s_{key}")) for lbl, key in _SLIDING]
    row["accepted_by_depth"] = receipt.get("accepted_by_depth") or []
    row["drafted_by_depth"] = receipt.get("drafted_by_depth") or []
    row["samples"] = [e for e in turn.get("flight", []) if e.get("ev") == "s"]
    return row


def _row_dur(r: dict) -> float:
    return ((r["ttft_s"] or 0.0) + (r["decode_elapsed_s"] or 0.0)) or r["wall_s"] or 1.0


def _share(r: dict) -> str:
    think = r["client_reasoning_tokens"] or 0
    denom = r["completion_tokens"] or (think + (r["client_output_tokens"] or 0))
    return f"{think / denom * 100:.0f}% think" if denom else ""


# ---------------------------------------------------------------------------
# sections


def _sec_timeline(rows: list[dict]) -> str:
    rows = [r for r in rows if r["start"]]
    if not rows:
        return _card("Session timeline", _QUIET.format("no dated turns"))
    t0 = min(r["start"] for r in rows)
    t1 = max(r["start"] + _row_dur(r) for r in rows)
    x0, x1, rh = 44.0, 856.0, 26.0
    h = len(rows) * rh + 42

    def sx(ts: float) -> float:
        return x0 + (ts - t0) / max(t1 - t0, 1e-9) * (x1 - x0)

    out = [f'<svg viewBox="0 0 {_W} {h}" role="img" aria-label="wall-clock timeline">']
    for tick in _time_ticks(t0, t1):
        out.append(f'<line x1="{sx(tick):.1f}" y1="4" x2="{sx(tick):.1f}" y2="{h - 34}" class="grid"/>'
                   f'<text x="{sx(tick):.1f}" y="{h - 18}" class="ax mid">{_hms(tick)}</text>')
    out.append(f'<line x1="{x0}" y1="{h - 34}" x2="{x1}" y2="{h - 34}" class="axis"/>')
    for i, r in enumerate(rows):
        y, dur = i * rh + 8, _row_dur(r)
        bx, bw = sx(r["start"]), max(sx(r["start"] + _row_dur(r)) - sx(r["start"]), 2.5)
        cancelled = r["status"] != "ok"
        out.append(f'<text x="{x0 - 8}" y="{y + 11}" class="lab end">t{r["turn"]}</text>')
        if cancelled:
            out.append(_rbar(bx, y, bw, 14, _CRIT))
        else:
            ttft_w = min(bw * (r["ttft_s"] or 0) / dur, bw) if r["ttft_s"] else 0.0
            if ttft_w >= 1:  # 2px surface gap between the segments when it fits
                gap = 2.0 if ttft_w > 6 and bw - ttft_w > 6 else 0.0
                out.append(f'<rect x="{bx:.1f}" y="{y}" width="{max(ttft_w - gap, 0.5):.1f}" height="14" fill="{_ORANGE}"/>')
            out.append(_rbar(bx + ttft_w, y, bw - ttft_w, 14, _BLUE))
        comp = r["completion_tokens"] or ((r["client_reasoning_tokens"] or 0) + (r["client_output_tokens"] or 0))
        ann = " · ".join(p for p in (f"{_fmt_tok(comp)} tok", _share(r), "cancelled" if cancelled else "") if p)
        out.append(f'<text x="{bx + bw + 8:.1f}" y="{y + 11}" class="ann">{_esc(ann)}</text>')
        ttft = f"{r['ttft_s']:.2f}s" if r["ttft_s"] is not None else "-"
        tip = (f"t{r['turn']} · {_hms(r['start'])} → {_hms(r['start'] + dur)}\n"
               f"wall {_fmt_dur(r['wall_s'])} · ttft {ttft} · decode {_fmt_dur(r['decode_elapsed_s'])}\n"
               f"completion {_fmt_tok(comp)} tok · {_share(r) or 'think -'}"
               + (f"\ndecode {r['decode_tok_s']:.1f} tok/s" if r["decode_tok_s"] else "")
               + ("\nCANCELLED / ERROR" if cancelled else "")
               + ("\nno matched server receipt" if r["receipt_missing"] else ""))
        out.append(f'<rect x="0" y="{y - 3}" width="{_W}" height="{rh}" fill="transparent" data-tip="{_esc(tip)}" tabindex="0"/>')
    legend = _chips([(_ORANGE, "TTFT (prefill / queue)"), (_BLUE, "decode"), (_CRIT, "cancelled / error")])
    return _card("Session timeline", legend + "".join(out) + "</svg>")


def _sec_cache(rows: list[dict]) -> str:
    rows = [r for r in rows if (r["prompt_tokens"] or 0) > 0]
    if not rows:
        return _card("Cache waterfall", _QUIET.format("no receipts with prompt tokens"))
    ticks = _num_ticks(max(float(r["prompt_tokens"]) for r in rows))
    xmax = ticks[-1] or 1.0
    x0, x1, rh = 44.0, 812.0, 24.0
    h = len(rows) * rh + 42

    def sx(v: float) -> float:
        return x0 + v / xmax * (x1 - x0)

    out = [f'<svg viewBox="0 0 {_W} {h}" role="img" aria-label="cache waterfall">']
    for tv in ticks:
        out.append(f'<line x1="{sx(tv):.1f}" y1="4" x2="{sx(tv):.1f}" y2="{h - 34}" class="grid"/>'
                   f'<text x="{sx(tv):.1f}" y="{h - 18}" class="ax mid">{_fmt_tok(tv)}</text>')
    out.append(f'<line x1="{x0}" y1="{h - 34}" x2="{sx(xmax):.1f}" y2="{h - 34}" class="axis"/>')
    for i, r in enumerate(rows):
        y = i * rh + 8
        cached, newpre = r["cached_tokens"], r["new_prefill_tokens"]
        out.append(f'<text x="{x0 - 8}" y="{y + 10}" class="lab end">t{r["turn"]}</text>')
        wall = r["turn"] > 1 and (newpre or 0) > 1_000
        if cached is None and newpre is None:
            out.append(_rbar(x0, y, sx(float(r["prompt_tokens"])) - x0, 12, _AXIS))
            note = "cache split unknown"
        else:
            cw = sx(float(cached or 0)) - x0
            if cw >= 1:
                gap = 2.0 if cw > 6 and (newpre or 0) > 0 else 0.0
                out.append(f'<rect x="{x0}" y="{y}" width="{max(cw - gap, 0.5):.1f}" height="12" fill="{_BLUE}"/>')
            if newpre:
                out.append(_rbar(x0 + cw, y, sx(float(newpre)) - x0, 12, _ORANGE))
            note = " · ".join(p for p in (
                f"+{_fmt_tok(newpre)} new" if wall else "",
                str(r["cache_source"] or ""), str(r["cache_miss_reason"] or "")) if p)
        cls = "wallnote" if wall else "ann"
        out.append(f'<text x="{sx(float(r["prompt_tokens"])) + 8:.1f}" y="{y + 10}" class="{cls}">{_esc(note)}</text>')
        tip = (f"t{r['turn']} · prompt {_fmt_tok(r['prompt_tokens'])} tok\n"
               + ("cache split unknown (receipt fields absent)" if cached is None and newpre is None
                  else f"cached {_fmt_tok(cached)} · new prefill {_fmt_tok(newpre)}")
               + (f"\nsource {r['cache_source']}" if r["cache_source"] else "")
               + (f"\nmiss reason {r['cache_miss_reason']}" if r["cache_miss_reason"] else ""))
        out.append(f'<rect x="0" y="{y - 3}" width="{_W}" height="{rh}" fill="transparent" data-tip="{_esc(tip)}" tabindex="0"/>')
    legend = _chips([(_BLUE, "cached (reused prefix)"), (_ORANGE, "new prefill"), (_AXIS, "split unknown")])
    return _card("Cache waterfall (prompt = cached + new prefill)", legend + "".join(out) + "</svg>")


def _tps_cell(r: dict) -> str | None:
    sliding = [(lbl, float(v)) for lbl, v in r["sliding"] if v is not None]
    samples, drafted = r["samples"], r["drafted_by_depth"]
    if not samples and not sliding and not drafted:
        return None
    w, px0, py0, py1 = 252, 30, 10, 86
    px1 = 166 if drafted else 240
    approx = not samples
    if samples:
        ts0 = float(samples[0].get("ts") or 0)
        pts = [(float(s.get("ts") or 0) - ts0, float(s.get("tps") or 0)) for s in samples]
        span = max(pts[-1][0], 1e-9)
    else:
        pts = [(float(i), v) for i, (_, v) in enumerate(sliding)]
        span = max(len(pts) - 1.0, 1.0)
    ticks = _num_ticks(max([v for _, v in pts] or [1.0]), 2)
    ymax = ticks[-1] or 1.0

    def spt(p: float, v: float) -> tuple[float, float]:
        return px0 + p / span * (px1 - px0), py1 - v / ymax * (py1 - py0)

    out = [f'<svg viewBox="0 0 {w} 116">']
    for tv in ticks:
        ty = spt(0, tv)[1]
        out.append(f'<line x1="{px0}" y1="{ty:.1f}" x2="{px1}" y2="{ty:.1f}" class="grid"/>'
                   f'<text x="{px0 - 4}" y="{ty + 3.5:.1f}" class="ax end">{tv:g}</text>')
    if pts:
        path = " ".join(f"{'M' if i == 0 else 'L'}{spt(p, v)[0]:.1f},{spt(p, v)[1]:.1f}" for i, (p, v) in enumerate(pts))
        dash = ' stroke-dasharray="5 4"' if approx else ""
        out.append(f'<path d="{path}" fill="none" stroke="{_BLUE}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"{dash}/>')
    if approx:
        for i, (lbl, v) in enumerate(sliding):
            cx, cy = spt(float(i), v)
            tip = f"{lbl}: {v:.1f} tok/s (receipt sliding window — approximation)"
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{_BLUE}" stroke="{_SURF}" stroke-width="2" data-tip="{_esc(tip)}" tabindex="0"/>'
                       f'<text x="{cx:.1f}" y="{py1 + 13}" class="ax mid">{_esc(lbl.replace("first ", "f").replace("last ", "l"))}</text>')
    elif pts:
        vals = [v for _, v in pts]
        tip = (f"flight samples n={len(vals)}\nmin {min(vals):.1f} · "
               f"mean {sum(vals) / len(vals):.1f} · max {max(vals):.1f} tok/s")
        out.append(f'<rect x="{px0}" y="{py0}" width="{px1 - px0}" height="{py1 - py0}" fill="transparent" data-tip="{_esc(tip)}" tabindex="0"/>'
                   f'<text x="{px0}" y="{py1 + 13}" class="ax">0s</text><text x="{px1}" y="{py1 + 13}" class="ax end">{span:.0f}s</text>')
    if drafted:
        bx0 = px1 + 18.0
        bw = max(6.0, min(14.0, (w - 12 - bx0) / len(drafted) - 6))
        for i, d in enumerate(drafted):
            acc = r["accepted_by_depth"][i] if i < len(r["accepted_by_depth"]) else 0
            rate = (acc / d) if d else 0.0
            x, bh = bx0 + i * (bw + 6), rate * (py1 - py0)
            out.append(_vbar(x, py1 - bh, bw, bh, _ORANGE, f"depth {i + 1}: {acc}/{d} drafts accepted ({rate * 100:.0f}%)"))
            out.append(f'<text x="{x + bw / 2:.1f}" y="{py1 - bh - 3:.1f}" class="ax mid">{rate * 100:.0f}</text>'
                       f'<text x="{x + bw / 2:.1f}" y="{py1 + 13}" class="ax mid">d{i + 1}</text>')
    out.append(f'<line x1="{px0}" y1="{py1}" x2="{w - 8}" y2="{py1}" class="axis"/></svg>')
    tok_s = f"{r['decode_tok_s']:.1f} tok/s" if r["decode_tok_s"] else "-"
    cancel = ' · <span class="crit">cancelled</span>' if r["status"] != "ok" else ""
    badge = f'<span class="badge">{"approx" if approx else "flight"}</span>'
    return (f'<div class="cell"><div class="ct"><span>t{r["turn"]} · {tok_s} · '
            f'{_fmt_tok(r["completion_tokens"])} tok{cancel}</span>{badge}</div>{"".join(out)}</div>')


def _sec_tps(rows: list[dict]) -> str:
    cells = [c for c in (_tps_cell(r) for r in rows) if c]
    if not cells:
        return _card("Per-request TPS", _QUIET.format("no flight samples or receipt sliding windows"))
    note = _QUIET.format(
        "solid line — flight recorder per-second samples · dashed line with markers — 4-point sketch "
        "from receipt sliding windows (approximation; no flight data recorded for these requests) · "
        "orange columns — draft tokens accepted per MTP depth")
    return _card("Per-request TPS", note + f'<div class="cells">{"".join(cells)}</div>')


def _sec_scatter(receipts: list[dict], session_ids: set[int], port: int) -> str:
    pts: list[tuple[float, float, bool, dict]] = []
    for rec in receipts:
        y = rec.get("decode_tok_s")
        x = rec.get("context_len")
        if x is None:
            x = ((rec.get("prompt_tokens") or 0) + (rec.get("completion_tokens") or 0)) or None
        if x and y:
            pts.append((float(x), float(y), id(rec) in session_ids, rec))
    if not pts:
        return _card("Context vs decode speed", _QUIET.format("no receipts with decode speed"))
    pts.sort(key=lambda p: p[2])  # history first, session points painted on top
    h, x0, x1, y0, y1 = 336, 56.0, 1092.0, 14.0, 284.0
    xticks, yticks = _num_ticks(max(p[0] for p in pts)), _num_ticks(max(p[1] for p in pts), 4)
    xmax, ymax = xticks[-1] or 1.0, yticks[-1] or 1.0
    out = [f'<svg id="scat" viewBox="0 0 {_W} {h}" role="img" aria-label="context vs decode speed">']
    for tv in yticks:
        ty = y1 - tv / ymax * (y1 - y0)
        out.append(f'<line x1="{x0}" y1="{ty:.1f}" x2="{x1}" y2="{ty:.1f}" class="grid"/>'
                   f'<text x="{x0 - 6}" y="{ty + 3.5:.1f}" class="ax end">{tv:g}</text>')
    for tv in xticks:
        tx = x0 + tv / xmax * (x1 - x0)
        out.append(f'<line x1="{tx:.1f}" y1="{y0}" x2="{tx:.1f}" y2="{y1}" class="grid"/>'
                   f'<text x="{tx:.1f}" y="{y1 + 16:.1f}" class="ax mid">{_fmt_tok(tv)}</text>')
    out.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>'
               f'<text x="{x0 + 4}" y="{y0 + 4}" class="ax">decode tok/s</text>'
               f'<text x="{(x0 + x1) / 2:.0f}" y="{h - 6}" class="ax mid">context length (tokens) — every receipt on port {port}</text>')
    for x, y, mine, rec in pts:
        cx, cy = x0 + x / xmax * (x1 - x0), y1 - y / ymax * (y1 - y0)
        tip = (f"{'this session · ' if mine else ''}{_hms(rec.get('logged_at_s'))}"
               f" · ctx {_fmt_tok(int(x))} tok\n{y:.1f} tok/s · completion {_fmt_tok(rec.get('completion_tokens'))} tok")
        style = (f'r="5" fill="{_BLUE}" stroke="{_SURF}" stroke-width="2"' if mine
                 else f'r="3" fill="{_MUT}" fill-opacity="0.5"')
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" {style} data-tip="{_esc(tip)}"/>')
    legend = _chips([(_BLUE, "this session"), (_MUT, f"history on port {port}")])
    return _card("Context vs decode speed (the decode cliff)", legend + "".join(out) + "</svg>")


def _sec_digest(digest: list[dict]) -> str:
    if not digest:
        return _card("Turn digest", _QUIET.format("no assistant turns"))
    head = ('<tr><th>t</th><th>start</th><th class="n">wall</th><th>status</th><th class="n">comp tok</th>'
            '<th class="n">think tok</th><th class="n">ttft s</th><th class="n">tok/s</th>'
            '<th class="n">reasoning chars</th><th class="n">output chars</th></tr>')
    body = []
    for d in digest:
        if d["prompt"]:
            body.append(f'<tr class="up"><td colspan="10">&#8220;{_esc(d["prompt"])}&#8221;</td></tr>')
        r = d["row"]
        status = "ok" if r["status"] == "ok" else '<span class="crit">cancel/err</span>'
        ttft = f"{r['ttft_s']:.2f}" if r["ttft_s"] is not None else "-"
        toks = f"{r['decode_tok_s']:.1f}" if r["decode_tok_s"] else "-"
        body.append(f'<tr><td>t{r["turn"]}</td><td>{_hms(r["start"])}</td><td class="n">{_fmt_dur(r["wall_s"])}</td>'
                    f'<td>{status}</td><td class="n">{_fmt_tok(r["completion_tokens"])}</td>'
                    f'<td class="n">{_fmt_tok(r["client_reasoning_tokens"])}</td><td class="n">{ttft}</td>'
                    f'<td class="n">{toks}</td><td class="n">{_fmt_tok(d["reasoning_chars"])}</td>'
                    f'<td class="n">{_fmt_tok(d["output_chars"])}</td></tr>')
    return _card("Turn digest", f'<div class="scroll"><table class="dg">{head}{"".join(body)}</table></div>')


# ---------------------------------------------------------------------------
# message-text helpers (defensive: schema may vary across OpenCode versions)


def _user_snippet(conn: Any, message: dict) -> str | None:
    texts: list[str] = []
    try:
        for part in _opencode_parts(conn, message.get("_id") or ""):
            if part.get("type") == "text" and part.get("text"):
                texts.append(str(part["text"]))
        if not texts:
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(str(p.get("text") or "") for p in content if isinstance(p, dict))
    except Exception:  # noqa: BLE001 — digest text is best-effort, never fatal
        return None
    text = " ".join(" ".join(texts).split())
    return (text[:200] + ("…" if len(text) > 200 else "")) or None


def _part_chars(conn: Any, message: dict) -> tuple[int | None, int | None]:
    try:
        parts = _opencode_parts(conn, message.get("_id") or "")
        return (sum(len(p.get("text") or "") for p in parts if p.get("type") == "reasoning"),
                sum(len(p.get("text") or "") for p in parts if p.get("type") == "text"))
    except Exception:  # noqa: BLE001
        return None, None


# ---------------------------------------------------------------------------
# page chrome (kept dense — every rule is chart chrome, not content)

_CSS = (
    "body{margin:0;background:#f9f9f7;color:#0b0b0b;font:14px/1.45 system-ui,-apple-system,'Segoe UI',sans-serif}"
    "main{max-width:1160px;margin:0 auto;padding:24px 20px 60px}h1{font-size:21px;margin:0 0 4px}"
    "h2{font-size:15px;font-weight:600;margin:0 0 8px}.meta b{font-weight:600}"
    ".meta{color:#52514e;font-size:12.5px;margin:0 0 14px;line-height:1.6}"
    "section{background:#fcfcfb;border:1px solid rgba(11,11,11,.1);border-radius:10px;padding:14px 16px;margin:14px 0}"
    ".tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0 2px}"
    ".tile{background:#fcfcfb;border:1px solid rgba(11,11,11,.1);border-radius:10px;padding:10px 12px}"
    ".tl{font-size:11.5px;color:#52514e}.tv{font-size:22px;font-weight:600;margin-top:2px}"
    ".legend{display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 10px;font-size:12px;color:#52514e}"
    ".sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}"
    ".flags{list-style:none;margin:0;padding:0}.bang{color:#d03b3b;font-weight:700;margin-right:6px}"
    ".flags li{border-left:3px solid #ec835a;background:rgba(236,131,90,.07);padding:7px 10px;margin:6px 0;border-radius:0 7px 7px 0;font-size:13px}"
    ".quiet{color:#898781;font-size:12.5px;margin:0 0 10px}.crit{color:#d03b3b;font-weight:600}"
    ".badge{font-size:10px;color:#898781;border:1px solid #e1e0d9;border-radius:4px;padding:1px 5px;height:fit-content}"
    ".cells{display:grid;grid-template-columns:repeat(auto-fill,minmax(252px,1fr));gap:12px}"
    ".cell{border:1px solid #efeee9;border-radius:8px;padding:8px 8px 4px}.scroll{overflow-x:auto}"
    ".ct{font-size:12px;color:#52514e;margin:0 0 4px;display:flex;justify-content:space-between;gap:6px}"
    ".dg{border-collapse:collapse;width:100%;font-size:12.5px}.dg td{border-bottom:1px solid #efeee9;padding:5px 8px;vertical-align:top}"
    ".dg th{text-align:left;color:#52514e;font-weight:600;border-bottom:1px solid #e1e0d9;padding:5px 8px;white-space:nowrap}"
    ".dg .n{text-align:right;font-variant-numeric:tabular-nums}.up td{color:#52514e;background:#f6f5f1;font-style:italic}"
    "svg{display:block;max-width:100%;height:auto}svg text{font:11px system-ui,-apple-system,sans-serif;fill:#52514e}"
    "text.ax{fill:#898781;font-variant-numeric:tabular-nums}text.mid{text-anchor:middle}text.end{text-anchor:end}text.lab{fill:#52514e}"
    "text.ann{font-size:10.5px;fill:#898781;paint-order:stroke;stroke:#fcfcfb;stroke-width:3px}"
    "text.wallnote{font-size:10.5px;fill:#0b0b0b;font-weight:600;paint-order:stroke;stroke:#fcfcfb;stroke-width:3px}"
    "line.grid{stroke:#e1e0d9;stroke-width:1}line.axis{stroke:#c3c2b7;stroke-width:1}"
    "#tip{position:fixed;display:none;background:#0b0b0b;color:#fcfcfb;font-size:12px;line-height:1.5;"
    "padding:7px 9px;border-radius:7px;white-space:pre-line;pointer-events:none;z-index:9;max-width:360px}"
)

# tooltip layer: delegated hover/focus on [data-tip]; nearest-point search inside #scat
_JS = (
    "const tip=document.getElementById('tip');const scat=document.getElementById('scat');"
    "const spts=scat?[...scat.querySelectorAll('circle[data-tip]')]:[];"
    "function show(t,x,y){tip.textContent=t;tip.style.display='block';const r=tip.getBoundingClientRect();"
    "tip.style.left=Math.min(x+14,innerWidth-r.width-8)+'px';tip.style.top=Math.min(y+14,innerHeight-r.height-8)+'px';}"
    "function hide(){tip.style.display='none';}"
    "function near(e){if(!scat)return false;const b=scat.getBoundingClientRect();"
    "if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)return false;"
    "let best=null,bd=26*26;for(const c of spts){const r=c.getBoundingClientRect();"
    "const dx=e.clientX-(r.left+r.width/2),dy=e.clientY-(r.top+r.height/2),d=dx*dx+dy*dy;if(d<bd){bd=d;best=c;}}"
    "if(best){show(best.getAttribute('data-tip'),e.clientX,e.clientY);return true;}return false;}"
    "document.addEventListener('mousemove',e=>{const el=e.target.closest?e.target.closest('[data-tip]'):null;"
    "if(el&&!(scat&&scat.contains(el))){show(el.getAttribute('data-tip'),e.clientX,e.clientY);}else if(!near(e)){hide();}});"
    "document.addEventListener('focusin',e=>{const el=e.target.closest?e.target.closest('[data-tip]'):null;"
    "if(el){const r=el.getBoundingClientRect();show(el.getAttribute('data-tip'),r.left,r.bottom+6);}});"
    "document.addEventListener('focusout',hide);"
)


# ---------------------------------------------------------------------------
# entry point


def cmd_trace_report(args: argparse.Namespace) -> int:
    conn = _opencode_connect(Path(args.db))
    if conn is None:
        print(f"opencode db not found: {args.db}", file=sys.stderr)
        return 1
    session_id = _resolve_session_arg(conn, getattr(args, "session", None))
    if not session_id:
        print("no opencode sessions found", file=sys.stderr)
        return 1
    port = _detect_port(args.port)
    if port is None:
        print("no request logs found under ~/.mtplx/logs", file=sys.stderr)
        return 1
    receipts = _load_receipts(port)
    flight = _load_flight(port)
    joined = _join_session(conn, session_id, receipts, flight)
    a_turns = [t for t in joined["turns"] if t["kind"] == "assistant"]
    rows = [_enrich(t) for t in a_turns]
    flags = _detect_pathologies(joined["turns"])
    session_ids = {id(t["receipt"]) for t in a_turns if t.get("receipt")}

    warm = [r for r in rows[1:] if r["prompt_tokens"] and r["cached_tokens"] is not None]
    reuse = (sum(r["cached_tokens"] or 0 for r in warm)
             / max(1, sum(r["prompt_tokens"] or 0 for r in warm))) if warm else None
    dec = [(r["completion_tokens"], r["decode_elapsed_s"]) for r in rows
           if r["completion_tokens"] and r["decode_elapsed_s"]]
    mean_dec = sum(c for c, _ in dec) / sum(s for _, s in dec) if dec else None
    starts = [r["start"] for r in rows if r["start"]]
    span = "-"
    if starts:
        lo = min(starts)
        hi = max(r["start"] + _row_dur(r) for r in rows if r["start"])
        day = _dt.datetime.fromtimestamp(lo, tz=_dt.UTC).astimezone()
        span = f"{day:%Y-%m-%d} {_hms(lo)} → {_hms(hi)}"
    cards = [
        ("Turns", str(len(rows))),
        ("Warm cache reuse", f"{reuse * 100:.1f}%" if reuse is not None else "-"),
        ("Completion tokens", _fmt_tok(sum(r["completion_tokens"] or 0 for r in rows))),
        ("Client think tokens", _fmt_tok(sum(r["client_reasoning_tokens"] or 0 for r in rows))),
        ("Wall time (turns)", _fmt_dur(sum(r["wall_s"] or 0 for r in rows))),
        ("Mean decode", f"{mean_dec:.1f} tok/s" if mean_dec else "-"),
    ]
    join_mode = "exact session_id" if joined["receipt_pool_scoped"] else "time+token fallback"
    session = joined["session"]
    header = (f'<h1>mtplx trace report</h1><p class="meta"><b>{_esc(session_id)}</b>'
              f' · {_esc(session.get("title") or "untitled")}<br>{_esc(session.get("directory") or "-")}'
              f' · {_esc(span)} · port {port} · join: {join_mode}</p><div class="tiles">'
              + "".join(f'<div class="tile"><div class="tl">{_esc(k)}</div><div class="tv">{_esc(v)}</div></div>'
                        for k, v in cards) + "</div>")
    pathology = _card("Pathology flags", '<ul class="flags">' + "".join(
        f'<li><span class="bang">!!</span>{_esc(f)}</li>' for f in flags) + "</ul>"
        if flags else _QUIET.format("none detected"))

    digest, pending, by_turn = [], None, {r["turn"]: r for r in rows}
    for turn in joined["turns"]:
        if turn["kind"] == "user":
            pending = _user_snippet(conn, turn["message"]) or pending
            continue
        reasoning_chars, output_chars = _part_chars(conn, turn["message"])
        digest.append({"row": by_turn[turn["turn"]], "prompt": pending,
                       "reasoning_chars": reasoning_chars, "output_chars": output_chars})
        pending = None

    page = ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>mtplx trace — {_esc(session_id)}</title><style>" + _CSS + "</style></head><body><main>"
            + header + pathology + _sec_timeline(rows) + _sec_cache(rows) + _sec_tps(rows)
            + _sec_scatter(receipts, session_ids, port) + _sec_digest(digest)
            + '</main><div id="tip"></div><script>' + _JS + "</script></body></html>")

    out_path = Path(args.out).expanduser() if args.out else METRICS_DIR / "reports" / f"{session_id}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    if getattr(args, "open", False):
        subprocess.run(["open", str(out_path)], check=False)
    return 0
