"""Head-cal leg 2a: MTPLX_DRAFT_CONFIDENCE_TRACE (default OFF).

The greedy trace's mean_accept_probability is tautologically the realized
accept rate (binary accepts), so head calibration needs the draft head's own
p(drafted token) attributed to accept/reject with its OWN denominators.
TinyMTP draft logits are one-hot-ish [0,1,0,0] over vocab 4, so p(argmax)
is exactly e/(3+e) — an analytically pinned expectation.
"""

from __future__ import annotations

import json
import math

import pytest

from tests.test_graphbank_compiled_verify import _run_tiny_mtpk

_EXPECTED_CONF = math.e / (3.0 + math.e)


def _clear(monkeypatch):
    for knob in (
        "MTPLX_DRAFT_CONFIDENCE_TRACE",
        "MTPLX_DRAFT_CONFIDENCE_WIDTH_THRESHOLD",
        "MTPLX_GREEDY_DRAFT_CHAIN",
        "MTPLX_BATCHED_GREEDY_ACCEPT",
        "MTPLX_BATCH_PAGED_OFFSETS",
    ):
        monkeypatch.delenv(knob, raising=False)


def _fingerprint(out):
    return {
        "tokens": list(out.tokens),
        "drafted_by_depth": list(out.stats.drafted_by_depth or []),
        "accepted_by_depth": list(out.stats.accepted_by_depth or []),
        "verify_calls": out.stats.verify_calls,
    }


def _traced_run(monkeypatch, tmp_path, *, mtp_token, flag, name):
    trace_path = tmp_path / f"{name}.jsonl"
    monkeypatch.setenv("MTPLX_DECODE_TRACE_JSONL", str(trace_path))
    monkeypatch.setenv("MTPLX_DECODE_TRACE_INTERVAL_S", "0.01")
    if flag:
        monkeypatch.setenv("MTPLX_DRAFT_CONFIDENCE_TRACE", "1")
    else:
        monkeypatch.delenv("MTPLX_DRAFT_CONFIDENCE_TRACE", raising=False)
    out, _model = _run_tiny_mtpk(max_tokens=6, mtp_token=mtp_token)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert rows, "decode trace emitted no rows"
    return out, rows


def _accumulate(rows, kind):
    """Sum count deltas and confidence mass (mean*count) across trace rows."""
    counts = None
    mass = None
    for row in rows:
        row_counts = row.get(f"draft_confidence_{kind}count_by_depth_delta")
        row_means = row.get(f"draft_confidence_{kind}mean_by_depth_delta")
        if row_counts is None:
            continue
        if counts is None:
            counts = [0] * len(row_counts)
            mass = [0.0] * len(row_counts)
        for i, (c, m) in enumerate(zip(row_counts, row_means)):
            counts[i] += int(c)
            if c and m is not None:
                mass[i] += float(m) * int(c)
    assert counts is not None, f"trace rows carry no {kind or 'total '}confidence keys"
    means = [(mass[i] / counts[i] if counts[i] else None) for i in range(len(counts))]
    return counts, means


def test_flag_off_emits_zero_counts_and_identical_tokens(monkeypatch, tmp_path):
    _clear(monkeypatch)
    out_off, rows_off = _traced_run(
        monkeypatch, tmp_path, mtp_token=1, flag=False, name="off"
    )
    counts_off, _ = _accumulate(rows_off, "")
    assert all(c == 0 for c in counts_off)

    out_on, _rows_on = _traced_run(
        monkeypatch, tmp_path, mtp_token=1, flag=True, name="on"
    )
    assert _fingerprint(out_on) == _fingerprint(out_off)


def test_stock_loop_attributes_accepts_with_pinned_confidence(monkeypatch, tmp_path):
    _clear(monkeypatch)
    _out, rows = _traced_run(
        monkeypatch, tmp_path, mtp_token=1, flag=True, name="accepts"
    )
    counts, means = _accumulate(rows, "")
    accepted_counts, accepted_means = _accumulate(rows, "accepted_")
    rejected_counts, _ = _accumulate(rows, "rejected_")
    assert counts[0] > 0
    assert accepted_counts == counts
    assert all(c == 0 for c in rejected_counts)
    for depth, (count, mean) in enumerate(zip(accepted_counts, accepted_means)):
        if count:
            assert mean == pytest.approx(_EXPECTED_CONF, abs=1e-4), (
                f"depth {depth}: mean {mean} != analytic {_EXPECTED_CONF}"
            )


def test_all_reject_attributes_only_evaluated_depths(monkeypatch, tmp_path):
    _clear(monkeypatch)
    _out, rows = _traced_run(
        monkeypatch, tmp_path, mtp_token=2, flag=True, name="rejects"
    )
    accepted_counts, _ = _accumulate(rows, "accepted_")
    rejected_counts, rejected_means = _accumulate(rows, "rejected_")
    assert rejected_counts[0] > 0
    assert rejected_means[0] == pytest.approx(_EXPECTED_CONF, abs=1e-4)
    # Own-denominator contract: depths drafted but never evaluated after the
    # depth-1 rejection are NOT counted (unlike drafted_by_depth).
    assert all(c == 0 for c in rejected_counts[1:])
    assert all(c == 0 for c in accepted_counts)


def test_confidence_rides_the_greedy_chain_eval(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("MTPLX_GREEDY_DRAFT_CHAIN", "1")
    _out, rows = _traced_run(
        monkeypatch, tmp_path, mtp_token=1, flag=True, name="chain"
    )
    counts, means = _accumulate(rows, "")
    assert counts[0] > 0
    for count, mean in zip(counts, means):
        if count:
            assert mean == pytest.approx(_EXPECTED_CONF, abs=1e-4)


def _width_fingerprint(out):
    return {
        "tokens": list(out.tokens),
        "drafted_by_depth": list(out.stats.drafted_by_depth or []),
        "verify_calls": out.stats.verify_calls,
    }


def test_width_gate_fires_above_analytic_confidence(monkeypatch, tmp_path):
    """Threshold above e/(3+e): every depth-1 draft gates the cycle — deeper
    depths are never drafted, width_stops counts fire, and the committed
    output tokens are invariant (verify corrects, width only costs speed)."""
    _clear(monkeypatch)
    baseline, _rows = _traced_run(
        monkeypatch, tmp_path, mtp_token=1, flag=False, name="width-base"
    )
    monkeypatch.setenv("MTPLX_DRAFT_CONFIDENCE_WIDTH_THRESHOLD", "0.6")
    gated, rows = _traced_run(
        monkeypatch, tmp_path, mtp_token=1, flag=False, name="width-gated"
    )
    assert list(gated.tokens) == list(baseline.tokens)
    drafted = list(gated.stats.drafted_by_depth or [])
    assert drafted[0] > 0
    assert all(d == 0 for d in drafted[1:])
    stops = sum(
        int(row.get("draft_confidence_width_stops_delta") or 0) for row in rows
    )
    assert stops == drafted[0]


def test_width_gate_inert_below_analytic_confidence(monkeypatch, tmp_path):
    _clear(monkeypatch)
    baseline, _rows = _traced_run(
        monkeypatch, tmp_path, mtp_token=1, flag=False, name="inert-base"
    )
    monkeypatch.setenv("MTPLX_DRAFT_CONFIDENCE_WIDTH_THRESHOLD", "0.3")
    gated, rows = _traced_run(
        monkeypatch, tmp_path, mtp_token=1, flag=False, name="inert-gated"
    )
    assert _width_fingerprint(gated) == _width_fingerprint(baseline)
    stops = sum(
        int(row.get("draft_confidence_width_stops_delta") or 0) for row in rows
    )
    assert stops == 0


@pytest.mark.parametrize("bad", ["", "abc", "0", "1", "1.5", "-0.2"])
def test_width_gate_invalid_values_stay_off(monkeypatch, tmp_path, bad):
    _clear(monkeypatch)
    baseline, _rows = _traced_run(
        monkeypatch, tmp_path, mtp_token=1, flag=False, name=f"bad-base-{bad or 'empty'}"
    )
    monkeypatch.setenv("MTPLX_DRAFT_CONFIDENCE_WIDTH_THRESHOLD", bad)
    gated, _rows2 = _traced_run(
        monkeypatch, tmp_path, mtp_token=1, flag=False, name=f"bad-gated-{bad or 'empty'}"
    )
    assert _width_fingerprint(gated) == _width_fingerprint(baseline)


def _hist_accumulate(rows, kind):
    hist = None
    for row in rows:
        flat = row.get(f"draft_confidence_{kind}_hist_flat_delta")
        if not flat:
            continue
        if hist is None:
            hist = [0] * len(flat)
        for i, v in enumerate(flat):
            hist[i] += int(v)
    assert hist is not None, f"no {kind} histogram keys in trace rows"
    return hist


def test_histograms_land_in_the_analytic_bucket(monkeypatch, tmp_path):
    """All tiny-lane confidence is exactly e/(3+e) ~ 0.4754 -> every
    attributed draft lands in bucket 4 of its depth's histogram."""
    _clear(monkeypatch)
    _out, rows = _traced_run(
        monkeypatch, tmp_path, mtp_token=1, flag=True, name="hist-accept"
    )
    accepted_hist = _hist_accumulate(rows, "accepted")
    rejected_hist = _hist_accumulate(rows, "rejected")
    accepted_counts, _ = _accumulate(rows, "accepted_")
    assert sum(rejected_hist) == 0
    depths = len(accepted_hist) // 10
    expected_bucket = int(_EXPECTED_CONF * 10)
    for depth in range(depths):
        row = accepted_hist[depth * 10 : depth * 10 + 10]
        assert sum(row) == accepted_counts[depth]
        for bucket, value in enumerate(row):
            if bucket != expected_bucket:
                assert value == 0, f"depth {depth} bucket {bucket} leaked {value}"

    _out2, rows2 = _traced_run(
        monkeypatch, tmp_path, mtp_token=2, flag=True, name="hist-reject"
    )
    rejected_hist2 = _hist_accumulate(rows2, "rejected")
    assert rejected_hist2[expected_bucket] > 0
    assert sum(rejected_hist2) == rejected_hist2[expected_bucket]
