"""Route Tape contracts: off-is-off, per-round completeness, fallback
visibility, dropped-event accounting, and Flight Recorder integration.

The E5 lesson applies doubly here: the tape is the diagnostic seam for the
flat-decode campaign — if it silently never fires, or silently perturbs, every
conclusion drawn from it is fabricated. These tests pin the seam.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from mtplx.route_tape import (  # noqa: E402
    BANK_FALLBACK_REASONS,
    COMMIT_ROUTES,
    GDN_CAPTURE_ROUTES,
    VERIFY_ROUTES,
    RouteTape,
    boot_id,
    counter_deltas,
    mono_ns,
    set_route_tape_sink,
    wall_ns,
)
from mtplx.server.flight_recorder import FlightRecorder  # noqa: E402


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestOffMeansOff:
    def test_disabled_without_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MTPLX_ROUTE_TAPE", raising=False)
        monkeypatch.setenv("MTPLX_ROUTE_TAPE_JSONL", str(tmp_path / "rt.jsonl"))
        tape = RouteTape("rid-1")
        assert not tape.enabled
        tape.emit("route", "round", 1, {"verify_route": "compiled_bank"})
        assert not (tmp_path / "rt.jsonl").exists()
        assert tape.dropped == 0

    def test_disabled_without_sink_or_path(self, monkeypatch):
        monkeypatch.setenv("MTPLX_ROUTE_TAPE", "1")
        monkeypatch.delenv("MTPLX_ROUTE_TAPE_JSONL", raising=False)
        set_route_tape_sink(None)
        tape = RouteTape("rid-1")
        assert not tape.enabled


class TestRecordShape:
    def test_round_record_complete(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MTPLX_ROUTE_TAPE", "1")
        path = tmp_path / "rt.jsonl"
        monkeypatch.setenv("MTPLX_ROUTE_TAPE_JSONL", str(path))
        tape = RouteTape("rid-7")
        assert tape.enabled
        tape.emit("route", "header", None, {"verify_strategy": "capture_commit"})
        tape.emit(
            "route",
            "round",
            3,
            {
                "verify_route": "compiled_bank",
                "commit_route": "capture_commit",
                "verify_width": 4,
                "accepted_depths": 2,
                "fallback_deltas": {"bank_fallback": {"post_restore_warmup": 1}},
            },
        )
        header, round_rec = _read_jsonl(path)
        for rec in (header, round_rec):
            assert rec["ev"] == "rt"
            assert rec["schema_version"] == 1
            assert rec["rid"] == "rid-7"
            assert rec["mono_ns"] > 0 and rec["wall_ns"] > 0
            assert rec["boot_id"] == boot_id()
        assert header["name"] == "header" and header["round"] is None
        assert round_rec["name"] == "round" and round_rec["round"] == 3
        attrs = round_rec["attrs"]
        assert attrs["verify_route"] in VERIFY_ROUTES
        assert attrs["commit_route"] in COMMIT_ROUTES
        assert attrs["fallback_deltas"]["bank_fallback"]["post_restore_warmup"] == 1

    def test_seq_monotonic(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MTPLX_ROUTE_TAPE", "1")
        path = tmp_path / "rt.jsonl"
        monkeypatch.setenv("MTPLX_ROUTE_TAPE_JSONL", str(path))
        tape = RouteTape("rid-8")
        for i in range(5):
            tape.emit("route", "round", i, {})
        seqs = [r["seq"] for r in _read_jsonl(path)]
        assert seqs == [1, 2, 3, 4, 5]


class TestDroppedAccounting:
    def test_sink_raise_counts_drop(self, monkeypatch):
        monkeypatch.setenv("MTPLX_ROUTE_TAPE", "1")

        def bad_sink(_rec):
            raise RuntimeError("writer exploded")

        set_route_tape_sink(bad_sink)
        try:
            tape = RouteTape("rid-9")
            assert tape.enabled
            tape.emit("route", "round", 1, {})
            tape.emit("route", "round", 2, {})
            assert tape.dropped == 2
        finally:
            set_route_tape_sink(None)


class TestFlightRecorderIntegration:
    def test_emit_route_lands_in_flight_jsonl(self, tmp_path):
        path = str(tmp_path / "flight-9999.jsonl")
        rec = FlightRecorder(path)
        assert rec.enabled
        rec.emit_route({"ev": "rt", "kind": "route", "name": "round", "round": 1,
                        "attrs": {"verify_route": "eager_plain"}})
        # writer thread is async; give it the one drain cycle
        import time

        deadline = time.time() + 5.0
        rows = []
        while time.time() < deadline:
            if os.path.exists(path):
                rows = [r for r in _read_jsonl(path) if r.get("ev") == "rt"]
                if rows:
                    break
            time.sleep(0.05)
        assert rows and rows[0]["attrs"]["verify_route"] == "eager_plain"

    def test_inert_recorder_swallows_route(self):
        rec = FlightRecorder(None)
        assert not rec.enabled
        rec.emit_route({"ev": "rt"})  # must not raise


class TestCounterDeltas:
    def test_delta_only_changes(self):
        assert counter_deltas({"a": 1, "b": 5}, {"a": 1, "b": 7}) == {"b": 2}
        assert counter_deltas({}, {"x": 4}) == {"x": 4}
        assert counter_deltas({"x": 4}, {"x": 4}) == {}

    def test_fallback_vocabularies_grounded(self):
        # Spot-pin the enums against the source vocabulary so a rename upstream
        # fails loudly here instead of silently drifting the tape schema.
        assert "compiled_bank" in VERIFY_ROUTES and "eager_plain" in VERIFY_ROUTES
        assert "linear_gdn_from_conv_tape" in GDN_CAPTURE_ROUTES
        assert "rollback_reforward" in COMMIT_ROUTES
        assert "post_restore_warmup" in BANK_FALLBACK_REASONS
        assert "capacity_overflow" in BANK_FALLBACK_REASONS


class TestClockDiscipline:
    def test_mono_ns_monotonic_and_timebase(self):
        a = mono_ns()
        b = mono_ns()
        assert b >= a
        assert wall_ns() > 0
        from mtplx.route_tape import TIMEBASE_DENOM, TIMEBASE_NUMER

        assert TIMEBASE_NUMER > 0 and TIMEBASE_DENOM > 0
