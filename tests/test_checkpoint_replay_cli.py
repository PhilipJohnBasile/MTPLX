from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from mtplx import checkpoint_replay, cli


_EMPTY_TRACE = {
    "schema_version": 2,
    "enabled": True,
    "max_events": 4096,
    "events_collected": 0,
    "events_dropped": 0,
    "pending_probe_count": 0,
    "sequence_high_watermark": 0,
    "bank_epoch": 0,
    "events": [],
}


def test_checkpoint_replay_parser_defaults_are_conservative() -> None:
    args = cli.build_parser().parse_args(["checkpoint-replay", "--trace", "trace.json"])

    assert args.block_size == 256
    assert args.decay == 1.0
    assert args.min_events == 100
    assert args.min_bucket_events == 20
    assert args.bootstrap_iterations == 10_000
    assert args.bootstrap_block_length is None
    assert args.seed == 0


def test_checkpoint_replay_command_requires_a_deployable_cpu_falsifier(
    monkeypatch, tmp_path, capsys
) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(_EMPTY_TRACE), encoding="utf-8")
    output_path = tmp_path / "report.json"
    args = SimpleNamespace(
        trace=str(trace_path),
        output=str(output_path),
        block_size=256,
        decay=1.0,
        min_events=100,
        min_bucket_events=20,
        bootstrap_iterations=10_000,
        bootstrap_block_length=None,
        seed=0,
    )

    monkeypatch.setattr(
        checkpoint_replay,
        "replay_checkpoint_trace",
        lambda *_args, **_kwargs: {
            "status": "amber",
            "runtime_go": False,
        },
    )
    assert cli._cmd_checkpoint_replay(args) == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "amber"
    assert json.loads(capsys.readouterr().out)["runtime_go"] is False

    monkeypatch.setattr(
        checkpoint_replay,
        "replay_checkpoint_trace",
        lambda *_args, **_kwargs: {
            "status": "counterfactual_only",
            "canonical_gate": True,
            "runtime_go": False,
            "deployable_go_available": False,
            "incumbent_provenance": {"gate_eligible": True},
        },
    )
    assert cli._cmd_checkpoint_replay(args) == 2

    monkeypatch.setattr(
        checkpoint_replay,
        "replay_checkpoint_trace",
        lambda *_args, **_kwargs: {
            "status": "passes_cpu_falsifier",
            "canonical_gate": True,
            "runtime_go": False,
            "deployable_go_available": True,
            "incumbent_provenance": {"gate_eligible": True},
        },
    )
    assert cli._cmd_checkpoint_replay(args) == 0

    monkeypatch.setattr(
        checkpoint_replay,
        "replay_checkpoint_trace",
        lambda *_args, **_kwargs: {
            "status": "passes_cpu_falsifier",
            "canonical_gate": False,
            "runtime_go": False,
            "deployable_go_available": True,
            "incumbent_provenance": {"gate_eligible": True},
        },
    )
    assert cli._cmd_checkpoint_replay(args) == 2


def test_checkpoint_replay_refuses_output_that_resolves_to_trace(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    original = json.dumps(_EMPTY_TRACE)
    trace_path.write_text(original, encoding="utf-8")
    args = SimpleNamespace(
        trace=str(trace_path),
        output=str(tmp_path / "." / "trace.json"),
        block_size=256,
        decay=1.0,
        min_events=100,
        min_bucket_events=20,
        bootstrap_iterations=10_000,
        bootstrap_block_length=None,
        seed=0,
    )

    with pytest.raises(ValueError, match="must not refer"):
        cli._cmd_checkpoint_replay(args)

    assert trace_path.read_text(encoding="utf-8") == original


def test_checkpoint_replay_refuses_output_hardlinked_to_trace(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    output_path = tmp_path / "report.json"
    original = json.dumps(_EMPTY_TRACE)
    trace_path.write_text(original, encoding="utf-8")
    os.link(trace_path, output_path)
    args = SimpleNamespace(
        trace=str(trace_path),
        output=str(output_path),
        block_size=256,
        decay=1.0,
        min_events=100,
        min_bucket_events=20,
        bootstrap_iterations=10_000,
        bootstrap_block_length=None,
        seed=0,
    )

    with pytest.raises(ValueError, match="must not refer"):
        cli._cmd_checkpoint_replay(args)

    assert trace_path.read_text(encoding="utf-8") == original
    assert output_path.read_text(encoding="utf-8") == original
