from __future__ import annotations

from pathlib import Path

import pytest

from mtplx.benchmarks.protocol import (
    WEIGHTED_TOK_S_AGGREGATION,
    assert_protocol_match,
    build_effective_run_record,
    protocol_mismatches,
    weighted_tok_s,
)


def _record(tmp_path: Path, **overrides):
    suite = tmp_path / "suite.jsonl"
    suite.write_text('{"id":"one","prompt":"hello"}\n', encoding="utf-8")
    kwargs = {
        "backend": "dflash_mlx_official",
        "model_ref": "target",
        "model_revision": "target-sha",
        "draft_ref": "draft",
        "draft_revision": "draft-sha",
        "prompt_suite": suite,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 256,
        "seed": 0,
        "enable_thinking": False,
        "block_size": 8,
        "generation_mode": "dflash",
        "nax_enabled": False,
        "verify_strategy": "official_dflash",
        "compiled_verify": "not_applicable",
        "runtime_switches": {"EXAMPLE_SWITCH": "off"},
    }
    kwargs.update(overrides)
    return build_effective_run_record(**kwargs)


def test_effective_record_pins_the_executed_protocol(tmp_path: Path) -> None:
    record = _record(tmp_path)

    assert record["prompt_suite_sha256"]
    assert record["enable_thinking"] is False
    assert record["model"]["revision"] == "target-sha"
    assert record["draft"]["revision"] == "draft-sha"
    assert record["verify_strategy"] == "official_dflash"
    assert record["compiled_verify"] == "not_applicable"
    assert record["runtime_switches"] == {"EXAMPLE_SWITCH": "off"}
    assert record["aggregation"] == WEIGHTED_TOK_S_AGGREGATION


def test_effective_record_requires_revisions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_revision"):
        _record(tmp_path, model_revision=None)
    with pytest.raises(ValueError, match="draft_revision"):
        _record(tmp_path, draft_revision=None)


def test_protocol_gate_catches_thinking_mismatch(tmp_path: Path) -> None:
    thinking_off = _record(tmp_path, enable_thinking=False)
    thinking_on = _record(tmp_path, enable_thinking=True)

    assert protocol_mismatches(thinking_off, thinking_on) == {
        "enable_thinking": (False, True)
    }
    with pytest.raises(ValueError, match="enable_thinking"):
        assert_protocol_match(thinking_off, thinking_on)


def test_arm_dimensions_do_not_make_protocol_incomparable(tmp_path: Path) -> None:
    dflash = _record(tmp_path, block_size=8, generation_mode="dflash")
    mtp = _record(
        tmp_path,
        backend="mtplx",
        draft_ref=None,
        draft_revision=None,
        block_size=1,
        generation_mode="mtp",
        nax_enabled=True,
    )

    assert protocol_mismatches(dflash, mtp) == {}


def test_weighted_tok_s_aggregates_tokens_over_decode_seconds() -> None:
    rows = [
        {"generated_tokens": 100, "tok_s": 100.0},
        {"generated_tokens": 100, "tok_s": 50.0},
    ]
    assert weighted_tok_s(rows) == pytest.approx(200.0 / 3.0)


def test_dflash_cli_requires_an_explicit_thinking_mode() -> None:
    from mtplx.cli import build_parser

    parser = build_parser()
    common = [
        "dflash-mlx-baseline",
        "--model-revision",
        "target-sha",
        "--draft-revision",
        "draft-sha",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(common)
    assert parser.parse_args([*common, "--disable-thinking"]).enable_thinking is False
    assert parser.parse_args([*common, "--enable-thinking"]).enable_thinking is True
