from __future__ import annotations

from pathlib import Path
import shlex

import pytest


def test_numerics_names_are_closed_and_normalized():
    from mtplx.mtp_batch_numerics import (
        MTPBatchNumerics,
        normalize_mtp_batch_numerics,
    )

    assert normalize_mtp_batch_numerics(None) is MTPBatchNumerics.THROUGHPUT
    assert (
        normalize_mtp_batch_numerics("balanced")
        is MTPBatchNumerics.BALANCED
    )
    assert (
        normalize_mtp_batch_numerics("b1-exact")
        is MTPBatchNumerics.B1_EXACT
    )
    with pytest.raises(ValueError, match="throughput, balanced, b1-exact"):
        normalize_mtp_batch_numerics("auto")


def test_public_serve_parser_exposes_mtp_batch_numerics():
    from mtplx.cli import build_parser

    args = build_parser().parse_args(
        ["serve", "--mtp-batch-numerics", "balanced"]
    )

    assert args.mtp_batch_numerics == "balanced"


def test_qwen35b_concurrency_guide_launch_command_is_public_and_complete():
    from mtplx.cli import build_parser

    guide = (
        Path(__file__).parents[1] / "docs/concurrency/qwen35b-mtp-batch.md"
    ).read_text(encoding="utf-8")
    launch_section = guide.split("Use the full construction contract.", 1)[1]
    command = launch_section.split("```bash\n", 1)[1].split("\n```", 1)[0]
    argv = shlex.split(command.replace("\\\n", " "))

    assert argv[:2] == ["mtplx", "serve"]
    args = build_parser().parse_args(argv[1:])
    assert args.model == "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"
    assert args.download is True
    assert args.scheduler_mode == "mtp_batch"
    assert args.batching_preset == "throughput"
    assert args.generation_mode == "mtp"
    assert args.load_mtp is True
    assert args.depth == 1
    assert args.max_active_requests == 8
    assert args.decode_batch_max == 8
    assert args.context_window == 131072
    assert args.profile == "turbo"
    assert args.verify_strategy == "target_prefix"
    assert args.verify_core == "stock"
    assert args.mtp_batch_numerics == "throughput"
