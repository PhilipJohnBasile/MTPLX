from __future__ import annotations

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
