"""Host-only boundary tests for the million-token sparse-prefill contract."""

from __future__ import annotations

import mlx.core as mx

import mtplx.kernels.qsa_prefill_flash as flash


def _inputs(capacity: int):
    rows = 2
    queries = mx.zeros((1, 24, rows, 256), dtype=mx.bfloat16)
    # Broadcast views carry the production logical shape without allocating a
    # multi-gigabyte physical K/V backing. This test calls only the host
    # validator and never dispatches the Metal kernel.
    one_token = mx.zeros((1, 2, 1, 256), dtype=mx.bfloat16)
    keys = mx.broadcast_to(one_token, (1, 2, capacity, 256))
    values = mx.broadcast_to(one_token, (1, 2, capacity, 256))
    block_ids = mx.zeros((rows, 512), dtype=mx.int32)
    block_valid = mx.ones((rows, 512), dtype=mx.bool_)
    return queries, keys, values, block_ids, block_valid


def test_flash_host_contract_accepts_exact_million_token_ceiling(monkeypatch):
    monkeypatch.setattr(flash, "_on_metal_device", lambda: True)
    monkeypatch.setattr(flash, "qsa_indexer_select_nax_available", lambda: True)
    inputs = _inputs(1_048_576)
    assert (
        flash._unsupported_reason(
            *inputs,
            pos_start=1_048_574,
            total_tokens=1_048_576,
            scale=0.0625,
        )
        is None
    )


def test_flash_host_contract_rejects_one_token_past_ceiling(monkeypatch):
    monkeypatch.setattr(flash, "_on_metal_device", lambda: True)
    monkeypatch.setattr(flash, "qsa_indexer_select_nax_available", lambda: True)
    inputs = _inputs(1_048_577)
    assert flash._unsupported_reason(
        *inputs,
        pos_start=1_048_575,
        total_tokens=1_048_577,
        scale=0.0625,
    ) == "the logical token count exceeds the production context limit"

