"""Numeric gates for the portable gathered flash_prefill tier.

The tier consumes the same compact per-row block selections as the NAX flash
kernel but runs on any Metal device (MTPLX_QSA_PREFILL_GATHER). Its contract:
attention over exactly (selected complete blocks ∪ visible tail) ∩ causal —
byte-for-byte the visible set of the dense-mask reconstruction — with fp32
softmax. These tests pin set semantics, tiling invariance, and the routing
order inside Attention.__call__.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import mlx.core as mx
import pytest

from mtplx.models.qwen4_exp import (
    _qsa_blocks_to_dense_mask,
    _qsa_prefill_gather_attention,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "mtplx" / "models" / "qwen4_exp.py"

_H = 4
_H_KV = 2
_D = 16
_RATIO = 4
_TOPK = 3


def _selection(pos_start: int, rows: int, total: int, seed: int):
    """Craft a valid per-row block selection like the eager selector's."""

    mx.random.seed(seed)
    nb_total = total // _RATIO
    ids = []
    valid = []
    for r in range(rows):
        qpos = pos_start + r
        complete = min((qpos + 1) // _RATIO, nb_total)
        take = min(_TOPK, complete)
        # Deterministic spread over the visible complete blocks; sorted
        # chronological ids like the production contract.
        chosen = sorted({(r * 7 + j * 3) % complete for j in range(take)} or set())
        row_ids = chosen + [0] * (_TOPK - len(chosen))
        row_valid = [True] * len(chosen) + [False] * (_TOPK - len(chosen))
        ids.append(row_ids)
        valid.append(row_valid)
    return (
        mx.array(ids, dtype=mx.int32),
        mx.array(valid, dtype=mx.bool_),
    )


def _qkv(rows: int, total: int, seed: int):
    mx.random.seed(seed)
    q = mx.random.normal((1, _H, rows, _D)).astype(mx.bfloat16)
    k = mx.random.normal((1, _H_KV, total, _D)).astype(mx.bfloat16)
    v = mx.random.normal((1, _H_KV, total, _D)).astype(mx.bfloat16)
    return q, k, v


def _dense_reference(q, k, v, mask, scale):
    """Explicit fp32 masked attention over the dense [1,1,S,T] bool mask."""

    rep = _H // _H_KV
    k_full = mx.repeat(k, rep, axis=1)
    v_full = mx.repeat(v, rep, axis=1)
    scores = (
        mx.matmul(q.astype(mx.float32), k_full.astype(mx.float32).swapaxes(-1, -2))
        * scale
    )
    neg = mx.array(-mx.inf, dtype=mx.float32)
    scores = mx.where(mask, scores, neg)
    probs = mx.softmax(scores, axis=-1)
    return mx.matmul(probs, v_full.astype(mx.float32))


@pytest.mark.parametrize(
    "pos_start,rows,total",
    [
        (13, 8, 21),  # mid-context: every row has >= TOPK complete blocks
        (1, 8, 9),  # boundary: early rows carry invalid (padded) slots
        (15, 4, 19),  # includes a (qpos+1) % ratio == 0 row (empty tail)
    ],
)
def test_gather_tier_matches_dense_mask_semantics(pos_start, rows, total):
    q, k, v = _qkv(rows, total, seed=pos_start)
    block_ids, block_valid = _selection(pos_start, rows, total, seed=pos_start)
    scale = 1.0 / math.sqrt(_D)

    mask = _qsa_blocks_to_dense_mask(
        block_ids,
        block_valid,
        pos_start=pos_start,
        total_tokens=total,
        compress_ratio=_RATIO,
    )
    reference = _dense_reference(q, k, v, mask, scale)

    out = _qsa_prefill_gather_attention(
        q,
        k,
        v,
        block_ids,
        block_valid,
        pos_start=pos_start,
        total_tokens=total,
        compress_ratio=_RATIO,
        scale=scale,
        tile_rows=64,
    )
    diff = mx.abs(out.astype(mx.float32) - reference).max()
    assert float(diff) < 2e-2, f"gather tier diverged from dense mask: {float(diff)}"


def test_gather_tier_is_tile_invariant():
    pos_start, rows, total = 13, 12, 25
    q, k, v = _qkv(rows, total, seed=99)
    block_ids, block_valid = _selection(pos_start, rows, total, seed=99)
    scale = 1.0 / math.sqrt(_D)
    kwargs = dict(
        pos_start=pos_start,
        total_tokens=total,
        compress_ratio=_RATIO,
        scale=scale,
    )
    small = _qsa_prefill_gather_attention(
        q, k, v, block_ids, block_valid, tile_rows=8, **kwargs
    )
    whole = _qsa_prefill_gather_attention(
        q, k, v, block_ids, block_valid, tile_rows=4096, **kwargs
    )
    assert bool(mx.array_equal(small, whole)), "row tiling changed the math"


def test_gather_tier_excludes_unselected_causal_tokens():
    """A causal-visible token outside (selected ∪ tail) must have no effect.

    This is the discriminator against any dense/causal fallback: full causal
    attention WOULD read the perturbed token, the sparse contract must not.
    """

    pos_start, rows, total = 13, 4, 21
    q, k, v = _qkv(rows, total, seed=7)
    block_ids, block_valid = _selection(pos_start, rows, total, seed=7)
    scale = 1.0 / math.sqrt(_D)
    kwargs = dict(
        pos_start=pos_start,
        total_tokens=total,
        compress_ratio=_RATIO,
        scale=scale,
        tile_rows=64,
    )

    row = 0
    qpos = pos_start + row
    visible = set()
    for slot in range(_TOPK):
        if bool(block_valid[row, slot]):
            b = int(block_ids[row, slot])
            visible.update(range(b * _RATIO, b * _RATIO + _RATIO))
    tail_start = ((qpos + 1) // _RATIO) * _RATIO
    visible.update(range(tail_start, qpos + 1))
    hidden_tokens = [t for t in range(qpos + 1) if t not in visible]
    assert hidden_tokens, "fixture must leave at least one invisible causal token"
    target = hidden_tokens[0]

    base = _qsa_prefill_gather_attention(q, k, v, block_ids, block_valid, **kwargs)
    v_perturbed = mx.array(v)
    v_perturbed[0, :, target, :] = v_perturbed[0, :, target, :] + 100.0
    perturbed = _qsa_prefill_gather_attention(
        q, k, v_perturbed, block_ids, block_valid, **kwargs
    )
    row_diff = mx.abs(
        base[0, :, row, :].astype(mx.float32)
        - perturbed[0, :, row, :].astype(mx.float32)
    ).max()
    assert float(row_diff) == 0.0, "row read a token outside its visible set"


def test_attention_routes_gather_tier_before_dense_reconstruction():
    """The portable tier must sit between the NAX kernel and the dense mask."""

    text = MODEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(MODEL_PATH))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Attention"
    )
    call = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    source = ast.get_source_segment(text, call)
    assert source is not None
    kernel = source.index("qsa_prefill_flash(")
    gather = source.index("_qsa_prefill_gather_attention(")
    dense = source.index("_qsa_blocks_to_dense_mask(")
    assert kernel < gather < dense
    assert "_qsa_prefill_gather_enabled()" in source


def test_gather_tier_knobs_are_registered_for_operator_overrides():
    profiles = (ROOT / "mtplx" / "profiles.py").read_text(encoding="utf-8")
    for key in ("MTPLX_QSA_PREFILL_GATHER", "MTPLX_QSA_PREFILL_GATHER_TILE"):
        assert f'"{key}"' in profiles
