"""Tiled indexer scoring (MTPLX_QSA_SCORE_TILE_ROWS) — exact-parity arm.

The whole-chunk scores matmul stages [1, S, H, nb] fp32 (+ relu twin): the
dominant #393 prefill transient. The tiled arm must produce the IDENTICAL
selection mask — row math never crosses rows — while bounding the live fp32
to one tile. Opt-in (default 0/off) until GPU-measured; these tests pin
parity (well-separated scores, exact-tie blocks, ragged last tile), the
default-off contract, and the rows-gather guard.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.models.qwen4_exp import (
    QSACache,
    TextArgs,
    QSAIndexer,
    _qsa_score_tile_rows,
)


def _args(budget=32, ratio=4):
    return TextArgs.from_dict(
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "layer_types": ["full_attention"] * 2,
            "rope_parameters": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 16,
            "indexer_budget": budget,
            "indexer_compress_ratio": ratio,
        }
    )


def _mask_for(indexer, hidden, pos_start, monkeypatch, tile):
    if tile:
        monkeypatch.setenv("MTPLX_QSA_SCORE_TILE_ROWS", str(tile))
    else:
        monkeypatch.delenv("MTPLX_QSA_SCORE_TILE_ROWS", raising=False)
    cache = QSACache(4)
    out = indexer(hidden, pos_start, cache)
    return out


class TestTileParity:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("MTPLX_QSA_SCORE_TILE_ROWS", raising=False)
        assert _qsa_score_tile_rows() == 0
        monkeypatch.setenv("MTPLX_QSA_SCORE_TILE_ROWS", "garbage")
        assert _qsa_score_tile_rows() == 0
        monkeypatch.setenv("MTPLX_QSA_SCORE_TILE_ROWS", "256")
        assert _qsa_score_tile_rows() == 256

    @pytest.mark.parametrize("tile", [7, 16, 33])
    def test_mask_identical_incl_ragged_tail(self, monkeypatch, tile):
        # S=100 rows over enough context that selection engages
        # (block_topk = 32/4 = 8 < visible blocks); random hidden makes
        # scores well-separated with prob ~1.
        mx.random.seed(11)
        args = _args()
        idx = QSAIndexer(args)
        hidden = mx.random.normal((1, 100, 64)).astype(mx.bfloat16)
        dense = _mask_for(idx, hidden, 0, monkeypatch, None)
        tiled = _mask_for(idx, hidden, 0, monkeypatch, tile)
        assert dense is not None and tiled is not None
        assert mx.array_equal(dense, tiled).item()

    def test_exact_tie_blocks_stable(self, monkeypatch):
        # All-negative projections relu to exactly 0.0 in every block:
        # maximal tie pressure. The 1e-12 index nudge must break ties the
        # same way in both paths (relu pins the values to exactly 0.0, so
        # no schedule delta can reorder them).
        args = _args()
        idx = QSAIndexer(args)
        hidden = mx.zeros((1, 96, 64), dtype=mx.bfloat16)
        dense = _mask_for(idx, hidden, 0, monkeypatch, None)
        tiled = _mask_for(idx, hidden, 0, monkeypatch, 16)
        assert dense is not None and tiled is not None
        assert mx.array_equal(dense, tiled).item()

    def test_small_s_never_tiles(self, monkeypatch):
        # S <= tile keeps the whole-chunk path (tile branch requires
        # tile < S), so decode-adjacent widths are untouched.
        mx.random.seed(3)
        args = _args()
        idx = QSAIndexer(args)
        hidden = mx.random.normal((1, 4, 64)).astype(mx.bfloat16)
        monkeypatch.setenv("MTPLX_QSA_SCORE_TILE_ROWS", "256")
        cache = QSACache(4)
        out = idx(hidden, 0, cache)
        # 4 tokens -> 1 visible block <= block_topk: dense==sparse regime.
        assert out is None

    def test_rows_gather_guard(self, monkeypatch):
        # A tile smaller than the rows-gather width must not crash into the
        # gather lane's top_idx dependency: the guard routes to the dense
        # mask instead.
        mx.random.seed(5)
        args = _args()
        idx = QSAIndexer(args)
        monkeypatch.setenv("MTPLX_QSA_SCORE_TILE_ROWS", "2")
        monkeypatch.setenv("MTPLX_QSA_GATHER", "1")
        monkeypatch.setenv("MTPLX_QSA_GATHER_MIN_CONTEXT", "1")
        cache = QSACache(4)
        warm = mx.random.normal((1, 96, 64)).astype(mx.bfloat16)
        idx(warm, 0, cache)
        cache.kv.offset = 96
        rows = mx.random.normal((1, 6, 64)).astype(mx.bfloat16)
        out = idx(rows, 96, cache)
        assert isinstance(out, mx.array), "guard must fall through to mask"
        assert out.dtype == mx.bool_
