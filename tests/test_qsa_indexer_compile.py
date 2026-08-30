"""Correctness and retrace gates for the compiled QSA indexer core.

These are synthetic array fixtures only.  They do not instantiate a model or
load checkpoint weights.  The eager oracle is the exact same v2.10 arithmetic
composition with ordinary Python staging; the system under test places that
composition behind one pure ``mx.compile`` boundary with explicit state.
"""

from __future__ import annotations

import inspect
import math

import mlx.core as mx
import pytest

import mtplx.kernels.qsa_indexer_compile as compile_module
from mtplx.kernels.qsa_indexer_compile import (
    QSACompileCapacityError,
    QSACompiledIndexerCore,
    qsa_indexer_capacity_bucket,
    qsa_indexer_dense_output_capacity,
    qsa_indexer_is_bucket_capacity,
    qsa_indexer_selector_chunk_rows,
)
from mtplx.kernels.qsa_indexer_prepare import (
    qsa_indexer_pool_keys_metal,
    qsa_indexer_prepare_queries_metal,
)
from mtplx.kernels.qsa_indexer_select import (
    qsa_indexer_select_blocks_metal,
    qsa_indexer_select_dense_mask_metal,
    qsa_indexer_select_row_tokens_metal,
)

requires_metal = pytest.mark.skipif(
    not mx.metal.is_available() or mx.default_device() != mx.gpu,
    reason="compiled QSA indexer tests require the Metal GPU",
)

HEADS = 2
HEAD_DIM = 8
RATIO = 2
TOPK = 4
QK_WIDTH = (HEADS + 1) * HEAD_DIM
RAW_CAPACITY = 256
POOLED_CAPACITY = 256
EPS = 1e-6


def _weights(dtype=mx.float16):
    q_norm = mx.linspace(0.75, 1.25, HEAD_DIM).astype(dtype)
    k_norm = mx.linspace(1.25, 0.75, HEAD_DIM).astype(dtype)
    inv_freq = mx.array([1.0, 0.1], dtype=mx.float32)
    return q_norm, k_norm, inv_freq


def _core(
    *,
    project=False,
    scratch_bytes=32 * 1024 * 1024,
    compile_factory=mx.compile,
    rope_attention_scaling=1.0,
):
    q_norm, k_norm, inv_freq = _weights()
    return QSACompiledIndexerCore(
        n_heads=HEADS,
        kv_heads=1,
        head_dim=HEAD_DIM,
        block_topk=TOPK,
        compress_ratio=RATIO,
        q_norm_weight=q_norm,
        k_norm_weight=k_norm,
        inv_freq=inv_freq,
        rms_norm_eps=EPS,
        rope_attention_scaling=rope_attention_scaling,
        project_qk=(lambda hidden: hidden) if project else None,
        selector_scratch_bytes=scratch_bytes,
        compile_factory=compile_factory,
    )


def _fixture(seed: int, rows: int):
    mx.random.seed(seed)
    qk_rows = (mx.random.normal((1, rows, QK_WIDTH)) * 0.2).astype(mx.float16)
    raw = (mx.random.normal((1, RAW_CAPACITY, HEAD_DIM)) * 0.2).astype(mx.float16)
    pooled = (mx.random.normal((1, POOLED_CAPACITY, HEAD_DIM)) * 0.2).astype(mx.float16)
    mx.eval(qk_rows, raw, pooled)
    return qk_rows, raw, pooled


def _fixture_with_capacities(
    seed: int,
    rows: int,
    *,
    raw_capacity: int,
    pooled_capacity: int,
):
    mx.random.seed(seed)
    qk_rows = (mx.random.normal((1, rows, QK_WIDTH)) * 0.2).astype(mx.float16)
    raw = (mx.random.normal((1, raw_capacity, HEAD_DIM)) * 0.2).astype(mx.float16)
    pooled = (mx.random.normal((1, pooled_capacity, HEAD_DIM)) * 0.2).astype(mx.float16)
    mx.eval(qk_rows, raw, pooled)
    return qk_rows, raw, pooled


def _oracle(
    qk_rows,
    raw,
    pooled,
    *,
    pos_start: int,
    total_tokens: int,
    logical_blocks: int,
    pooled_len: int,
    mode: str,
    rope_attention_scaling: float = 1.0,
):
    rows = int(qk_rows.shape[1])
    q_norm, k_norm, inv_freq = _weights(qk_rows.dtype)
    q_raw = qk_rows[..., : HEADS * HEAD_DIM].reshape(
        1,
        rows,
        HEADS,
        HEAD_DIM,
    )
    k_raw = qk_rows[..., HEADS * HEAD_DIM :].reshape(1, rows, HEAD_DIM)
    q = (
        None
        if mode == "update_only"
        else qsa_indexer_prepare_queries_metal(
            q_raw,
            q_norm,
            inv_freq,
            pos_start=pos_start,
            eps=EPS,
            attention_scaling=rope_attention_scaling,
        )
    )
    raw_next = mx.slice_update(
        raw,
        k_raw,
        mx.array([pos_start], dtype=mx.int32),
        axes=(1,),
    )

    max_new = (rows + RATIO - 1) // RATIO
    max_start = min(
        int(raw.shape[1]) // RATIO - max_new,
        int(pooled.shape[1]) - max_new,
    )
    block_start = min(max(logical_blocks - max_new, 0), pooled_len, max_start)
    raw_window = raw_next[
        :,
        block_start * RATIO : (block_start + max_new) * RATIO,
        :,
    ]
    pooled_window = qsa_indexer_pool_keys_metal(
        raw_window,
        k_norm,
        inv_freq,
        block_start=block_start,
        compress_ratio=RATIO,
        eps=EPS,
        attention_scaling=rope_attention_scaling,
    )
    pooled_next = mx.slice_update(
        pooled,
        pooled_window,
        mx.array([block_start], dtype=mx.int32),
        axes=(1,),
    )
    common = {
        "pos_start": pos_start,
        "total_tokens": total_tokens,
        "logical_blocks": logical_blocks,
        "block_topk": TOPK,
        "compress_ratio": RATIO,
    }
    if mode == "update_only":
        selection = None
    elif mode == "blocks":
        selection = qsa_indexer_select_blocks_metal(q, pooled_next, **common)
    elif mode == "row_tokens":
        selection = qsa_indexer_select_row_tokens_metal(q, pooled_next, **common)
    elif mode == "dense_mask":
        selection = qsa_indexer_select_dense_mask_metal(
            q,
            pooled_next,
            output_total_tokens=qsa_indexer_dense_output_capacity(
                int(pooled.shape[1]),
                RATIO,
            ),
            **common,
        )
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(mode)
    return selection, raw_next, pooled_next


def _array_equal(actual, expected):
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype
    assert tuple(actual.shape) == tuple(expected.shape)
    assert bool(mx.array_equal(actual, expected).item())


def _selection_equal(actual, expected):
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple)
        assert len(actual) == len(expected)
        for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
            _array_equal(actual_leaf, expected_leaf)
    else:
        assert isinstance(expected, mx.array)
        _array_equal(actual, expected)


def test_shared_capacity_bucket_contract_is_logarithmic():
    expected = {
        0: 0,
        1: 256,
        256: 256,
        257: 512,
        512: 512,
        513: 1024,
        768: 1024,
        65_536: 65_536,
        125_000: 131_072,
        250_000: 262_144,
        500_000: 524_288,
        1_000_000: 1_048_576,
    }
    assert {value: qsa_indexer_capacity_bucket(value) for value in expected} == expected
    assert qsa_indexer_is_bucket_capacity(0)
    assert qsa_indexer_is_bucket_capacity(256)
    assert qsa_indexer_is_bucket_capacity(65_536)
    assert qsa_indexer_is_bucket_capacity(1_048_576)
    assert not qsa_indexer_is_bucket_capacity(768)


def test_production_prefill_chunk_geometry_caps_hidden_score_scratch():
    rows = qsa_indexer_selector_chunk_rows(
        2048,
        65_536,
        32 * 1024 * 1024,
    )
    assert rows == 128
    assert rows * 65_536 * 4 == 32 * 1024 * 1024


def test_compiled_body_has_no_host_sync_or_shapeless_escape_hatch():
    source = inspect.getsource(QSACompiledIndexerCore._make_compiled)
    assert ".item(" not in source
    assert "mx.eval(" not in source
    assert "shapeless=True" not in source
    assert "mx.slice_update" in source
    assert "qsa_indexer_prepare_queries_metal" in source
    assert "qsa_indexer_pool_keys_metal" in source


@requires_metal
@pytest.mark.parametrize("mode", ["blocks", "dense_mask", "row_tokens"])
def test_compiled_complete_core_matches_eager_composition(mode):
    rows = 3
    pos = 20
    total = pos + rows
    logical = total // RATIO
    pooled_len = pos // RATIO
    qk_rows, raw, pooled = _fixture(100 + len(mode), rows)
    core = _core()

    actual = core.select_qk_rows(
        qk_rows,
        raw,
        pooled,
        pos_start=pos,
        total_tokens=total,
        logical_blocks=logical,
        pooled_len=pooled_len,
        mode=mode,
    )
    expected_selection, expected_raw, expected_pooled = _oracle(
        qk_rows,
        raw,
        pooled,
        pos_start=pos,
        total_tokens=total,
        logical_blocks=logical,
        pooled_len=pooled_len,
        mode=mode,
    )

    _selection_equal(actual.selection, expected_selection)
    _array_equal(actual.raw_keys, expected_raw)
    _array_equal(actual.pooled, expected_pooled)
    mx.eval(actual.pooled_len, actual.offset)
    assert actual.pooled_len.tolist() == [logical]
    assert actual.offset.tolist() == [total]

    report = core.to_dict()
    assert report["calls"] == report["compiled_calls"] == 1
    assert report["traces"] == report["entry_count"] == 1
    assert report["modes"][mode] == 1
    assert report["compiled_keys"][0]["source"] == "qk_rows"
    assert report["compiled_keys"][0]["mode"] == mode


@requires_metal
def test_compiled_core_preserves_static_yarn_attention_scaling():
    rows = 3
    pos = 1_000_000
    total = pos + rows
    logical = total // RATIO
    pooled_len = pos // RATIO
    rope_scale = 1.0 + 0.1 * math.log(4.0)
    qk_rows, raw, pooled = _fixture_with_capacities(
        177,
        rows,
        raw_capacity=1_048_576,
        pooled_capacity=524_288,
    )
    core = _core(rope_attention_scaling=rope_scale)

    actual = core.select_qk_rows(
        qk_rows,
        raw,
        pooled,
        pos_start=pos,
        total_tokens=total,
        logical_blocks=logical,
        pooled_len=pooled_len,
        mode="blocks",
    )
    expected_selection, expected_raw, expected_pooled = _oracle(
        qk_rows,
        raw,
        pooled,
        pos_start=pos,
        total_tokens=total,
        logical_blocks=logical,
        pooled_len=pooled_len,
        mode="blocks",
        rope_attention_scaling=rope_scale,
    )
    _selection_equal(actual.selection, expected_selection)
    _array_equal(actual.raw_keys, expected_raw)
    _array_equal(actual.pooled, expected_pooled)


@requires_metal
def test_update_only_captures_first_full_prefill_without_q_prep_or_selector(
    monkeypatch,
):
    rows = 2048
    qk_rows, raw, pooled = _fixture_with_capacities(
        190,
        rows,
        raw_capacity=2048,
        pooled_capacity=1024,
    )
    expected_selection, expected_raw, expected_pooled = _oracle(
        qk_rows,
        raw,
        pooled,
        pos_start=0,
        total_tokens=rows,
        logical_blocks=rows // RATIO,
        pooled_len=0,
        mode="update_only",
    )
    assert expected_selection is None

    def unexpected(*_args, **_kwargs):
        raise AssertionError("update_only must not trace query prep or selection")

    monkeypatch.setattr(
        compile_module,
        "qsa_indexer_prepare_queries_metal",
        unexpected,
    )
    monkeypatch.setattr(
        compile_module,
        "qsa_indexer_select_blocks_metal",
        unexpected,
    )
    monkeypatch.setattr(
        compile_module,
        "qsa_indexer_select_dense_mask_metal",
        unexpected,
    )
    monkeypatch.setattr(
        compile_module,
        "qsa_indexer_select_row_tokens_metal",
        unexpected,
    )
    core = _core()
    actual = core.select_qk_rows(
        qk_rows,
        raw,
        pooled,
        pos_start=0,
        total_tokens=rows,
        logical_blocks=rows // RATIO,
        pooled_len=0,
        mode="update_only",
    )
    assert actual.selection is None
    _array_equal(actual.raw_keys, expected_raw)
    _array_equal(actual.pooled, expected_pooled)
    mx.eval(actual.pooled_len, actual.offset)
    assert actual.pooled_len.tolist() == [rows // RATIO]
    assert actual.offset.tolist() == [rows]

    report = core.to_dict()
    assert report["calls"] == report["compiled_calls"] == 1
    assert report["traces"] == report["entry_count"] == 1
    assert report["selector_dispatches"] == 0
    assert report["modes"]["update_only"] == 1
    assert report["compiled_keys"][0]["selector_chunk_rows"] == 0


@requires_metal
def test_changed_tensor_frontiers_replay_one_compiled_entry_and_advance_state():
    rows = 3
    qk0, raw, pooled = _fixture(211, rows)
    qk1, _, _ = _fixture(212, rows)
    core = _core()

    first_scalars = tuple(
        mx.array([value], dtype=mx.int32) for value in (20, 23, 11, 10)
    )
    first = core.select_qk_rows(
        qk0,
        raw,
        pooled,
        pos_start=first_scalars[0],
        total_tokens=first_scalars[1],
        logical_blocks=first_scalars[2],
        pooled_len=first_scalars[3],
        mode="blocks",
    )
    mx.eval(*first[1:])

    second = core.select_qk_rows(
        qk1,
        first.raw_keys,
        first.pooled,
        pos_start=mx.array([23], dtype=mx.int32),
        total_tokens=mx.array([26], dtype=mx.int32),
        logical_blocks=mx.array([13], dtype=mx.int32),
        pooled_len=first.pooled_len,
        mode="blocks",
    )
    expected_selection, expected_raw, expected_pooled = _oracle(
        qk1,
        first.raw_keys,
        first.pooled,
        pos_start=23,
        total_tokens=26,
        logical_blocks=13,
        pooled_len=11,
        mode="blocks",
    )
    _selection_equal(second.selection, expected_selection)
    _array_equal(second.raw_keys, expected_raw)
    _array_equal(second.pooled, expected_pooled)
    mx.eval(second.pooled_len, second.offset)
    assert second.pooled_len.tolist() == [13]
    assert second.offset.tolist() == [26]

    report = core.to_dict()
    assert report["calls"] == report["compiled_calls"] == 2
    assert report["traces"] == report["entry_count"] == 1
    assert report["capacity_transitions"] == 0


@requires_metal
def test_hidden_and_shared_qk_sources_have_separate_fixed_arity_graphs():
    rows = 2
    pos = 20
    total = 22
    logical = 11
    pooled_len = 10
    qk_rows, raw, pooled = _fixture(313, rows)
    core = _core(project=True)

    from_qk = core.select_qk_rows(
        qk_rows,
        raw,
        pooled,
        pos_start=pos,
        total_tokens=total,
        logical_blocks=logical,
        pooled_len=pooled_len,
        mode="row_tokens",
    )
    from_hidden = core.select_hidden(
        qk_rows,
        raw,
        pooled,
        pos_start=pos,
        total_tokens=total,
        logical_blocks=logical,
        pooled_len=pooled_len,
        mode="row_tokens",
    )
    _selection_equal(from_hidden.selection, from_qk.selection)
    _array_equal(from_hidden.raw_keys, from_qk.raw_keys)
    _array_equal(from_hidden.pooled, from_qk.pooled)

    report = core.to_dict()
    assert report["qk_rows_calls"] == 1
    assert report["hidden_calls"] == 1
    assert report["traces"] == report["entry_count"] == 2
    assert {key["source"] for key in report["compiled_keys"]} == {
        "hidden",
        "qk_rows",
    }


@requires_metal
def test_compile_graph_captures_shape_stable_selector_row_chunking():
    rows = 5
    # Exactly two score rows fit in each selector dispatch.
    scratch_bytes = 2 * POOLED_CAPACITY * 4
    qk_rows, raw, pooled = _fixture(401, rows)
    core = _core(scratch_bytes=scratch_bytes)
    pos = 20
    total = pos + rows
    logical = total // RATIO
    pooled_len = pos // RATIO

    actual = core.select_qk_rows(
        qk_rows,
        raw,
        pooled,
        pos_start=pos,
        total_tokens=total,
        logical_blocks=logical,
        pooled_len=pooled_len,
        mode="dense_mask",
    )
    expected_selection, expected_raw, expected_pooled = _oracle(
        qk_rows,
        raw,
        pooled,
        pos_start=pos,
        total_tokens=total,
        logical_blocks=logical,
        pooled_len=pooled_len,
        mode="dense_mask",
    )
    _selection_equal(actual.selection, expected_selection)
    _array_equal(actual.raw_keys, expected_raw)
    _array_equal(actual.pooled, expected_pooled)

    report = core.to_dict()
    assert report["selector_dispatches"] == 3
    assert report["compiled_keys"][0]["selector_chunk_rows"] == 2
    assert tuple(actual.selection.shape) == (
        1,
        1,
        rows,
        qsa_indexer_dense_output_capacity(POOLED_CAPACITY, RATIO),
    )


@requires_metal
def test_full_pooled_bucket_with_no_new_block_recomputes_in_bounds():
    # The logical pooled prefix occupies every backing row.  This decode row
    # completes no block, so naively staging one candidate at pooled_len=256
    # would write out of bounds.  The compiled core moves its fixed window
    # backward and recomputes block 255 exactly instead.
    qk_rows, raw, pooled = _fixture_with_capacities(
        451,
        1,
        raw_capacity=1024,
        pooled_capacity=256,
    )
    core = _core()
    actual = core.select_qk_rows(
        qk_rows,
        raw,
        pooled,
        pos_start=512,
        total_tokens=513,
        logical_blocks=256,
        pooled_len=256,
        mode="blocks",
    )
    expected_selection, expected_raw, expected_pooled = _oracle(
        qk_rows,
        raw,
        pooled,
        pos_start=512,
        total_tokens=513,
        logical_blocks=256,
        pooled_len=256,
        mode="blocks",
    )
    _selection_equal(actual.selection, expected_selection)
    _array_equal(actual.raw_keys, expected_raw)
    _array_equal(actual.pooled, expected_pooled)


@requires_metal
def test_rollback_decreases_tensor_frontiers_without_retrace_or_stale_suffix():
    rows = 3
    qk_spec, raw, pooled = _fixture_with_capacities(
        461,
        rows,
        raw_capacity=1024,
        pooled_capacity=256,
    )
    qk_replay, _, _ = _fixture_with_capacities(
        462,
        rows,
        raw_capacity=1024,
        pooled_capacity=256,
    )
    core = _core()

    speculative = core.select_qk_rows(
        qk_spec,
        raw,
        pooled,
        pos_start=mx.array([510], dtype=mx.int32),
        total_tokens=mx.array([513], dtype=mx.int32),
        logical_blocks=mx.array([256], dtype=mx.int32),
        pooled_len=mx.array([255], dtype=mx.int32),
        mode="row_tokens",
    )
    mx.eval(*speculative[1:])

    # Roll back two tokens/one complete block, then overwrite the rejected
    # raw rows.  Block 255 remains physically present in the pooled suffix but
    # the lowered logical frontier must make it unobservable.
    replay = core.select_qk_rows(
        qk_replay,
        speculative.raw_keys,
        speculative.pooled,
        pos_start=mx.array([508], dtype=mx.int32),
        total_tokens=mx.array([511], dtype=mx.int32),
        logical_blocks=mx.array([255], dtype=mx.int32),
        pooled_len=mx.array([254], dtype=mx.int32),
        mode="row_tokens",
    )
    expected_selection, expected_raw, expected_pooled = _oracle(
        qk_replay,
        speculative.raw_keys,
        speculative.pooled,
        pos_start=508,
        total_tokens=511,
        logical_blocks=255,
        pooled_len=254,
        mode="row_tokens",
    )
    _selection_equal(replay.selection, expected_selection)
    _array_equal(replay.raw_keys, expected_raw)
    _array_equal(replay.pooled, expected_pooled)
    mx.eval(replay.pooled_len, replay.offset)
    assert replay.pooled_len.tolist() == [255]
    assert replay.offset.tolist() == [511]

    report = core.to_dict()
    assert report["calls"] == report["compiled_calls"] == 2
    assert report["traces"] == report["entry_count"] == 1


@requires_metal
def test_host_capacity_preflight_refuses_before_compile_or_custom_dispatch():
    created = []

    def record_compile(fn):
        created.append(fn)
        return fn

    qk_rows, raw, pooled = _fixture(509, 2)
    core = _core(compile_factory=record_compile)
    with pytest.raises(QSACompileCapacityError, match="must be reserved"):
        core.select_qk_rows(
            qk_rows,
            raw,
            pooled,
            pos_start=255,
            total_tokens=257,
            logical_blocks=128,
            pooled_len=127,
            mode="blocks",
        )
    assert created == []
    assert core.to_dict()["calls"] == 0


@requires_metal
def test_unbucketed_restored_backing_is_a_visible_preflight_error():
    qk_rows, raw, pooled = _fixture(610, 1)
    unbucketed = mx.zeros((1, 768, HEAD_DIM), dtype=raw.dtype)
    core = _core(compile_factory=lambda fn: fn)
    with pytest.raises(QSACompileCapacityError, match="power-of-two bucket"):
        core.select_qk_rows(
            qk_rows,
            unbucketed,
            pooled,
            pos_start=20,
            total_tokens=21,
            logical_blocks=10,
            pooled_len=10,
            mode="blocks",
        )


def test_compile_module_reexports_the_shared_mtp_bucket_functions():
    from mtplx import qsa_mtp_precompute

    assert (
        compile_module.qsa_indexer_capacity_bucket
        is qsa_mtp_precompute.qsa_indexer_capacity_bucket
    )
    assert (
        compile_module.qsa_indexer_is_bucket_capacity
        is qsa_mtp_precompute.qsa_indexer_is_bucket_capacity
    )
