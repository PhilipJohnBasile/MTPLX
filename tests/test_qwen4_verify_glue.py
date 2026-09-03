"""Tests for ``MTPLX_QWEN4_VERIFY_GLUE`` (W70).

PURE PYTHON ON PURPOSE.  Nothing here evaluates an MLX array: a benchmark
queue owns the GPU lock while this branch is built, and one un-flocked
``mx.eval`` corrupts a measurement window.  The MLX-evaluating checks --
bit-exact parity at the real layer shapes and the queued-lane timing -- live
in ``the PR #391 harness micro_verify_glue_a.py``, which runs inside a guarded
window.

What IS falsifiable without a GPU, and is tested here:

* the flag/item parser, including the typo-raises rule that keeps an arm from
  silently measuring the control twice;
* the contract check -- every geometry this kernel is not wired for RAISES,
  and the message names the offending shape;
* the install state machine: pending RAISES at the hot-path gate, an
  exactness failure DISABLES and reports the reason, a contract failure
  propagates;
* the generated Metal source: the operand order that makes the rotation
  bit-exact against the stock chain, and the pass-through copy;
* the arithmetic of the engagement line and the node accounting.
"""

from __future__ import annotations

import re

import pytest


# --------------------------------------------------------------------------
# Flag parsing -- no MLX needed
# --------------------------------------------------------------------------
from mtplx.runtime_options import (
    QWEN4_VERIFY_GLUE_ITEMS,
    parse_verify_glue_items,
)


def test_items_are_the_two_built_ones():
    assert QWEN4_VERIFY_GLUE_ITEMS == ("qsa_rope", "qsa_rope_idx")


def test_unset_selects_everything():
    assert parse_verify_glue_items(None) == frozenset(QWEN4_VERIFY_GLUE_ITEMS)


def test_empty_and_all_select_everything():
    for raw in ("", "   ", ",,", "all"):
        assert parse_verify_glue_items(raw) == frozenset(QWEN4_VERIFY_GLUE_ITEMS)


def test_single_item_selects_only_it():
    assert parse_verify_glue_items("qsa_rope") == frozenset({"qsa_rope"})
    assert parse_verify_glue_items(" QSA_ROPE_IDX ") == frozenset(
        {"qsa_rope_idx"}
    )


def test_unknown_item_raises_rather_than_being_dropped():
    # A typo that silently disabled the item under test would make the arm
    # measure the control twice.
    with pytest.raises(ValueError) as excinfo:
        parse_verify_glue_items("qsa_rope,qsa_ropes")
    assert "qsa_ropes" in str(excinfo.value)
    assert "qsa_rope_idx" in str(excinfo.value)


def test_hc_triple_is_not_an_item():
    # W69 ranked it second; it is structurally not fusable (grid-wide
    # read-after-write between the three hyper-connection kernels), and the
    # reason is recorded next to the item list.
    assert "hc_triple" not in QWEN4_VERIFY_GLUE_ITEMS
    with pytest.raises(ValueError):
        parse_verify_glue_items("hc_triple")


def test_env_gate_defaults_off_and_item_names_are_checked():
    from mtplx import runtime_options

    runtime_options.reset_qwen4_verify_glue_cache(env={})
    assert runtime_options.qwen4_verify_glue_enabled() is False
    assert runtime_options.qwen4_verify_glue_enabled("qsa_rope") is False

    runtime_options.reset_qwen4_verify_glue_cache(
        env={"MTPLX_QWEN4_VERIFY_GLUE": "1"}
    )
    try:
        assert runtime_options.qwen4_verify_glue_enabled() is True
        assert runtime_options.qwen4_verify_glue_enabled("qsa_rope") is True
        assert runtime_options.qwen4_verify_glue_enabled("qsa_rope_idx") is True
        with pytest.raises(ValueError):
            runtime_options.qwen4_verify_glue_enabled("nope")
    finally:
        runtime_options.reset_qwen4_verify_glue_cache(env={})


def test_item_selection_isolates_one_rewrite():
    from mtplx import runtime_options

    runtime_options.reset_qwen4_verify_glue_cache(
        env={
            "MTPLX_QWEN4_VERIFY_GLUE": "1",
            "MTPLX_QWEN4_VERIFY_GLUE_ITEMS": "qsa_rope",
        }
    )
    try:
        assert runtime_options.qwen4_verify_glue_enabled("qsa_rope") is True
        assert runtime_options.qwen4_verify_glue_enabled("qsa_rope_idx") is False
    finally:
        runtime_options.reset_qwen4_verify_glue_cache(env={})


# --------------------------------------------------------------------------
# The kernel module.  Importing MLX is fine; EVALUATING is not, so every
# array below stays lazy and no test reads a value off the device.
# --------------------------------------------------------------------------
mx = pytest.importorskip("mlx.core")

from mtplx.kernels import qwen4_m4_rope as rope  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_state():
    rope.reset_for_tests()
    yield
    rope.reset_for_tests()


def _lazy(shape, dtype=None):
    """A lazy array of the given shape.  Never evaluated."""

    return mx.zeros(shape, dtype=dtype or mx.bfloat16)


def _production_operands():
    """The 16 K decode cell's real QSA attention rope geometry."""

    queries = _lazy((1, 4, 24, 256))
    keys = _lazy((1, 4, 2, 256))
    inv_freq = mx.zeros((32,), dtype=mx.float32)
    return queries, keys, inv_freq


def test_contract_accepts_the_production_geometry():
    queries, keys, inv_freq = _production_operands()
    assert rope.check_contract(queries, keys, inv_freq) == (4, 24, 2, 64)


def test_contract_accepts_keys_only():
    _, keys, inv_freq = _production_operands()
    assert rope.check_contract(None, keys, inv_freq) == (4, 0, 2, 64)


@pytest.mark.parametrize(
    "queries,keys,inv_freq,needle",
    [
        # batch > 1
        (None, "b2", None, "[1, S, H, D]"),
        # 3-D keys
        (None, "k3", None, "[1, S, H, D]"),
        # rotary wider than the head dimension
        (None, None, "wide", "rotary_dim"),
        # odd rotary width
        (None, None, "odd", "rotary_dim"),
        # float64-ish inv_freq
        (None, None, "bf16", "float32"),
        # q/k dtype split
        ("f16", None, None, "share a dtype"),
        # q/k head-dim split
        ("d128", None, None, "share a head dimension"),
        # q/k row split
        ("rows2", None, None, "same row count"),
    ],
)
def test_contract_raises_and_names_the_mismatch(queries, keys, inv_freq, needle):
    q, k, freq = _production_operands()
    if keys == "b2":
        k = _lazy((2, 4, 2, 256))
    elif keys == "k3":
        k = _lazy((1, 4, 256))
    if inv_freq == "wide":
        freq = mx.zeros((256,), dtype=mx.float32)
    elif inv_freq == "odd":
        # rotary_dim = 2 * size is always even; force the odd path by making
        # the head dimension smaller than the rotary width.
        freq = mx.zeros((200,), dtype=mx.float32)
    elif inv_freq == "bf16":
        freq = mx.zeros((32,), dtype=mx.bfloat16)
    if queries == "f16":
        q = _lazy((1, 4, 24, 256), dtype=mx.float16)
    elif queries == "d128":
        q = _lazy((1, 4, 24, 128))
    elif queries == "rows2":
        q = _lazy((1, 2, 24, 256))

    with pytest.raises(rope.RopeGlueContractError) as excinfo:
        rope.check_contract(q, k, freq)
    assert needle in str(excinfo.value)


def test_contract_counter_advances():
    queries, keys, inv_freq = _production_operands()
    before = rope.counters()["contract_checks"]
    rope.check_contract(queries, keys, inv_freq)
    assert rope.counters()["contract_checks"] == before + 1


def test_pos_start_tensor_must_be_one_int32():
    with pytest.raises(rope.RopeGlueContractError):
        rope._as_i32_scalar(mx.zeros((4,), dtype=mx.int32), "pos_start")
    with pytest.raises(rope.RopeGlueContractError):
        rope._as_i32_scalar(mx.zeros((1,), dtype=mx.float32), "pos_start")
    assert rope._as_i32_scalar(7, "pos_start").shape == (1,)


@pytest.mark.parametrize("scaling", [0.0, -1.0, float("nan"), float("inf")])
def test_attention_scaling_must_be_finite_and_positive(scaling):
    with pytest.raises(rope.RopeGlueContractError):
        rope._attention_scaling(scaling)


# --------------------------------------------------------------------------
# The generated Metal source: the parts that make it bit-exact
# --------------------------------------------------------------------------
def test_generated_constants_carry_the_geometry():
    header = rope._constants(24, 2, 256, 64, 1.0)
    for fragment in (
        "constant constexpr uint HQ = 24;",
        "constant constexpr uint HK = 2;",
        "constant constexpr uint HEAD_DIM = 256;",
        "constant constexpr uint ROTARY_DIM = 64;",
        "constant constexpr uint HALF_ROTARY = 32;",
    ):
        assert fragment in header
    assert "ROPE_ATTENTION_SCALE = 1.0f;" in header


def test_scaling_literal_round_trips_exactly():
    # A truncated literal would silently change every cos/sin in the table.
    header = rope._constants(24, 2, 256, 64, 0.8956339385554523)
    assert "0.8956339385554523f" in header


def test_rotation_keeps_the_four_products_distinct():
    # Letting Metal contract a*b - c*d into an FMA moves bf16 cutoff values,
    # which is exactly the difference the stock graph does not have.
    source = rope._ROTATE
    for fragment in (
        "const float first_cosine = first * cosine;",
        "const float second_sine = second * sine;",
        "const float second_cosine = second * cosine;",
        "const float first_sine = first * sine;",
        "static_cast<T>(first_cosine - second_sine)",
        "static_cast<T>(second_cosine + first_sine)",
    ):
        assert fragment in source


def test_rotation_uses_precise_trig_like_mx_cos():
    # mx.cos/mx.sin lower to the precise variants; mx.fast.rope deliberately
    # does not, which is why this kernel is not written on top of it.
    assert "metal::precise::cos" in rope._ROTATE
    assert "metal::precise::sin" in rope._ROTATE
    assert "fast::" not in rope._ROTATE


def test_pass_through_half_is_copied_not_recomputed():
    assert "for (uint dim = ROTARY_DIM + lane; dim < HEAD_DIM; dim += 32u)" in (
        rope._ROTATE
    )
    assert "dst[dim] = src[(size_t)dim * d_stride];" in rope._ROTATE


def test_position_is_int32_then_float_like_the_stock_chain():
    # The stock chain is (pos_start + arange(S)).astype(float32); rounding the
    # sum in float first would differ past 2**24.
    assert "float(pos_start[0] + int(row))" in rope._SOURCE_QK
    assert "float(pos_start[0] + int(row))" in rope._SOURCE_K


def test_qk_variant_packs_both_tensors_into_one_grid():
    assert "const uint PER_ROW = HQ + HK;" in rope._SOURCE_QK
    assert "if (slot < HQ)" in rope._SOURCE_QK
    # ...and the keys-only variant never binds a query buffer.
    assert "q_out" not in rope._SOURCE_K
    assert re.search(r"\bq_strides\b", rope._SOURCE_K) is None
    assert re.search(r"\bq_strides\b", rope._SOURCE_QK) is not None


def test_kernel_variants_have_distinct_cache_keys():
    # MLX caches compiled libraries by NAME; two geometries sharing a name
    # would silently run each other's code.
    names = set()
    for heads_q, heads_k, dim, rot, scale, dtype in (
        (24, 2, 256, 64, 1.0, mx.bfloat16),
        (24, 2, 256, 64, 1.0, mx.float16),
        (24, 2, 256, 64, 0.9, mx.bfloat16),
        (12, 2, 256, 64, 1.0, mx.bfloat16),
        (24, 2, 128, 64, 1.0, mx.bfloat16),
        (24, 2, 256, 32, 1.0, mx.bfloat16),
    ):
        names.add(
            f"hq{heads_q}_hk{heads_k}_d{dim}_r{rot}_"
            f"s{rope._float_tag(scale)}_{rope._dtype_tag(dtype)}"
        )
    assert len(names) == 6


# --------------------------------------------------------------------------
# Install state machine
# --------------------------------------------------------------------------
def test_pending_until_the_probe_runs():
    assert rope.pending() is True
    assert rope.installed() is False
    assert rope.disabled_reason() is None


def test_hot_path_gate_raises_while_pending():
    from mtplx import qwen4_verify_glue as glue
    from mtplx.qwen4_verify_glue import VerifyGlueContractError

    glue.reset_for_tests()
    with pytest.raises(VerifyGlueContractError) as excinfo:
        glue.qsa_rope_installed()
    assert "never ran" in str(excinfo.value)
    with pytest.raises(VerifyGlueContractError):
        glue.qsa_rope_idx_installed()


def test_install_with_no_qsa_layer_is_a_contract_error():
    with pytest.raises(rope.RopeGlueContractError) as excinfo:
        rope.install(())
    assert "no QSA attention layer" in str(excinfo.value)


class _FakeNorm:
    def __init__(self, dtype=None):
        self.weight = mx.zeros((256,), dtype=dtype or mx.bfloat16)


class _FakeAttention:
    def __init__(self, *, q_dtype=None, k_dtype=None, head_dim=256):
        self.n_heads = 24
        self.n_kv_heads = 2
        self.head_dim = head_dim
        self.q_norm = _FakeNorm(q_dtype)
        self.k_norm = _FakeNorm(k_dtype)
        self._inv_freq = mx.zeros((32,), dtype=mx.float32)
        self._rope_attention_scaling = 1.0


def test_install_raises_when_the_head_norms_disagree_on_dtype():
    layers = ((0, _FakeAttention(q_dtype=mx.bfloat16, k_dtype=mx.float16)),)
    with pytest.raises(rope.RopeGlueContractError) as excinfo:
        rope.install(layers)
    assert "one kernel rotates both tensors" in str(excinfo.value)


def test_install_raises_when_a_head_norm_is_missing():
    class _NoNorm(_FakeAttention):
        def __init__(self):
            super().__init__()
            self.q_norm = None

    with pytest.raises(rope.RopeGlueContractError) as excinfo:
        rope.install(((3, _NoNorm()),))
    assert "q_norm/k_norm" in str(excinfo.value)


def test_exactness_failure_disables_the_item_and_reports_the_reason(monkeypatch):
    """A probe miss is a NUMERICAL verdict: disable and log, never raise."""

    calls = {"n": 0}

    def _fake_reference(queries, keys, inv_freq, *, pos_start, attention_scaling):
        return queries, keys

    def _fake_rope(queries, keys, inv_freq, *, pos_start, attention_scaling):
        return queries, keys

    class _AlwaysDifferent:
        def item(self):
            calls["n"] += 1
            return False

    monkeypatch.setattr(rope, "stock_reference", _fake_reference)
    monkeypatch.setattr(rope, "rope_qk", _fake_rope)
    monkeypatch.setattr(rope.mx, "array_equal", lambda a, b: _AlwaysDifferent())
    monkeypatch.setattr(rope.mx, "eval", lambda *a, **k: None)

    warnings: list = []

    class _Logger:
        def warning(self, *args):
            warnings.append(args)

    ok = rope.install(((7, _FakeAttention()),), logger=_Logger())
    assert ok is False
    assert rope.installed() is False
    assert rope.pending() is False
    reason = rope.disabled_reason()
    assert reason is not None and "layer 7" in reason and "not bit-exact" in reason
    assert rope.counters()["probe_failures"] == 1
    assert warnings, "an exactness failure must be logged, not swallowed"
    report = rope.engagement()
    assert report["installed"] is False
    assert report["disabled_reason"] == reason


def test_probe_pass_installs_and_is_idempotent(monkeypatch):
    class _Same:
        def item(self):
            return True

    monkeypatch.setattr(
        rope, "stock_reference", lambda q, k, f, **kw: (q, k)
    )
    monkeypatch.setattr(rope, "rope_qk", lambda q, k, f, **kw: (q, k))
    monkeypatch.setattr(rope.mx, "array_equal", lambda a, b: _Same())
    monkeypatch.setattr(rope.mx, "eval", lambda *a, **k: None)

    layers = ((0, _FakeAttention()), (4, _FakeAttention()))
    assert rope.install(layers) is True
    assert rope.installed() is True
    runs = rope.counters()["probe_runs"]
    assert runs == 4  # two layers x two pos_start cells
    # A second install must not re-probe: the verdict is per process.
    assert rope.install(layers) is True
    assert rope.counters()["probe_runs"] == runs


def test_probe_runs_once_per_geometry_not_once_per_layer(monkeypatch):
    """The kernel reads no per-layer parameter, so 12 identical layers are
    one probe -- unlike the route kernel, where every layer owns weights."""

    class _Same:
        def item(self):
            return True

    monkeypatch.setattr(rope, "stock_reference", lambda q, k, f, **kw: (q, k))
    monkeypatch.setattr(rope, "rope_qk", lambda q, k, f, **kw: (q, k))
    monkeypatch.setattr(rope.mx, "array_equal", lambda a, b: _Same())
    monkeypatch.setattr(rope.mx, "eval", lambda *a, **k: None)

    shared_inv_freq = mx.zeros((32,), dtype=mx.float32)
    layers = []
    for index in range(0, 48, 4):
        attention = _FakeAttention()
        attention._inv_freq = shared_inv_freq
        layers.append((index, attention))

    assert rope.install(tuple(layers)) is True
    # Twelve layers, one signature, two pos_start cells.
    assert rope.counters()["probe_runs"] == 2
    # ...but the CONTRACT is checked on every one of them.
    assert rope.counters()["contract_checks"] == 12


def test_engagement_line_reports_the_counters_and_the_off_reason(monkeypatch):
    off = rope.engagement_line(layers=12, enabled=False)
    assert off == "[PR #391] verify-glue qsa_rope: off"

    on = rope.engagement_line(layers=12, enabled=True)
    assert "layers=12" in on
    assert "dispatches/layer 16->1" in on
    assert "dependent_levels/layer 7->1" in on
    assert "qk_calls=0" in on
    assert "probe_failures=0" in on


def test_node_accounting_matches_the_engagement_line():
    assert rope.dispatches_removed_per_layer() == 15
    assert rope.dispatches_removed_per_layer(with_positions=False) == 13
    # Twelve QSA layers per verify cycle.
    assert 12 * rope.dispatches_removed_per_layer() == 180


def test_rows_narrowing_is_decode_and_verify_only():
    from mtplx import qwen4_verify_glue as glue

    assert glue.serves_rows(1) is True
    assert glue.serves_rows(4) is True
    assert glue.serves_rows(8) is True
    assert glue.serves_rows(9) is False
    assert glue.serves_rows(16384) is False


# --------------------------------------------------------------------------
# The microbench's CLI and arm table.  ``micro_verify_glue_a`` imports MLX
# lazily, so everything below runs without touching the device.
# --------------------------------------------------------------------------
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

MICRO_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "PR #391"
    / "micro_verify_glue_a.py"
)


