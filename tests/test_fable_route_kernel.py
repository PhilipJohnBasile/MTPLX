"""MTPLX_FABLE_ROUTE_KERNEL — the routing head's op contract, gate, and refusals.

Everything here runs on the CPU stream with toy tensors: no Metal, no kernel
dispatch, no model.  That bounds what can be proved, and the boundary matters:

  * The KERNEL cannot be executed here, so nothing below claims the Metal
    source is correct.  What is proved is the *contract the source is written
    against* — every MLX rounding and ordering boundary the two kernels
    transcribe.  If MLX changes one (bf16 sum widening to fp32, the sigmoid
    decomposition, argsort losing stability), these fail instead of the kernel
    silently routing to different experts.
  * The tie rule is the METAL rule.  ``mx.argpartition`` on the CPU backend is
    an unstable partial selection: same top-k SET, different slot ORDER.  On
    Metal ``argpartition`` and ``argsort`` are literally the same kernel
    (``carg_block_sort``), so ``argsort`` is the only CPU-side spelling of what
    the verifier actually does.  Both halves of that are asserted, so nobody
    "simplifies" the reference back to ``argpartition``.
  * The gate: flag off, nothing changes; flag on with the wrong pack, geometry
    or companion flag, it RAISES with the offending field named.  The one
    check that needs a GPU — the per-layer ``mx.array_equal`` of the whole
    emitted tuple against the stock scaffold — lives in
    ``install_qwen4_m4_stage3`` and runs at model build.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

import mtplx.kernels.qwen4_m4_route as route_kernel
import mtplx.qwen4_m4_stage3 as stage3

NUM_EXPERTS = route_kernel.NUM_EXPERTS
TOP_K = route_kernel.TOP_K
HIDDEN = route_kernel.HIDDEN
ROWS = route_kernel.ROWS


@pytest.fixture(autouse=True)
def _cpu_stream():
    """Confine every op in this module to the CPU stream."""

    with mx.stream(mx.cpu):
        yield


@pytest.fixture(autouse=True)
def _gates_off(monkeypatch):
    """Default every test to the shipped state, whatever the session env is."""

    monkeypatch.setattr(stage3, "_ROUTE_KERNEL_CACHE", False)
    monkeypatch.setattr(stage3, "_ROUTE_KERNEL_VEC_LANES_CACHE", None)


# --------------------------------------------------------------------------
# toy fixtures
# --------------------------------------------------------------------------


def _q8_linear(in_dims: int, out_dims: int, seed: int):
    """A q8/g64 affine projection shaped like the router / shared gate."""

    mx.random.seed(seed)
    layer = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=64, bits=8)
    layer.scales = layer.scales.astype(mx.bfloat16)
    layer.biases = layer.biases.astype(mx.bfloat16)
    return layer


class _ToyBlock:
    """The four attributes ``stock_route``/``check_contract`` actually read."""

    def __init__(self, *, experts: int = NUM_EXPERTS, top_k: int = TOP_K):
        self.num_experts = experts
        self.top_k = top_k
        self.norm_topk_prob = True
        self.gate = _q8_linear(HIDDEN, experts, seed=11)
        self.shared_expert_gate = _q8_linear(HIDDEN, 1, seed=12)


@pytest.fixture(scope="module")
def toy_block():
    with mx.stream(mx.cpu):
        return _ToyBlock()


@pytest.fixture(scope="module")
def toy_x():
    with mx.stream(mx.cpu):
        mx.random.seed(7)
        return mx.random.normal((1, ROWS, HIDDEN)).astype(mx.bfloat16)


def _bits(value: mx.array) -> mx.array:
    widths = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32}
    return value.view(widths[value.dtype])


def _same_bits(a: mx.array, b: mx.array) -> bool:
    return bool(mx.all(_bits(a) == _bits(b)).item())


# --------------------------------------------------------------------------
# 1. the tie rule the Metal top-10 encodes
# --------------------------------------------------------------------------


def _sort_key_topk(gates: mx.array, k: int = TOP_K) -> np.ndarray:
    """The kernel's rule, in numpy: the k greatest under the uint32 key
    ``(order_preserving_u16(value) << 16) | index``, emitted ascending.

    This is a model of ``route_sort_key`` + the ten max-extract rounds in
    ``_TAIL_SOURCE``, written independently of ``mx.argsort`` so the assertion
    that they agree has content.
    """

    u = np.asarray(_bits(gates.astype(mx.bfloat16)))
    key = np.where(u & 0x8000, (~u) & 0xFFFF, u | 0x8000).astype(np.uint32)
    idx = np.broadcast_to(
        np.arange(gates.shape[-1], dtype=np.uint32), gates.shape
    )
    packed = (key << np.uint32(16)) | idx
    order = np.argsort(packed, axis=-1, kind="stable")
    return (packed[np.arange(packed.shape[0])[:, None], order[:, -k:]] & 0xFFFF)


def test_metal_topk_rule_is_argsort_last_k_on_random_gates():
    mx.random.seed(3)
    logits = (mx.random.normal((8, NUM_EXPERTS)) * 3).astype(mx.bfloat16)
    gates = mx.softmax(logits, axis=-1, precise=True)
    want = np.asarray(mx.argsort(gates, axis=-1)[..., -TOP_K:])
    assert np.array_equal(_sort_key_topk(gates), want.astype(np.uint32))


def test_metal_topk_rule_holds_on_constructed_exact_ties():
    """Every gate identical: the rule must keep the TEN HIGHEST indices, in
    ascending index order.  This is the case a value-only selector gets wrong.
    """

    gates = mx.full((2, NUM_EXPERTS), 0.5, dtype=mx.bfloat16)
    got = _sort_key_topk(gates)
    want = np.arange(NUM_EXPERTS - TOP_K, NUM_EXPERTS, dtype=np.uint32)
    assert np.array_equal(got[0], want)
    assert np.array_equal(np.asarray(mx.argsort(gates, axis=-1)[0, -TOP_K:]), want)


def test_metal_topk_rule_holds_on_a_tie_straddling_the_boundary():
    """A tie run that spans the top-10 cut: the kept half is the HIGH indices,
    and the dropped half the low ones."""

    values = np.zeros((1, NUM_EXPERTS), dtype=np.float32)
    values[0, :6] = 0.75  # six strictly-largest, always kept
    values[0, 100:110] = 0.5  # a ten-wide tie run; only four survive the cut
    gates = mx.array(values).astype(mx.bfloat16)
    got = _sort_key_topk(gates)
    want = np.asarray(mx.argsort(gates, axis=-1)[..., -TOP_K:]).astype(np.uint32)
    assert np.array_equal(got, want)
    # slots 0..3 are the tie's four HIGHEST indices, ascending.
    assert list(got[0, :4]) == [106, 107, 108, 109]


def test_argsort_is_stable_ascending_on_exact_ties():
    """sort.h's BlockMergeSort is stable; ``mx.argsort`` is the CPU spelling of
    the same rule.  If this ever fails the kernel's tie-break is wrong."""

    values = np.concatenate(
        [np.full(8, 1.0, np.float32), np.full(8, 2.0, np.float32)]
    )[None, :]
    order = np.asarray(mx.argsort(mx.array(values).astype(mx.bfloat16), axis=-1))
    assert list(order[0]) == list(range(16))


def test_cpu_argpartition_agrees_on_the_set_but_not_the_slot_order():
    """The reason the reference is argsort and not argpartition.

    On Metal both dispatch ``carg_block_sort`` and are identical.  On CPU
    argpartition is ``nth_element``-class: the ten indices are the same ten,
    but their slot order is not the ascending order the retained gather and
    reduction kernels are validated against.
    """

    mx.random.seed(5)
    gates = mx.softmax(
        (mx.random.normal((8, NUM_EXPERTS)) * 3).astype(mx.bfloat16),
        axis=-1,
        precise=True,
    )
    partitioned = mx.argpartition(gates, kth=-TOP_K, axis=-1)[..., -TOP_K:]
    sorted_ids = mx.argsort(gates, axis=-1)[..., -TOP_K:]
    assert partitioned.dtype == mx.uint32 and sorted_ids.dtype == mx.uint32
    a = np.sort(np.asarray(partitioned), axis=-1)
    b = np.sort(np.asarray(sorted_ids), axis=-1)
    assert np.array_equal(a, b), "top-k SET must agree"
    assert not bool(mx.array_equal(partitioned, sorted_ids).item()), (
        "if CPU argpartition ever became order-identical to argsort this test "
        "should be revisited, not deleted: the kernel targets the Metal rule"
    )


# --------------------------------------------------------------------------
# 2. the rounding boundaries the Metal source copies
# --------------------------------------------------------------------------


def test_topk_sum_accumulates_in_bf16_in_slot_order():
    """``row_reduce_small_1_reduce_sumbfloat16`` instantiates T == U == bf16."""

    mx.random.seed(9)
    a = (mx.random.normal((256, TOP_K)) * 0.1).astype(mx.bfloat16)
    sequential = a[:, 0]
    for i in range(1, TOP_K):
        sequential = (sequential + a[:, i]).astype(mx.bfloat16)
    assert _same_bits(sequential, mx.sum(a, axis=-1))
    widened = mx.sum(a.astype(mx.float32), axis=-1).astype(mx.bfloat16)
    assert not _same_bits(widened, mx.sum(a, axis=-1)), (
        "an fp32 accumulation is NOT the same tensor; the kernel's bf16 loop "
        "is not cosmetic"
    )


def test_renormalize_divide_rounds_once():
    """``Divide`` at T = bf16 widens for the divide and rounds on the store."""

    mx.random.seed(13)
    num = (mx.random.uniform(shape=(512,)) + 0.05).astype(mx.bfloat16)
    den = (mx.random.uniform(shape=(512,)) + 1.0).astype(mx.bfloat16)
    once = (num.astype(mx.float32) / den.astype(mx.float32)).astype(mx.bfloat16)
    assert _same_bits(num / den, once)


def test_mlx_sigmoid_is_the_bf16_exp_decomposition():
    """``unary_ops.h`` Sigmoid at T = bf16 rounds ``exp`` to bf16 BEFORE the
    reciprocal.  The shared-gate factor is that value, not bf16(fp32 sigmoid).
    """

    mx.random.seed(17)
    x = (mx.random.normal((4096,)) * 4).astype(mx.bfloat16)
    xf = x.astype(mx.float32)
    e = mx.exp(mx.abs(xf)).astype(mx.bfloat16).astype(mx.float32)
    y = 1.0 / (1.0 + e)
    decomposed = mx.where(xf < 0, y, 1.0 - y).astype(mx.bfloat16)
    assert _same_bits(decomposed, mx.sigmoid(x))
    naive = mx.sigmoid(xf).astype(mx.bfloat16)
    disagreement = float(
        mx.mean((_bits(naive) != _bits(mx.sigmoid(x))).astype(mx.float32)).item()
    )
    assert disagreement > 0.05, (
        "a once-rounded fp32 sigmoid must visibly disagree, otherwise this "
        "contract has stopped being load-bearing"
    )


def test_precise_softmax_returns_bf16(toy_block, toy_x):
    gates = mx.softmax(toy_block.gate(toy_x), axis=-1, precise=True)
    assert gates.dtype == mx.bfloat16
    assert gates.shape == (1, ROWS, NUM_EXPERTS)


# --------------------------------------------------------------------------
# 3. the emitted tuple
# --------------------------------------------------------------------------


def test_stock_route_emits_the_downstream_tuple(toy_block, toy_x):
    expert_ids, route_scores, shared_factor = route_kernel.stock_route(
        toy_block, toy_x
    )
    assert expert_ids.shape == (ROWS, TOP_K) and expert_ids.dtype == mx.uint32
    assert route_scores.shape == (ROWS, TOP_K)
    assert route_scores.dtype == mx.bfloat16
    assert shared_factor.shape == (ROWS,) and shared_factor.dtype == mx.bfloat16


def test_metal_order_reference_selects_the_same_experts_as_the_stock_head(
    toy_block, toy_x
):
    """Same ten experts per row.  Only the SET is comparable across backends:
    ``argpartition`` and ``argsort`` are the same kernel on Metal, but on CPU
    they emit the ten in different slots -- and the next test shows the slot
    order changes the scores, not just their arrangement."""

    stock = route_kernel.stock_route(toy_block, toy_x)
    metal = route_kernel.metal_order_reference(toy_block, toy_x)
    assert metal[0].dtype == mx.uint32
    assert np.array_equal(
        np.sort(np.asarray(stock[0]), axis=-1),
        np.sort(np.asarray(metal[0]), axis=-1),
    )
    assert _same_bits(stock[2], metal[2])


def test_slot_order_changes_the_renormalised_scores(toy_block, toy_x):
    """Why reproducing argsort's ORDER matters, not only its set.

    ``route_scores.sum(-1)`` accumulates in bf16 and rounds at every add, so a
    permutation of the same ten gates gives a different denominator and a
    different bf16 quotient.  A kernel that got the right ten experts in the
    wrong slots would therefore also emit different weights -- and feed the
    retained routed-down kernel a reduction tree it was not validated against.
    """

    gates = mx.softmax(toy_block.gate(toy_x), axis=-1, precise=True).reshape(
        ROWS, NUM_EXPERTS
    )
    ids = mx.argsort(gates, axis=-1)[..., -TOP_K:]
    picked = mx.take_along_axis(gates, ids, axis=-1)
    ascending = picked / picked.sum(axis=-1, keepdims=True)
    reversed_ = picked[:, ::-1]
    descending = (reversed_ / reversed_.sum(axis=-1, keepdims=True))[:, ::-1]
    assert not _same_bits(ascending, descending)


def test_metal_order_reference_scores_are_ascending(toy_block, toy_x):
    """Slot 9 is the largest: the retained kernels' fixed SLOT_ORDER reduction
    tree is validated against that layout."""

    _, route_scores, _ = route_kernel.metal_order_reference(toy_block, toy_x)
    scores = np.asarray(route_scores.astype(mx.float32))
    assert np.all(np.diff(scores, axis=-1) >= 0)


# --------------------------------------------------------------------------
# 4. geometry invariants
# --------------------------------------------------------------------------


def test_k_lanes_is_pinned_by_the_traced_threadgroup_shape():
    """``qmv_wide_impl`` fixes num_simdgroups = 2 and
    results_per_simdgroup = 32 / k_lanes; the trace's ``[1,64,1]`` grid over
    512 rows is 8 rows per threadgroup.  That leaves exactly one k_lanes, and
    k_lanes is what fixes which groups each lane accumulates."""

    assert 2 * (32 // route_kernel.K_LANES) == route_kernel.ROWS_PER_TG
    assert NUM_EXPERTS // route_kernel.ROWS_PER_TG == 64


@pytest.mark.parametrize("vec_lanes", route_kernel.VEC_LANES_CHOICES)
def test_logits_geometry_covers_every_output_row(vec_lanes):
    grid, threadgroup = route_kernel.logits_geometry(vec_lanes)
    threads = threadgroup[0]
    assert threads == route_kernel.ROWS_PER_TG * route_kernel.K_LANES * vec_lanes
    assert threads % 32 == 0
    blocks = grid[0] // threads
    assert grid[0] % threads == 0
    assert blocks * route_kernel.ROWS_PER_TG >= route_kernel.NTOT
    assert (blocks - 1) * route_kernel.ROWS_PER_TG < route_kernel.NTOT
    # Every lane octet must sit inside one simdgroup for the shuffle ladder.
    assert (route_kernel.K_LANES * vec_lanes) in (8, 32)


def test_tail_geometry_matches_the_softmax_kernel_it_replaces():
    grid, threadgroup = route_kernel.tail_geometry()
    assert threadgroup == (route_kernel.TAIL_THREADS, 1, 1)
    assert threadgroup[0] * route_kernel.SOFTMAX_N_READS == NUM_EXPERTS
    assert grid == (route_kernel.TAIL_THREADS * ROWS, 1, 1)


def test_logits_geometry_rejects_an_unbuilt_vec_lane_count():
    with pytest.raises(ValueError, match="vec_lanes"):
        route_kernel.logits_geometry(2)
    with pytest.raises(ValueError, match="vec_lanes"):
        route_kernel.bind(vec_lanes=3)


def test_router_bytes_match_the_census_line():
    """The census's ``router 1.393 MB`` per layer, plus the folded gate."""

    assert route_kernel.router_bytes_per_layer() == 1_392_640 + 2_720


def test_dispatch_accounting_is_ten_to_two():
    assert route_kernel.EAGER_DISPATCHES_PER_LAYER == 10
    assert route_kernel.FUSED_DISPATCHES_PER_LAYER == 2


# --------------------------------------------------------------------------
# 5. the Metal source's load-bearing spellings (tripwires)
#
# None of these can be executed here, and every one of them is a boundary a
# well-meaning cleanup would move.  Pin the text.
# --------------------------------------------------------------------------


def test_logits_source_keeps_the_shuffle_ladder_not_simd_sum():
    src = route_kernel.logits_source()
    for offset in (4, 2, 1):
        assert f"simd_shuffle_down(result[j], {offset})" in src
    assert "simd_sum(" not in src, (
        "qmv_wide_impl deliberately does NOT use simd_sum: a simdgroup spans "
        "four output rows"
    )


def test_logits_source_indexes_off_the_simdgroup_attributes():
    """A linear-thread-id derivation would only accidentally put the eight
    k-lanes of one output row in one simdgroup; the shuffle ladder needs that
    guaranteed."""

    src = route_kernel.logits_source()
    assert "const uint sgid = simdgroup_index_in_threadgroup;" in src
    assert "const uint slid = thread_index_in_simdgroup;" in src
    assert "const int rem = int(slid) % RPR;" in src


@pytest.mark.parametrize("vec_lanes", route_kernel.VEC_LANES_CHOICES)
def test_lane_octets_tile_a_simdgroup_exactly(vec_lanes):
    threads_per_row = route_kernel.K_LANES * vec_lanes
    assert 32 % threads_per_row == 0, "an output row must not straddle a simdgroup"
    rows_per_simd = 32 // threads_per_row
    assert route_kernel.ROWS_PER_TG % rows_per_simd == 0


def test_logits_source_keeps_the_q8_dequantize_expression():
    src = route_kernel.logits_source()
    assert "w_dq[i] = float(scale * wc[i] + bias);" in src
    assert "acc += float(xc[i]) * w_dq[i];" in src


def test_tail_source_uses_fast_exp_in_the_softmax():
    src = route_kernel.tail_source()
    assert "fast::exp(ld[i] - maxval)" in src, (
        "softmax.h's softmax_exp is fast::exp; the precise flavour is a "
        "different function"
    )


def test_tail_source_uses_the_mlx_bf16_sigmoid_decomposition():
    src = route_kernel.tail_source()
    assert "1 / (1 + metal::exp(metal::abs(sx)))" in src
    assert "bfloat(1 - sigmoid_y)" in src


def test_tail_source_accumulates_the_renormaliser_in_bf16():
    src = route_kernel.tail_source()
    assert "bfloat total = bfloat(0.0f);" in src
    assert "total = picked[j] + total;" in src


def test_tail_source_writes_the_top_k_slots_in_ascending_order():
    src = route_kernel.tail_source()
    assert "selected[TTOPK - 1 - s] = winner;" in src


# --------------------------------------------------------------------------
# 6. construction-bound refusals
# --------------------------------------------------------------------------


def test_contract_accepts_the_shipped_pack(toy_block):
    route_kernel.check_contract(toy_block, index=0)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("num_experts", 128, "512 experts"),
        ("top_k", 8, "top-k 10"),
        ("norm_topk_prob", False, "normalization"),
    ),
)
def test_contract_refuses_a_wrong_block_shape(toy_block, field, value, match):
    block = _ToyBlock.__new__(_ToyBlock)
    block.__dict__.update(toy_block.__dict__)
    setattr(block, field, value)
    with pytest.raises(ValueError, match=match):
        route_kernel.check_contract(block, index=3)


@pytest.mark.parametrize(
    ("attr", "value", "match"),
    (
        ("bits", 4, "bits=4"),
        ("group_size", 32, "group_size=32"),
        ("mode", "mxfp4", "mode="),
    ),
)
def test_contract_refuses_a_wrong_router_quantization(toy_block, attr, value, match):
    block = _ToyBlock.__new__(_ToyBlock)
    block.__dict__.update(toy_block.__dict__)
    gate = _q8_linear(HIDDEN, NUM_EXPERTS, seed=11)
    setattr(gate, attr, value)
    block.gate = gate
    with pytest.raises(ValueError, match=match):
        route_kernel.check_contract(block, index=1)


def test_contract_refuses_fp32_scales(toy_block):
    block = _ToyBlock.__new__(_ToyBlock)
    block.__dict__.update(toy_block.__dict__)
    gate = _q8_linear(HIDDEN, NUM_EXPERTS, seed=11)
    gate.scales = gate.scales.astype(mx.float32)
    block.gate = gate
    with pytest.raises(ValueError, match=r"gate\.scales dtype"):
        route_kernel.check_contract(block, index=2)


def test_contract_refuses_a_wrong_shared_gate_width(toy_block):
    block = _ToyBlock.__new__(_ToyBlock)
    block.__dict__.update(toy_block.__dict__)
    block.shared_expert_gate = _q8_linear(HIDDEN, 2, seed=12)
    with pytest.raises(ValueError, match=r"shared_expert_gate\.weight shape"):
        route_kernel.check_contract(block, index=4)


def test_contract_refuses_an_unquantized_router(toy_block):
    block = _ToyBlock.__new__(_ToyBlock)
    block.__dict__.update(toy_block.__dict__)
    dense = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
    dense.bits = 8
    dense.group_size = 64
    dense.mode = "affine"
    block.gate = dense
    with pytest.raises(ValueError, match="not an affine-quantized"):
        route_kernel.check_contract(block, index=5)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    (
        ((2, HIDDEN), mx.bfloat16),
        ((ROWS, 1024), mx.bfloat16),
        ((ROWS, HIDDEN), mx.float32),
    ),
)
def test_check_input_refuses_a_shape_or_dtype_the_kernel_cannot_serve(shape, dtype):
    with pytest.raises(ValueError, match="MTPLX_FABLE_ROUTE_KERNEL"):
        route_kernel.check_input(mx.zeros(shape, dtype=dtype))


def test_check_input_accepts_the_verify_width():
    route_kernel.check_input(mx.zeros((ROWS, HIDDEN), dtype=mx.bfloat16))
    route_kernel.check_input(mx.zeros((1, ROWS, HIDDEN), dtype=mx.bfloat16))


# --------------------------------------------------------------------------
# 7. the env gate
# --------------------------------------------------------------------------


def test_gate_defaults_off(monkeypatch):
    monkeypatch.setattr(stage3, "_ROUTE_KERNEL_CACHE", None)
    monkeypatch.delenv(stage3.FABLE_ROUTE_KERNEL_ENV, raising=False)
    assert stage3.fable_route_kernel_enabled() is False


def test_gate_reads_the_environment_once(monkeypatch):
    monkeypatch.setattr(stage3, "_ROUTE_KERNEL_CACHE", None)
    monkeypatch.setenv(stage3.FABLE_ROUTE_KERNEL_ENV, "1")
    assert stage3.fable_route_kernel_enabled() is True
    monkeypatch.delenv(stage3.FABLE_ROUTE_KERNEL_ENV)
    assert stage3.fable_route_kernel_enabled() is True  # memoized
    stage3.reset_fable_route_kernel_cache()
    assert stage3.fable_route_kernel_enabled() is False


def test_vec_lanes_defaults_and_validates(monkeypatch):
    monkeypatch.setattr(stage3, "_ROUTE_KERNEL_VEC_LANES_CACHE", None)
    monkeypatch.delenv(stage3.FABLE_ROUTE_KERNEL_VEC_LANES_ENV, raising=False)
    assert stage3.fable_route_kernel_vec_lanes() == route_kernel.DEFAULT_VEC_LANES

    for raw, match in (("2", "is not one of"), ("wide", "not an integer")):
        stage3.reset_fable_route_kernel_cache()
        monkeypatch.setenv(stage3.FABLE_ROUTE_KERNEL_VEC_LANES_ENV, raw)
        with pytest.raises(ValueError, match=match):
            stage3.fable_route_kernel_vec_lanes()

    stage3.reset_fable_route_kernel_cache()
    monkeypatch.setenv(stage3.FABLE_ROUTE_KERNEL_VEC_LANES_ENV, "1")
    assert stage3.fable_route_kernel_vec_lanes() == 1


def test_route_kernel_requires_the_paired_routed_glu_lane():
    with pytest.raises(ValueError, match="requires MTPLX_QWEN4_M4_ROUTED_GLU"):
        stage3._validate_feature_combination(
            routed_down_reduce_enabled=True,
            routed_down_residual_tail_enabled=True,
            routed_glu_enabled=False,
            route_kernel_enabled=True,
        )


def test_route_kernel_requires_stage3(monkeypatch):
    monkeypatch.setattr(stage3, "_ROUTE_KERNEL_CACHE", True)
    monkeypatch.setattr(stage3, "_ROUTE_KERNEL_VEC_LANES_CACHE", 4)
    monkeypatch.setattr(stage3, "qwen4_m4_stage3_enabled", lambda: False)
    monkeypatch.setattr(stage3, "qwen4_m4_routed_down_reduce_enabled", lambda: False)
    monkeypatch.setattr(
        stage3, "qwen4_m4_routed_down_residual_tail_enabled", lambda: False
    )
    monkeypatch.setattr(stage3, "qwen4_m4_routed_glu_enabled", lambda: False)
    with pytest.raises(ValueError, match="child routes require M4 stage3"):
        stage3.qwen4_m4_stage3_flags()


def test_flags_capture_a_bad_vec_lane_value_before_any_weight(monkeypatch):
    monkeypatch.setattr(stage3, "_ROUTE_KERNEL_CACHE", True)
    monkeypatch.setattr(stage3, "_ROUTE_KERNEL_VEC_LANES_CACHE", None)
    monkeypatch.setenv(stage3.FABLE_ROUTE_KERNEL_VEC_LANES_ENV, "16")
    monkeypatch.setattr(stage3, "qwen4_m4_stage3_enabled", lambda: True)
    monkeypatch.setattr(stage3, "qwen4_m4_routed_down_reduce_enabled", lambda: True)
    monkeypatch.setattr(
        stage3, "qwen4_m4_routed_down_residual_tail_enabled", lambda: True
    )
    monkeypatch.setattr(stage3, "qwen4_m4_routed_glu_enabled", lambda: True)
    with pytest.raises(ValueError, match="is not one of"):
        stage3.qwen4_m4_stage3_flags()


# --------------------------------------------------------------------------
# 8. the receipt
# --------------------------------------------------------------------------


def test_report_carries_the_route_kernel_receipt():
    on = stage3._installation_report(
        layer_count=48,
        max_delta=0.0,
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
        routed_glu_enabled=True,
        route_kernel_enabled=True,
        route_kernel_vec_lanes=4,
    )
    assert on["route_kernel"] == {
        "installed": True,
        "layers": 48,
        "vec_lanes": 4,
        "dispatches_per_layer": 2,
    }
    off = stage3._installation_report(
        layer_count=48,
        max_delta=0.0,
        routed_down_reduce_enabled=True,
        routed_down_residual_tail_enabled=True,
        routed_glu_enabled=True,
    )
    assert off["route_kernel"]["installed"] is False
    assert off["route_kernel"]["layers"] == 0
    assert off["route_kernel"]["dispatches_per_layer"] == 10


def test_paired_forward_defaults_to_the_stock_head():
    """``route=None`` is the shipped path: the forward must still be the ten
    dispatches, not a half-installed hybrid."""

    import inspect

    sig = inspect.signature(
        stage3._m4_paired_routed_glu_residual_tail_forward
    )
    assert sig.parameters["route"].default is None


def test_the_forward_gate_is_install_bound_not_request_bound():
    """MTPLX_FABLE_ROUTE_KERNEL is decided once, when the layer is installed.

    `install_qwen4_m4_stage3` validates the kernel bit-exact per layer and
    binds it; the forward then branches on whether a `route` callable exists,
    never on anything about the request. So greedy and temperature-1 requests
    take the identical path, and there is no per-request check that could
    raise. A pack the kernel cannot serve fails at install, which is where a
    deployment error belongs.
    """

    import ast
    import inspect

    from mtplx import qwen4_m4_stage3

    source = inspect.getsource(qwen4_m4_stage3)
    tree = ast.parse(source)
    gates = [
        ast.unparse(node.test)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "route is None"
    ]
    assert gates, "the route-kernel forward gate moved"
    forward = inspect.getsource(qwen4_m4_stage3)
    for request_term in ("sampler", "temperature", "draft_sampler"):
        assert f"if {request_term}" not in forward
