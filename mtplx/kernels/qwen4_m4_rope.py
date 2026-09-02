"""One-dispatch partial RoPE for the Qwen3.8 Flash-Next fixed-M4 verifier.

WHY THIS EXISTS
---------------
W69's node census of the compiled fixed-M4 verify body found ``qsa_rope`` --
the rotary embedding applied inside the twelve QSA layers -- to be the largest
fusable glue group in the graph.  On today's stack
(``MTPLX_FABLE_OPDIET=1`` with the ``rope`` item, ``MTPLX_FABLE_QSA_M4``
**off**) one QSA layer issues its rotation as a chain of small elementwise and
copy dispatches:

    positions = pos_start + arange(S)                       arange, add
    angles    = positions.astype(f32)[:,None] * inv_freq    2 fused elementwise
    cos, sin  = mx.cos(angles), mx.sin(angles)              2 (siblings)
    lo, hi    = the two half-width multiply-adds            2 per rotated tensor
    out       = concatenate([lo, hi, pass])                 3 copies per tensor

``Attention.__call__`` rotates TWO tensors with that one table -- the query
``[1, S, 24, 256]`` and the key ``[1, S, 2, 256]`` -- so the attention half of
a QSA layer is 4 table dispatches + 2 x 5 rotation dispatches (+ the 2 that
build ``positions``), all of it zero-byte work around 6 k of activations.

This module issues the same arithmetic as ONE dispatch that rotates both
tensors: 14-16 dispatches and ~7 dependent levels become 1 and 1.

WHAT IT DOES NOT DO
-------------------
* **No RMSNorm.**  ``qsa_indexer_prepare.py`` fuses norm + rope because the
  indexer's ``head_dim`` is 128 and MLX's ``rms_single_row`` reduction (32
  lanes x 4 contiguous values) is reproducible exactly only up to 128.  The
  attention head_dim here is **256**, so ``q_norm``/``k_norm`` stay outside.
* **No M-RoPE.**  The vision (t, h, w) path keeps the stock chain; the caller
  routes around this kernel when ``vision_rope_state()`` is live.
* **No indexer query prep and no pooled bank row.**  Those two rope sites of
  the same QSA layer already have shipped, pinned-bit-exact kernels
  (``qsa_indexer_prepare_queries_metal`` and ``qsa_m4_pooled_row``); see
  ``MTPLX_FABLE_VERIFY_GLUE_ITEMS=qsa_rope_idx`` and ``MTPLX_FABLE_QSA_M4``.

NUMERICS
--------
The target is the LIVE stock expression, which is
``_rope_cos_sin_half`` + ``_apply_partial_rope_half`` when the op diet's
``rope`` item is armed and ``_rope_cos_sin`` + ``_apply_partial_rope``
otherwise.  The two are bitwise-identical to each other by construction
(``mtplx/models/qwen4_exp.py``), and this kernel reproduces either:

    theta   = (float)(int32(pos_start) + row) * (float)inv_freq[pair]
    cosine  = precise::cos(theta) * ROPE_ATTENTION_SCALE
    sine    = precise::sin(theta) * ROPE_ATTENTION_SCALE
    lo      = (T)(first * cosine  - second * sine)
    hi      = (T)(second * cosine + first  * sine)
    pass    = the input value, copied

``mx.cos``/``mx.sin`` lower to the Metal *precise* variants (unlike
``mx.fast.rope``, which deliberately uses fast trig), so this kernel uses
them too.  ``lo`` matches the stock ``x1*cos + (-x2)*sin`` bit for bit because
IEEE negation is exact: ``(-x2)*sin == -(x2*sin)`` and ``a + (-b) == a - b``.
The four products are kept as distinct fp32 locals for the same reason
``qsa_indexer_prepare.py`` does: letting Metal contract ``a*b - c*d`` into an
FMA changes a handful of bf16 cutoff values.

That is the same construction whose bit-exactness against the eager chain is
pinned in ``tests/test_qsa_indexer_prepare_metal.py``.  It is NOT a proof
against the *compiled* body, where MLX fuses the multiply-add chain into a
kernel of its own whose contraction behaviour we do not control.  So the
install probe in this module is ``mx.array_equal`` against the live eager
expression and the microbench (``scripts/fable/micro_verify_glue_a.py``)
re-checks against the compiled one; a probe miss DISABLES the item and logs
rather than raising, because rope rounding is a numerical verdict, while a
contract miss (geometry, dtype, rotary width) always RAISES.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, Optional

import mlx.core as mx

#: Rows this lane is wired for (the physical-M4 verify window).  A narrowing,
#: not a contract: the S=1 draft route keeps the stock chain.
VERIFY_ROWS = 4

#: Head dimension above which the fused-norm kernels stop being exact.  Quoted
#: here only to record WHY the norm is not folded in.
MAX_EXACT_NORM_HEAD_DIM = 128

#: What one QSA layer's attention rope costs today, and after.  Used by the
#: engagement line and by the microbench's accounting.
#:
#: Derived from W69's census of the COMPILED body (``qsa_rope``, 288 matched
#: dispatches over 12 QSA layers = 24/layer for three table builds and four
#: three-copy concatenates, plus the four applications' eight fused
#: multiply-adds, which the census counts in "elementwise, fused" rather than
#: in the group).  The attention share of that is one table (4) plus two
#: applications (2 x [2 fused multiply-adds + 3 concatenate copies]) = 14, or
#: 16 counting the ``arange`` + ``add`` that build ``positions``.
#:
#: For scale, the same two rotations built on the HOST and counted off the
#: unevaluated tape in the EAGER lane are 61 dispatches (49 pre-op-diet)
#: against the fused kernel's 1.  The eager lane is not the number to quote:
#: ``mx.compile`` collapses each half's multiply-add chain into one kernel,
#: which is why the compiled census says 14-16 and not 61.  It is quoted here
#: only as the upper bound the microbench's ``disp`` column reproduces.
STOCK_DISPATCHES_PER_LAYER = 14
STOCK_DISPATCHES_PER_LAYER_WITH_POSITIONS = 16
FUSED_DISPATCHES_PER_LAYER = 1
#: Read-after-write levels the stock chain has (positions -> astype/broadcast
#: -> multiply -> {cos,sin} -> {lo,hi} -> concat copies).  The fused kernel is
#: one level.  This -- not the dispatch count -- is what the critical-path
#: model charges 1.83 us each.
STOCK_DEPENDENT_LEVELS_PER_LAYER = 7
FUSED_DEPENDENT_LEVELS_PER_LAYER = 1

_SIMD = 32

_COUNTS: Dict[str, int] = {
    "contract_checks": 0,
    "probe_runs": 0,
    "probe_failures": 0,
    "qk_calls": 0,
    "k_only_calls": 0,
}

#: ``None`` until the probe has run, ``""`` once it passed, otherwise the
#: reason this lane is off for the process.
_DISABLED_REASON: Optional[str] = None
_PROBE_REPORT: Dict[str, Any] = {}


class RopeGlueContractError(RuntimeError):
    """The lane was armed against a geometry it is not contracted for."""


def counters() -> Dict[str, int]:
    """A copy of the engagement counters."""

    return dict(_COUNTS)


def engagement() -> Dict[str, Any]:
    """Snapshot of counters plus the install verdict.

    ``qk_calls``/``k_only_calls`` are the ENGAGEMENT LINE: an ABBA that
    reports a win with these at zero measured something else.

    They count TRACES, not decode cycles: under ``mx.compile`` the Python
    body runs once per retrace and the C++ replay never touches it.  Read
    them as "did this lane get into the graph at all", never as a cycle
    count.
    """

    report = dict(_COUNTS)
    report["installed"] = _DISABLED_REASON == ""
    report["disabled_reason"] = _DISABLED_REASON or None
    report["probe"] = dict(_PROBE_REPORT)
    return report


def disabled_reason() -> Optional[str]:
    """The reason the lane is off, or ``None`` while it is usable/pending."""

    return _DISABLED_REASON or None


def installed() -> bool:
    """True once the probe has run and passed."""

    return _DISABLED_REASON == ""


def pending() -> bool:
    """True while the install probe has not run yet.

    An armed flag that reaches the hot path still pending means the install
    site never ran -- the arm is measuring the control while its receipt says
    otherwise.  Callers RAISE on this rather than serving the stock chain.
    """

    return _DISABLED_REASON is None


def reset_for_tests() -> None:
    """Clear the process verdict and counters.  Tests only."""

    global _DISABLED_REASON
    _DISABLED_REASON = None
    _PROBE_REPORT.clear()
    for key in _COUNTS:
        _COUNTS[key] = 0


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16, mx.float32)


def _float_tag(value: float) -> str:
    return (
        format(float(value), ".9g")
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "d")
    )


def _dtype_tag(dtype: mx.Dtype) -> str:
    return {mx.float16: "f16", mx.bfloat16: "bf16", mx.float32: "f32"}[dtype]


def _as_i32_scalar(value, name: str) -> mx.array:
    """Normalize a host or traced scalar without synchronizing an array."""

    if isinstance(value, mx.array):
        if value.dtype != mx.int32 or int(value.size) != 1:
            raise RopeGlueContractError(f"{name} tensor must be one int32 value")
        return value.reshape((1,))
    return mx.array([int(value)], dtype=mx.int32)


def _attention_scaling(value: float) -> float:
    scaling = float(value)
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise RopeGlueContractError(
            f"attention_scaling must be finite and positive; got {scaling}"
        )
    return scaling


def check_contract(
    queries: Optional[mx.array],
    keys: mx.array,
    inv_freq: mx.array,
) -> tuple[int, int, int, int]:
    """Validate the geometry and return ``(rows, heads_q, heads_k, rotary)``.

    RAISES on every mismatch -- an armed flag that cannot apply is a
    configuration error, never a quiet revert to the stock chain (the failure
    mode that left ``MTPLX_FUSED_HC_V3`` armed-but-dead at M=4).
    """

    _COUNTS["contract_checks"] += 1
    if not mx.metal.is_available():
        raise RopeGlueContractError(
            "the fused QSA rope has no portable spelling; it is a Metal kernel"
        )
    if keys.ndim != 4 or int(keys.shape[0]) != 1:
        raise RopeGlueContractError(
            f"keys must be [1, S, H, D]; got {tuple(keys.shape)}"
        )
    if keys.dtype not in _SUPPORTED_DTYPES:
        raise RopeGlueContractError(
            f"keys must be float16, bfloat16 or float32; got {keys.dtype}"
        )
    if inv_freq.ndim != 1 or inv_freq.dtype != mx.float32:
        raise RopeGlueContractError(
            "inv_freq must be a one-dimensional float32 array; got "
            f"shape={tuple(inv_freq.shape)}, dtype={inv_freq.dtype}"
        )
    rows = int(keys.shape[1])
    heads_k = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    rotary = 2 * int(inv_freq.shape[0])
    if rotary <= 0 or rotary % 2 or rotary > head_dim:
        raise RopeGlueContractError(
            f"rotary_dim {rotary} must be even, positive and at most the head "
            f"dimension {head_dim}"
        )
    heads_q = 0
    if queries is not None:
        if queries.ndim != 4 or int(queries.shape[0]) != 1:
            raise RopeGlueContractError(
                f"queries must be [1, S, H, D]; got {tuple(queries.shape)}"
            )
        if queries.dtype != keys.dtype:
            raise RopeGlueContractError(
                "queries and keys must share a dtype (one kernel rotates "
                f"both); got {queries.dtype} and {keys.dtype}"
            )
        if int(queries.shape[1]) != rows:
            raise RopeGlueContractError(
                f"queries and keys must have the same row count; got "
                f"{int(queries.shape[1])} and {rows}"
            )
        if int(queries.shape[3]) != head_dim:
            raise RopeGlueContractError(
                "queries and keys must share a head dimension (one table "
                f"serves both); got {int(queries.shape[3])} and {head_dim}"
            )
        heads_q = int(queries.shape[2])
    if rows <= 0 or heads_k <= 0 or (queries is not None and heads_q <= 0):
        raise RopeGlueContractError(
            f"every dimension must be positive; got rows={rows}, "
            f"heads_q={heads_q}, heads_k={heads_k}"
        )
    return rows, heads_q, heads_k, rotary


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def _constants(
    heads_q: int,
    heads_k: int,
    head_dim: int,
    rotary_dim: int,
    attention_scaling: float,
) -> str:
    return (
        _HEADER
        + f"constant constexpr uint HQ = {heads_q};\n"
        + f"constant constexpr uint HK = {heads_k};\n"
        + f"constant constexpr uint HEAD_DIM = {head_dim};\n"
        + f"constant constexpr uint ROTARY_DIM = {rotary_dim};\n"
        + f"constant constexpr uint HALF_ROTARY = {rotary_dim // 2};\n"
        + "constant constexpr float ROPE_ATTENTION_SCALE = "
        + f"{float(attention_scaling)!r}f;\n"
    )


#: The rotation itself, shared by both variants.  ``src``/``dst``/``d_stride``
#: are set by the caller-side prologue, so the arithmetic below has exactly
#: one definition.
_ROTATE = """
    for (uint pair = lane; pair < HALF_ROTARY; pair += 32u) {
        const float theta = position *
            float(inv_freq[(size_t)pair * inv_freq_strides[0]]);
        const float cosine =
            metal::precise::cos(theta) * ROPE_ATTENTION_SCALE;
        const float sine =
            metal::precise::sin(theta) * ROPE_ATTENTION_SCALE;
        const float first = float(src[(size_t)pair * d_stride]);
        const float second =
            float(src[(size_t)(pair + HALF_ROTARY) * d_stride]);
        // Distinct fp32 products: the stock graph rounds both multiplies
        // before its add/subtract, and letting Metal contract this into an
        // FMA moves a handful of bf16 cutoff values.
        const float first_cosine = first * cosine;
        const float second_sine = second * sine;
        const float second_cosine = second * cosine;
        const float first_sine = first * sine;
        dst[pair] = static_cast<T>(first_cosine - second_sine);
        dst[pair + HALF_ROTARY] = static_cast<T>(second_cosine + first_sine);
    }

    // The pass-through half is a copy in the stock chain too (the third
    // operand of its concatenate), so this is exact by inspection.
    for (uint dim = ROTARY_DIM + lane; dim < HEAD_DIM; dim += 32u) {
        dst[dim] = src[(size_t)dim * d_stride];
    }
"""

#: One simdgroup per (row, head).  Slots ``[0, HQ)`` are the query's, slots
#: ``[HQ, HQ + HK)`` the key's, so ONE dispatch rotates both tensors with one
#: table -- and the two rotations stop being siblings that each cost a launch.
_SOURCE_QK = """
    const uint unit = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint PER_ROW = HQ + HK;
    const uint row = unit / PER_ROW;
    const uint slot = unit - row * PER_ROW;

    // int32 first, then float: the stock chain is
    // (pos_start + arange(S)).astype(float32).
    const float position = float(pos_start[0] + int(row));

    device const T* src;
    device T* dst;
    size_t d_stride;
    if (slot < HQ) {
        src = q + (size_t)row * (size_t)q_strides[1]
                + (size_t)slot * (size_t)q_strides[2];
        dst = q_out + ((size_t)row * HQ + slot) * HEAD_DIM;
        d_stride = (size_t)q_strides[3];
    } else {
        const uint head = slot - HQ;
        src = k + (size_t)row * (size_t)k_strides[1]
                + (size_t)head * (size_t)k_strides[2];
        dst = k_out + ((size_t)row * HK + head) * HEAD_DIM;
        d_stride = (size_t)k_strides[3];
    }
""" + _ROTATE

#: The ``kv_only`` append path never computes a query, so its variant binds
#: only the key tensor -- a dummy query output would be an allocation and a
#: write the stock path does not make.
_SOURCE_K = """
    const uint unit = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint row = unit / HK;
    const uint head = unit - row * HK;

    const float position = float(pos_start[0] + int(row));

    device const T* src = k + (size_t)row * (size_t)k_strides[1]
                            + (size_t)head * (size_t)k_strides[2];
    device T* dst = k_out + ((size_t)row * HK + head) * HEAD_DIM;
    const size_t d_stride = (size_t)k_strides[3];
""" + _ROTATE


@lru_cache(maxsize=64)
def _kernel_qk(
    heads_q: int,
    heads_k: int,
    head_dim: int,
    rotary_dim: int,
    attention_scaling: float,
    dtype: mx.Dtype,
):
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_qwen4_m4_rope_qk_hq{heads_q}_hk{heads_k}_d{head_dim}_"
            f"r{rotary_dim}_s{_float_tag(attention_scaling)}_{_dtype_tag(dtype)}"
        ),
        input_names=["q", "k", "inv_freq", "pos_start"],
        output_names=["q_out", "k_out"],
        header=_constants(
            heads_q, heads_k, head_dim, rotary_dim, attention_scaling
        ),
        source=_SOURCE_QK,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=64)
def _kernel_k(
    heads_k: int,
    head_dim: int,
    rotary_dim: int,
    attention_scaling: float,
    dtype: mx.Dtype,
):
    return mx.fast.metal_kernel(
        name=(
            f"mtplx_qwen4_m4_rope_k_hk{heads_k}_d{head_dim}_"
            f"r{rotary_dim}_s{_float_tag(attention_scaling)}_{_dtype_tag(dtype)}"
        ),
        input_names=["k", "inv_freq", "pos_start"],
        output_names=["k_out"],
        header=_constants(0, heads_k, head_dim, rotary_dim, attention_scaling),
        source=_SOURCE_K,
        ensure_row_contiguous=False,
    )


def rope_qk(
    queries: Optional[mx.array],
    keys: mx.array,
    inv_freq: mx.array,
    *,
    pos_start,
    attention_scaling: float = 1.0,
):
    """Rotate ``queries`` and ``keys`` in ONE dispatch.

    ``queries`` may be ``None`` (the ``kv_only`` append path, where the query
    half of the layer is never computed); the keys-only variant is then
    dispatched and the query output is ``None``.  Both tensors are
    ``[1, S, H, D]`` and share ``D`` and their dtype; ``pos_start`` may be a
    one-element int32 array so a compiled graph replays at new absolute
    positions without baking an offset into its trace.
    """

    rows, heads_q, heads_k, rotary = check_contract(queries, keys, inv_freq)
    head_dim = int(keys.shape[3])
    scale = _attention_scaling(attention_scaling)
    start = _as_i32_scalar(pos_start, "pos_start")
    dtype = keys.dtype
    if queries is None:
        kernel = _kernel_k(heads_k, head_dim, rotary, scale, dtype)
        (k_out,) = kernel(
            inputs=[keys, inv_freq, start],
            template=[("T", dtype)],
            grid=(rows * heads_k * _SIMD, 1, 1),
            threadgroup=(_SIMD, 1, 1),
            output_shapes=[tuple(keys.shape)],
            output_dtypes=[dtype],
        )
        _COUNTS["k_only_calls"] += 1
        return None, k_out
    kernel = _kernel_qk(heads_q, heads_k, head_dim, rotary, scale, dtype)
    q_out, k_out = kernel(
        inputs=[queries, keys, inv_freq, start],
        template=[("T", dtype)],
        grid=(rows * (heads_q + heads_k) * _SIMD, 1, 1),
        threadgroup=(_SIMD, 1, 1),
        output_shapes=[tuple(queries.shape), tuple(keys.shape)],
        output_dtypes=[dtype, dtype],
    )
    _COUNTS["qk_calls"] += 1
    return q_out, k_out


# ---------------------------------------------------------------------------
# The reference, and the install probe
# ---------------------------------------------------------------------------
def stock_reference(
    queries: Optional[mx.array],
    keys: mx.array,
    inv_freq: mx.array,
    *,
    pos_start,
    attention_scaling: float = 1.0,
):
    """The LIVE stock rope expression for the same operands.

    Calls ``mtplx.models.qwen4_exp`` rather than restating its arithmetic, so
    the probe cannot drift from the thing it certifies.  Which spelling is
    live is the op diet's ``rope`` item -- the two are bitwise-identical by
    construction, and the probe follows whichever the process armed.
    """

    from mtplx.models import qwen4_exp as _model
    from mtplx.runtime_options import fable_opdiet_enabled

    rows = int(keys.shape[1])
    positions = pos_start + mx.arange(rows, dtype=mx.int32)
    if fable_opdiet_enabled("rope"):
        cos, sin = _model._rope_cos_sin_half(
            positions, inv_freq, attention_scaling
        )
        apply_rope = _model._apply_partial_rope_half
    else:
        cos, sin = _model._rope_cos_sin(positions, inv_freq, attention_scaling)
        apply_rope = _model._apply_partial_rope
    rotated_q = None if queries is None else apply_rope(queries, cos, sin)
    return rotated_q, apply_rope(keys, cos, sin)


def _probe_cell(dtype: mx.Dtype, rows: int, heads_q: int, heads_k: int, head_dim: int):
    """Deterministic operands at the caller's real geometry.

    Built from ``sin``/``cos`` ramps rather than an RNG so two processes probe
    the identical numbers and a failure is reproducible from the log line.
    """

    n_q = rows * heads_q * head_dim
    n_k = rows * heads_k * head_dim
    queries = (
        mx.sin(mx.arange(n_q, dtype=mx.float32) * 0.00048828125)
        .reshape(1, rows, heads_q, head_dim)
        .astype(dtype)
    )
    keys = (
        mx.cos(mx.arange(n_k, dtype=mx.float32) * 0.0009765625)
        .reshape(1, rows, heads_k, head_dim)
        .astype(dtype)
    )
    return queries, keys


def install(
    layers,
    *,
    rows: int = VERIFY_ROWS,
    logger=None,
) -> bool:
    """Contract-check and bit-exactness-probe the lane once per process.

    ``layers`` is an iterable of ``(index, attention_module)`` for the QSA
    layers.  Returns True when the lane is usable.  RAISES on a contract
    failure -- an armed flag on a pack this kernel is not wired for means the
    arm measured a different model.  DISABLES (returns False, records the
    reason, logs) on an exactness failure: rope rounding is a numerical
    verdict, and taking the whole model down for it would be the wrong trade.

    Called from ``install_qwen4_fixed_verify_route`` -- model build time,
    outside any ``mx.compile`` trace, the same place every other fixed-M4
    lane validates itself.
    """

    global _DISABLED_REASON
    if _DISABLED_REASON is not None:
        return _DISABLED_REASON == ""

    seen = 0
    #: The probe's cost is per DISTINCT geometry, not per layer.  Unlike the
    #: route kernel -- where every layer carries its own weights and every
    #: layer must therefore be proved -- this kernel reads no per-layer
    #: parameter: its output is a function of the shapes, the shared
    #: ``inv_freq`` object and the scaling.  Contract-check every layer,
    #: probe one representative of each signature.
    probed: set = set()
    for index, attention in layers:
        heads_q = int(attention.n_heads)
        heads_k = int(attention.n_kv_heads)
        head_dim = int(attention.head_dim)
        inv_freq = attention._inv_freq
        scaling = float(attention._rope_attention_scaling)
        # The rope sees the activation dtype, which is the dtype of the head
        # norms that immediately precede it and are unquantized on every pack
        # this lane serves.
        q_norm = getattr(attention, "q_norm", None)
        k_norm = getattr(attention, "k_norm", None)
        if q_norm is None or k_norm is None:
            raise RopeGlueContractError(
                f"QSA layer {index} has no q_norm/k_norm; this lane is wired "
                "for the Flash-Next attention block"
            )
        dtype = q_norm.weight.dtype
        if k_norm.weight.dtype != dtype:
            raise RopeGlueContractError(
                f"QSA layer {index}: q_norm dtype {dtype} != k_norm dtype "
                f"{k_norm.weight.dtype}; one kernel rotates both tensors"
            )
        queries, keys = _probe_cell(dtype, rows, heads_q, heads_k, head_dim)
        # Raises with the offending geometry named.  Every layer, always.
        check_contract(queries, keys, inv_freq)
        signature = (
            heads_q, heads_k, head_dim, int(inv_freq.shape[0]),
            scaling, str(dtype), id(inv_freq),
        )
        if signature in probed:
            seen += 1
            continue
        probed.add(signature)
        for pos_start in (0, 17_405):
            _COUNTS["probe_runs"] += 1
            want_q, want_k = stock_reference(
                queries,
                keys,
                inv_freq,
                pos_start=pos_start,
                attention_scaling=scaling,
            )
            got_q, got_k = rope_qk(
                queries,
                keys,
                inv_freq,
                pos_start=pos_start,
                attention_scaling=scaling,
            )
            same_q = mx.array_equal(want_q, got_q)
            same_k = mx.array_equal(want_k, got_k)
            mx.eval(same_q, same_k)
            failed = [
                name
                for name, ok in (("queries", same_q), ("keys", same_k))
                if not bool(ok.item())
            ]
            if failed:
                _COUNTS["probe_failures"] += 1
                _DISABLED_REASON = (
                    f"layer {index} pos_start={pos_start}: "
                    + ", ".join(failed)
                    + " are not bit-exact with the stock rope chain"
                )
                _PROBE_REPORT["failed"] = _DISABLED_REASON
                if logger is not None:
                    logger.warning(
                        "MTPLX_FABLE_VERIFY_GLUE item 'qsa_rope': %s; "
                        "disabling the item for every layer (this arm now "
                        "measures the stock chain)",
                        _DISABLED_REASON,
                    )
                return False
        seen += 1

    if seen == 0:
        # Nothing to serve is a configuration error, not a numerical verdict:
        # the flag was armed against a model with no QSA layers.
        raise RopeGlueContractError(
            "MTPLX_FABLE_VERIFY_GLUE item 'qsa_rope' found no QSA attention "
            "layer to install on"
        )
    _PROBE_REPORT["layers"] = seen
    _PROBE_REPORT["rows"] = int(rows)
    _DISABLED_REASON = ""
    return True


def dispatches_removed_per_layer(*, with_positions: bool = True) -> int:
    """Dispatches this kernel deletes from one QSA layer's attention rope."""

    stock = (
        STOCK_DISPATCHES_PER_LAYER_WITH_POSITIONS
        if with_positions
        else STOCK_DISPATCHES_PER_LAYER
    )
    return stock - FUSED_DISPATCHES_PER_LAYER


def engagement_line(*, layers: int, enabled: bool) -> str:
    """One-line engagement receipt for the serving log."""

    if not enabled:
        reason = disabled_reason()
        suffix = f" ({reason})" if reason else ""
        return f"[fable] verify-glue qsa_rope: off{suffix}"
    return (
        "[fable] verify-glue qsa_rope: on, "
        f"layers={layers}, "
        f"dispatches/layer {STOCK_DISPATCHES_PER_LAYER_WITH_POSITIONS}->"
        f"{FUSED_DISPATCHES_PER_LAYER}, "
        f"dependent_levels/layer {STOCK_DEPENDENT_LEVELS_PER_LAYER}->"
        f"{FUSED_DEPENDENT_LEVELS_PER_LAYER}, "
        f"qk_calls={_COUNTS['qk_calls']}, "
        f"k_only_calls={_COUNTS['k_only_calls']}, "
        f"probe_failures={_COUNTS['probe_failures']}"
    )


__all__ = [
    "FUSED_DEPENDENT_LEVELS_PER_LAYER",
    "FUSED_DISPATCHES_PER_LAYER",
    "MAX_EXACT_NORM_HEAD_DIM",
    "STOCK_DEPENDENT_LEVELS_PER_LAYER",
    "STOCK_DISPATCHES_PER_LAYER",
    "STOCK_DISPATCHES_PER_LAYER_WITH_POSITIONS",
    "VERIFY_ROWS",
    "RopeGlueContractError",
    "check_contract",
    "counters",
    "disabled_reason",
    "dispatches_removed_per_layer",
    "engagement",
    "engagement_line",
    "install",
    "installed",
    "pending",
    "reset_for_tests",
    "rope_qk",
    "stock_reference",
]
