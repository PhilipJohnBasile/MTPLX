"""W70 -- ``MTPLX_FABLE_VERIFY_GLUE``: install gate and engagement receipts.

The compiled fixed-M4 verify body is one replayed graph of ~2,750 dispatch
nodes per decode cycle (``docs/perf/verify-node-census.md``).  Half of them
live in the twelve QSA layers and most of THOSE are glue: rope, masks, cache
offsets, index arithmetic and layout copies around ten load-bearing
operations.  This flag fuses that glue, one selectable item at a time.

TWO ITEMS, both off by default:

``qsa_rope``      The attention query/key rotation.  ``Attention.__call__``
                  builds one RoPE table and rotates two tensors with it; on
                  today's stack that is 14-16 dispatches over ~7
                  read-after-write levels per QSA layer.
                  ``kernels/qwen4_m4_rope.rope_qk`` issues the identical
                  arithmetic as ONE dispatch, one level.
``qsa_rope_idx``  The indexer's query preparation (RMSNorm + partial RoPE).
                  Runs the SHIPPED ``qsa_indexer_prepare_queries_metal``,
                  whose bit-exactness against ``_prepare_queries_eager`` is
                  pinned in ``tests/test_qsa_indexer_prepare_metal.py``.  The
                  fixed-M4 lane never called it: ``_prepare_queries`` gates on
                  ``MTPLX_QSA_FUSED_INDEXER``, and the fixed-capacity branch
                  goes straight to the eager chain.  ``MTPLX_FABLE_QSA_M4``
                  also uses this kernel, bundled with three other rewrites
                  (one of which carries an fp32 reassociation assumption);
                  this item is that flag's exact subset, alone.

THE GATE (modelled on ``kernels/qwen4_m4_route`` + ``kernels/qsa_sparse_decode``)

* CONTRACT failures RAISE.  An armed flag on a pack the kernel is not wired
  for means the arm measured a different model -- the failure mode that left
  ``MTPLX_FUSED_HC_V3`` armed-but-inert at M=4 and cost a window.
* EXACTNESS failures DISABLE the item for the process and log the reason.
  Rope rounding is a numerical verdict, not a broken contract, and taking the
  model down for it would be the wrong trade.  ``engagement()`` then reports
  ``installed=False`` with the reason, so a receipt can never claim a win
  from an item that was off.
* The probe runs at model build, inside
  ``install_qwen4_fixed_verify_route`` -- outside any ``mx.compile`` trace,
  the same place every other fixed-M4 lane validates itself.

ENGAGEMENT.  ``qk_calls`` / ``prep_calls`` are the line to read first: an ABBA
that reports a win with those at zero measured something else.  They count
TRACES, not decode cycles -- under ``mx.compile`` the Python body runs once per
retrace and the C++ replay never touches it -- so read them as "did this lane
get into the graph at all".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

import mlx.core as mx

from mtplx.runtime_options import (
    FABLE_VERIFY_GLUE_ITEMS,
    fable_verify_glue_enabled,
)

logger = logging.getLogger(__name__)

#: Verify/draft widths the items serve.  Prefill keeps the stock chain: these
#: kernels are latency plays on a 4-row window, and a 16 K prefill row count
#: is a different regime that nothing here has measured.
MIN_ROWS = 1
MAX_ROWS = 8

_IDX_COUNTS: Dict[str, int] = {
    "contract_checks": 0,
    "probe_runs": 0,
    "probe_failures": 0,
    "prep_calls": 0,
}
_IDX_DISABLED_REASON: Optional[str] = None
_IDX_PROBE_REPORT: Dict[str, Any] = {}

_REPORT: Dict[str, Any] = {}


class VerifyGlueContractError(RuntimeError):
    """An item was armed against a geometry it is not contracted for."""


# ---------------------------------------------------------------------------
# qsa_rope -- delegates to the kernel module that owns the arithmetic
# ---------------------------------------------------------------------------
def qsa_rope_installed() -> bool:
    """True when the fused attention rope passed its probe this process.

    RAISES when the item is armed but the probe never ran: a flag that is
    armed-but-inert makes an arm measure the control while its receipt claims
    the candidate, which is exactly how ``MTPLX_FUSED_HC_V3`` burned a window.
    """

    from mtplx.kernels import qwen4_m4_rope as _rope

    if _rope.pending():
        raise VerifyGlueContractError(
            "MTPLX_FABLE_VERIFY_GLUE item 'qsa_rope' is armed but its install "
            "probe never ran: this route did not go through "
            "install_qwen4_fixed_verify_route, so the item would be inert"
        )
    return _rope.installed()


def serves_rows(rows: int) -> bool:
    """True for the widths these items are wired for."""

    return MIN_ROWS <= int(rows) <= MAX_ROWS


# ---------------------------------------------------------------------------
# qsa_rope_idx -- the shipped indexer preparation kernel, on the fixed lane
# ---------------------------------------------------------------------------
def idx_counters() -> Dict[str, int]:
    return dict(_IDX_COUNTS)


def qsa_rope_idx_installed() -> bool:
    """True when the indexer preparation item passed its probe.

    RAISES while the probe is still pending, for the same reason
    :func:`qsa_rope_installed` does.
    """

    if _IDX_DISABLED_REASON is None:
        raise VerifyGlueContractError(
            "MTPLX_FABLE_VERIFY_GLUE item 'qsa_rope_idx' is armed but its "
            "install probe never ran: this route did not go through "
            "install_qwen4_fixed_verify_route, so the item would be inert"
        )
    return _IDX_DISABLED_REASON == ""


def qsa_rope_idx_disabled_reason() -> Optional[str]:
    return _IDX_DISABLED_REASON or None


def note_prep_call() -> None:
    """Engagement counter for the indexer preparation item."""

    _IDX_COUNTS["prep_calls"] += 1


def reset_for_tests() -> None:
    """Clear every item's verdict and counters.  Tests only."""

    global _IDX_DISABLED_REASON
    _IDX_DISABLED_REASON = None
    _IDX_PROBE_REPORT.clear()
    _REPORT.clear()
    for key in _IDX_COUNTS:
        _IDX_COUNTS[key] = 0
    from mtplx.kernels import qwen4_m4_rope as _rope

    _rope.reset_for_tests()


def _check_indexer_contract(indexer: Any, *, index: int) -> mx.Dtype:
    """Validate the indexer half.  Raises; never returns False."""

    from mtplx.kernels.qwen4_qsa_m4_indexer import MAX_EXACT_HEAD_DIM

    _IDX_COUNTS["contract_checks"] += 1
    where = f"MTPLX_FABLE_VERIFY_GLUE item 'qsa_rope_idx' layer {index}"
    if not mx.metal.is_available():
        raise VerifyGlueContractError(
            f"{where}: the fused indexer preparation is a Metal kernel and "
            "has no portable spelling"
        )
    if int(indexer.kv_heads) != 1:
        raise VerifyGlueContractError(
            f"{where}: wired for a single indexer KV head; got "
            f"{int(indexer.kv_heads)}"
        )
    head_dim = int(indexer.head_dim)
    if not (0 < head_dim <= MAX_EXACT_HEAD_DIM):
        raise VerifyGlueContractError(
            f"{where}: the kernel reproduces MLX's rms_single_row reduction "
            f"exactly only for head_dim <= {MAX_EXACT_HEAD_DIM}; got "
            f"{head_dim}"
        )
    weight = indexer.q_layernorm.weight
    if weight.ndim != 1 or int(weight.shape[0]) != head_dim:
        raise VerifyGlueContractError(
            f"{where}: q_layernorm weight must be one head-dimension vector; "
            f"got {tuple(weight.shape)} for head_dim {head_dim}"
        )
    inv_freq = indexer._inv_freq
    if inv_freq.ndim != 1 or inv_freq.dtype != mx.float32:
        raise VerifyGlueContractError(
            f"{where}: inv_freq must be a 1-D float32 array; got "
            f"shape={tuple(inv_freq.shape)}, dtype={inv_freq.dtype}"
        )
    if 2 * int(inv_freq.shape[0]) > head_dim:
        raise VerifyGlueContractError(
            f"{where}: rotary_dim {2 * int(inv_freq.shape[0])} exceeds "
            f"head_dim {head_dim}"
        )
    return weight.dtype


def _probe_indexer(indexer: Any, *, index: int, rows: int) -> Optional[str]:
    """Bit-exactness probe for one indexer.  Returns a reason, or ``None``."""

    dtype = _check_indexer_contract(indexer, index=index)
    heads = int(indexer.n_heads)
    head_dim = int(indexer.head_dim)
    n = rows * heads * head_dim
    sample = (
        mx.sin(mx.arange(n, dtype=mx.float32) * 0.00048828125)
        .reshape(1, rows, heads, head_dim)
        .astype(dtype)
    )
    for pos_start in (0, 17_405):
        _IDX_COUNTS["probe_runs"] += 1
        want = indexer._prepare_queries_eager(sample, pos_start)
        got = indexer._prepare_queries_m4(sample, pos_start)
        same = mx.array_equal(want, got)
        mx.eval(same)
        if not bool(same.item()):
            return (
                f"layer {index} pos_start={pos_start}: the fused indexer "
                "query preparation is not bit-exact with the eager chain"
            )
    return None


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
def install(
    qsa_layers: Iterable[tuple[int, Any]],
    *,
    rows: int = 4,
) -> Dict[str, Any]:
    """Contract-check and probe every armed item once, at model build.

    ``qsa_layers`` is ``(layer_index, attention_module)`` for the QSA layers.
    Returns the install report; also stored for ``engagement()``.
    """

    global _IDX_DISABLED_REASON

    layers = tuple(qsa_layers)
    report: Dict[str, Any] = {
        "armed": bool(fable_verify_glue_enabled()),
        "items": {},
        "qsa_layers": len(layers),
        "rows": int(rows),
    }

    if fable_verify_glue_enabled("qsa_rope"):
        from mtplx.kernels import qwen4_m4_rope as _rope

        ok = _rope.install(layers, rows=int(rows), logger=logger)
        report["items"]["qsa_rope"] = _rope.engagement()
        logger.info(
            "%s", _rope.engagement_line(layers=len(layers), enabled=ok)
        )

    if fable_verify_glue_enabled("qsa_rope_idx"):
        if _IDX_DISABLED_REASON is None:
            reason = None
            seen = 0
            # The shipped kernel reads the layer's OWN q_layernorm weight, so
            # unlike the rope item this cannot be deduped by geometry: every
            # layer is probed.
            for index, attention in layers:
                indexer = getattr(attention, "indexer", None)
                if indexer is None:
                    raise VerifyGlueContractError(
                        "MTPLX_FABLE_VERIFY_GLUE item 'qsa_rope_idx' layer "
                        f"{index}: this attention block has no QSA indexer"
                    )
                reason = _probe_indexer(indexer, index=index, rows=int(rows))
                if reason is not None:
                    break
                seen += 1
            if seen == 0 and reason is None:
                raise VerifyGlueContractError(
                    "MTPLX_FABLE_VERIFY_GLUE item 'qsa_rope_idx' found no QSA "
                    "indexer to install on"
                )
            if reason is None:
                _IDX_PROBE_REPORT["layers"] = seen
                _IDX_DISABLED_REASON = ""
            else:
                _IDX_COUNTS["probe_failures"] += 1
                _IDX_DISABLED_REASON = reason
                _IDX_PROBE_REPORT["failed"] = reason
                logger.warning(
                    "MTPLX_FABLE_VERIFY_GLUE item 'qsa_rope_idx': %s; "
                    "disabling the item for every layer (this arm now "
                    "measures the eager chain)",
                    reason,
                )
        report["items"]["qsa_rope_idx"] = _idx_engagement()
        logger.info(
            "%s",
            idx_engagement_line(
                layers=len(layers), enabled=qsa_rope_idx_installed()
            ),
        )

    _REPORT.clear()
    _REPORT.update(report)
    return report


def _idx_engagement() -> Dict[str, Any]:
    out = dict(_IDX_COUNTS)
    out["installed"] = _IDX_DISABLED_REASON == ""
    out["disabled_reason"] = _IDX_DISABLED_REASON or None
    out["probe"] = dict(_IDX_PROBE_REPORT)
    return out


def idx_engagement_line(*, layers: int, enabled: bool) -> str:
    """One-line engagement receipt for the indexer preparation item."""

    if not enabled:
        reason = qsa_rope_idx_disabled_reason()
        suffix = f" ({reason})" if reason else ""
        return f"[fable] verify-glue qsa_rope_idx: off{suffix}"
    return (
        "[fable] verify-glue qsa_rope_idx: on, "
        f"layers={layers}, dispatches/layer 10->1, "
        f"dependent_levels/layer 7->1, "
        f"prep_calls={_IDX_COUNTS['prep_calls']}, "
        f"probe_failures={_IDX_COUNTS['probe_failures']}"
    )


def engagement() -> Dict[str, Any]:
    """Counters and install verdict for every item, for the receipt."""

    from mtplx.kernels import qwen4_m4_rope as _rope

    return {
        "armed": bool(fable_verify_glue_enabled()),
        "known_items": list(FABLE_VERIFY_GLUE_ITEMS),
        "selected": [
            item
            for item in FABLE_VERIFY_GLUE_ITEMS
            if fable_verify_glue_enabled(item)
        ],
        "install": dict(_REPORT),
        "qsa_rope": _rope.engagement(),
        "qsa_rope_idx": _idx_engagement(),
    }


__all__ = [
    "MAX_ROWS",
    "MIN_ROWS",
    "VerifyGlueContractError",
    "engagement",
    "idx_counters",
    "idx_engagement_line",
    "install",
    "note_prep_call",
    "qsa_rope_idx_disabled_reason",
    "qsa_rope_idx_installed",
    "qsa_rope_installed",
    "reset_for_tests",
    "serves_rows",
]
