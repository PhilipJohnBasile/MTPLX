"""Install the vendored ``ArraysCache`` advance() leak fix into mlx-lm.

Stock mlx-lm 0.31.x ``ArraysCache.advance()`` decrements its metadata with
lazy mx.array arithmetic, leaking one dead graph node (and its live Metal
buffer object) per layer per decode token until the process dies at
Metal's buffer-count limit. The fix (mlx-lm PR #1642 @ 985af30) is
vendored in :mod:`mtplx.vendored_arrays_cache`; this module decides at
runtime whether it is needed, following the ``install_hy_v3_model_shim``
precedent: prefer upstream when it already has the capability, otherwise
install the vendored implementation, idempotently.

Call :func:`install_arrays_cache_fix` before any code constructs GDN /
linear-attention caches — in particular before the mtp_batch launcher
gate (``a3b_mtp_batch._require_mlx_lm_arrays_cache_fix``), which probes a
fresh ``ArraysCache(1)`` for the fix's ``_lp_advance``/``_len_advance``
bookkeeping attributes and aborts startup when they are missing.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

# install_arrays_cache_fix() results.
UPSTREAM_FIXED = "upstream_fixed"
VENDORED_INSTALLED = "vendored_installed"
ALREADY_INSTALLED = "already_installed"


def install_arrays_cache_fix() -> str:
    """Make ``mlx_lm.models.cache.ArraysCache`` leak-free. Idempotent.

    Returns ``"upstream_fixed"`` when the installed mlx-lm already
    carries the deferred-advance bookkeeping natively (nothing is
    touched), ``"vendored_installed"`` when the vendored subclass was
    installed over a stock class, and ``"already_installed"`` when a
    previous call in this process already installed it.
    """
    import mlx_lm.models.cache as cache_module

    from .vendored_arrays_cache import FixedArraysCache

    current = cache_module.ArraysCache
    if current is FixedArraysCache:
        return ALREADY_INSTALLED
    # Capability probe, not a version gate: the fixed tree still reports
    # 0.31.3, so only construction tells the two apart. Same probe as the
    # mtp_batch launcher gate.
    probe = current(1)
    if hasattr(probe, "_lp_advance") and hasattr(probe, "_len_advance"):
        return UPSTREAM_FIXED

    cache_module.ArraysCache = FixedArraysCache
    # save_prompt_cache records type(c).__name__ and load_prompt_cache
    # resolves it via this module's globals(); make "FixedArraysCache"
    # resolvable so saved caches round-trip. (Old files naming
    # "ArraysCache" now resolve to the fixed class — a safe superset.)
    cache_module.FixedArraysCache = FixedArraysCache
    # Rebind the name where it was imported with ``from ... import``
    # before this call (model modules and the generate loop hold their
    # own reference); modules imported later resolve the patched name
    # from cache_module automatically.
    rebound = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if name != "mlx_lm.generate" and not name.startswith("mlx_lm.models"):
            continue
        if getattr(module, "ArraysCache", None) is current:
            module.ArraysCache = FixedArraysCache
            rebound.append(name)
    logger.info(
        "[ArraysCache fix] vendored PR #1642 class installed over stock "
        "mlx-lm %s (rebound: %s)",
        getattr(sys.modules.get("mlx_lm"), "__version__", "unknown"),
        ", ".join(rebound) or "none",
    )
    return VENDORED_INSTALLED
