"""Startup compile probe for the QSA sparse-prefill MPP pipelines (issue #404).

The macOS 27 MetalPerformancePrimitives SDK rejects address-space-qualified
cooperative-tensor template operands at Metal *compile* time, so 2.10.1 armed
the QSA sparse prefill lane at startup and then answered the first 33K+ prompt
with a mid-request HTTP 500 (issues #404/#405/#407; fix credit mrmurphy +
sunnybluesea).  The kernel sources now strip the qualifier with
``metal::remove_addrspace_t`` like mlx's own ``steel/gemm/nax.h``, and this
probe is the second layer: if a future SDK still refuses to build the
pipeline, the lane must degrade to dense prefill with a diagnostic instead of
a mid-request 500.

This lives outside ``qsa_indexer_prefill.py`` deliberately: that backend
module is contractually sync-free (no ``mx.eval``), while a compile probe
must force compilation by dispatching once.  Issue-#400 precedent: only the
real pipeline proves anything, so the probe dispatches the production score
kernel on a minimal family-shaped dummy and caches the verdict for the
process lifetime.  All three QSA prefill pipelines share the same MPP
cooperative-tensor construction, so this compile gate is representative of
the failure class it exists for.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def qsa_prefill_mpp_compile_supported() -> bool:
    try:
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            return False
        from mtplx.kernels.qsa_indexer_prefill import (
            qsa_indexer_prefill_scores_mpp,
        )

        q = mx.zeros((1, 16, 4, 128), dtype=mx.bfloat16)
        pooled = mx.zeros((1, 32, 128), dtype=mx.bfloat16)
        mx.eval(qsa_indexer_prefill_scores_mpp(q, pooled))
        return True
    except Exception as exc:
        print(
            "[mtplx] QSA sparse prefill disabled: this Metal SDK cannot "
            f"compile the MPP score pipeline; using dense prefill ({exc})",
            flush=True,
        )
        return False
