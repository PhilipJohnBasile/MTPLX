"""Greedy exactness route: t<=0 verify forwards must use stock matmuls.

The product promise is MTP output == AR output at temperature 0. AR decode
runs M=1 stock kernels; the turbo verify patch routes M=4..16 through vk/nax
kernels that are argmax-validated but not bit-exact vs stock (~6e-3 dmax,
flip band ~1.6e-2 measured 2026-08-29). While `exact_verify` is set the
QuantizedLinear patch must fall through to stock so both paths share one
numeric frame. These tests fail against the unfixed tree (no contextvar, no
fall-through).
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.attention_context import exact_verify, exact_verify_required


def test_exact_verify_contextvar_scopes_and_defaults():
    assert exact_verify_required() is False
    with exact_verify(True):
        assert exact_verify_required() is True
        with exact_verify(False):
            assert exact_verify_required() is False
        assert exact_verify_required() is True
    assert exact_verify_required() is False


def test_patched_qlinear_falls_to_stock_under_exact_verify(monkeypatch):
    from mtplx import nax_verify

    installed = nax_verify.install_nax_qlinear_patch()
    assert installed["installed"] is True
    try:
        # Verify-shaped call: bits=4, M=4, decode_verify phase — the exact
        # geometry the vk_k lane owns on the 27B packs.
        layer = nn.QuantizedLinear(512, 2048, bits=4, group_size=64)
        x = mx.random.normal((1, 4, 512)).astype(mx.bfloat16)

        from mtplx.attention_context import attention_phase

        nax_verify.nax_qlinear_fallback_counts.pop("exact_t0", None)
        with attention_phase("decode_verify"), exact_verify(True):
            y_exact = layer(x)
            mx.eval(y_exact)
        assert nax_verify.nax_qlinear_fallback_counts.get("exact_t0", 0) > 0

        # Stock reference computed with the patch uninstalled must match the
        # exact-route output bit-for-bit: that equality IS the contract.
        nax_verify.uninstall_nax_qlinear_patch()
        y_stock = layer(x)
        mx.eval(y_stock)
        assert mx.array_equal(y_exact, y_stock).item()
    finally:
        nax_verify.uninstall_nax_qlinear_patch()


def test_generation_wraps_verify_forward_with_exact_verify():
    # The verify forward in generate_mtpk must arm the route from the live
    # sampler temperature. Pin the wiring at the source level so a revert of
    # the with-block (while the contextvar machinery stays) cannot pass.
    import inspect

    import mtplx.generation as generation

    src = inspect.getsource(generation.generate_mtpk)
    assert "exact_verify(sampler.temperature <= 0)" in src


def test_shared_verify_trace_key_carries_exactness_route():
    # A t>0 compiled verify trace bakes vk/nax kernels into the graph; the
    # shared-trace key must therefore differ between routes or a greedy
    # request replays non-exact kernels. Pin the key construction.
    import inspect

    import mtplx.graphbank as graphbank

    src = inspect.getsource(graphbank.CompiledVerifyBank._shared_or_new_verify_step)
    assert "exact_verify_required()" in src
