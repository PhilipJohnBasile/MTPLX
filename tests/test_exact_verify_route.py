"""Greedy exactness guard: OPT-IN stock-matmul frame at t<=0.

2026-08-31 contract flip (founder order — restore turbo at greedy): the
shipping default runs the turbo vk/nax verify kernels at every temperature;
MTPLX_EXACT_T0_GUARD=1 re-arms the stock frame. Receipts for the flip live
in SPEEDWAR-20260831/turbo-guard-27b: the guard cost 5-21% greedy decode on
the 27B turbo serve quad and delivered identity on neither route (5/6
default-suite prompts diverge from greedy AR at identical indexes with the
guard on or off — the flips live in the cross-M numeric frame that stock
kernels share).

These tests pin BOTH modes: dark default (guard never fires) and armed
(bit-exact stock fall-through, the original contract).
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

import mtplx.attention_context as attention_context
from mtplx.attention_context import exact_verify, exact_verify_required


def test_guard_dark_by_default_even_inside_exact_scope(monkeypatch):
    monkeypatch.setattr(attention_context, "_EXACT_T0_GUARD_ARMED", False)
    assert exact_verify_required() is False
    with exact_verify(True):
        # The generation loop still marks greedy scopes; the dark guard
        # ignores them so turbo kernels run at t<=0.
        assert exact_verify_required() is False
    assert exact_verify_required() is False


def test_guard_armed_scopes_and_defaults(monkeypatch):
    monkeypatch.setattr(attention_context, "_EXACT_T0_GUARD_ARMED", True)
    assert exact_verify_required() is False
    with exact_verify(True):
        assert exact_verify_required() is True
        with exact_verify(False):
            assert exact_verify_required() is False
        assert exact_verify_required() is True
    assert exact_verify_required() is False


def test_env_spelling_arms_the_guard(monkeypatch):
    import importlib

    monkeypatch.setenv("MTPLX_EXACT_T0_GUARD", "1")
    module = importlib.reload(attention_context)
    try:
        assert module._EXACT_T0_GUARD_ARMED is True
        monkeypatch.setenv("MTPLX_EXACT_T0_GUARD", "0")
        module = importlib.reload(attention_context)
        assert module._EXACT_T0_GUARD_ARMED is False
        monkeypatch.delenv("MTPLX_EXACT_T0_GUARD")
        module = importlib.reload(attention_context)
        assert module._EXACT_T0_GUARD_ARMED is False
    finally:
        monkeypatch.delenv("MTPLX_EXACT_T0_GUARD", raising=False)
        importlib.reload(attention_context)


def test_patched_qlinear_falls_to_stock_when_guard_armed(monkeypatch):
    monkeypatch.setattr(attention_context, "_EXACT_T0_GUARD_ARMED", True)
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


def test_patched_qlinear_rides_turbo_when_guard_dark(monkeypatch):
    monkeypatch.setattr(attention_context, "_EXACT_T0_GUARD_ARMED", False)
    from mtplx import nax_verify

    installed = nax_verify.install_nax_qlinear_patch()
    assert installed["installed"] is True
    try:
        layer = nn.QuantizedLinear(512, 2048, bits=4, group_size=64)
        x = mx.random.normal((1, 4, 512)).astype(mx.bfloat16)

        from mtplx.attention_context import attention_phase

        nax_verify.nax_qlinear_fallback_counts.pop("exact_t0", None)
        with attention_phase("decode_verify"), exact_verify(True):
            y = layer(x)
            mx.eval(y)
        # Dark guard: the greedy scope must NOT force the stock bail.
        assert nax_verify.nax_qlinear_fallback_counts.get("exact_t0", 0) == 0
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
    # An ARMED-guard t>0 compiled verify trace bakes vk/nax kernels into the
    # graph; the shared-trace key must therefore differ between routes or a
    # greedy request replays non-exact kernels. Pin the key construction.
    import inspect

    import mtplx.graphbank as graphbank

    src = inspect.getsource(graphbank.CompiledVerifyBank._shared_or_new_verify_step)
    assert "exact_verify_required()" in src
