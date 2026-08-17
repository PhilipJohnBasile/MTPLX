"""Speed-path env knobs must bind at call time, not import time (F26).

An import-time ``os.environ`` snapshot silently pins whichever value was set
when the module first loaded — the turbo profile's ``MTPLX_NAX_M4_IMPL=vk_k``
export worked only by import-order accident. These tests set env AFTER import
(the profile/server boot pattern) and require the reader to see it. All tests
are CPU-only: kernel entry points are intercepted before any Metal dispatch.

Also covers the install-report honesty fix: ``install_nax_qlinear_patch``
must report the real ``nax_available()`` probe, never a hardcoded True.
"""

from __future__ import annotations

import platform

import mlx.core as mx
import pytest

import mtplx.nax_verify as nax_verify
import mtplx.verify_kernels as verify_kernels


_SENTINEL = object()


def test_m4_impl_env_is_read_at_call_time(monkeypatch):
    # The turbo-profile pattern: env exported after mtplx.nax_verify import.
    monkeypatch.setenv("MTPLX_NAX_M4_IMPL", "vk_k")

    dispatched: list[str] = []
    monkeypatch.setattr(verify_kernels, "vk_eligible_ksplit", lambda *a, **k: True)
    monkeypatch.setattr(
        verify_kernels,
        "vk_qmm_m4_impl",
        lambda impl, *a, **k: dispatched.append(impl) or _SENTINEL,
    )
    # Any legacy-kernel build means the env was ignored; fail before Metal.
    for name in (
        "_build_kernel_m4_ksplit_np",
        "_build_kernel_m4_bn6",
        "_build_kernel_m4_kp1",
    ):
        monkeypatch.setattr(
            nax_verify,
            name,
            lambda *a, _n=name, **k: pytest.fail(
                f"{_n} invoked: MTPLX_NAX_M4_IMPL=vk_k was ignored (frozen at import)"
            ),
        )

    x2 = mx.zeros((4, 64), dtype=mx.bfloat16)
    w_q = mx.zeros((32, 8), dtype=mx.uint32)
    scales = mx.zeros((32, 1), dtype=mx.bfloat16)
    biases = mx.zeros((32, 1), dtype=mx.bfloat16)
    result = nax_verify.nax_qmm_m4(x2, w_q, scales, biases, group_size=64)

    assert result is _SENTINEL
    assert dispatched == ["vk_k"]


def test_vk_nsg_env_is_read_at_call_time(monkeypatch):
    # Defaults: M4 NSG=8 (N % 32), M6 NSG=4 (N % 16). Change after import and
    # the eligibility predicates must follow — in both directions.
    monkeypatch.setenv("MTPLX_VK_M4_NSG", "12")  # N % 48
    assert verify_kernels.vk_eligible_m4(4, 64, 48, 4, 64, mx.bfloat16) is True
    assert verify_kernels.vk_eligible_m4(4, 64, 32, 4, 64, mx.bfloat16) is False

    monkeypatch.setenv("MTPLX_VK_M6_NSG", "6")  # N % 24
    assert verify_kernels.vk_eligible_m6(6, 64, 24, 4, 64, mx.bfloat16) is True
    assert verify_kernels.vk_eligible_m6(6, 64, 16, 4, 64, mx.bfloat16) is False

    # Restoring the default env restores the default predicates.
    monkeypatch.delenv("MTPLX_VK_M4_NSG")
    monkeypatch.delenv("MTPLX_VK_M6_NSG")
    assert verify_kernels.vk_eligible_m4(4, 64, 32, 4, 64, mx.bfloat16) is True
    assert verify_kernels.vk_eligible_m6(6, 64, 16, 4, 64, mx.bfloat16) is True


def test_force_gpu_family_fallback_env_is_read_per_call(monkeypatch):
    # Deterministic on any machine: fake a G17 + macOS 26.2 hardware probe so
    # only the env switch decides the outcome.
    monkeypatch.setattr(
        mx, "device_info", lambda: {"architecture": "applegpu_g17s"}
    )
    monkeypatch.setattr(
        platform, "mac_ver", lambda: ("26.2.1", ("", "", ""), "arm64")
    )
    monkeypatch.delenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", raising=False)
    nax_verify.nax_available.cache_clear()
    try:
        assert nax_verify.nax_available() is True
        # Setting the QA switch after the first probe must take effect
        # immediately — no cache_clear() required.
        monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
        assert nax_verify.nax_available() is False
        monkeypatch.delenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK")
        assert nax_verify.nax_available() is True
    finally:
        # Drop the fake-hardware memo so later tests probe the real machine.
        nax_verify.nax_available.cache_clear()


def test_install_report_tells_the_truth_about_nax(monkeypatch):
    # With the fallback switch on, the probe is False on every machine; the
    # install report must say so instead of hardcoding True.
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    assert not nax_verify._QLINEAR_PATCH["installed"], (
        "test requires a pristine QuantizedLinear patch state"
    )
    report = nax_verify.install_nax_qlinear_patch()
    try:
        assert report["installed"] is True
        assert report["nax_available"] is False
        # The already-installed path must agree with the live probe too.
        again = nax_verify.install_nax_qlinear_patch()
        assert again["already"] is True
        assert again["nax_available"] is False
    finally:
        nax_verify.uninstall_nax_qlinear_patch()
