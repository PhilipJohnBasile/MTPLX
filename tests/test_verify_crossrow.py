"""CPU-side gates for the crossrow verify QMV port.

Metal behavior (parity vs stock, table==no-table bitwise, perturbed-table
positive control, timing) runs on-device via the fan-gated crossrow_check
probe; these tests pin the host-side contract: eligibility geometry, env
gating, and the group-divisor math baked into the generated source.
"""

from __future__ import annotations

import mlx.core as mx

from mtplx.verify_crossrow import (
    _qmm_kernel,
    crossrow_eligible,
    crossrow_enabled,
)


def test_env_gate(monkeypatch):
    monkeypatch.delenv("MTPLX_VK_CROSSROW", raising=False)
    assert not crossrow_enabled()
    monkeypatch.setenv("MTPLX_VK_CROSSROW", "1")
    assert crossrow_enabled()
    monkeypatch.setenv("MTPLX_VK_CROSSROW", "0")
    assert not crossrow_enabled()


def test_eligibility_geometry():
    ok = dict(bits=4, group_size=32, dtype=mx.bfloat16)
    assert crossrow_eligible(4, 5120, 16480, **ok)
    assert crossrow_eligible(5, 17408, 5120, **ok)
    assert crossrow_eligible(2, 5120, 48, **ok)  # N=48 % 8 == 0
    assert not crossrow_eligible(1, 5120, 16480, **ok)  # serial stays stock
    assert not crossrow_eligible(6, 5120, 16480, **ok)  # v1 single-group only
    assert not crossrow_eligible(4, 5120, 16481, **ok)  # N % 8
    assert not crossrow_eligible(4, 5000, 16480, **ok)  # K % 512
    assert not crossrow_eligible(4, 5120, 16480, bits=8, group_size=64, dtype=mx.bfloat16)
    assert not crossrow_eligible(4, 5120, 16480, bits=4, group_size=128, dtype=mx.bfloat16)
    assert not crossrow_eligible(4, 5120, 16480, bits=4, group_size=32, dtype=mx.float32)


def test_generated_source_group_divisors():
    """The lane->group map must match the quant layout: g64 -> lid/4 with
    K/64 groups per row, g32 -> lid/2 with K/32 groups per row; the table
    variant reads xsums and never re-forms sums inline (and vice versa)."""

    from mtplx.verify_crossrow import _qmm_source

    s32 = _qmm_source(4, 32, mx.bfloat16, use_table=True)
    assert "int(lid) / 2" in s32 and "K / 32" in s32
    assert "xsums" in s32 and "sums[mm] += float(xv[0])" not in s32

    s64 = _qmm_source(4, 64, mx.bfloat16, use_table=False)
    assert "int(lid) / 4" in s64 and "K / 64" in s64
    assert "sums[mm] += float(xv[0])" in s64 and "xsums" not in s64


def test_kernel_cache_keys_distinct():
    a = _qmm_kernel(4, 32, mx.bfloat16, use_table=True)
    b = _qmm_kernel(4, 32, mx.bfloat16, use_table=False)
    c = _qmm_kernel(5, 32, mx.bfloat16, use_table=True)
    assert a is not b and a is not c and b is not c
    assert a is _qmm_kernel(4, 32, mx.bfloat16, use_table=True)
