"""The startup probe must test the producer the hardware will actually run."""

import pytest

from mtplx.kernels import qsa_indexer_select
from mtplx.models import qwen4_exp
from mtplx.attention_context import attention_phase


@pytest.mark.parametrize("nax,consumer,mpp,expected", [
    (False, True, False, True),
    (False, False, False, False),
    (True, True, False, False),
    (True, True, True, True),
])
@pytest.mark.parametrize("mode", ["auto", "1"])
def test_auto_gate_keeps_portable_lane_and_rejects_broken_mpp(
    monkeypatch, nax, consumer, mpp, expected, mode,
):
    monkeypatch.setenv("MTPLX_QSA_PREFILL", mode)
    monkeypatch.setattr(qsa_indexer_select, "qsa_indexer_select_nax_available", lambda: nax)
    monkeypatch.setattr(qwen4_exp, "qsa_prefill_lane_auto_supported", lambda: consumer)
    calls = []
    monkeypatch.setattr(qwen4_exp, "_qsa_prefill_mpp_compile_ok", lambda: calls.append(True) or mpp)
    assert qwen4_exp._qsa_prefill_enabled() is (expected if mode == "auto" else (not nax or mpp))
    assert bool(calls) is (nax and (consumer or mode == "1"))
    monkeypatch.setenv("MTPLX_QSA_PREFILL", "0")
    assert qwen4_exp._qsa_prefill_enabled() is False


@pytest.mark.parametrize("phase,rows,total", [
    ("ar_decode", 1, 100000), ("decode_verify", 4, 100000),
    ("prefill", 1, 100000), ("prefill", 2048, 2048),
])
def test_decode_and_small_chunks_do_not_resolve_prefill_pipelines(monkeypatch, phase, rows, total):
    calls = []
    monkeypatch.setattr(qwen4_exp, "_qsa_prefill_enabled", lambda: calls.append(True) or True)
    with attention_phase(phase):
        assert qwen4_exp._qsa_large_prefill_enabled(rows, total) is False
    assert not calls
    with attention_phase("prefill"):
        assert qwen4_exp._qsa_large_prefill_enabled(2048, 100000) is True
    assert len(calls) == 1
