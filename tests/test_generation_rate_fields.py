"""Decode-rate attribution: restore overhead must never read as decode time.

Probe receipt 2026-08-30 (M5 Max, Flash-Next 91K warm turn): a 384-token
completion whose session restore stalled 10.7 s in snapshot-view COW
divergence read decode_tok_s=21.3 while its own sliding decode windows ran
50-62 tok/s. The restore machinery outside measured prefill/restore compute
was being charged into decode_elapsed_s. `non_decode_extra_s` carries it.
"""

from mtplx.generation import _generation_rate_fields


def test_restore_overhead_excluded_from_decode():
    # Turn-4 replay: 18.48 s wall, 0.43 s measured prefill+restore,
    # 10.7 s unattributed restore machinery, 384 tokens.
    fields = _generation_rate_fields(
        generated_tokens=384,
        elapsed_s=18.48,
        prompt_eval_time_s=0.425,
        cache_restore_time_s=0.004,
        non_decode_extra_s=10.696,
    )
    assert 51.0 < fields["decode_tok_s"] < 53.0
    assert abs(fields["decode_elapsed_s"] - 7.355) < 1e-6
    # End-to-end keeps telling the whole-wall truth.
    assert abs(fields["end_to_end_tok_s"] - 384 / 18.48) < 1e-9


def test_default_extra_keeps_previous_semantics():
    old = _generation_rate_fields(
        generated_tokens=100,
        elapsed_s=10.0,
        prompt_eval_time_s=2.0,
        cache_restore_time_s=0.5,
    )
    assert abs(old["decode_elapsed_s"] - 7.5) < 1e-9
    assert abs(old["decode_tok_s"] - 100 / 7.5) < 1e-9


def test_non_decode_clamped_at_elapsed():
    fields = _generation_rate_fields(
        generated_tokens=10,
        elapsed_s=5.0,
        prompt_eval_time_s=3.0,
        cache_restore_time_s=1.0,
        non_decode_extra_s=9.0,
    )
    assert fields["decode_elapsed_s"] == 0.0
    assert fields["decode_tok_s"] == 0.0
