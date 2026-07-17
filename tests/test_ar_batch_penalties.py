"""Issue #156: presence/frequency penalties must apply under ar_batch.

The batched-AR pump (`_BatchedARGenerationService`) hands per-job sampler
closures to mlx-lm's BatchGenerator. Before the fix, those closures called
`_sample_from_logits` without completion token counts, so penalties silently
no-op'd in exactly the lane the opencode quickstart auto-selects — while the
serial and MTP paths honored them (tests/test_penalties.py).

These tests drive the closure the way BatchGenerator does: one call per
emitted token with a [1, vocab] logprobs row. `_make_sampler` never touches
`self`, so it is exercised directly against a real `_BatchedARJob`.
"""

from __future__ import annotations

import mlx.core as mx

from mtplx.sampling import SamplerConfig
from mtplx.server.openai import _BatchedARGenerationService, _BatchedARJob


def _job(sampler: SamplerConfig, *, seed: int = 0, seed_is_explicit: bool = False):
    return _BatchedARJob(
        request_id="req-penalties-test",
        prompt_ids=[1, 2, 3],
        max_tokens=8,
        sampler=sampler,
        seed=seed,
        stop_token_ids=set(),
        token_callback=None,
        prefill_callback=None,
        request_observability=None,
        mtp_disabled_reason=None,
        generation_limits={},
        seed_is_explicit=seed_is_explicit,
    )


def _sampler_for(job) -> object:
    return _BatchedARGenerationService._make_sampler(None, job)


def _row(*vals: float) -> mx.array:
    return mx.array([list(vals)])


def test_greedy_presence_penalty_flips_repeated_token():
    # Serial parity: generate_ar applies penalties before argmax at temp 0.
    # Token 0 leads token 1 by 1.0; once token 0 has been emitted, a one-off
    # presence penalty of 2.0 drops it below token 1 and the argmax flips.
    job = _job(SamplerConfig(temperature=0.0, presence_penalty=2.0))
    sample = _sampler_for(job)
    logprobs = _row(5.0, 4.0, 0.0, 0.0)
    assert int(sample(logprobs).item()) == 0
    assert int(sample(logprobs).item()) == 1


def test_greedy_frequency_penalty_scales_with_count():
    # frequency is linear in count: with a 0.4 gap and 0.5/occurrence the
    # repeated token flips after ONE emission, and the penalty tracks each
    # token independently (alternation), proving per-token counts are live.
    job = _job(SamplerConfig(temperature=0.0, frequency_penalty=0.5))
    sample = _sampler_for(job)
    logprobs = _row(5.0, 4.6, 0.0, 0.0)
    assert int(sample(logprobs).item()) == 0  # counts {0:1}
    assert int(sample(logprobs).item()) == 1  # 5-0.5=4.5 < 4.6; counts {0:1, 1:1}
    assert int(sample(logprobs).item()) == 0  # 4.6-0.5=4.1 < 4.5


def test_temperature_path_applies_penalties_with_top_k1():
    # temp > 0 goes through the numpy sampling path; top_k=1 makes the flip
    # deterministic while still exercising the distribution lane. Penalties
    # are applied before top-k filtering (see test_penalties.py).
    job = _job(
        SamplerConfig(temperature=1.0, top_p=1.0, top_k=1, presence_penalty=2.0),
        seed=1234,
        seed_is_explicit=True,
    )
    sample = _sampler_for(job)
    logprobs = _row(5.0, 4.0, 0.0, 0.0)
    assert int(sample(logprobs).item()) == 0
    assert int(sample(logprobs).item()) == 1


def test_counts_live_on_the_job_and_survive_sampler_rebuild():
    # The counter is job state, not closure state: a re-built sampler for the
    # same job must keep penalizing tokens emitted through the old closure.
    job = _job(SamplerConfig(temperature=0.0, presence_penalty=2.0))
    first = _sampler_for(job)
    logprobs = _row(5.0, 4.0, 0.0, 0.0)
    assert int(first(logprobs).item()) == 0
    rebuilt = _sampler_for(job)
    assert int(rebuilt(logprobs).item()) == 1
    assert job.completion_token_counts == {0: 1, 1: 1}


def test_zero_penalties_keep_pure_argmax_fast_path():
    # No penalties at temp 0 must stay the raw argmax lambda: no counter
    # writes, no numpy round-trip (the fast paths are perf-critical).
    job = _job(SamplerConfig(temperature=0.0))
    sample = _sampler_for(job)
    logprobs = _row(5.0, 4.0, 0.0, 0.0)
    assert int(sample(logprobs).item()) == 0
    assert int(sample(logprobs).item()) == 0
    assert not job.completion_token_counts


def test_zero_penalties_unseeded_temperature_keeps_fused_sampler():
    # Unseeded, penalty-free temperature jobs must keep mlx-lm's fused GPU
    # sampler (the batch pump's concurrency fast path), not our numpy closure.
    job = _job(SamplerConfig(temperature=0.7, top_p=0.95, top_k=20))
    sample = _sampler_for(job)
    assert getattr(sample, "__name__", "") != "sample_one"
    assert not job.completion_token_counts
