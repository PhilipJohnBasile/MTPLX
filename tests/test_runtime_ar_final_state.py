"""AR final-state capture must never bill or destroy a finished response (F31).

``generate_ar``'s ``capture_final_state`` tail forward extends the cache for
the session-bank committer AFTER the response is complete. Two invariants:

1. Timing: the tail pass is bank bookkeeping, not decode — it must be
   excluded from the reported ``elapsed_s`` (billing it to AR inflates AR
   time and flatters every MTP-vs-AR multiplier).
2. Survival: a failure inside the tail (OOM, kernel error) may degrade only
   the final-state capture — the finished response's tokens, text, timing
   and finish_reason must survive, with the degradation visible in events.

Both tests calibrate the tail's position with a baseline run: generation is
deterministic (greedy, fixed seed), so the tail is the N-th ``forward_ar``
call where N is the baseline's total.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mtplx.generation import generate_ar
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(f"<{int(token)}>" for token in tokens)


class _RampModel:
    """Greedy argmax always walks t -> t+1 over an 8-token vocab."""

    vocab = 8

    def __init__(self):
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def make_cache(self):
        return []

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        tokens = [int(token) for token in np.asarray(input_ids).reshape(-1)]
        row = [0.0] * self.vocab
        row[(tokens[-1] + 1) % self.vocab] = 10.0
        logits = mx.array([[row]], dtype=mx.float32)
        if return_hidden:
            return logits, mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        return logits


def _runtime() -> MTPLXRuntime:
    return MTPLXRuntime(
        model=_RampModel(),
        tokenizer=_Tokenizer(),
        model_path=Path("tiny-ar-final-state"),
        mtp_enabled=False,
        contract=MTPContract(),
    )


def _generate(rt: MTPLXRuntime):
    return generate_ar(
        rt,
        [1],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        seed=0,
        stop_token_ids=set(),
        capture_final_state=True,
    )


def _count_forward_ar_calls(rt: MTPLXRuntime) -> list[int]:
    calls: list[int] = []
    original = rt.forward_ar

    def counting(*args, **kwargs):
        calls.append(len(calls) + 1)
        return original(*args, **kwargs)

    rt.forward_ar = counting
    return calls


def _baseline():
    rt = _runtime()
    calls = _count_forward_ar_calls(rt)
    out = _generate(rt)
    assert out.final_state is not None, "baseline must capture a final state"
    assert out.tokens == [2, 3, 4]
    return out, len(calls)


def test_tail_forward_failure_preserves_completed_response():
    baseline, tail_call = _baseline()

    rt = _runtime()
    original = rt.forward_ar
    seen: list[int] = []

    def failing(*args, **kwargs):
        seen.append(len(seen) + 1)
        if len(seen) == tail_call:
            raise RuntimeError("injected tail OOM")
        return original(*args, **kwargs)

    rt.forward_ar = failing
    out = _generate(rt)

    # The finished response survives intact.
    assert out.tokens == baseline.tokens
    assert out.text == baseline.text
    assert out.finish_reason == baseline.finish_reason
    assert out.stats.generated_tokens == baseline.stats.generated_tokens
    assert out.stats.elapsed_s > 0
    # Only the capture degrades, and visibly.
    assert out.final_state is None
    assert any("final_state_capture_error" in event for event in out.stats.events)
    assert len(seen) == tail_call, "the injected failure must hit the tail call"


def test_tail_forward_is_excluded_from_elapsed():
    _, tail_call = _baseline()

    rt = _runtime()
    original = rt.forward_ar
    seen: list[int] = []

    def slow_tail(*args, **kwargs):
        seen.append(len(seen) + 1)
        if len(seen) == tail_call:
            time.sleep(0.5)
        return original(*args, **kwargs)

    rt.forward_ar = slow_tail
    out = _generate(rt)

    # Capture succeeds, but its cost is not billed to the AR decode window.
    assert out.final_state is not None
    assert len(seen) == tail_call
    assert out.stats.elapsed_s < 0.4, (
        "final-state tail forward leaked into elapsed_s: "
        f"{out.stats.elapsed_s:.3f}s for a ~ms decode"
    )
