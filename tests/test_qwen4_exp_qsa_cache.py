"""QSACache must be a full citizen of the cache contract.

The QSA indexer keeps its own raw-key stream (and derived pooled block keys)
next to the attention KV. The serve loop rolls caches back after every
speculative verify round (``rollback_after_verify``: trim for trimmable
entries, snapshot-restore for the rest) and resumes banked sessions through
``state``. A raw-key stream that only ever appends desyncs from the KV on the
first rollback; once the context crosses the indexer's engage threshold the
selection mask is built from the raw-stream length while attention keys come
from the KV — the ``broadcast_shapes (1,1,4,3719) vs (1,24,4,3715)`` crash
OpenCode hit live at 3.7k ctx (2026-08-27). Below the threshold the same
desync corrupts pooled blocks silently instead of crashing.

All runs are CPU (M-series GPU fp32 matmul is reduced-precision; CPU is the
parity surface).
"""

import mlx.core as mx
import pytest

import mtplx.graphbank as graphbank
from mtplx.cache_state import (
    rollback_after_verify,
    snapshot_untrimmable_cache,
)
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )


@pytest.fixture()
def attn():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    layer = Attention(_tiny_args())
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


def _hidden(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(mx.float32)


PREFILL = 12  # engage threshold with budget=8/ratio=2 is >8 visible tokens
STEP = 4  # a depth-3 verify round: 1 committed + 3 drafts


def test_rollback_then_forward_matches_fresh_run(attn):
    """A rejected verify round must leave the QSA layer exactly where a run
    that never saw the rejected tokens would be."""
    x_pre = _hidden(PREFILL, seed=1)
    x_rejected = _hidden(STEP, seed=2)
    x_next = _hidden(STEP, seed=3)

    cache = [QSACache()]
    attn(x_pre, cache[0])
    snap = snapshot_untrimmable_cache(cache)
    attn(x_rejected, cache[0])
    rollback_after_verify(cache, snap, verified_tokens=STEP)
    assert cache[0].offset == PREFILL
    out = attn(x_next, cache[0])

    fresh = QSACache()
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert out.shape == golden.shape
    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_state_roundtrip_resumes_identically(attn):
    """Bank restore: ``state`` must carry everything the layer needs — a
    resumed session past the engage threshold selects the same blocks and
    produces the same output as the uninterrupted run."""
    x_pre = _hidden(PREFILL, seed=4)
    x_next = _hidden(STEP, seed=5)

    live = QSACache()
    attn(x_pre, live)
    golden = attn(x_next, live)

    donor = QSACache()
    attn(x_pre, donor)
    resumed = QSACache()
    resumed.state = donor.state
    assert resumed.offset == PREFILL
    out = attn(x_next, resumed)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_trim_contract(attn):
    """QSACache is trimmable: trim rolls the layer back token-exactly,
    including through a pooled-block boundary."""
    cache = QSACache()
    assert cache.is_trimmable()

    x_pre = _hidden(PREFILL, seed=6)
    x_tail = _hidden(3, seed=7)  # odd length: trims back through a block edge
    x_next = _hidden(STEP, seed=8)

    attn(x_pre, cache)
    attn(x_tail, cache)
    assert cache.trim(3) == 3
    assert cache.offset == PREFILL
    out = attn(x_next, cache)

    fresh = QSACache()
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_rollback_below_engage_threshold_still_exact(attn):
    """The desync is silent below the engage threshold (dense mask hides it);
    the pooled stream must still be positionally correct once the session
    grows past it."""
    x_pre = _hidden(4, seed=9)
    x_rejected = _hidden(STEP, seed=10)
    # two accepted rounds carry the session across the threshold
    x_a = _hidden(STEP, seed=11)
    x_b = _hidden(STEP, seed=12)
    x_c = _hidden(STEP, seed=13)

    cache = [QSACache()]
    attn(x_pre, cache[0])
    snap = snapshot_untrimmable_cache(cache)
    attn(x_rejected, cache[0])
    rollback_after_verify(cache, snap, verified_tokens=STEP)
    for chunk in (x_a, x_b, x_c):
        out = attn(chunk, cache[0])

    fresh = QSACache()
    attn(x_pre, fresh)
    for chunk in (x_a, x_b, x_c):
        golden = attn(chunk, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_tensor_offset_qsa_cache_trim_matches_stock(attn):
    """The compiled-verifier cache owns fixed banks without changing QSA math."""
    x_pre = _hidden(PREFILL, seed=14)
    x_rejected = _hidden(STEP, seed=15)
    x_next = _hidden(STEP, seed=16)

    cache = [QSACache(compress_ratio=attn.indexer.ratio)]
    attn(x_pre, cache[0])
    promoted, failures = graphbank.promote_kv_cache_offsets(
        cache,
        reserve_tokens=STEP,
        initial_reserve_tokens=16,
    )

    assert promoted == 1
    assert failures == {}
    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    assert cache[0].size() == PREFILL

    attn(x_rejected, cache[0])
    assert cache[0].trim(STEP) == STEP
    out = attn(x_next, cache[0])

    fresh = QSACache(compress_ratio=attn.indexer.ratio)
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_compiled_verify_bank_threads_qsa_state_without_fallback(attn):
    class TinyQSARuntime:
        def __init__(self):
            mx.random.seed(17)
            self.attn = attn
            self.embed = mx.random.normal((32, 64)).astype(mx.float32)
            self.head = mx.random.normal((64, 32)).astype(mx.float32)

        def forward_ar_capture(
            self,
            input_ids,
            *,
            cache,
            return_hidden=True,
            hidden_variant=None,
            capture_backend=None,
        ):
            del hidden_variant, capture_backend
            hidden = self.attn(self.embed[input_ids], cache[0])
            logits = hidden @ self.head
            return logits, hidden, {}

    rt = TinyQSARuntime()
    cache = [QSACache(compress_ratio=attn.indexer.ratio)]
    rt.forward_ar_capture(
        mx.arange(PREFILL, dtype=mx.int32).reshape(1, -1), cache=cache
    )
    bank = graphbank.CompiledVerifyBank(rt, request_max_tokens=16)

    bank.forward_ar_capture(mx.array([[1, 2, 3, 4]]), cache=cache)
    bank.forward_ar_capture(mx.array([[5, 6, 7, 8]]), cache=cache)

    assert bank.stats["fallback_calls"] == 0, bank.stats["fallback_reasons"]
    assert bank.stats["compiled_calls"] == 2
    assert bank.stats["traces"] == 1
    assert isinstance(cache[0], graphbank.TensorOffsetQSACache)
    assert cache[0].size() == PREFILL + 8
