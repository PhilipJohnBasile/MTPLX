"""Family layer-owned capture-commit: repair-free verify rollback.

A rejected speculative window must commit by replaying ONLY the GDN
recurrences (and trimming trimmable entries) from the pre-verify snapshot,
matching a run that never saw the rejected rows to fp32 ulp-class tolerance.
Not bitwise: the chunked gated-delta scan reassociates when the kept rows
ride a wider verify window, so captured activations differ from a fresh
narrow forward's at the last float — the same noise class the fallback
path's own rollback+re-forward produces relative to the verify pass. The
acceptance decision itself always uses the verify pass's own logits, so
this tolerance never touches sampling exactness.

CPU-only (parity surface).
"""

import mlx.core as mx
import numpy as np
import pytest
from types import SimpleNamespace

from mtplx.cache_state import snapshot_untrimmable_cache_lazy
from mtplx.models.qwen4_exp import (
    TextArgs,
    TextModel,
    verify_capture_scope,
)


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=2,
        hc_lowrank=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[2],
        ngram_vocab_size_base=512,
        heads_per_ngram=2,
        ple_embed_dim=64,
    )


@pytest.fixture()
def tm():
    import mlx_lm.models.cache as cache_module
    import mtplx.models.qwen4_exp as qwen4_exp

    prev = mx.default_device()
    previous_arrays_cache = qwen4_exp.ArraysCache
    qwen4_exp.ArraysCache = cache_module.ArraysCache
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    model = TextModel(_tiny_args())
    mx.eval(model.parameters())
    yield model
    qwen4_exp.ArraysCache = previous_arrays_cache
    mx.set_default_device(prev)


def _ids(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.randint(0, 128, (1, tokens))


def _host_ids(ids: mx.array) -> list[int]:
    return [int(token) for token in np.asarray(ids).reshape(-1)]


PREFILL = 12
WINDOW = 4
KEEP = 2


class _FakeSidecar:
    def __init__(self):
        self.direct_inputs = []

    def gather_np(self, flat):
        flat = np.asarray(flat, dtype=np.int64)
        self.direct_inputs.append(flat.copy())
        rows = np.repeat(flat[:, None], 16, axis=1).astype(np.float32)
        return mx.array(rows)

    def __call__(self, ids, dim):
        flat = np.asarray(ids.reshape(-1), dtype=np.int64)
        return self.gather_np(flat).reshape(*ids.shape, dim)


def _run(tm, chunks, cache):
    out = None
    for ids in chunks:
        out = tm.model(ids, cache)
    return out


def test_capture_commit_matches_fresh_run_eager(tm):
    ids_pre = _ids(PREFILL, seed=1)
    ids_verify = _ids(WINDOW, seed=2)
    ids_next = _ids(3, seed=3)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    snap = snapshot_untrimmable_cache_lazy(cache)
    with verify_capture_scope():
        tm.model(ids_verify, cache)
    assert tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    out = _run(tm, [ids_next], cache)

    golden_cache = tm.make_cache()
    tm.model(ids_pre, golden_cache)
    tm.model(ids_verify[:, :KEEP], golden_cache)
    golden = _run(tm, [ids_next], golden_cache)

    assert mx.allclose(out, golden, atol=2e-5, rtol=2e-5).item()


def test_capture_commit_matches_fresh_run_compiled(tm):
    tm.model._gdn_compiled_env = True

    ids_pre = _ids(PREFILL, seed=4)
    ids_verify = _ids(WINDOW, seed=5)
    ids_next = _ids(3, seed=6)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    snap = snapshot_untrimmable_cache_lazy(cache)
    with verify_capture_scope():
        tm.model(ids_verify, cache)
    assert tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    out = _run(tm, [ids_next], cache)

    golden_cache = tm.make_cache()
    tm.model(ids_pre, golden_cache)
    tm.model(ids_verify[:, :KEEP], golden_cache)
    golden = _run(tm, [ids_next], golden_cache)

    # The compiled/eager boundary may reorder float ops; the commit itself
    # must still be exact relative to the same-lane golden (golden ran
    # eager S=2 which the compiled gate also serves) — compare through the
    # compiled lane end to end.
    assert mx.allclose(out, golden, atol=2e-5, rtol=2e-5).item()


def test_commit_refuses_without_capture_and_leaves_cache_intact(tm):
    ids_pre = _ids(PREFILL, seed=7)
    ids_verify = _ids(WINDOW, seed=8)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    snap = snapshot_untrimmable_cache_lazy(cache)
    tm.model(ids_verify, cache)  # NOT captured
    offsets_before = [getattr(c, "offset", None) for c in cache]
    assert not tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    assert [getattr(c, "offset", None) for c in cache] == offsets_before


def test_full_accept_needs_no_commit_and_next_round_overwrites_rows(tm):
    ids_pre = _ids(PREFILL, seed=9)
    ids_v1 = _ids(WINDOW, seed=10)
    ids_v2 = _ids(WINDOW, seed=11)

    cache = tm.make_cache()
    tm.model(ids_pre, cache)
    with verify_capture_scope():
        tm.model(ids_v1, cache)  # full accept: no commit
    snap = snapshot_untrimmable_cache_lazy(cache)
    with verify_capture_scope():
        tm.model(ids_v2, cache)
    assert tm.model.commit_verified_window(
        cache, snap.states, keep_tokens=KEEP, verified_tokens=WINDOW
    )
    out = _run(tm, [_ids(2, seed=12)], cache)

    golden_cache = tm.make_cache()
    tm.model(ids_pre, golden_cache)
    tm.model(ids_v1, golden_cache)
    tm.model(ids_v2[:, :KEEP], golden_cache)
    golden = _run(tm, [_ids(2, seed=12)], golden_cache)

    assert mx.allclose(out, golden, atol=2e-5, rtol=2e-5).item()


def test_fixed_m4_capture_route_returns_family_commit_rows(tm):
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    report = install_qwen4_fixed_verify_route(runtime)
    cache = tm.make_cache()
    ids_pre = _ids(PREFILL, seed=18)
    ids_verify = _ids(WINDOW, seed=19)
    tm(ids_pre, cache=cache)

    aux = runtime.prepare_compiled_verify_aux(ids_verify, cache)
    logits, hidden, captures = runtime.forward_ar_capture(
        ids_verify,
        cache=cache,
        return_hidden=True,
        compiled_aux=aux,
    )
    mx.eval(logits, hidden)

    linear = [i for i, layer in enumerate(tm.model.layers) if layer.is_linear]
    ple_index = next(
        i for i, layer in enumerate(tm.model.layers) if getattr(layer, "ple", None)
    )
    assert report == {"installed": True, "linear_layers": len(linear), "rows": 4}
    assert all(
        tuple(captures[i])[:6] == ("qkv", "q", "k", "v", "a", "b") for i in linear
    )
    assert {"ple_hidden", "ple_ids"}.issubset(captures[ple_index])

    tm.model.clear_verify_capture(cache)
    runtime.commit_compiled_verify_captures(cache, captures)
    assert all(cache[i]._mtplx_verify_rows is not None for i in linear)
    assert cache[ple_index]._mtplx_verify_ple is not None


@pytest.mark.parametrize(
    ("tokens", "previous"),
    (
        ((3, 4, 5, 6), None),
        ((0, 4, 5, 6), None),
        ((3, 0, 5, 6), None),
        ((3, 4, 0, 6), None),
        ((3, 4, 5, 0), None),
        ((3, 4, 5, 6), (0, 9)),
    ),
)
def test_fixed_m4_sidecar_aux_stages_exact_rows_without_mutating_history(
    tm, tokens, previous
):
    from mtplx.qwen4_fixed_verify import (
        _prepare_compiled_verify_aux,
        install_qwen4_fixed_verify_route,
    )

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    cache = tm.make_cache()
    prefill = _ids(PREFILL, seed=28)
    tm(prefill, cache=cache)
    ple_index = next(
        i for i, layer in enumerate(tm.model.layers) if getattr(layer, "ple", None)
    )
    ple = tm.model.layers[ple_index].ple
    if previous is not None:
        cache[ple_index][ple.NGRAM_IDX] = mx.array([previous], dtype=mx.int64)
    sidecar = _FakeSidecar()
    ple.ple_embedding.ngram_embedding._sidecar = sidecar
    install_qwen4_fixed_verify_route(runtime)

    prompt_ids = list(previous) if previous is not None else _host_ids(prefill)
    prepare = runtime.build_fixed_m4_compiled_verify_aux(cache, prompt_ids)
    ids = mx.array([tokens])
    history_before = cache[ple_index][ple.NGRAM_IDX]
    reference = _prepare_compiled_verify_aux(runtime, ids, cache)
    candidate = prepare(ids, list(tokens), [int(tokens[0])], 0)
    mx.eval(reference, candidate)

    assert mx.array_equal(candidate, reference).item()
    assert candidate.shape == (1, WINDOW, 64)
    assert cache[ple_index][ple.NGRAM_IDX] is history_before
    assert len(sidecar.direct_inputs) == 2
    assert np.array_equal(sidecar.direct_inputs[0], sidecar.direct_inputs[1])

    rebound_history = mx.array([[0, 11]])
    cache[ple_index][ple.NGRAM_IDX] = rebound_history
    second_ids = mx.array([[9, 8, 7, 6]])
    second_reference = _prepare_compiled_verify_aux(runtime, second_ids, cache)
    second_candidate = prepare(second_ids, [9, 8, 7, 6], [0, 11, 9], 2)
    mx.eval(second_reference, second_candidate)

    assert mx.array_equal(second_candidate, second_reference).item()
    assert cache[ple_index][ple.NGRAM_IDX] is rebound_history
    assert len(sidecar.direct_inputs) == 4
    assert np.array_equal(sidecar.direct_inputs[2], sidecar.direct_inputs[3])


def test_compiled_fixed_m4_route_preserves_family_prefix_commit(tm, monkeypatch):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)

    cache = tm.make_cache()
    ids_pre = _ids(PREFILL, seed=20)
    ids_verify = _ids(WINDOW, seed=21)
    tm(ids_pre, cache=cache)
    snap = snapshot_untrimmable_cache_lazy(cache)
    bank = graphbank.CompiledVerifyBank(
        runtime,
        max_verify_len=WINDOW,
        request_max_tokens=16,
    )

    logits, hidden, captures = bank.forward_ar_capture(ids_verify, cache=cache)
    mx.eval(logits, hidden)

    assert bank.stats["fallback_calls"] == 0, bank.stats["fallback_reasons"]
    assert bank.stats["compiled_calls"] == 1
    assert captures
    assert tm.model.commit_verified_window(
        cache,
        snap.states,
        keep_tokens=KEEP,
        verified_tokens=WINDOW,
    )


def test_installed_fixed_m4_replay_preserves_compiled_gdn_schedule(tm, monkeypatch):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)

    cache = tm.make_cache()
    prefill = _ids(PREFILL, seed=23)
    tm(prefill, cache=cache)
    tm.model._gdn_compiled_env = True

    compiled_gdn_calls = []
    original_compiled_gdn = tm.model._decode_layers_compiled

    def observed_compiled_gdn(*args, **kwargs):
        compiled_gdn_calls.append(True)
        return original_compiled_gdn(*args, **kwargs)

    monkeypatch.setattr(tm.model, "_decode_layers_compiled", observed_compiled_gdn)
    bank = graphbank.CompiledVerifyBank(
        runtime,
        max_verify_len=WINDOW,
        request_max_tokens=16,
    )
    bank.install_fixed_m4(
        cache,
        prompt_ids=_host_ids(prefill),
        hidden_variant=None,
    )

    def repeated_check(*_args, **_kwargs):
        raise AssertionError("installed M4 replay re-entered generic dispatch")

    monkeypatch.setattr(bank, "_fallback_reason", repeated_check)
    monkeypatch.setattr(bank, "_resolve_bucket", repeated_check)
    monkeypatch.setattr(bank, "_ensure_shadow", repeated_check)
    monkeypatch.setattr(bank, "_paged_ineligibility", repeated_check)

    completion_tokens = []
    for seed in (24, 25):
        snap = snapshot_untrimmable_cache_lazy(cache)
        ids = _ids(WINDOW, seed=seed)
        host_ids = _host_ids(ids)
        completion_tokens.append(host_ids[0])
        logits, hidden, captures = bank.forward_fixed_m4(
            ids,
            host_input_ids=host_ids,
            completion_tokens=completion_tokens,
            committed_count=len(completion_tokens) - 1,
            cache=cache,
        )
        mx.eval(logits, hidden)
        assert captures == {}
        assert tm.model.commit_verified_window(
            cache,
            snap.states,
            keep_tokens=KEEP,
            verified_tokens=WINDOW,
        )
        completion_tokens.extend(host_ids[1:KEEP])

    assert bank.stats["compiled_calls"] == 2
    assert compiled_gdn_calls


@pytest.mark.parametrize(
    ("boundary", "staged_builds"),
    (("both", 1), ("pre", 1), ("post", 0), ("none", 0)),
)
def test_fixed_m4_staged_aux_requires_pre_schedule(
    tm, monkeypatch, boundary, staged_builds
):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_BOUNDARY", boundary)
    cache = tm.make_cache()
    prefill = _ids(PREFILL, seed=30)
    tm(prefill, cache=cache)
    ple_index = next(
        i for i, layer in enumerate(tm.model.layers) if getattr(layer, "ple", None)
    )
    tm.model.layers[
        ple_index
    ].ple.ple_embedding.ngram_embedding._sidecar = _FakeSidecar()
    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    real_build = runtime.build_fixed_m4_compiled_verify_aux
    builds = []

    def build(_cache, prompt_ids):
        builds.append(True)
        return real_build(_cache, prompt_ids)

    runtime.build_fixed_m4_compiled_verify_aux = build
    bank = graphbank.CompiledVerifyBank(
        runtime,
        max_verify_len=WINDOW,
        request_max_tokens=16,
    )
    bank.install_fixed_m4(
        cache,
        prompt_ids=_host_ids(prefill),
        hidden_variant=None,
    )

    ids = _ids(WINDOW, seed=31)
    host_ids = _host_ids(ids)
    logits, hidden, captures = bank.forward_fixed_m4(
        ids,
        host_input_ids=host_ids,
        completion_tokens=[host_ids[0]],
        committed_count=0,
        cache=cache,
    )
    mx.eval(logits, hidden)

    assert len(builds) == staged_builds
    assert captures == {}
    assert bank.stats["compiled_calls"] == 1


def test_fixed_m4_staged_sidecar_matches_materialized_route_across_windows(
    tm, monkeypatch
):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    prefill = _ids(PREFILL, seed=32)
    staged_cache = tm.make_cache()
    materialized_cache = tm.make_cache()
    tm(prefill, cache=staged_cache)
    tm(prefill, cache=materialized_cache)
    ple_index = next(
        i for i, layer in enumerate(tm.model.layers) if getattr(layer, "ple", None)
    )
    ple = tm.model.layers[ple_index].ple
    ple.ple_embedding.ngram_embedding._sidecar = _FakeSidecar()
    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_BOUNDARY", "both")

    staged_bank = graphbank.CompiledVerifyBank(
        runtime, max_verify_len=WINDOW, request_max_tokens=16
    )
    prompt_ids = _host_ids(prefill)
    staged_bank.install_fixed_m4(
        staged_cache,
        prompt_ids=prompt_ids,
        hidden_variant=None,
    )
    materialized_bank = graphbank.CompiledVerifyBank(
        runtime, max_verify_len=WINDOW, request_max_tokens=16
    )
    materialized_bank._build_fixed_m4_aux = None
    materialized_bank.install_fixed_m4(
        materialized_cache,
        prompt_ids=prompt_ids,
        hidden_variant=None,
    )

    completion_tokens = []
    for seed in (33, 34):
        ids = _ids(WINDOW, seed=seed)
        host_ids = _host_ids(ids)
        completion_tokens.append(host_ids[0])
        staged_logits, staged_hidden, staged_captures = staged_bank.forward_fixed_m4(
            ids,
            host_input_ids=host_ids,
            completion_tokens=completion_tokens,
            committed_count=len(completion_tokens) - 1,
            cache=staged_cache,
        )
        reference_logits, reference_hidden, reference_captures = (
            materialized_bank.forward_fixed_m4(
                ids,
                host_input_ids=host_ids,
                completion_tokens=completion_tokens,
                committed_count=len(completion_tokens) - 1,
                cache=materialized_cache,
            )
        )
        mx.eval(staged_logits, staged_hidden, reference_logits, reference_hidden)

        assert mx.array_equal(staged_logits, reference_logits).item()
        assert mx.array_equal(staged_hidden, reference_hidden).item()
        assert mx.array_equal(
            staged_cache[ple_index][ple.NGRAM_IDX],
            materialized_cache[ple_index][ple.NGRAM_IDX],
        ).item()
        assert staged_captures == reference_captures == {}
        completion_tokens.extend(host_ids[1:])

    assert staged_bank.stats["compiled_calls"] == 2
    assert materialized_bank.stats["compiled_calls"] == 2


def test_installed_fixed_m4_routes_shorter_windows_to_family_capture(tm, monkeypatch):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_PREWARM_DONE", True)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)

    cache = tm.make_cache()
    prefill = _ids(PREFILL, seed=26)
    tm(prefill, cache=cache)
    bank = graphbank.CompiledVerifyBank(
        runtime,
        max_verify_len=WINDOW,
        request_max_tokens=16,
    )
    bank.install_fixed_m4(
        cache,
        prompt_ids=_host_ids(prefill),
        hidden_variant=None,
    )

    logits, hidden, captures = bank.forward_ar_capture(_ids(2, seed=27), cache=cache)
    mx.eval(logits, hidden)

    assert captures
    assert bank.stats["compiled_calls"] == 0
    assert bank.stats["fallback_calls"] == 0


def test_fixed_m4_bank_is_selected_only_for_its_construction_contract():
    from mtplx.generation import _qwen4_fixed_m4_compiled_verify_requested

    runtime = SimpleNamespace(qwen4_fixed_m4_compiled_verify=True)
    assert _qwen4_fixed_m4_compiled_verify_requested(
        runtime,
        verify_strategy="batched",
        compiled_mode="on",
        max_tokens=1024,
        cached_tokens=0,
    )
    assert not _qwen4_fixed_m4_compiled_verify_requested(
        runtime,
        verify_strategy="capture_commit",
        compiled_mode="on",
        max_tokens=1024,
        cached_tokens=0,
    )
    assert not _qwen4_fixed_m4_compiled_verify_requested(
        runtime,
        verify_strategy="batched",
        compiled_mode="on",
        max_tokens=1025,
        cached_tokens=0,
    )
    assert not _qwen4_fixed_m4_compiled_verify_requested(
        runtime,
        verify_strategy="batched",
        compiled_mode="on",
        max_tokens=1024,
        cached_tokens=512,
    )


def test_fixed_m4_bank_fails_loud_instead_of_falling_back(tm, monkeypatch):
    import mtplx.graphbank as graphbank
    from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route

    class TinyRuntime:
        pass

    runtime = TinyRuntime()
    runtime.model = SimpleNamespace(language_model=tm)
    install_qwen4_fixed_verify_route(runtime)
    monkeypatch.setattr(graphbank, "_compiled_verify_bits_gate_ok", lambda _rt: True)
    bank = graphbank.CompiledVerifyBank(
        runtime, max_verify_len=4, request_max_tokens=16
    )
    monkeypatch.setattr(bank, "_fallback_reason", lambda *args, **kwargs: "forced")

    with pytest.raises(RuntimeError, match="fixed-M4 verifier refused: forced"):
        bank.forward_ar_capture(_ids(WINDOW, seed=22), cache=tm.make_cache())
