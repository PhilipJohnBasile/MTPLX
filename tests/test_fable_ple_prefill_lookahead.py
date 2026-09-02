"""MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD: one-slot lifecycle and exactness.

Pure Python + NumPy.  ``mtplx.ple_prefill_lookahead`` imports no MLX, and the
sidecar half is exercised against a real on-disk memmap built in tmp_path at
the production row geometry -- no GPU, no MLX array, no model load.
"""

from __future__ import annotations

import ast
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from mtplx import ple_prefill_lookahead as lookahead_mod
from mtplx.ple_prefill_lookahead import PrefillLookahead, prefill_lookahead_scope


ROOT = Path(__file__).resolve().parents[1]
MODEL_TEXT = (ROOT / "mtplx" / "models" / "qwen4_exp.py").read_text("utf-8")
GENERATION_TEXT = (ROOT / "mtplx" / "generation.py").read_text("utf-8")
PROFILES_TEXT = (ROOT / "mtplx" / "profiles.py").read_text("utf-8")


class _Immediate:
    """A future whose work already ran -- keeps the tests single-threaded."""

    def __init__(self, fn, *args):
        self._exc = None
        try:
            self._value = fn(*args)
        except BaseException as exc:  # noqa: BLE001 - re-raised in result()
            self._value, self._exc = None, exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._value

    def cancel(self):
        return False


def immediate_submit(fn, *args):
    return _Immediate(fn, *args)


@pytest.fixture(autouse=True)
def _clean_counters():
    lookahead_mod.reset_counters()
    yield
    lookahead_mod.reset_counters()


def spans_for(total: int, chunk: int) -> list[tuple[int, int]]:
    return [
        (start, min(total, start + chunk)) for start in range(0, total, chunk)
    ]


def build(total=64, chunk=16, prepare=None, submit=immediate_submit):
    ids = np.arange(1000, 1000 + total, dtype=np.int64)
    calls: list[tuple[int, int]] = []

    def default_prepare(start, end):
        calls.append((start, end))
        return (np.asarray(ids[start:end]), {"weight": ids[start:end] * 2})

    return (
        PrefillLookahead(
            ids,
            spans_for(total, chunk),
            prepare=prepare or default_prepare,
            submit=submit,
        ),
        ids,
        calls,
    )


# ---------------------------------------------------------------------------
# Env resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "ON"])
def test_flag_true_spellings(monkeypatch, value):
    monkeypatch.setenv(lookahead_mod.ENV_FLAG, value)
    lookahead_mod.enabled.cache_clear()
    try:
        assert lookahead_mod.enabled() is True
    finally:
        lookahead_mod.enabled.cache_clear()


@pytest.mark.parametrize("value", ["", "0", "off", "NO"])
def test_flag_false_spellings(monkeypatch, value):
    monkeypatch.setenv(lookahead_mod.ENV_FLAG, value)
    lookahead_mod.enabled.cache_clear()
    try:
        assert lookahead_mod.enabled() is False
    finally:
        lookahead_mod.enabled.cache_clear()


def test_flag_default_is_off(monkeypatch):
    monkeypatch.delenv(lookahead_mod.ENV_FLAG, raising=False)
    lookahead_mod.enabled.cache_clear()
    try:
        assert lookahead_mod.enabled() is False
    finally:
        lookahead_mod.enabled.cache_clear()


def test_unparseable_flag_raises(monkeypatch):
    monkeypatch.setenv(lookahead_mod.ENV_FLAG, "sometimes")
    lookahead_mod.enabled.cache_clear()
    try:
        with pytest.raises(ValueError):
            lookahead_mod.enabled()
    finally:
        lookahead_mod.enabled.cache_clear()


def test_flag_is_read_once(monkeypatch):
    monkeypatch.setenv(lookahead_mod.ENV_FLAG, "1")
    lookahead_mod.enabled.cache_clear()
    try:
        assert lookahead_mod.enabled() is True
        monkeypatch.setenv(lookahead_mod.ENV_FLAG, "0")
        assert lookahead_mod.enabled() is True
    finally:
        lookahead_mod.enabled.cache_clear()


# ---------------------------------------------------------------------------
# One-slot lifecycle
# ---------------------------------------------------------------------------


def test_span_index_of_matches_by_content_and_advances_the_hint():
    look, ids, _ = build()
    assert look.span_index_of(ids[16:32]) == 1
    assert look.span_index_of(ids[32:48]) == 2
    assert look.span_index_of(ids[0:16]) == 0  # wraps, still correct


def test_span_index_of_rejects_tokens_that_are_not_in_the_plan():
    look, ids, _ = build()
    assert look.span_index_of(np.zeros(16, dtype=np.int64)) is None
    assert look.span_index_of(ids[:8]) is None  # wrong width


def test_take_returns_the_prepared_payload_for_the_matching_span():
    look, ids, calls = build()
    look.submit(0)
    flat, mats = look.take(0)
    assert calls == [(0, 16)]
    np.testing.assert_array_equal(flat, ids[0:16])
    np.testing.assert_array_equal(mats["weight"], ids[0:16] * 2)
    assert lookahead_mod.COUNTERS["hit"] == 1


def test_only_one_slot_is_ever_held():
    look, _ids, calls = build()
    look.submit(0)
    look.submit(1)  # ignored: the slot is taken
    assert calls == [(0, 16)]
    assert lookahead_mod.COUNTERS["submitted"] == 1


def test_take_of_a_different_span_discards_and_counts_the_miss():
    look, _ids, _calls = build()
    look.submit(0)
    assert look.take(2) is None
    assert lookahead_mod.COUNTERS["miss_wrong_span"] == 1
    assert look.take(0) is None  # the slot was released, not re-served
    assert lookahead_mod.COUNTERS["miss_empty"] == 1


def test_take_on_an_empty_slot_counts_a_miss():
    look, _ids, _calls = build()
    assert look.take(0) is None
    assert lookahead_mod.COUNTERS["miss_empty"] == 1


def test_ineligible_preparation_is_a_counted_miss_not_a_crash():
    look, _ids, _calls = build(prepare=lambda start, end: None)
    look.submit(0)
    assert look.take(0) is None
    assert lookahead_mod.COUNTERS["miss_ineligible"] == 1


def test_worker_exception_surfaces_and_is_counted():
    def boom(start, end):
        raise RuntimeError("sidecar read failed")

    look, _ids, _calls = build(prepare=boom)
    look.submit(0)
    with pytest.raises(RuntimeError, match="sidecar read failed"):
        look.take(0)
    assert lookahead_mod.COUNTERS["worker_error"] == 1


def test_next_index_stops_at_the_last_span():
    look, _ids, _calls = build(total=64, chunk=16)
    assert look.next_index(0) == 1
    assert look.next_index(3) is None
    look.submit(None)  # a None index is a no-op, not a crash
    assert "submitted" not in lookahead_mod.COUNTERS


def test_close_discards_a_pending_slot_and_is_idempotent():
    look, _ids, _calls = build()
    look.submit(0)
    look.close()
    assert lookahead_mod.COUNTERS["discarded_on_close"] == 1
    look.close()
    assert lookahead_mod.COUNTERS["discarded_on_close"] == 1


def test_submit_after_close_is_inert():
    look, _ids, calls = build()
    look.close()
    look.submit(0)
    assert calls == []


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_scope_publishes_the_lookahead_and_primes_chunk_zero():
    look, ids, calls = build()
    assert lookahead_mod.active_lookahead() is None
    with prefill_lookahead_scope(look) as active:
        assert active is look
        assert lookahead_mod.active_lookahead() is look
        # Chunk 0 is prepared before the caller does anything else.
        assert calls == [(0, 16)]
        for start in range(0, 64, 16):
            index = look.span_index_of(ids[start : start + 16])
            look.take(index)
            look.submit(look.next_index(index))
    assert lookahead_mod.active_lookahead() is None


def test_scope_closes_the_lookahead_even_when_the_prefill_raises():
    look, _ids, _calls = build()
    with pytest.raises(ValueError):
        with prefill_lookahead_scope(look):
            raise ValueError("aborted prefill")
    assert lookahead_mod.active_lookahead() is None
    assert look._closed is True


def test_none_scope_is_a_no_op():
    with prefill_lookahead_scope(None) as active:
        assert active is None
        assert lookahead_mod.active_lookahead() is None


def test_real_worker_thread_completes_and_shuts_down():
    ids = np.arange(64, dtype=np.int64)
    seen: list[str] = []

    def prepare(start, end):
        seen.append(os.environ.get("__unused__", "") or "ran")
        return (ids[start:end], {"weight": ids[start:end]})

    look = PrefillLookahead(ids, spans_for(64, 16), prepare=prepare)
    try:
        look.submit(0)
        flat, _mats = look.take(0)
        np.testing.assert_array_equal(flat, ids[:16])
        assert seen == ["ran"]
    finally:
        look.close()
    assert look._pool is None


# ---------------------------------------------------------------------------
# Sidecar: the worker half must return exactly the owner's bytes
# ---------------------------------------------------------------------------

# Production geometry: 16 ngram heads/token, head_dim 160, q4/g32.
ROW_U32, ROW_SCALES, ROW_BIASES = 20, 5, 5


class FakeSidecar:
    """`_SidecarGather.prepare_rows_np` transplanted onto real memmaps.

    The method under test only touches ``_maps``, ``_row_meta``, ``_pool``,
    ``_hot_cap_rows``, ``_HOT_PATH_MAX_ROWS`` and the two batch counters, so
    binding the real implementation to this stand-in exercises the shipped
    code without importing MLX.
    """

    _HOT_PATH_MAX_ROWS = 4096

    def __init__(self, tmp_path: Path, rows: int, *, hot_cap_rows: int = 1000):
        self.bits = 4
        self.group_size = 32
        self._maps = {}
        self._row_meta = []
        rng = np.random.default_rng(20260901)
        for name, width, dtype, tag in (
            ("weight", ROW_U32, np.uint32, "U32"),
            ("scales", ROW_SCALES, np.uint16, "BF16"),
            ("biases", ROW_BIASES, np.uint16, "BF16"),
        ):
            path = tmp_path / f"{name}.bin"
            data = rng.integers(
                0, np.iinfo(dtype).max, size=(rows, width), dtype=dtype
            )
            path.write_bytes(data.tobytes())
            mm = np.memmap(path, mode="r", dtype=dtype, shape=(rows, width))
            self._maps[name] = (mm, tag)
            self._row_meta.append((0, width * dtype().itemsize))
        self._fd = os.open(str(tmp_path / "weight.bin"), os.O_RDONLY)
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._hot_cap_rows = hot_cap_rows
        self.prefetch_batches = 0
        self.lookahead_batches = 0

    def close(self):
        self._pool.shutdown(wait=True)
        os.close(self._fd)


def _bind(method_name: str):
    """One `_SidecarGather` method, compiled from the SHIPPED source.

    Calling it with a FakeSidecar as ``self`` runs the real implementation, so
    these tests cannot pass against a stale copy of it.
    """

    return _bind_method("_SidecarGather", method_name)


@pytest.fixture
def sidecar(tmp_path):
    sc = FakeSidecar(tmp_path, rows=200_000)
    try:
        yield sc
    finally:
        sc.close()


def owner_matrices(sidecar, flat, names=("weight", "scales", "biases")):
    """What `_rows_matrices`'s big-gather branch produces on the owner thread."""

    uniq, inverse = np.unique(flat, return_inverse=True)
    return {
        name: np.ascontiguousarray(sidecar._maps[name][0][uniq])[inverse]
        for name in names
    }


def test_prepared_rows_are_bit_identical_to_the_owner_path(sidecar):
    prepare = _bind("prepare_rows_np")
    warm = _bind("_warm")
    submit = _bind("_submit_warm")
    sidecar._warm = lambda rows, counted=True: warm(
        sidecar, rows, counted=counted
    )
    sidecar._submit_warm = lambda rows, counted: submit(
        sidecar, rows, counted=counted
    )

    rng = np.random.default_rng(3)
    flat = rng.integers(0, 200_000, size=32_768, dtype=np.int64)
    unique, mats = prepare(sidecar, flat)
    expected = owner_matrices(sidecar, flat)
    assert unique == len(np.unique(flat))
    for name, rows in expected.items():
        assert mats[name].dtype == rows.dtype
        assert mats[name].shape == rows.shape
        np.testing.assert_array_equal(mats[name], rows)


def test_worker_batches_are_counted_apart_from_prefetch_batches(sidecar):
    prepare = _bind("prepare_rows_np")
    warm = _bind("_warm")
    submit = _bind("_submit_warm")
    sidecar._warm = lambda rows, counted=True: warm(
        sidecar, rows, counted=counted
    )
    sidecar._submit_warm = lambda rows, counted: submit(
        sidecar, rows, counted=counted
    )
    prepare(sidecar, np.arange(20_000, dtype=np.int64))
    assert sidecar.lookahead_batches == 1
    assert sidecar.prefetch_batches == 0


def test_hot_lru_sized_gathers_are_refused_by_the_worker(sidecar):
    """The LRU is owner-thread-only state; the worker must not reach it."""

    prepare = _bind("prepare_rows_np")
    small = np.arange(sidecar._HOT_PATH_MAX_ROWS, dtype=np.int64)
    assert prepare(sidecar, small) is None


def test_worker_serves_a_big_gather_when_the_lru_is_disabled(sidecar):
    prepare = _bind("prepare_rows_np")
    warm = _bind("_warm")
    submit = _bind("_submit_warm")
    sidecar._warm = lambda rows, counted=True: warm(
        sidecar, rows, counted=counted
    )
    sidecar._submit_warm = lambda rows, counted: submit(
        sidecar, rows, counted=counted
    )
    sidecar._hot_cap_rows = 0
    small = np.arange(64, dtype=np.int64)
    unique, mats = prepare(sidecar, small)
    assert unique == 64
    np.testing.assert_array_equal(
        mats["weight"], owner_matrices(sidecar, small)["weight"]
    )


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_stage_consumes_the_lookahead_before_it_queues_the_next_chunk():
    helper = MODEL_TEXT.split("def _take_prefill_lookahead", 1)[1].split(
        "\n    def ", 1
    )[0]
    take = helper.index("lookahead.take(index)")
    submit = helper.index("lookahead.submit(lookahead.next_index(index))")
    assert take < submit, "take must free the slot before submit fills it"
    assert "np.array_equal(worker_flat, flat)" in helper
    assert 'count("miss_row_mismatch")' in helper
    assert 'count("miss_unknown_span")' in helper


def test_worker_entry_creates_no_mlx_array():
    body = MODEL_TEXT.split("def prefill_lookahead_prepare", 1)[1].split(
        "\n    def ", 1
    )[0]
    assert "mx." not in body
    assert "sidecar.prepare_rows_np(flat, names)" in body


def test_lookahead_module_never_imports_mlx():
    text = (ROOT / "mtplx" / "ple_prefill_lookahead.py").read_text("utf-8")
    assert "mlx" not in text
    assert "import mx" not in text


def test_generation_wraps_the_chunk_loop_and_the_scope_owns_the_worker():
    assert "_ple_prefill_lookahead_scope(rt, body, mtp_streaming_spans)" in (
        GENERATION_TEXT
    )
    helper = GENERATION_TEXT.split("def _ple_prefill_lookahead_scope", 1)[
        1
    ].split("\ndef ", 1)[0]
    assert "_resolve_ple_lookahead_hook(rt)" in helper
    assert "prefill_lookahead_scope(hook(body, list(spans)))" in helper


def test_model_builder_refuses_to_run_without_the_staged_sidecar():
    builder = MODEL_TEXT.split("def ple_prefill_lookahead", 1)[1].split(
        "\n    def ", 1
    )[0]
    assert "if not lookahead_mod.enabled():" in builder
    assert builder.count("raise RuntimeError") == 3
    assert "embedding._np_consts()" in builder


def test_flag_is_registered_for_validated_operator_overrides():
    assert f'"{lookahead_mod.ENV_FLAG}"' in PROFILES_TEXT


# ---------------------------------------------------------------------------
# The exactness claim: the worker rebuilds the SAME rows the owner would
# ---------------------------------------------------------------------------


def _compile_from_source(node) -> dict:
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {"np": np, "os": os}
    exec(compile(module, "<qwen4_exp>", "exec"), namespace)
    return namespace


def _bind_top_level(name: str):
    """Compile one module-level function out of the real qwen4_exp source."""

    node = next(
        n
        for n in ast.parse(MODEL_TEXT).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return _compile_from_source(node)[name]


def _bind_method(class_name: str, method_name: str):
    """Compile one class method out of the real qwen4_exp source, unbound."""

    cls = next(
        n
        for n in ast.walk(ast.parse(MODEL_TEXT))
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    node = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    return _compile_from_source(node)[method_name]


class _FakeEmbedding:
    """Just enough of NGramEmbedding to run the shipped row arithmetic.

    Production geometry for Qwen3.8 Flash-Next: ngram_size 3, 8 heads per
    ngram -> 16 rows per token; EOS 248044 from the model's config.json.
    """

    ngram_size = 3
    heads_per_ngram = 8
    context_len = ngram_size - 1
    eos_id = 248_044

    def __init__(self, ngram_heads=16, vocab=200_000):
        self._mult = np.array([2_654_435_761, 40_503, 1_337], dtype=np.int64)
        self._sizes = np.full(ngram_heads, vocab // ngram_heads, dtype=np.int64)
        self._offs = (
            np.arange(ngram_heads, dtype=np.int64) * (vocab // ngram_heads)
        )
        self._ngram_rows_np = _bind_top_level("_ngram_rows_np")

    def _rows_np(self, ids_np, prev_np):
        return self._ngram_rows_np(
            ids_np,
            prev_np,
            mult=self._mult,
            sizes=self._sizes,
            offs=self._offs,
            eos=self.eos_id,
            ngram_size=self.ngram_size,
            heads_per_ngram=self.heads_per_ngram,
        )


class _EchoSidecar:
    """Returns the ids themselves, so the test compares row ids, not bytes."""

    bits = 4

    def prepare_rows_np(self, flat, names):
        return (int(len(np.unique(flat))), {name: flat for name in names})


def _embedding_with(sidecar):
    embedding = _FakeEmbedding()
    embedding.ngram_embedding = type("_T", (), {"_sidecar": sidecar})()
    return embedding


def test_worker_history_matches_the_owner_cache_history_chunk_for_chunk():
    """The claim the whole lane rests on, at the production chunk geometry.

    The owner derives each chunk's PLE history from the live state cache; the
    worker derives it from the plan.  If they ever disagreed, the mismatch
    guard in `_take_prefill_lookahead` would silently kill every hit -- so
    prove they agree rather than relying on the guard to hide it.
    """

    prepare = _bind_method("NGramEmbedding", "prefill_lookahead_prepare")
    embedding = _embedding_with(_EchoSidecar())

    rng = np.random.default_rng(11)
    total, chunk = 16_383, 2_048
    plan = rng.integers(0, 150_000, size=total, dtype=np.int64)
    # EOS tokens mid-prompt: `_ngram_rows_np` restarts its segment scan there,
    # which is exactly where a history reconstruction can go wrong.
    plan[3_000] = embedding.eos_id
    plan[9_001] = embedding.eos_id

    history = np.full((1, embedding.context_len), embedding.eos_id, np.int64)
    chunks = 0
    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        # Owner: exactly what stage() does with the live PLE state cache.
        owner_rows, history = embedding._rows_np(
            plan[start:end].reshape(1, -1), history
        )
        worker_flat, _mats = prepare(embedding, plan, start, end)
        np.testing.assert_array_equal(worker_flat, owner_rows.reshape(-1))
        chunks += 1
    assert chunks == 8
    assert owner_rows.size == 2047 * 16


def test_worker_pads_the_prompt_head_with_eos_like_the_empty_state_cache():
    prepare = _bind_method("NGramEmbedding", "prefill_lookahead_prepare")
    embedding = _embedding_with(_EchoSidecar())
    plan = np.arange(100, 116, dtype=np.int64)
    expected, _ = embedding._rows_np(
        plan[:8].reshape(1, -1),
        np.full((1, embedding.context_len), embedding.eos_id, np.int64),
    )
    worker_flat, _mats = prepare(embedding, plan, 0, 8)
    np.testing.assert_array_equal(worker_flat, expected.reshape(-1))


def test_worker_returns_none_when_the_sidecar_never_attached():
    prepare = _bind_method("NGramEmbedding", "prefill_lookahead_prepare")
    embedding = _embedding_with(None)
    assert prepare(embedding, np.arange(64, dtype=np.int64), 0, 16) is None


def test_worker_returns_none_when_the_gather_would_take_the_hot_lru():
    class _LruSidecar(_EchoSidecar):
        def prepare_rows_np(self, flat, names):
            return None

    prepare = _bind_method("NGramEmbedding", "prefill_lookahead_prepare")
    embedding = _embedding_with(_LruSidecar())
    assert prepare(embedding, np.arange(64, dtype=np.int64), 0, 16) is None


def test_engagement_snapshot_is_per_scope_not_module_global():
    look, ids, _calls = build()
    look.submit(0)
    look.take(0)
    assert look.engagement() == {
        "spans": 4,
        "submits": 1,
        "hits": 1,
        "misses": 0,
        "ineligible": 0,
    }
    other, _ids, _c = build()
    assert other.engagement()["hits"] == 0, "counters must not leak across scopes"


def test_verify_full_engagement_raises_when_the_lane_served_nothing():
    look, _ids, _calls = build()
    with pytest.raises(RuntimeError, match="did not engage"):
        look.verify_full_engagement()


def test_verify_full_engagement_raises_on_partial_engagement():
    look, ids, _calls = build()
    look.submit(0)
    look.take(0)
    with pytest.raises(RuntimeError) as excinfo:
        look.verify_full_engagement()
    assert "'hits': 1" in str(excinfo.value)
    assert "'spans': 4" in str(excinfo.value)


def test_verify_full_engagement_passes_when_every_chunk_hit():
    look, ids, _calls = build(total=64, chunk=16)
    for index in range(4):
        look.submit(index)
        assert look.take(index) is not None
    look.verify_full_engagement()


def test_scope_raises_on_a_clean_prefill_that_never_engaged():
    """The exact 2026-09-01 outcome must now be impossible to report."""

    look, _ids, _calls = build()
    with pytest.raises(RuntimeError, match="did not engage"):
        with prefill_lookahead_scope(look):
            pass  # a "prefill" that never consumed a chunk
    assert lookahead_mod.active_lookahead() is None


def test_scope_does_not_mask_the_real_failure_of_an_aborted_prefill():
    look, _ids, _calls = build()
    with pytest.raises(ValueError, match="aborted prefill"):
        with prefill_lookahead_scope(look):
            raise ValueError("aborted prefill")


def test_scope_accepts_a_prefill_that_engaged_every_chunk():
    look, ids, _calls = build(total=64, chunk=16)
    with prefill_lookahead_scope(look):
        for start in range(0, 64, 16):
            index = look.span_index_of(ids[start : start + 16])
            look.take(index)
            look.submit(look.next_index(index))
    assert look.engagement()["hits"] == 4


# ---------------------------------------------------------------------------
# The model-resolution walk that made the lane inert
# ---------------------------------------------------------------------------


def _generation_function(name: str):
    """Compile one generation.py function from source (generation imports MLX)."""

    node = next(
        n
        for n in ast.parse(GENERATION_TEXT).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {}
    exec(compile(module, "<generation>", "exec"), namespace)
    return namespace[name]


class _Node:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_hook_is_found_two_wrappers_below_the_runtime():
    """rt.model -> language_model -> model is the production house shape.

    Walking one level finds TextModel, which has no hook -- the 2026-09-01
    non-engagement, exactly.
    """

    resolve = _generation_function("_resolve_ple_lookahead_hook")
    inner = _Node(ple_prefill_lookahead=lambda ids, spans: "built")
    runtime = _Node(model=_Node(language_model=_Node(model=inner)))
    assert resolve(runtime)(None, None) == "built"


def test_hook_is_found_on_a_text_only_shape_without_language_model():
    resolve = _generation_function("_resolve_ple_lookahead_hook")
    inner = _Node(ple_prefill_lookahead=lambda ids, spans: "built")
    assert resolve(_Node(model=_Node(model=inner)))(None, None) == "built"


def test_hook_is_found_directly_on_the_runtime_model():
    resolve = _generation_function("_resolve_ple_lookahead_hook")
    inner = _Node(ple_prefill_lookahead=lambda ids, spans: "built")
    assert resolve(_Node(model=inner))(None, None) == "built"


def test_hook_resolution_returns_none_for_an_architecture_without_it():
    resolve = _generation_function("_resolve_ple_lookahead_hook")
    assert resolve(_Node(model=_Node(language_model=_Node(model=_Node())))) is None
    assert resolve(_Node()) is None


def test_hook_resolution_terminates_on_a_self_referential_wrapper():
    resolve = _generation_function("_resolve_ple_lookahead_hook")
    node = _Node()
    node.model = node
    assert resolve(_Node(model=node)) is None


def test_scope_raises_rather_than_running_an_armed_lane_it_cannot_serve():
    scope_source = GENERATION_TEXT.split(
        "def _ple_prefill_lookahead_scope", 1
    )[1].split("\ndef ", 1)[0]
    assert "_resolve_ple_lookahead_hook(rt)" in scope_source
    assert "raise RuntimeError" in scope_source
    assert "_lookahead_enabled()" in scope_source
    assert "_PREFILL_CHUNK_RECORDS.clear()" in scope_source


# ---------------------------------------------------------------------------
# Per-chunk timings, on both arms
# ---------------------------------------------------------------------------


def test_prefill_loop_records_chunk_wall_and_gather_time():
    assert "gather_before = _ple_stage_seconds()" in GENERATION_TEXT
    assert "_record_prefill_chunk(" in GENERATION_TEXT
    assert "ple_gather_s=_ple_stage_seconds() - gather_before" in GENERATION_TEXT
    assert "wall_s=chunk_wall_s" in GENERATION_TEXT


def test_chunk_record_buffer_is_bounded_and_reset_per_prefill():
    record = _generation_function("_record_prefill_chunk")
    assert "_PREFILL_CHUNK_RECORD_CAP" in GENERATION_TEXT
    # The cap guards a long-lived server process; the clear() guards A/B rows.
    assert "_PREFILL_CHUNK_RECORDS.clear()" in GENERATION_TEXT
    assert record is not None


def test_stage_timing_wraps_the_whole_gather_and_survives_early_returns():
    stage = MODEL_TEXT.split("def stage(self, input_ids: mx.array", 1)[1].split(
        "\n    def ", 1
    )[0]
    assert "started = time.perf_counter()" in stage
    assert "self._stage_body(input_ids, cache, state_idx)" in stage
    assert "finally:" in stage
    assert "_PLE_STAGE_SECONDS[0] +=" in stage
    assert "_PLE_STAGE_CALLS[0] +=" in stage


def test_reject_unwired_prefill_loop_is_silent_when_the_flag_is_off(monkeypatch):
    monkeypatch.delenv(lookahead_mod.ENV_FLAG, raising=False)
    lookahead_mod.enabled.cache_clear()
    try:
        assert lookahead_mod.reject_unwired_prefill_loop("_prefill") is None
    finally:
        lookahead_mod.enabled.cache_clear()


def test_reject_unwired_prefill_loop_raises_when_armed(monkeypatch):
    monkeypatch.setenv(lookahead_mod.ENV_FLAG, "1")
    lookahead_mod.enabled.cache_clear()
    try:
        with pytest.raises(RuntimeError) as excinfo:
            lookahead_mod.reject_unwired_prefill_loop("_prefill")
        message = str(excinfo.value)
        assert "'_prefill'" in message
        assert "control under the candidate" in message
    finally:
        lookahead_mod.enabled.cache_clear()


UNWIRED_LOOPS = (
    "_prefill",
    "_prefill_with_hidden_sequence",
    "_prefill_restored_prompt_suffix",
)

@pytest.mark.parametrize("loop", UNWIRED_LOOPS)
def test_every_unwired_chunked_prefill_loop_carries_the_tripwire(loop):
    """No chunked prefill loop may quietly serve an armed lane's request."""

    node = next(
        n
        for n in ast.parse(GENERATION_TEXT).body
        if isinstance(n, ast.FunctionDef) and n.name == loop
    )
    body = ast.unparse(node)
    assert f'_reject_unwired_ple_lookahead({loop!r})' in body


def test_the_wired_loop_opens_the_scope_instead_of_the_tripwire():
    node = next(
        n
        for n in ast.parse(GENERATION_TEXT).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_prefill_committed_mtp_history_streaming"
    )
    body = ast.unparse(node)
    assert "_ple_prefill_lookahead_scope(rt, body, mtp_streaming_spans)" in body
    assert "_reject_unwired_ple_lookahead" not in body


def test_every_chunked_prefill_loop_is_either_wired_or_tripwired():
    """The invariant that makes 'armed but inert' unreachable.

    Any function that iterates prefill chunk spans either opens the lookahead
    scope or refuses an armed lane. A new loop added without either lands
    here, not in a wasted GPU window.
    """

    tree = ast.parse(GENERATION_TEXT)
    unhandled = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.unparse(node)
        iterates = (
            "_iter_prefill_chunk_spans(" in body
            or "_prefill_spans_with_tail_grid(" in body
        )
        if not iterates or node.name in {
            "_iter_prefill_chunk_spans",
            "_prefill_spans_with_tail_grid",
        }:
            continue
        if (
            "_ple_prefill_lookahead_scope(" in body
            or "_reject_unwired_ple_lookahead(" in body
        ):
            continue
        unhandled.append(node.name)
    assert not unhandled, f"chunked prefill loops with no lookahead policy: {unhandled}"


# ---------------------------------------------------------------------------
# 2026-09-02 regression: every SHORT prompt 500ed on {'spans': 1, 'hits': 0}
#
# One prefill chunk has nothing to look ahead FROM, and its sub-4,096-row
# gather is exactly what the owner's hot-row LRU serves -- which the worker
# is forbidden to touch, so it declined and the scope called that inert.
# ---------------------------------------------------------------------------


def drive(look, ids, chunk=16, *, skip_submit_for=()):
    """Run the shipped prefill order over a lookahead: take, then submit next."""

    total = len(ids)
    for start in range(0, total, chunk):
        index = look.span_index_of(ids[start : start + chunk])
        look.take(index)
        nxt = look.next_index(index)
        if nxt not in skip_submit_for:
            look.submit(nxt)


def test_a_single_span_prefill_does_not_arm_the_lane():
    look, _ids, _calls = build(total=16, chunk=16)
    assert look.spans == [(0, 16)]
    assert look.armed is False
    assert look.inert_reason == "single_span"


def test_a_multi_span_prefill_is_armed():
    look, _ids, _calls = build(total=64, chunk=16)
    assert look.armed is True
    assert look.inert_reason is None
    assert look.overlappable_spans() == [1, 2, 3]


def test_single_span_scope_is_a_no_op_and_records_the_reason():
    """The exact production failure: one chunk must not 500."""

    look, _ids, calls = build(total=16, chunk=16)
    with prefill_lookahead_scope(look) as active:
        # No worker, no contextvar: `_take_prefill_lookahead` sees no active
        # lane and the owner takes the ordinary (hot-LRU) gather.
        assert active is None
        assert lookahead_mod.active_lookahead() is None
        assert calls == []
    assert lookahead_mod.COUNTERS["scope_skipped_single_span"] == 1
    assert lookahead_mod.last_scope_status() == {
        "armed": False,
        "reason": "single_span",
        "spans": 1,
    }
    assert look._closed is True


def test_single_span_verify_is_trivially_satisfied():
    look, _ids, _calls = build(total=16, chunk=16)
    look.verify_full_engagement()  # nothing to look ahead from


def test_multi_span_scope_records_that_it_armed():
    look, ids, _calls = build(total=64, chunk=16)
    with prefill_lookahead_scope(look) as active:
        assert active is look
        drive(look, ids)
    assert lookahead_mod.last_scope_status() == {
        "armed": True,
        "reason": None,
        "spans": 4,
    }
    assert "scope_skipped_single_span" not in lookahead_mod.COUNTERS


def test_eight_spans_with_hits_on_every_span_pass_unchanged():
    """The 16K cell: 8 spans, 8 submits, 8 hits, receipts untouched."""

    look, ids, calls = build(total=128, chunk=16)
    assert len(look.spans) == 8
    with prefill_lookahead_scope(look):
        drive(look, ids)
    assert look.engagement() == {
        "spans": 8,
        "submits": 8,
        "hits": 8,
        "misses": 0,
        "ineligible": 0,
    }
    assert calls == [(i, i + 16) for i in range(0, 128, 16)]
    assert lookahead_mod.COUNTERS["hit"] == 8


def test_eight_spans_pass_when_only_span_zero_missed():
    """Span 0 has no forward to overlap with; its miss cannot fail the lane."""

    def prepare(start, end):
        if start == 0:
            return None  # the hot-LRU decline, on the first chunk
        return (np.arange(1000 + start, 1000 + end, dtype=np.int64), {})

    look, ids, _calls = build(total=128, chunk=16, prepare=prepare)
    with prefill_lookahead_scope(look):
        drive(look, ids)
    assert look.engagement()["hits"] == 7
    assert look.engagement()["ineligible"] == 1
    look.verify_full_engagement()


def test_eight_spans_raise_when_span_three_missed():
    look, ids, _calls = build(total=128, chunk=16)
    with pytest.raises(RuntimeError) as excinfo:
        with prefill_lookahead_scope(look):
            # Span 3's rows were never queued: the slot is empty when the
            # forward asks for them. A real inert step, still a hard failure.
            drive(look, ids, skip_submit_for=(3,))
    message = str(excinfo.value)
    assert "did not engage" in message
    assert "spans with no worker-prepared rows: [3]" in message


def test_a_short_trailing_span_the_worker_declined_is_not_a_lane_failure():
    """2049 tokens -> a 1-token tail chunk -> a hot-LRU-sized gather.

    The worker declines those by design (the LRU is owner-thread-only), so
    the tail must not be read as the lane sitting inert.
    """

    ids = np.arange(1000, 1000 + 40, dtype=np.int64)
    spans = [(0, 16), (16, 32), (32, 40)]

    def prepare(start, end):
        if end - start < 16:
            return None  # below _HOT_PATH_MAX_ROWS: the owner's LRU serves it
        return (np.asarray(ids[start:end]), {})

    look = PrefillLookahead(ids, spans, prepare=prepare, submit=immediate_submit)
    with prefill_lookahead_scope(look):
        for index, (start, end) in enumerate(spans):
            assert look.span_index_of(ids[start:end]) == index
            look.take(index)
            look.submit(look.next_index(index))
    assert look.engagement()["hits"] == 2
    assert look.engagement()["ineligible"] == 1


def test_a_lane_the_worker_declined_everywhere_still_raises():
    """Indistinguishable from inert, so it keeps failing closed."""

    look, ids, _calls = build(total=64, chunk=16, prepare=lambda s, e: None)
    with pytest.raises(RuntimeError) as excinfo:
        with prefill_lookahead_scope(look):
            drive(look, ids)
    assert "did not engage" in str(excinfo.value)
    assert "declined by the worker's eligibility rule" in str(excinfo.value)


def test_a_wholly_inert_multi_span_lane_still_raises_as_before():
    look, _ids, _calls = build(total=128, chunk=16)
    with pytest.raises(RuntimeError, match="did not engage"):
        with prefill_lookahead_scope(look):
            pass


def test_scope_status_resets_with_the_counters():
    look, _ids, _calls = build(total=16, chunk=16)
    with prefill_lookahead_scope(look):
        pass
    assert lookahead_mod.last_scope_status()["reason"] == "single_span"
    lookahead_mod.reset_counters()
    assert lookahead_mod.last_scope_status() == {
        "armed": None,
        "reason": None,
        "spans": 0,
    }


