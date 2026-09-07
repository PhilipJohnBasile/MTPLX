"""MTPLX_QWEN4_PLE_PREFILL_LOOKAHEAD on the RESTORED-SUFFIX prefill loop.

The 2026-09-02 served battery armed the lane and then 500ed every cell whose
prompt shared a prefix with an earlier one: the session bank restored the
prefix, `_prefill_restored_prompt_suffix` took over, and that loop answered an
armed flag with a RuntimeError.  A raise per request is a serving outage for
any client whose prompt shares a cached prefix, which is most real traffic.

The restored-suffix loop is the same chunked prefill over a shorter span -- the
same two grid helpers, one `stage()` per chunk, in chunk order -- so the lane
applies to its chunks exactly as to a fresh prompt's.  The one thing that is
NOT the same is where a chunk's n-gram history comes from: the owner reads it
off the RESTORED PLE state cache, so the worker has to rebuild it from the
whole prompt.  These tests drive the real loop with a stub runtime and the
SHIPPED consumption path (`NGramEmbedding._take_prefill_lookahead`, compiled
out of qwen4_exp.py), on the CPU device: no GPU, no Metal, no model load.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mlx.core as mx

from mtplx import generation as generation_mod
from mtplx import ple_prefill_lookahead as lookahead_mod


ROOT = Path(__file__).resolve().parents[1]
MODEL_TEXT = (ROOT / "mtplx" / "models" / "qwen4_exp.py").read_text("utf-8")
GENERATION_TEXT = (ROOT / "mtplx" / "generation.py").read_text("utf-8")


# ---------------------------------------------------------------------------
# The shipped halves, compiled out of source so a stale copy cannot pass
# ---------------------------------------------------------------------------


def _compile_from_source(node) -> dict:
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {"np": np, "os": os}
    exec(compile(module, "<qwen4_exp>", "exec"), namespace)
    return namespace


def _bind_top_level(name: str):
    node = next(
        n
        for n in ast.parse(MODEL_TEXT).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return _compile_from_source(node)[name]


def _bind_method(class_name: str, method_name: str):
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


def _generation_source(name: str) -> str:
    node = next(
        n
        for n in ast.parse(GENERATION_TEXT).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.unparse(node)


class _EchoSidecar:
    """Returns the row ids themselves, so the test compares ids, not bytes."""

    bits = 4

    def prepare_rows_np(self, flat, names, *, vectorized=False, record=None):
        return (int(len(np.unique(flat))), {name: flat for name in names})


class _Embedding:
    """Just enough of NGramEmbedding to run the shipped row arithmetic.

    Production geometry for Qwen3.8 Flash-Next: ngram_size 3, 8 heads per
    ngram -> 16 rows per token; the PLE history is the previous two tokens,
    which on a warm restore live in the RESTORED state cache.
    """

    ngram_size = 3
    heads_per_ngram = 8
    context_len = ngram_size - 1
    eos_id = 248_044
    ngram_heads = 16

    def __init__(self):
        self._mult = np.array([2_654_435_761, 40_503, 1_337], dtype=np.int64)
        self._sizes = np.full(self.ngram_heads, 200_000 // self.ngram_heads, np.int64)
        self._offs = (
            np.arange(self.ngram_heads, dtype=np.int64) * (200_000 // self.ngram_heads)
        )
        self._ngram_rows_np = _bind_top_level("_ngram_rows_np")
        self.ngram_embedding = SimpleNamespace(_sidecar=_EchoSidecar())
        for name in ("_prefill_span_rows", "_sidecar_map_names"):
            setattr(self, name, _bind_method("NGramEmbedding", name).__get__(self))
        self.prefill_lookahead_prepare = _bind_method(
            "NGramEmbedding", "prefill_lookahead_prepare"
        ).__get__(self)
        # The shipped consumption path, verbatim: `span_index_of` -> `take` ->
        # `submit(next)`, then the row-equality proof.
        self._take_prefill_lookahead = _bind_method(
            "NGramEmbedding", "_take_prefill_lookahead"
        ).__get__(self)
        self._take_first_gather_early = _bind_method(
            "NGramEmbedding", "_take_first_gather_early"
        ).__get__(self)

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


# ---------------------------------------------------------------------------
# A runtime whose forward_ar stages exactly like the served one
# ---------------------------------------------------------------------------


class _StagingModel:
    """`rt.model`: exposes the lane hook and stages once per forward."""

    def __init__(self, embedding: _Embedding, prefix_ids: list[int]):
        self.embedding = embedding
        # What a session-bank restore leaves behind: the PLE state cache holds
        # the last `context_len` tokens of the RESTORED PREFIX.
        self.ple_state = np.asarray(
            prefix_ids[-embedding.context_len :], dtype=np.int64
        ).reshape(1, -1)
        self.stages: list[dict] = []

    def ple_prefill_lookahead(self, token_ids, spans):
        if not lookahead_mod.enabled():
            return None
        embedding = self.embedding
        lookahead = lookahead_mod.PrefillLookahead(
            token_ids,
            spans,
            prepare=lambda start, end: embedding.prefill_lookahead_prepare(
                lookahead.token_ids, start, end
            ),
            rows_per_token=int(embedding.ngram_heads),
            min_servable_rows=0,
        )
        return lookahead

    def stage(self, ids_np: np.ndarray) -> None:
        """`NGramEmbedding._stage_body`, minus the MLX half."""

        rows, new_hist = self.embedding._rows_np(ids_np, self.ple_state)
        flat = rows.reshape(-1)
        prepared = self.embedding._take_prefill_lookahead(ids_np, flat)
        self.ple_state = np.asarray(new_hist, dtype=np.int64)
        self.stages.append(
            {
                "tokens": int(ids_np.shape[1]),
                "ids": ids_np.reshape(-1).tolist(),
                "served_by_worker": prepared is not None,
            }
        )


class _StubRuntime:
    """Target-only AR runtime: forward_ar returns logits alone."""

    def __init__(self, model: _StagingModel):
        self.model = model
        self.mtp_enabled = False
        self.model_path = Path("tiny-restored-suffix")
        self.diagnostic_counters: dict[str, int] = {}

    def forward_ar(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
        input_embeddings=None,
    ):
        assert not return_hidden
        self.model.stage(np.asarray(input_ids, dtype=np.int64))
        if not emit_logits:
            return None
        return mx.zeros((1, 1, 4), dtype=mx.float32)

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def repage_target_prefill_cache(self, _cache):
        return False


@pytest.fixture(autouse=True)
def _cpu_and_clean_counters(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ["MTPLX_SUSTAINED_PREFILL"] = "1"
    os.environ["MTPLX_PREFILL_CHUNK_SIZE"] = "4"
    os.environ[lookahead_mod.ENV_FLAG] = "1"
    lookahead_mod.enabled.cache_clear()
    lookahead_mod.reset_counters()
    try:
        yield
    finally:
        mx.set_default_device(previous)
        lookahead_mod.enabled.cache_clear()
        lookahead_mod.reset_counters()


def _run_suffix(
    prefix_len: int,
    suffix_len: int,
    *,
    fused_max: int = 0,
    hand_down_plan: bool = True,
    plan_override: list[int] | None = None,
):
    """Drive the real restored-suffix prefill for one warm request."""

    rng = np.random.default_rng(90)
    prompt_ids = [int(t) for t in rng.integers(10, 150_000, size=prefix_len + suffix_len)]
    suffix = list(prompt_ids[prefix_len:])
    embedding = _Embedding()
    model = _StagingModel(embedding, prompt_ids[:prefix_len])
    rt = _StubRuntime(model)
    os.environ["MTPLX_SMALL_SUFFIX_FUSED_MAX"] = str(fused_max)
    restored = SimpleNamespace(
        entry=SimpleNamespace(prefix_len=prefix_len),
        cache=[],
        logits=None,
        hidden=None,
        mtp_history_cache=None,
        restore_mode="clone",
    )
    generation_mod._prefill_restored_prompt_suffix(
        rt,
        restored,
        suffix,
        base_hidden_variant="post_norm",
        mtp_hidden_variant="post_norm",
        mtp_history_policy="cycle",
        tokens_total=len(prompt_ids),
        cached_tokens=prefix_len,
        plan_ids=(
            plan_override
            if plan_override is not None
            else (prompt_ids if hand_down_plan else None)
        ),
    )
    return model, prompt_ids


# ---------------------------------------------------------------------------
# (a) a multi-chunk suffix ENGAGES the lane
# ---------------------------------------------------------------------------


def test_the_worker_is_given_the_prompt_not_the_suffix():
    """The chunk whose n-gram history lives in the restored prefix.

    Chunk 0 of a restored suffix reaches back two tokens into the PREFIX. Hand
    the worker the suffix alone and it EOS-pads that head instead, its rows
    disagree with the owner's, and `_take_prefill_lookahead` throws the
    payload away on the row-equality check -- a lane that is armed, busy and
    buys nothing on the one chunk with the largest stall.
    """

    model, prompt_ids = _run_suffix(prefix_len=9, suffix_len=13)
    first_body = next(s for s in model.stages if s["tokens"] > 1)
    assert first_body["ids"] == prompt_ids[9:13]
    assert first_body["served_by_worker"] is True

    embedding = _Embedding()
    eos_padded, _ = embedding._rows_np(
        np.asarray(prompt_ids[9:13], np.int64).reshape(1, -1),
        np.full((1, embedding.context_len), embedding.eos_id, np.int64),
    )
    restored_history, _ = embedding._rows_np(
        np.asarray(prompt_ids[9:13], np.int64).reshape(1, -1),
        np.asarray(prompt_ids[7:9], np.int64).reshape(1, -1),
    )
    assert not np.array_equal(eos_padded, restored_history)


def test_without_the_plan_the_first_chunk_falls_back_and_is_counted():
    """The documented fallback, so the plan-passing is not a vacuous claim.

    A caller that does not hand down the prompt leaves the worker EOS-padding
    the first chunk's history.  Its rows then disagree with the owner's and
    `_take_prefill_lookahead` throws the payload away on the row-equality
    check -- exact, counted (`miss_row_mismatch`), never silent -- while the
    later chunks, whose history is inside the suffix, still hit.
    """

    model, _prompt_ids = _run_suffix(
        prefix_len=9, suffix_len=13, hand_down_plan=False
    )
    body_stages = [s for s in model.stages if s["tokens"] > 1]
    assert [s["served_by_worker"] for s in body_stages] == [False, True, True]
    assert lookahead_mod.snapshot_counters().get("miss_row_mismatch") == 1
    # Still an engagement, still no raise: the lane ran, one chunk declined.
    assert lookahead_mod.last_scope_status()["armed"] is True


def test_a_plan_that_is_not_this_suffixs_prompt_falls_back_instead_of_500ing():
    """The plan is compared, not assumed.

    Absolute spans over a plan whose tail is NOT what the chunks carry would
    make every `span_index_of` miss; the lane would then read spans it was
    designed to serve but never took as inertness and raise at scope exit --
    the same 500 this whole change removes, one layer down.
    """

    rng = np.random.default_rng(7)
    wrong = [int(t) for t in rng.integers(10, 150_000, size=22)]
    model, _prompt_ids = _run_suffix(
        prefix_len=9, suffix_len=13, plan_override=wrong
    )
    body_stages = [s for s in model.stages if s["tokens"] > 1]
    assert [s["served_by_worker"] for s in body_stages] == [False, True, True]
    assert lookahead_mod.snapshot_counters().get("miss_unknown_span", 0) == 0


# ---------------------------------------------------------------------------
# (b) a one-chunk suffix DECLINES, by design, counted
# ---------------------------------------------------------------------------


def test_a_one_token_suffix_declines_rather_than_falling_through():
    """No chunk loop at all: the final logits pass is the whole prefill."""

    model, _prompt_ids = _run_suffix(prefix_len=9, suffix_len=1, fused_max=0)

    assert [s["tokens"] for s in model.stages] == [1]
    scope = lookahead_mod.last_scope_status()
    assert scope["armed"] is False
    assert scope["reason"] == "single_span"


# ---------------------------------------------------------------------------
# (c) no request path raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prefix_len", "suffix_len", "fused_max"),
    [(9, 13, 0), (9, 5, 512), (9, 3, 0), (9, 1, 0), (2, 21, 0), (2, 2, 0)],
)
def test_no_restored_suffix_shape_raises_under_an_armed_flag(
    prefix_len, suffix_len, fused_max
):
    """The battery's failure, in every shape the loop accepts."""

    _run_suffix(prefix_len=prefix_len, suffix_len=suffix_len, fused_max=fused_max)


def test_the_suffix_loop_no_longer_carries_the_tripwire():
    node = next(
        n
        for n in ast.parse(GENERATION_TEXT).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_prefill_restored_prompt_suffix"
    )
    body = ast.unparse(node)
    assert "_reject_unwired_ple_lookahead" not in body
    assert "_ple_prefill_lookahead_scope(" in body


def test_both_restored_suffix_callers_hand_down_the_whole_prompt():
    """Without the plan the worker cannot rebuild the first chunk's history."""

    calls = [
        block
        for block in GENERATION_TEXT.split("_prefill_restored_prompt_suffix(")[1:]
        if "cached_tokens=" in block.split("\n            )")[0]
    ]
    assert len(calls) == 2
    for block in calls:
        assert "plan_ids=prompt_ids," in block.split(")\n")[0]


def test_first_gather_early_never_carried_the_tripwire():
    """The other PLE key does not share the guard -- checked, not assumed."""

    assert "reject_unwired" not in _generation_source("_ple_first_gather_early_scope")
    assert "reject_unwired" not in _generation_source("_with_ple_first_gather_early")
    # Its only raise is the install-shaped one: an armed flag on an
    # architecture that cannot serve the lane. Per-request conditions route
    # through `first_gather_early_scope(None, reason)`, which counts a decline.
    scope_src = _generation_source("_ple_first_gather_early_scope")
    assert "first_gather_early_scope(None, 'unpredictable_first_span')" in scope_src
    assert "'model_declined_span'" in scope_src


# ---------------------------------------------------------------------------
# The receipt must say the warm loop is covered
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The warm loop must leave a per-chunk receipt, like the cold one
# ---------------------------------------------------------------------------


def test_the_warm_loop_records_per_chunk_wall_and_gather_time():
    _run_suffix(prefix_len=9, suffix_len=13)
    records = generation_mod.prefill_chunk_records()
    assert [(r["start"], r["end"]) for r in records] == [
        (9.0, 13.0),
        (13.0, 17.0),
        (17.0, 21.0),
    ]
    assert all(r["wall_s"] >= 0.0 and r["ple_gather_s"] >= 0.0 for r in records)
