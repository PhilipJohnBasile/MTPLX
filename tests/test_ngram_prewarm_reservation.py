"""The n-gram pre-read reserves the engine's whole growth to its budget, not the KV estimate alone.

2026-09-03: a 23.4 GiB pre-read (free 38.9 GiB minus a 9.5 GiB KV reservation minus 6 GiB) on a 128 GB Mac
left the kernel with no free pages once a 206k prefill grew the engine to its budget; the machine panicked
on the watchdog. The reservation is now max(KV estimate, budget minus weights on disk).
"""

from __future__ import annotations

import types

import mtplx.ple_row_gather as prg
from mtplx.memory_plan import NGRAM_TABLE_FILENAME


def _pack(tmp_path, weights_sizes, table_size=4096):
    for i, n in enumerate(weights_sizes):
        (tmp_path / f"model-{i}.safetensors").write_bytes(b"w" * n)
    (tmp_path / NGRAM_TABLE_FILENAME).write_bytes(b"t" * table_size)
    return tmp_path


def test_growth_is_budget_minus_weights_and_excludes_the_streamed_table(tmp_path, monkeypatch):
    import mtplx.memory_plan as mp

    monkeypatch.setattr(mp, "detect_total_ram_bytes", lambda: 1000)
    monkeypatch.setattr(mp, "usable_engine_bytes", lambda total: 750)
    monkeypatch.delenv("MTPLX_MEMORY_LIMIT_BYTES", raising=False)
    pack = _pack(tmp_path, [100, 150], table_size=5000)
    growth, source = prg.estimate_engine_growth_bytes(pack)
    assert growth == 750 - 250
    assert source.startswith("engine_growth(")


def test_an_allocator_cap_bounds_the_growth(tmp_path, monkeypatch):
    import mtplx.memory_plan as mp

    monkeypatch.setattr(mp, "detect_total_ram_bytes", lambda: 1000)
    monkeypatch.setattr(mp, "usable_engine_bytes", lambda total: 750)
    monkeypatch.setenv("MTPLX_MEMORY_LIMIT_BYTES", "600")
    pack = _pack(tmp_path, [100, 150])
    growth, _ = prg.estimate_engine_growth_bytes(pack)
    assert growth == 600 - 250


def test_unknown_inputs_reserve_nothing_and_say_why(tmp_path, monkeypatch):
    import mtplx.memory_plan as mp

    monkeypatch.setattr(mp, "detect_total_ram_bytes", lambda: None)
    assert prg.estimate_engine_growth_bytes(tmp_path) == (0, "ram_unknown")
    monkeypatch.setattr(mp, "detect_total_ram_bytes", lambda: 1000)
    monkeypatch.setattr(mp, "usable_engine_bytes", lambda total: 750)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert prg.estimate_engine_growth_bytes(empty) == (0, "weights_unknown")


def test_the_server_reserves_the_larger_of_kv_and_growth(monkeypatch):
    from mtplx.server import openai as srv

    seen = {}
    monkeypatch.setattr(prg, "estimate_kv_reservation_bytes", lambda model, tokens: (30, "kv"))
    monkeypatch.setattr(prg, "estimate_engine_growth_bytes", lambda model: (80, "growth"))
    monkeypatch.setattr(prg, "set_prewarm_reservation", lambda b, s: seen.update(bytes=b, source=s))
    out = srv._publish_ngram_prewarm_reservation(types.SimpleNamespace(model="/pack", context_window=4096))
    assert (out["bytes"], out["source"]) == (80, "growth")
    assert seen == {"bytes": 80, "source": "growth"}

    monkeypatch.setattr(prg, "estimate_engine_growth_bytes", lambda model: (10, "growth"))
    out = srv._publish_ngram_prewarm_reservation(types.SimpleNamespace(model="/pack", context_window=4096))
    assert (out["bytes"], out["source"]) == (30, "kv")


def test_the_budget_shrinks_with_the_larger_reservation():
    # free 38.9 GiB, table 29.8 GiB, margin 6 GiB: a 9.5 GiB KV reservation buys 23.4 GiB, the 18.7 GiB growth
    # reservation (96 GiB budget minus 77.3 GiB of weights) buys 14.2 GiB.
    gib = 1024**3
    small = prg.resolve_budget("auto", table_bytes=int(29.8 * gib), free_bytes=int(38.9 * gib), reserved_bytes=int(9.5 * gib))
    large = prg.resolve_budget("auto", table_bytes=int(29.8 * gib), free_bytes=int(38.9 * gib), reserved_bytes=int(18.7 * gib))
    assert small["budget_bytes"] > large["budget_bytes"] > 0
    assert abs(large["budget_bytes"] - 14.2 * gib) < 0.2 * gib
