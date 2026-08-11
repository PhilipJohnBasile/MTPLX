"""Focused tests for the CPU-only staged rotating-KV reference oracle."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ORACLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/research_rotating_staged_kv_cpu_oracle.py"
)
ORACLE_SPEC = importlib.util.spec_from_file_location("rotating_staged_kv_cpu_oracle", ORACLE_PATH)
assert ORACLE_SPEC is not None and ORACLE_SPEC.loader is not None
oracle = importlib.util.module_from_spec(ORACLE_SPEC)
sys.modules[ORACLE_SPEC.name] = oracle
ORACLE_SPEC.loader.exec_module(oracle)


def test_prefix_matrix_covers_all_capacities_prefixes_and_wrap_boundaries() -> None:
    result = oracle.run_prefix_matrix()

    assert result["capacities"] == list(range(1, 17))
    assert result["case_count"] == sum(capacity + 1 for capacity in range(1, 17))
    assert result["scenario_counts"] == {
        "before_first_wrap": 16,
        "exactly_at_first_wrap": 16,
        "post_first_wrap": sum(capacity - 1 for capacity in range(1, 17)),
    }

    traces = {trace["accepted_prefix"]: trace for trace in result["selected_capacity_4_traces"]}
    assert traces[0]["snapshot"]["offset"] == 3
    assert traces[1]["snapshot"]["offset"] == 4
    assert traces[1]["snapshot"]["next_write_slot"] == 0
    assert traces[2]["snapshot"]["next_token"]["visible_positions"] == (2, 3, 4, 5)
    assert traces[4]["snapshot"]["next_token"]["write_slot"] == 3


def test_rejected_suffix_is_invisible_and_cleanup_preserves_live_ring() -> None:
    cache, history = oracle._seed_before_wrap(4)
    proposal = [oracle._pair("proposal", 4, index) for index in range(4)]
    cache.stage(proposal)
    commit = cache.commit(2)
    history.extend(proposal[:2])
    oracle.assert_contract(cache, history)

    visible_tags = {item.pair.tag for item in cache.visible()}
    assert visible_tags.isdisjoint(pair.tag for pair in proposal[2:])
    assert commit.discarded_suffix == 2

    before = cache.snapshot()
    cache.stage(proposal[2:])
    assert cache.cleanup() == 2
    assert cache.snapshot() == before


def test_cleanup_reset_and_bonus_contracts_cover_each_capacity() -> None:
    assert oracle.run_cleanup_and_reset_checks() == {"cleanup_cases": 16, "reset_cases": 16}

    bonus = oracle.run_bonus_checks()
    assert bonus["swift_516_staged_round"]["cases"] == 16
    assert bonus["mtplx_gemma4_exact_round"]["cases"] == 16


def test_transaction_state_rejects_nested_rounds_and_invalid_prefixes() -> None:
    cache = oracle.RotatingStagedKVReference(2)
    pair = oracle._pair("proposal", 2, 0)
    cache.stage((pair,))
    with pytest.raises(RuntimeError, match="already open"):
        cache.stage((pair,))
    with pytest.raises(ValueError, match="outside"):
        cache.commit(2)
    assert cache.cleanup() == 1
    with pytest.raises(RuntimeError, match="no staged"):
        cache.commit(0)


def test_oracle_source_has_no_mlx_or_metal_imports() -> None:
    source_path = Path(oracle.__file__)
    parsed = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(parsed)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [node])
        if isinstance(alias, ast.alias)
    }

    assert "mlx" not in imported
    assert "metal" not in imported


def test_receipt_is_deterministic_and_reports_reference_scope(tmp_path: Path) -> None:
    first = oracle.render_receipt()
    second = oracle.render_receipt()
    assert first == second

    destination = oracle.write_receipt(tmp_path / "receipt.json")
    assert destination.read_text(encoding="utf-8") == first
    payload = json.loads(first)
    payload_without_hash = dict(payload)
    recorded_hash = payload_without_hash.pop("payload_sha256")
    canonical = json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert recorded_hash == hashlib.sha256(canonical).hexdigest()
    assert payload["result_kind"] == "cpu_reference_oracle"
    assert payload["scope"]["mlx"] == "not imported"
    assert payload["scope"]["production_cache_integration"] == "not performed"
