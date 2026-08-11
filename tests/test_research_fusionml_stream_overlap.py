"""CPU-only focused checks for the FusionML overlap-probe harness."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "benchmarks/research_fusionml_stream_overlap.py"
SPEC = importlib.util.spec_from_file_location("research_fusionml_stream_overlap", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
MLX_MODULES_BEFORE_IMPORT = {name for name in sys.modules if name == "mlx" or name.startswith("mlx.")}
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)
MLX_MODULES_AFTER_IMPORT = {name for name in sys.modules if name == "mlx" or name.startswith("mlx.")}


def test_module_imports_no_mlx_or_metal_at_import_time() -> None:
    parsed = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    top_level_imports = {
        alias.name.split(".")[0]
        for node in parsed.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [node])
        if isinstance(alias, ast.alias)
    }

    assert "mlx" not in top_level_imports
    assert "metal" not in top_level_imports
    assert MLX_MODULES_AFTER_IMPORT == MLX_MODULES_BEFORE_IMPORT


def test_help_and_argument_validation_do_not_require_mlx() -> None:
    with pytest.raises(SystemExit) as help_exit:
        probe.parse_args(["--help"])
    assert help_exit.value.code == 0
    args = probe.parse_args(["--output", "receipt.json", "--rounds", "8"])
    assert args.execute is False
    assert args.rounds == 8
    assert MLX_MODULES_AFTER_IMPORT == MLX_MODULES_BEFORE_IMPORT


def test_counterbalanced_schedule_places_every_case_in_every_position() -> None:
    names = [f"case_{index}" for index in range(8)]
    schedule = probe.counterbalanced_schedule(names, rounds=16, seed=77)

    assert len(schedule) == 16
    assert all(len(row) == 8 and sorted(row) == names for row in schedule)
    for cycle in (schedule[:8], schedule[8:]):
        for ordinal in range(8):
            assert sorted(row[ordinal] for row in cycle) == names
    assert probe._canonical_sha256(schedule) == probe._canonical_sha256(schedule)


def test_schedule_rejects_invalid_case_shapes() -> None:
    with pytest.raises(ValueError, match="even"):
        probe.counterbalanced_schedule(["a", "b", "c"], rounds=1, seed=0)
    with pytest.raises(ValueError, match="unique"):
        probe.counterbalanced_schedule(["a", "a", "b", "c"], rounds=1, seed=0)
    with pytest.raises(ValueError, match="positive"):
        probe.counterbalanced_schedule(["a", "b", "c", "d"], rounds=0, seed=0)


def test_summary_preserves_raw_samples_and_reports_spread() -> None:
    result = probe._summary([8.0, 10.0, 12.0, 14.0])

    assert result["samples_ms"] == [8.0, 10.0, 12.0, 14.0]
    assert result["median_ms"] == 11.0
    assert result["min_ms"] == 8.0
    assert result["max_ms"] == 14.0
    assert result["spread_percent_of_median"] == pytest.approx(54.5454545)


def test_public_provenance_and_scope_do_not_contain_local_paths() -> None:
    source = probe.SOURCE_PROVENANCE
    assert source["fusionml_main_commit"] == "2e36b64da1fc8e00ea90afa2ec7cc1b9744f0464"
    assert source["fusionml_benchmark_commit"] == "4e7121a8f9c70fd6512b233c5705374626323471"
    assert source["upstream_script"] == "benchmarks/python/stream_parallelism_probe.py"
    assert all("/Users/" not in str(value) and "/private/" not in str(value) for value in source.values())
    assert probe.RESULT_KIND == "synthetic_stream_overlap_mechanism_probe"


def test_process_snapshot_redacts_executable_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess(
        args=["ps"],
        returncode=0,
        stdout="12.5  2.0 /opt/synthetic/private-serve-process\n",
        stderr="",
    )
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: completed)

    snapshot = probe._process_snapshot()

    assert snapshot["available"] is True
    candidate = snapshot["candidate_compute_processes"][0]
    assert candidate == {
        "cpu_percent": 12.5,
        "memory_percent": 2.0,
        "executable_category": "serving_process",
    }
    assert "/opt/" not in str(snapshot)


def test_hardware_run_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "_run_hardware_probe", lambda _: (_ for _ in ()).throw(AssertionError("must not run")))
    with pytest.raises(SystemExit, match="--execute"):
        probe.main(["--output", "receipt.json"])
