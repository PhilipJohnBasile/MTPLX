"""CPU-only tests for the ParoQuant HumanEval likelihood probe."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "benchmarks/research_paroquant_q35_nll.py"
)
SPEC = importlib.util.spec_from_file_location("research_paroquant_q35_nll", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_summary_is_token_weighted() -> None:
    result = probe._summary([1.0, 3.0], [3, 1])
    assert result["task_count"] == 2
    assert result["scored_token_count"] == 4
    assert result["mean_nll_nats_per_token"] == pytest.approx(1.5)
    assert result["perplexity"] == pytest.approx(math.exp(1.5))
    assert result["median_task_nll_nats_per_token"] == 2.0


def test_load_rows_preserves_file_order_and_limit(tmp_path: Path) -> None:
    path = tmp_path / "HumanEval.jsonl"
    rows = [
        {
            "task_id": f"HumanEval/{index}",
            "prompt": f"def f{index}():\n",
            "canonical_solution": f"    return {index}\n",
        }
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = probe._load_rows(path, 2)
    assert [row["task_id"] for row in selected] == ["HumanEval/0", "HumanEval/1"]


def test_load_rows_rejects_missing_canonical_solution(tmp_path: Path) -> None:
    path = tmp_path / "HumanEval.jsonl"
    path.write_text(
        json.dumps({"task_id": "HumanEval/0", "prompt": "def f():\n"}) + "\n"
    )
    with pytest.raises(ValueError, match="canonical_solution"):
        probe._load_rows(path, 1)


def test_summary_rejects_misaligned_vectors() -> None:
    with pytest.raises(ValueError, match="aligned"):
        probe._summary([1.0], [])


def _receipt(label: str, manifest: str, nlls: list[float]) -> dict:
    tasks = [
        {
            "task_id": f"HumanEval/{index}",
            "canonical_solution_tokens": 2,
            "prompt_token_ids_sha256": f"prompt-{index}",
            "canonical_solution_token_ids_sha256": f"solution-{index}",
            "nll_nats_per_token": nll,
        }
        for index, nll in enumerate(nlls)
    ]
    aggregate = probe._summary(nlls, [2] * len(nlls))
    return {
        "schema": "mtplx-q35-compressed-artifact-humaneval-nll-v3",
        "label": label,
        "dataset": {"sha256": "dataset", "selection": "first_2_file_order"},
        "model_identity": {"weight_manifest_after_sha256": manifest},
        "tokenizer_identity": {
            "asset_manifest_sha256": "tokenizer",
            "assets": [{"filename": "tokenizer.json", "sha256": "tokenizer"}],
        },
        "aggregate": aggregate,
        "tasks": tasks,
    }


def _generic_receipt(label: str, manifest: str, nlls: list[float]) -> dict:
    return _receipt(label, manifest, nlls)


def test_compare_receipts_is_repeat_checked_and_directional(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for label, manifest, nlls in (
        ("stock_a1", "stock", [1.0, 2.0]),
        ("stock_a2", "stock", [1.0, 2.0]),
        ("paro_b1", "paro", [1.5, 1.0]),
        ("paro_b2", "paro", [1.5, 1.0]),
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(_receipt(label, manifest, nlls)))
        paths[label] = path
    output = tmp_path / "comparison.json"
    result = probe.compare_receipts(
        [paths["stock_a1"], paths["stock_a2"]],
        [paths["paro_b1"], paths["paro_b2"]],
        output,
    )
    assert result["stock"]["exact_task_nll_repeat"] is True
    assert result["paro"]["exact_task_nll_repeat"] is True
    assert result["comparison"]["tasks_worse"] == 1
    assert result["comparison"]["tasks_improved"] == 1
    assert result["status"] == "directional_quality_improvement"
    assert result["repeat_gate_pass"] is True
    assert result["promotion_decision"] == "not_decided_by_teacher_forced_nll_alone"
    assert json.loads(output.read_text())["schema"].endswith("comparison-v2")


@pytest.mark.parametrize(
    ("candidate_nlls", "expected_status"),
    [
        ([1.2, 2.2], "directional_quality_regression"),
        ([0.8, 1.8], "directional_quality_improvement"),
        ([1.0, 2.0], "no_directional_quality_change"),
    ],
)
def test_compare_receipts_classifies_all_directions(
    tmp_path: Path, candidate_nlls: list[float], expected_status: str
) -> None:
    paths: list[Path] = []
    for label, manifest, nlls in (
        ("stock_a1", "stock", [1.0, 2.0]),
        ("stock_a2", "stock", [1.0, 2.0]),
        ("paro_b1", "paro", candidate_nlls),
        ("paro_b2", "paro", candidate_nlls),
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(_receipt(label, manifest, nlls)))
        paths.append(path)

    result = probe.compare_receipts(paths[:2], paths[2:], tmp_path / "comparison.json")
    assert result["status"] == expected_status
    assert result["promotion_decision"] == "not_decided_by_teacher_forced_nll_alone"


def test_compare_receipts_rejects_dataset_mismatch(tmp_path: Path) -> None:
    paths = []
    for index in range(4):
        report = _receipt(f"r{index}", "stock" if index < 2 else "paro", [1.0])
        if index == 3:
            report["dataset"]["sha256"] = "different"
        path = tmp_path / f"r{index}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    with pytest.raises(ValueError, match="dataset mismatch"):
        probe.compare_receipts(paths[:2], paths[2:], tmp_path / "out.json")


def test_compare_artifacts_rejects_tokenizer_or_token_sequence_mismatch(
    tmp_path: Path,
) -> None:
    paths = []
    for index, manifest in enumerate(
        ("baseline", "baseline", "candidate", "candidate")
    ):
        report = _generic_receipt(f"r{index}", manifest, [1.0])
        if index == 2:
            report["tokenizer_identity"]["asset_manifest_sha256"] = "different"
        path = tmp_path / f"r{index}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    with pytest.raises(ValueError, match="tokenizer identity mismatch"):
        probe.compare_artifacts(
            paths[:2], paths[2:], tmp_path / "out.json", "baseline", "candidate"
        )


def test_compare_artifacts_uses_neutral_names_and_status(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for label, manifest, nlls in (
        ("baseline_a1", "baseline", [1.0, 2.0]),
        ("baseline_a2", "baseline", [1.0, 2.0]),
        ("candidate_b1", "candidate", [1.2, 2.2]),
        ("candidate_b2", "candidate", [1.2, 2.2]),
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(_generic_receipt(label, manifest, nlls)))
        paths[label] = path

    output = tmp_path / "comparison.json"
    result = probe.compare_artifacts(
        [paths["baseline_a1"], paths["baseline_a2"]],
        [paths["candidate_b1"], paths["candidate_b2"]],
        output,
        "stock-4bit",
        "candidate-nvfp4",
    )
    assert result["status"] == "directional_quality_regression"
    assert result["promotion_decision"] == "not_decided_by_teacher_forced_nll_alone"
    assert result["baseline_name"] == "stock-4bit"
    assert result["candidate_name"] == "candidate-nvfp4"
    assert result["baseline"]["exact_task_nll_repeat"] is True
    assert result["candidate"]["exact_task_nll_repeat"] is True
    assert result["repeat_gate_pass"] is True
    assert result["comparison"]["tasks_worse"] == 2
    assert json.loads(output.read_text())["schema"].endswith("comparison-v2")


def test_compare_artifacts_marks_repeat_mismatch(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for label, manifest, nlls in (
        ("baseline_a1", "baseline", [1.0]),
        ("baseline_a2", "baseline", [1.0]),
        ("candidate_b1", "candidate", [1.1]),
        ("candidate_b2", "candidate", [1.2]),
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(_generic_receipt(label, manifest, nlls)))
        paths[label] = path

    result = probe.compare_artifacts(
        [paths["baseline_a1"], paths["baseline_a2"]],
        [paths["candidate_b1"], paths["candidate_b2"]],
        tmp_path / "comparison.json",
        "baseline",
        "candidate",
    )
    assert result["repeat_gate_pass"] is False
    assert result["status"].endswith("_with_repeat_mismatch")


def test_compare_artifacts_averages_each_task_across_repeats(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for label, manifest, nlls in (
        ("baseline_a1", "baseline", [1.0, 4.0]),
        ("baseline_a2", "baseline", [3.0, 2.0]),
        ("candidate_b1", "candidate", [3.0, 3.0]),
        ("candidate_b2", "candidate", [1.0, 5.0]),
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(_generic_receipt(label, manifest, nlls)))
        paths[label] = path

    result = probe.compare_artifacts(
        [paths["baseline_a1"], paths["baseline_a2"]],
        [paths["candidate_b1"], paths["candidate_b2"]],
        tmp_path / "comparison.json",
        "baseline",
        "candidate",
    )
    assert result["comparison"]["tasks_equal"] == 1
    assert result["comparison"]["tasks_worse"] == 1
    assert result["comparison"]["tasks_improved"] == 0
    assert result["task_delta_basis"] == (
        "per-task mean across all receipts in each arm"
    )


def test_downgrade_historical_comparison_preserves_numbers_without_authority(
    tmp_path: Path,
) -> None:
    legacy = {
        "schema": "mtplx-q35-compressed-artifact-nll-comparison-v2",
        "baseline_name": "stock",
        "candidate_name": "candidate",
        "comparison_script_sha256": "legacy-script",
        "repeat_gate_pass": True,
        "status": "directional_quality_regression",
        "comparison": {"mean_nll_percent_change": 9.0, "tasks_worse": 2},
        "task_delta_basis": "legacy",
        "task_deltas_descending": [{"task_id": "HumanEval/0"}],
        "baseline": {"mean_nll_nats_per_token": 1.0},
        "candidate": {"mean_nll_nats_per_token": 1.1},
    }
    source = tmp_path / "legacy.json"
    output = tmp_path / "downgraded.json"
    source.write_text(json.dumps(legacy))

    result = probe.downgrade_historical_comparison(source, output)

    assert result["status"].startswith("historical_not_comparable")
    assert (
        result["quality_evidence"] == "not_usable_for_cross_artifact_quality_direction"
    )
    assert (
        result["historical_unverified_numeric_output"]["comparison"]["tasks_worse"] == 2
    )
    assert (
        json.loads(output.read_text())["promotion_decision"]
        == "not_decided_by_historical_nll"
    )
