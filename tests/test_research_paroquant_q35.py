"""CPU-only tests for compressed-artifact performance comparison receipts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/research_paroquant_q35.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("research_paroquant_q35", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _receipt(
    label: str,
    manifest: str,
    prefill_ms: float,
    decode_tps: float,
    fixed_source_hash: str = "source-receipt",
    campaign_id: str = "campaign",
    ordinal: int = 1,
    arm: str | None = None,
    continuation_tokens: int = 128,
    decode_forward_steps: int | None = None,
    continuation_token_ids: list[int] | None = None,
) -> dict:
    if arm is None:
        arm = "stock-4bit" if manifest == "stock" else "nvfp4"
    if decode_forward_steps is None:
        decode_forward_steps = continuation_tokens - 1
    if continuation_token_ids is None:
        continuation_token_ids = list(range(1_000, 1_000 + continuation_tokens))
    token_hash = hashlib.sha256(
        np.asarray(continuation_token_ids, dtype=np.int32).tobytes()
    ).hexdigest()
    return {
        "schema": "mtplx-q35-compressed-artifact-process-v4",
        "label": label,
        "campaign": {
            "id": campaign_id,
            "arm": arm,
            "ordinal": ordinal,
            "started_at_utc": "2026-08-11T00:00:00+00:00",
            "finished_at_utc": "2026-08-11T00:00:01+00:00",
            "command_contract": {
                "continuation_tokens": continuation_tokens,
                "prefill_repetitions": 2,
                "prompt_length": 512,
            },
        },
        "model_identity": {"weight_manifest_after_sha256": manifest},
        "prompt": {"length": 512, "token_ids_sha256": "prompt"},
        "prefill": {"median_ms": prefill_ms, "samples_ms": [prefill_ms, prefill_ms]},
        "continuation": {
            "tokens": continuation_tokens,
            "decode_forward_steps": decode_forward_steps,
            "decode_tokens_per_second": decode_tps,
            "token_ids_sha256": token_hash,
            "token_ids": continuation_token_ids,
            "decode_workload": "teacher_forced_fixed_continuation",
            "fixed_work_token_ids_sha256": token_hash,
            "fixed_work_source_receipt_sha256": fixed_source_hash,
        },
        "load_seconds": 1.0,
        "peak_memory_bytes_after_weight_residency": 100,
    }


def test_compare_performance_artifacts_reports_speed_and_repeat(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for label, manifest, prefill_ms, decode_tps, ordinal in (
        ("stock_a1", "stock", 100.0, 50.0, 1),
        ("stock_a2", "stock", 102.0, 52.0, 4),
        ("nv_b1", "nv", 80.0, 60.0, 2),
        ("nv_b2", "nv", 82.0, 62.0, 3),
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(
            json.dumps(
                _receipt(
                    label,
                    manifest,
                    prefill_ms,
                    decode_tps,
                    ordinal=ordinal,
                )
            )
        )
        paths[label] = path

    output = tmp_path / "comparison.json"
    result = probe.compare_performance_artifacts(
        [paths["stock_a1"], paths["stock_a2"]],
        [paths["nv_b1"], paths["nv_b2"]],
        output,
        "stock-4bit",
        "nvfp4",
        ["stock-4bit", "nvfp4", "nvfp4", "stock-4bit"],
    )
    assert result["baseline"]["exact_continuation_repeat"] is True
    assert result["candidate"]["exact_continuation_repeat"] is True
    assert result["comparison"][
        "prefill_speedup_candidate_over_baseline"
    ] == pytest.approx(101.0 / 81.0)
    assert result["comparison"][
        "decode_speedup_candidate_over_baseline"
    ] == pytest.approx(61.0 / 51.0)
    assert result["comparison"]["same_fixed_token_sequence_across_artifacts"] is True
    assert result["comparison"]["same_decode_forward_count_across_artifacts"] is True
    assert result["comparison"]["same_routed_moe_work"] == "unobserved_not_established"
    assert json.loads(output.read_text())["promotion_decision"].startswith(
        "not_decided"
    )


def test_compare_performance_artifacts_rejects_bad_order(tmp_path: Path) -> None:
    paths = []
    for index, manifest in enumerate(("stock", "stock", "nv", "nv")):
        path = tmp_path / f"r{index}.json"
        path.write_text(
            json.dumps(_receipt(f"r{index}", manifest, 10.0, 20.0, ordinal=index + 1))
        )
        paths.append(path)
    with pytest.raises(ValueError, match="length"):
        probe.compare_performance_artifacts(
            paths[:2],
            paths[2:],
            tmp_path / "out.json",
            "stock",
            "nv",
            ["stock", "nv"],
        )


def test_compare_rejects_greedy_or_missing_fixed_work(tmp_path: Path) -> None:
    paths = []
    for index, manifest in enumerate(("stock", "stock", "nv", "nv")):
        receipt = _receipt(f"r{index}", manifest, 10.0, 20.0, ordinal=index + 1)
        if index == 0:
            receipt["continuation"].pop("decode_workload")
        path = tmp_path / f"r{index}.json"
        path.write_text(json.dumps(receipt))
        paths.append(path)
    with pytest.raises(ValueError, match="fixed-work"):
        probe.compare_performance_artifacts(
            paths[:2],
            paths[2:],
            tmp_path / "out.json",
            "stock",
            "nv",
            ["stock", "nv", "nv", "stock"],
        )


def test_compare_rejects_different_fixed_work_sources(tmp_path: Path) -> None:
    paths = []
    for index, manifest in enumerate(("stock", "stock", "nv", "nv")):
        source_hash = "source-a" if index < 2 else "source-b"
        path = tmp_path / f"r{index}.json"
        path.write_text(
            json.dumps(
                _receipt(
                    f"r{index}",
                    manifest,
                    10.0,
                    20.0,
                    source_hash,
                    ordinal=index + 1,
                )
            )
        )
        paths.append(path)
    with pytest.raises(ValueError, match="fixed-work sources differ"):
        probe.compare_performance_artifacts(
            paths[:2],
            paths[2:],
            tmp_path / "out.json",
            "stock",
            "nv",
            ["stock", "nv", "nv", "stock"],
        )


def test_compare_rejects_cross_arm_decode_forward_count_mismatch(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for index, manifest in enumerate(("stock", "stock", "nv", "nv")):
        candidate = index >= 2
        path = tmp_path / f"r{index}.json"
        path.write_text(
            json.dumps(
                _receipt(
                    f"r{index}",
                    manifest,
                    10.0,
                    20.0,
                    ordinal=index + 1,
                    continuation_tokens=127 if candidate else 128,
                    decode_forward_steps=126 if candidate else 127,
                )
            )
        )
        paths.append(path)

    with pytest.raises(ValueError, match="decode-forward counts differ: 127 != 126"):
        probe.compare_performance_artifacts(
            paths[:2],
            paths[2:],
            tmp_path / "out.json",
            "stock-4bit",
            "nvfp4",
            ["stock-4bit", "nvfp4", "nvfp4", "stock-4bit"],
        )


def test_performance_arm_rejects_campaign_count_mismatch(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index in range(2):
        receipt = _receipt(f"r{index}", "stock", 10.0, 20.0)
        if index == 1:
            receipt["campaign"]["command_contract"]["prefill_repetitions"] = 3
        path = tmp_path / f"r{index}.json"
        path.write_text(json.dumps(receipt))
        paths.append(path)

    with pytest.raises(ValueError, match="prefill-repetition count mismatch"):
        probe._performance_arm(paths)


def test_fixed_continuation_rejects_tampered_token_list(tmp_path: Path) -> None:
    source = _receipt("stock", "stock", 10.0, 20.0)
    source["continuation"]["token_ids"] = [1, 2, 3]
    source["continuation"]["tokens"] = 3
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="token hash"):
        probe.fixed_continuation(path, 3)


def test_compare_rejects_asserted_order_that_disagrees_with_receipts(
    tmp_path: Path,
) -> None:
    paths = []
    for index, manifest in enumerate(("stock", "stock", "nv", "nv")):
        ordinal = (1, 4, 2, 3)[index]
        receipt = _receipt(f"r{index}", manifest, 10.0, 20.0, ordinal=ordinal)
        if index == 2:
            receipt["campaign"]["arm"] = "stock-4bit"
        path = tmp_path / f"r{index}.json"
        path.write_text(json.dumps(receipt))
        paths.append(path)
    with pytest.raises(ValueError, match="recorded campaign arms"):
        probe.compare_performance_artifacts(
            paths[:2],
            paths[2:],
            tmp_path / "out.json",
            "stock-4bit",
            "nvfp4",
            ["stock-4bit", "nvfp4", "nvfp4", "stock-4bit"],
        )


def test_performance_arm_rejects_tampered_v4_token_ids_with_metadata_unchanged(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for index in range(2):
        receipt = _receipt(f"r{index}", "stock", 10.0, 20.0, ordinal=index + 1)
        if index == 1:
            receipt["continuation"]["token_ids"][0] += 1
        path = tmp_path / f"r{index}.json"
        path.write_text(json.dumps(receipt))
        paths.append(path)

    with pytest.raises(ValueError, match="continuation token hash"):
        probe._performance_arm(paths)


def test_performance_arm_rejects_tampered_v4_token_ids_with_fixed_work_stale(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for index in range(2):
        receipt = _receipt(f"r{index}", "stock", 10.0, 20.0, ordinal=index + 1)
        if index == 1:
            token_ids = receipt["continuation"]["token_ids"]
            token_ids[0] += 1
            receipt["continuation"]["token_ids_sha256"] = hashlib.sha256(
                np.asarray(token_ids, dtype=np.int32).tobytes()
            ).hexdigest()
        path = tmp_path / f"r{index}.json"
        path.write_text(json.dumps(receipt))
        paths.append(path)

    with pytest.raises(ValueError, match="fixed-work token hash"):
        probe._performance_arm(paths)


def test_performance_arm_rejects_nonrepeating_valid_v4_continuations(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for index in range(2):
        token_ids = list(range(2_000 + index, 2_128 + index))
        path = tmp_path / f"r{index}.json"
        path.write_text(
            json.dumps(
                _receipt(
                    f"r{index}",
                    "stock",
                    10.0,
                    20.0,
                    ordinal=index + 1,
                    continuation_token_ids=token_ids,
                )
            )
        )
        paths.append(path)

    with pytest.raises(ValueError, match="exact continuation repeat"):
        probe._performance_arm(paths)


def test_compare_rejects_cross_arm_validated_fixed_token_mismatch(
    tmp_path: Path,
) -> None:
    paths: list[Path] = []
    for index, manifest in enumerate(("stock", "stock", "nv", "nv")):
        candidate = index >= 2
        token_ids = list(
            range(3_000 if candidate else 1_000, 3_128 if candidate else 1_128)
        )
        path = tmp_path / f"r{index}.json"
        path.write_text(
            json.dumps(
                _receipt(
                    f"r{index}",
                    manifest,
                    10.0,
                    20.0,
                    ordinal=index + 1,
                    continuation_token_ids=token_ids,
                )
            )
        )
        paths.append(path)

    with pytest.raises(ValueError, match="fixed-work continuations differ"):
        probe.compare_performance_artifacts(
            paths[:2],
            paths[2:],
            tmp_path / "out.json",
            "stock-4bit",
            "nvfp4",
            ["stock-4bit", "nvfp4", "nvfp4", "stock-4bit"],
        )
