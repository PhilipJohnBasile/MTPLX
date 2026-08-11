"""CPU-only tests for the Qwen3.6 norm-shift spot-check report."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/research_qwen36_norm_shift_audit.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("research_qwen36_norm_shift_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_parse_artifact() -> None:
    label, path, mode, expected = probe._parse_artifact(
        "stock::/models/stock::indexed::abc"
    )
    assert (label, path, mode, expected) == (
        "stock",
        Path("/models/stock"),
        "indexed",
        "abc",
    )


def test_parse_artifact_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="invalid artifact"):
        probe._parse_artifact("stock::/models/stock::wrong::abc")


def test_comparison_reports_per_key_mismatch() -> None:
    artifacts = [
        {
            "label": "stock",
            "tensors": {
                "a": {"float32_sha256": "same"},
                "b": {"float32_sha256": "base"},
            },
        },
        {
            "label": "candidate",
            "tensors": {
                "a": {"float32_sha256": "same"},
                "b": {"float32_sha256": "changed"},
            },
        },
    ]
    result = probe._comparison(artifacts)
    assert result["candidate_tensor_matches"]["candidate"] == {
        "a": True,
        "b": False,
    }
    assert result["all_representative_tensors_match"] is False
