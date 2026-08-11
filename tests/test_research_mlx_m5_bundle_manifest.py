"""Tests for the deterministic M5 research-bundle manifest."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/research_mlx_m5_bundle_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("research_mlx_m5_bundle_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _fixture(root: Path, payload: str = '{"schema":"receipt-v1"}\n') -> Path:
    receipt_dir = root / "benchmarks/results/mlx-m5-research-20260810"
    receipt_dir.mkdir(parents=True)
    receipt = receipt_dir / "receipt.json"
    receipt.write_text(payload)
    summary = {
        "compressed_q35_fixed_work": {
            "artifacts": {
                "fixture": {
                    "performance_receipt": receipt.relative_to(root).as_posix(),
                    "performance_receipt_sha256": hashlib.sha256(
                        payload.encode()
                    ).hexdigest(),
                }
            }
        }
    }
    (receipt_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    (root / "benchmarks/research_probe.py").write_text("# probe\n")
    (root / "tests").mkdir()
    (root / "tests/test_research_probe.py").write_text("# test\n")
    (root / "MANIFEST.in").write_text("include benchmarks/research_*.py\n")
    docs = root / "docs/research"
    docs.mkdir(parents=True)
    (docs / "mlx-lm-1709-exactness-audit-20260810.md").write_text("audit\n")
    (docs / "mlx-m5max-speed-without-quality-loss-20260810.md").write_text("research\n")
    (docs / "native-mtp-on-mlx.md").write_text("unrelated research\n")
    (root / "docs/README.md").write_text("docs\n")
    provenance = root / "benchmarks/provenance"
    provenance.mkdir()
    (provenance / "README.md").write_text("historical source notes\n")
    (
        provenance / "research_paroquant_q35_fixed_work_v4_producer_1578f6f8.py"
    ).write_text("# historical producer\n")
    return receipt_dir / "bundle_manifest.json"


def test_manifest_is_deterministic_and_excludes_itself(tmp_path: Path) -> None:
    output = _fixture(tmp_path)
    output.write_text('{"schema":"old-manifest"}\n')
    first = probe.build_manifest(tmp_path, output)
    second = probe.build_manifest(tmp_path, output)
    assert first == second
    assert first["receipt_count"] == 2
    assert first["source_count"] == 8
    assert (
        first["validation"]["summary_content_addressed_receipt_references_validated"]
        == 1
    )
    assert (
        first["validation"]["research_bundle_distribution_scope"]
        == "repository_only_excluded_from_sdist_and_wheel"
    )
    source_paths = {row["path"] for row in first["sources"]}
    assert {
        "MANIFEST.in",
        "docs/README.md",
        "docs/research/mlx-lm-1709-exactness-audit-20260810.md",
        "docs/research/mlx-m5max-speed-without-quality-loss-20260810.md",
    } <= source_paths
    assert "docs/research/native-mtp-on-mlx.md" not in source_paths
    assert all(row["path"] != str(output) for row in first["receipts"])


def test_repository_manifest_scopes_research_bundle_to_repository() -> None:
    manifest = SCRIPT.parents[1] / "MANIFEST.in"
    directives = set(manifest.read_text(encoding="utf-8").splitlines())

    assert {
        "include CITATION.cff",
        "include CHANGELOG.md",
        "recursive-include scripts *.py *.sh",
        "exclude tests/test_research_*.py",
    } <= directives
    research_inclusion_directives = {
        "include benchmarks/provenance/README.md",
        "include benchmarks/provenance/"
        "research_paroquant_q35_fixed_work_v4_producer_1578f6f8.py",
        "include benchmarks/research_*.py",
        "recursive-include benchmarks/results/mlx-m5-research-20260810 *.json",
        "include docs/README.md",
        "include docs/research/mlx-lm-1709-exactness-audit-20260810.md",
        "include docs/research/mlx-m5max-speed-without-quality-loss-20260810.md",
        "recursive-include docs/research *.md",
        "include tests/test_research_*.py",
    }
    assert not directives & research_inclusion_directives


def test_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    output = _fixture(tmp_path, '{"schema":"one","schema":"two"}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        probe.build_manifest(tmp_path, output)


def test_manifest_rejects_stale_summary_receipt_hash(tmp_path: Path) -> None:
    output = _fixture(tmp_path)
    summary_path = output.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["compressed_q35_fixed_work"]["artifacts"]["fixture"][
        "performance_receipt_sha256"
    ] = "0" * 64
    summary_path.write_text(json.dumps(summary) + "\n")

    with pytest.raises(ValueError, match="content-addressed receipt hash mismatch"):
        probe.build_manifest(tmp_path, output)


def test_manifest_rejects_summary_receipt_path_escape(tmp_path: Path) -> None:
    output = _fixture(tmp_path)
    summary_path = output.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["compressed_q35_fixed_work"]["artifacts"]["fixture"][
        "performance_receipt"
    ] = "../../../outside.json"
    summary_path.write_text(json.dumps(summary) + "\n")

    with pytest.raises(ValueError, match="escapes root"):
        probe.build_manifest(tmp_path, output)


def test_manifest_rejects_private_paths(tmp_path: Path) -> None:
    output = _fixture(tmp_path, json.dumps({"path": "/" + "Users/pjb/model"}))
    with pytest.raises(ValueError, match="private marker"):
        probe.build_manifest(tmp_path, output)


def test_manifest_rejects_any_user_home_path(tmp_path: Path) -> None:
    output = _fixture(tmp_path, json.dumps({"path": "/" + "Users/another-user/model"}))
    with pytest.raises(ValueError, match="user_home_path"):
        probe.build_manifest(tmp_path, output)


@pytest.mark.parametrize(
    ("marker_class", "sensitive_value"),
    [
        ("macos_var_folders_path", "/var/" + "folders/ab/cdef/T/private.json"),
        ("hugging_face_access_token", "hf" + "_" + "a" * 32),
        ("github_classic_access_token", "gh" + "p_" + "b" * 36),
        (
            "github_fine_grained_access_token",
            "github" + "_pat_" + "c" * 40,
        ),
        ("bearer_credential", "Bearer " + "d" * 32),
    ],
)
def test_manifest_rejects_actual_credential_and_temp_forms(
    tmp_path: Path, marker_class: str, sensitive_value: str
) -> None:
    output = _fixture(tmp_path, json.dumps({"value": sensitive_value}))
    with pytest.raises(ValueError, match=marker_class):
        probe.build_manifest(tmp_path, output)


def test_scanner_source_and_fixture_literals_do_not_self_match() -> None:
    probe.assert_public_safe(SCRIPT)
    probe.assert_public_safe(Path(__file__).resolve())
