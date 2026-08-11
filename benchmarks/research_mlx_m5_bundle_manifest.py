#!/usr/bin/env python3
"""Build a deterministic, public-safe manifest for the M5 research bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PRIVATE_MARKER_PATTERNS = {
    "user_home_path": re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home)/[^/\s]+(?:/|$)"),
    "private_tmp_path": re.compile(r"(?<![A-Za-z0-9_])/private/(?:tmp|var)/"),
    "macos_var_folders_path": re.compile(r"(?<![A-Za-z0-9_])/var/" r"folders/"),
    "hugging_face_env_token": re.compile(r"\bHF" r"_TOKEN\b"),
    "hugging_face_access_token": re.compile(r"\bhf" r"_[A-Za-z0-9]{20,}\b"),
    "github_env_token": re.compile(r"\bGITHUB" r"_TOKEN\b"),
    "github_classic_access_token": re.compile(
        r"\bgh" r"(?:p|o|u|s|r)_[A-Za-z0-9]{20,}\b"
    ),
    "github_fine_grained_access_token": re.compile(
        r"\bgithub" r"_pat_[A-Za-z0-9_]{20,}\b"
    ),
    "authorization_header": re.compile(r"\bAuthor" r"ization\s*:"),
    "bearer_credential": re.compile(r"\bBear" r"er\s+[A-Za-z0-9._~+/=-]{16,}\b"),
}

PROVENANCE_SOURCES = (
    "benchmarks/provenance/README.md",
    "benchmarks/provenance/research_paroquant_q35_fixed_work_v4_producer_1578f6f8.py",
)

SUMMARY_CONTENT_ADDRESSED_RECEIPT_FIELDS = {
    "compressed_q35_fixed_work": ("performance_receipt",),
    "dflash_ddtree_qwen36_35b_a3b": ("receipt",),
    "exact_kernel_stack": ("dispatch_matrix_v2",),
    "mlx_lm_pr_1709_exactness_audit": ("receipt",),
    "mlx_pr_3882_q35": ("public_safe_raw_receipt", "receipt"),
    "mlx_pr_4048_resource_retention": ("receipt",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def assert_public_safe(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for name, pattern in PRIVATE_MARKER_PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"private marker class {name!r} found in {path}")


def validate_content_addressed_references(
    root: Path,
    bundle_root: Path,
    value: Any,
    *,
    path_fields: tuple[str, ...],
    location: str,
) -> int:
    """Recursively validate explicit path-plus-hash fields in a JSON value."""
    if isinstance(value, list):
        return sum(
            validate_content_addressed_references(
                root,
                bundle_root,
                item,
                path_fields=path_fields,
                location=f"{location}[{index}]",
            )
            for index, item in enumerate(value)
        )
    if not isinstance(value, dict):
        return 0

    reference_count = 0
    for path_field in path_fields:
        hash_field = f"{path_field}_sha256"
        has_path = path_field in value
        has_hash = hash_field in value
        if has_path != has_hash:
            raise ValueError(
                f"{location} must define {path_field} and {hash_field} together"
            )
        if not has_path:
            continue

        receipt_path = value[path_field]
        expected_sha256 = value[hash_field]
        if not isinstance(receipt_path, str) or not isinstance(expected_sha256, str):
            raise ValueError(
                f"{location} {path_field} and {hash_field} must both be strings"
            )

        resolved_path = (root / receipt_path).resolve()
        try:
            resolved_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"content-addressed receipt path escapes root: {receipt_path}"
            ) from exc
        try:
            resolved_path.relative_to(bundle_root)
        except ValueError as exc:
            raise ValueError(
                f"content-addressed receipt path is outside result bundle: {receipt_path}"
            ) from exc
        if not resolved_path.is_file():
            raise ValueError(f"content-addressed receipt is not a file: {receipt_path}")

        actual_sha256 = sha256(resolved_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "content-addressed receipt hash mismatch for "
                f"{location}.{path_field}: expected {expected_sha256}, "
                f"actual {actual_sha256}"
            )
        reference_count += 1

    for key, item in value.items():
        reference_count += validate_content_addressed_references(
            root,
            bundle_root,
            item,
            path_fields=path_fields,
            location=f"{location}.{key}",
        )
    return reference_count


def validate_summary_content_addressed_receipts(
    root: Path, bundle_root: Path, summary: dict[str, Any]
) -> int:
    """Validate the summary's explicitly contracted receipt references."""
    reference_count = 0
    for section, path_fields in SUMMARY_CONTENT_ADDRESSED_RECEIPT_FIELDS.items():
        section_value = summary.get(section)
        if section_value is None:
            continue
        reference_count += validate_content_addressed_references(
            root,
            bundle_root,
            section_value,
            path_fields=path_fields,
            location=f"summary.{section}",
        )
    return reference_count


def _entry(root: Path, path: Path, schema: str | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        **({"schema": schema} if schema is not None else {}),
    }


def build_manifest(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    results = root / "benchmarks/results/mlx-m5-research-20260810"
    summary_path = results / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing bundle summary: {summary_path}")
    assert_public_safe(summary_path)
    summary = load_json_strict(summary_path)
    summary_reference_count = validate_summary_content_addressed_receipts(
        root, results, summary
    )

    receipt_entries = []
    for path in sorted(results.rglob("*.json")):
        if path.resolve() == output:
            continue
        assert_public_safe(path)
        report = load_json_strict(path)
        receipt_entries.append(_entry(root, path, report.get("schema")))

    source_paths = [root / "MANIFEST.in"]
    source_paths += sorted((root / "benchmarks").glob("research_*.py"))
    source_paths += sorted((root / "tests").glob("test_research_*.py"))
    source_paths += [
        root / "docs/README.md",
        root / "docs/research/mlx-lm-1709-exactness-audit-20260810.md",
        root / "docs/research/mlx-m5max-speed-without-quality-loss-20260810.md",
    ]
    source_paths += [root / relative for relative in PROVENANCE_SOURCES]
    source_entries = []
    for path in source_paths:
        if path.resolve() == output:
            continue
        assert_public_safe(path)
        source_entries.append(_entry(root, path))

    manifest = {
        "schema": "mtplx-mlx-m5-research-bundle-manifest-v2",
        "status": "identity_and_public_safety_manifest_not_release_evidence",
        "research_date": "2026-08-10-to-2026-08-11",
        "base_commit": "5262443fd4b4b04faab5690346d1225ce6ee5b0e",
        "receipt_count": len(receipt_entries),
        "source_count": len(source_entries),
        "receipts": receipt_entries,
        "sources": source_entries,
        "validation": {
            "json_duplicate_keys_rejected": True,
            "private_marker_classes": list(PRIVATE_MARKER_PATTERNS),
            "provenance_sources_included": list(PROVENANCE_SOURCES),
            "manifest_excludes_itself": True,
            "research_bundle_distribution_scope": (
                "repository_only_excluded_from_sdist_and_wheel"
            ),
            "summary_content_addressed_receipt_references_validated": summary_reference_count,
        },
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results/mlx-m5-research-20260810/bundle_manifest.json",
    )
    args = parser.parse_args()
    report = build_manifest(args.root, args.json_out)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
