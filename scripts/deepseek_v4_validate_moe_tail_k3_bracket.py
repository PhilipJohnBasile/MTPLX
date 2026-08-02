#!/usr/bin/env python3
"""Validate the guarded DeepSeek-V4 MoE-tail primer/C0/B/C1 K3 bracket.

The verdict is persisted before a loss returns nonzero.  Correct-but-slower
results therefore remain auditable and can never be mistaken for a promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


_MODEL_PATH = "/Users/davidtai/models/DeepSeek-V4-Flash-2bit-DQ-mtp"
_PROMPT_PATH = (
    "/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4/"
    "smoke-2bitdq-20260731-prompt2.txt"
)
_PROMPT_SHA256 = "ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33"
_CONFIG_SHA256 = "c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f"
_INDEX_SHA256 = "c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8"
_MLX_CORE_SHA256 = "d7bd29fc20b4a08318d21161c3dfb340889cc9454c5e554ad749eb0127cfa2d6"
_MLX_LIB_SHA256 = "2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd"
_LOCK_PATH = "/tmp/mtplx-gpu-exclusive.lock"
_CONTRACT = {
    "prompt_tokens": 328,
    "max_tokens": 256,
    "depths": [3],
    "verify_strategy": "capture_commit",
    "verify_core": "stock",
    "mtp_history_policy": "committed",
}
_ARTIFACT = {
    "config_sha256": _CONFIG_SHA256,
    "index_sha256": _INDEX_SHA256,
    "model_type": "deepseek_v4",
    "num_hidden_layers": 43,
    "num_nextn_predict_layers": 1,
    "body_q2_routed_projections": 129,
    "body_q2_manifest_tensors": 387,
    "mtp_manifest_tensors": 35,
    "index_weight_count": 2645,
}
_LOADED = {
    "runtime_mtp_enabled": True,
    "body_layers_loaded": 43,
    "mtp_blocks_bound": 1,
    "body_q2_routed_projections": 129,
    "body_q2_weight_dtype": "uint32",
    "mtp_mxfp4_routed_projections": 3,
    "mtp_routed_weight_dtype": "uint32",
}
_TAIL_REPORT = {
    "route": "decode_verify_m4",
    "body_layers_installed": 43,
    "mtp_layers_stock": 1,
    "verify_rows": 4,
    "repair_rows": 1,
    "topk": 6,
    "hidden_size": 4096,
    "kernel_selfcheck_exact": True,
}
_COUNTERS = (
    "accepted_by_depth",
    "drafted_by_depth",
    "accepted_drafts",
    "rejected_drafts",
    "drafted_tokens",
    "skipped_drafts",
    "bonus_tokens",
    "correction_tokens",
    "verify_calls",
    "mtp_forward_calls",
    "make_mtp_cache_calls",
    "update_mtp_cache_calls",
    "mtp_history_append_calls",
    "forward_ar_hidden_calls",
    "forward_ar_plain_calls",
)
_WINDOW_KEYS = {
    "schema_version",
    "kind",
    "verified",
    "verified_monotonic_ns",
    "window_id",
    "attestation",
}


def _stage4_env(candidate: bool) -> dict[str, str]:
    return {
        "MTPLX_COMPILED_VERIFY": "off",
        "MTPLX_DSV4_ATTN": "fused",
        "MTPLX_DSV4_FP32_ACTIVATIONS": "0",
        "MTPLX_DSV4_HC_COMPILE": "1",
        "MTPLX_DSV4_MOE_TAIL": "1" if candidate else "0",
        "MTPLX_DSV4_O_LORA": "cached",
        "MTPLX_DSV4_SINKHORN_KERNEL": "1",
    }


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _guard_errors(window: Any, label: str) -> list[str]:
    prefix = f"{label}.guard_window"
    if not isinstance(window, dict):
        return [f"{prefix} is absent or not an object"]
    if set(window) != _WINDOW_KEYS | {"receipt_path", "receipt_sha256"}:
        return [f"{prefix} has an unexpected shape"]
    document = {key: window[key] for key in _WINDOW_KEYS}
    attestation = document.get("attestation")
    if not isinstance(attestation, dict):
        return [f"{prefix}.attestation is absent or not an object"]
    errors = []
    integers = (
        "guard_pid",
        "child_pid",
        "issued_monotonic_ns",
        "expires_monotonic_ns",
        "lock_device",
        "lock_inode",
    )
    if document.get("schema_version") != 1:
        errors.append(f"{prefix}.schema_version is not 1")
    if document.get("kind") != "mtplx_verified_guard_window":
        errors.append(f"{prefix}.kind is invalid")
    if document.get("verified") is not True:
        errors.append(f"{prefix} is not verified")
    if attestation.get("schema_version") != 1:
        errors.append(f"{prefix}.attestation schema is invalid")
    if any(
        isinstance(attestation.get(key), bool)
        or not isinstance(attestation.get(key), int)
        for key in integers
    ):
        errors.append(f"{prefix}.attestation integer identity is malformed")
    else:
        issued = attestation["issued_monotonic_ns"]
        expires = attestation["expires_monotonic_ns"]
        verified = document.get("verified_monotonic_ns")
        if (
            isinstance(verified, bool)
            or not isinstance(verified, int)
            or not issued <= verified <= expires
            or expires - issued > 60_000_000_000
        ):
            errors.append(f"{prefix} verification is outside the attestation expiry")
    if attestation.get("lock_path") != _LOCK_PATH:
        errors.append(f"{prefix} did not attest the canonical GPU lock")
    if not _valid_sha256(attestation.get("nonce_sha256")):
        errors.append(f"{prefix} nonce digest is malformed")
    if document.get("window_id") != _canonical_digest(attestation):
        errors.append(f"{prefix}.window_id does not bind the attestation")
    receipt_path = window.get("receipt_path")
    if not isinstance(receipt_path, str) or not Path(receipt_path).is_absolute():
        errors.append(f"{prefix}.receipt_path is not absolute")
    if (
        not _valid_sha256(window.get("receipt_sha256"))
        or window.get("receipt_sha256") != _canonical_digest(document)
    ):
        errors.append(f"{prefix}.receipt_sha256 does not bind the document")
    return errors


def _identity_errors(actual: Any, expected: dict, prefix: str) -> list[str]:
    if not isinstance(actual, dict):
        return [f"{prefix} is absent or not an object"]
    return [
        f"{prefix}.{key}={actual.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]


def _receipt_errors(
    receipt: dict[str, Any], label: str, *, candidate: bool, role: str
) -> list[str]:
    errors = []
    for key, expected in _CONTRACT.items():
        if receipt.get(key) != expected:
            errors.append(f"{label}.{key}={receipt.get(key)!r}, expected {expected!r}")
    for key, expected in (
        ("status", 0),
        ("model_path", _MODEL_PATH),
        ("model_type", "deepseek_v4"),
        ("num_hidden_layers", 43),
        ("num_nextn_predict_layers", 1),
        ("receipt_role", role),
        ("performance_eligible", role == "measurement"),
    ):
        if receipt.get(key) != expected:
            errors.append(f"{label}.{key}={receipt.get(key)!r}, expected {expected!r}")
    source_commit = receipt.get("source_commit")
    if not (
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit)
    ):
        errors.append(f"{label}.source_commit is absent or malformed")
    host = receipt.get("host") or {}
    if host.get("mlx_version") != "0.31.2":
        errors.append(f"{label}.host.mlx_version is not official 0.31.2")
    errors.extend(
        _identity_errors(
            receipt.get("mlx_identity"),
            {
                "version": "0.31.2",
                "core_sha256": _MLX_CORE_SHA256,
                "lib_sha256": _MLX_LIB_SHA256,
            },
            f"{label}.mlx_identity",
        )
    )
    errors.extend(
        _identity_errors(
            receipt.get("artifact_identity"), _ARTIFACT, f"{label}.artifact_identity"
        )
    )
    errors.extend(
        _identity_errors(
            receipt.get("loaded_runtime_identity"),
            _LOADED,
            f"{label}.loaded_runtime_identity",
        )
    )
    if receipt.get("prompt_file") != _PROMPT_PATH:
        errors.append(f"{label}.prompt_file is not canonical")
    errors.extend(
        _identity_errors(
            receipt.get("prompt"),
            {"path": _PROMPT_PATH, "sha256": _PROMPT_SHA256, "tokens": 328},
            f"{label}.prompt",
        )
    )
    expected_env = _stage4_env(candidate)
    if receipt.get("launch_mtplx_env") != expected_env:
        errors.append(
            f"{label}.launch_mtplx_env={receipt.get('launch_mtplx_env')!r}, "
            f"expected {expected_env!r}"
        )
    return errors


def _k3_arm(receipt: dict[str, Any], label: str) -> tuple[dict[str, Any] | None, list[str]]:
    arms = [
        arm
        for arm in receipt.get("arms", [])
        if arm.get("speculative_depth") == 3
    ]
    if len(arms) != 1:
        return None, [f"{label} must contain exactly one K3 arm; found {len(arms)}"]
    arm = arms[0]
    errors = []
    if arm.get("error"):
        errors.append(f"{label}.K3 reported error: {arm['error']}")
    tokens = arm.get("tokens")
    if (
        arm.get("generated_tokens") != 256
        or not isinstance(tokens, list)
        or len(tokens) != 256
        or not all(isinstance(token, int) and not isinstance(token, bool) for token in tokens)
    ):
        errors.append(f"{label}.K3 did not persist exactly 256 integer tokens")
    stats = arm.get("stats")
    if not isinstance(stats, dict):
        errors.append(f"{label}.K3 stats are absent")
    else:
        missing = [key for key in _COUNTERS if key not in stats]
        if missing:
            errors.append(f"{label}.K3 stats missing counters {missing}")
        drafted = stats.get("drafted_by_depth")
        if (
            not isinstance(drafted, list)
            or len(drafted) < 3
            or drafted[2] <= 0
            or stats.get("verify_calls", 0) <= 0
        ):
            errors.append(f"{label}.K3 did not execute the physical M4 target workload")
    return arm, errors


def validate_moe_tail_k3_bracket(
    primer: dict[str, Any],
    before: dict[str, Any],
    candidate: dict[str, Any],
    after: dict[str, Any],
    *,
    peak_ceiling_gib: float,
) -> dict[str, Any]:
    receipts = {
        "primer": primer,
        "C0": before,
        "candidate": candidate,
        "C1": after,
    }
    errors = []
    for label, receipt in receipts.items():
        errors.extend(
            _receipt_errors(
                receipt,
                label,
                candidate=label == "candidate",
                role=(
                    "discarded_control_primer"
                    if label == "primer"
                    else "measurement"
                ),
            )
        )
        errors.extend(_guard_errors(receipt.get("guard_window"), label))
    windows = [receipt.get("guard_window") for receipt in receipts.values()]
    same_guard = all(window == windows[0] for window in windows[1:])
    if not same_guard:
        errors.append("guard window differs across primer/C0/candidate/C1")

    arms = {}
    tokens = {}
    counters = {}
    peaks = {}
    measured_tps = {"C0": None, "candidate": None, "C1": None}
    for label, receipt in receipts.items():
        arm, arm_errors = _k3_arm(receipt, label)
        errors.extend(arm_errors)
        if arm is None:
            continue
        arms[label] = arm
        persisted = arm.get("tokens")
        if isinstance(persisted, list):
            tokens[label] = hashlib.sha256(
                json.dumps(persisted, separators=(",", ":")).encode()
            ).hexdigest()
        stats = arm.get("stats")
        if isinstance(stats, dict) and all(key in stats for key in _COUNTERS):
            counters[label] = {key: stats[key] for key in _COUNTERS}
        try:
            peak = float(arm["peak_gib"])
            peaks[label] = peak
            if not 0.0 < peak < peak_ceiling_gib:
                errors.append(
                    f"{label}.K3 peak_gib={peak:g} is outside (0, {peak_ceiling_gib:g})"
                )
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}.K3 peak_gib is invalid")
        if label != "primer":
            try:
                tps = float(arm["decode_tokens_per_second"])
                if tps <= 0:
                    raise ValueError
                measured_tps[label] = tps
            except (KeyError, TypeError, ValueError):
                errors.append(f"{label}.K3 decode_tokens_per_second is invalid")

    token_equal = len(tokens) == 4 and len(set(tokens.values())) == 1
    if not token_equal:
        errors.append("K3 token digest differs across primer/C0/candidate/C1")
    counter_equal = len(counters) == 4 and all(
        value == next(iter(counters.values())) for value in counters.values()
    )
    if not counter_equal:
        errors.append("K3 counters differ across primer/C0/candidate/C1")

    if candidate.get("deepseek_v4_moe_tail") != _TAIL_REPORT:
        errors.append("candidate has no valid MoE-tail installation report")
    for label in ("primer", "C0", "C1"):
        if receipts[label].get("deepseek_v4_moe_tail") is not None:
            errors.append(f"{label} control is not stock: MoE-tail report is present")
    commits = {receipt.get("source_commit") for receipt in receipts.values()}
    if len(commits) != 1 or None in commits:
        errors.append("source_commit differs across primer/C0/candidate/C1")

    drift = None
    candidate_delta = None
    performance_pass = False
    if all(value is not None for value in measured_tps.values()):
        control_mean = (measured_tps["C0"] + measured_tps["C1"]) / 2.0
        drift = abs(measured_tps["C1"] - measured_tps["C0"]) / control_mean
        candidate_delta = (
            measured_tps["candidate"] - control_mean
        ) / control_mean
        performance_pass = candidate_delta > drift
    integrity_pass = not errors
    status = (
        "INVALID_BRACKET"
        if not integrity_pass
        else "PASS"
        if performance_pass
        else "LOSS"
    )
    return {
        "schema_version": 1,
        "kind": "deepseek_v4_moe_tail_k3_bracket",
        "status": status,
        "integrity_pass": integrity_pass,
        "performance_pass": performance_pass if integrity_pass else False,
        "errors": errors,
        "peak_ceiling_gib": peak_ceiling_gib,
        "tokens": {"digests": tokens, "all_equal": token_equal},
        "counters": {"values": counters, "all_equal": counter_equal},
        "peak_gib": peaks,
        "guard_window": {
            "window_id": (
                primer.get("guard_window", {}).get("window_id")
                if isinstance(primer.get("guard_window"), dict)
                else None
            ),
            "all_equal_and_valid": same_guard
            and not any("guard_window" in error for error in errors),
        },
        "primer": {
            "receipt_role": primer.get("receipt_role"),
            "performance_data_used": False,
        },
        "k3_tps": measured_tps,
        "control": {
            "mean_tps": (
                None
                if drift is None
                else (measured_tps["C0"] + measured_tps["C1"]) / 2.0
            ),
            "drift_fraction": drift,
            "candidate_delta_fraction": candidate_delta,
        },
        "source_commit": next(iter(commits)) if len(commits) == 1 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primer", required=True, type=Path)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--peak-ceiling-gib", type=float, default=108.0)
    args = parser.parse_args()
    if args.peak_ceiling_gib <= 0:
        parser.error("--peak-ceiling-gib must be positive")
    result = validate_moe_tail_k3_bracket(
        json.loads(args.primer.read_text()),
        json.loads(args.before.read_text()),
        json.loads(args.candidate.read_text()),
        json.loads(args.after.read_text()),
        peak_ceiling_gib=args.peak_ceiling_gib,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2 if result["status"] == "INVALID_BRACKET" else 1


if __name__ == "__main__":
    raise SystemExit(main())
