"""Canonical effective-run records for comparable benchmark arms.

Benchmark labels describe intent.  This module records what the runner
actually executed and provides a strict comparison gate so a result cannot be
silently compared across thinking modes, prompt suites, sampler settings, or
aggregation methods.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_SCHEMA = 1
WEIGHTED_TOK_S_AGGREGATION = "sum_generated_tokens/sum_decode_seconds"

_COMPARISON_FIELDS = (
    "prompt_suite_sha256",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "seed_policy",
    "enable_thinking",
    "runtime_contract",
    "aggregation",
)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_effective_run_record(
    *,
    backend: str,
    model_ref: Path | str,
    model_revision: str | None,
    draft_ref: Path | str | None,
    draft_revision: str | None,
    prompt_suite: Path | str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: int,
    seed_policy: str = "fixed",
    enable_thinking: bool,
    block_size: int | None,
    generation_mode: str,
    nax_enabled: bool,
    verify_strategy: str,
    compiled_verify: str,
    runtime_switches: Mapping[str, str],
    runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    suite = Path(prompt_suite).expanduser().resolve()
    return {
        "schema": PROTOCOL_SCHEMA,
        "backend": str(backend),
        "model": {
            "ref": str(model_ref),
            "revision": _required_revision(model_revision, "model_revision"),
        },
        "draft": (
            None
            if draft_ref is None
            else {
                "ref": str(draft_ref),
                "revision": _required_revision(draft_revision, "draft_revision"),
            }
        ),
        "prompt_suite": str(suite),
        "prompt_suite_sha256": file_sha256(suite),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_tokens": int(max_tokens),
        "seed": int(seed),
        "seed_policy": str(seed_policy),
        "enable_thinking": bool(enable_thinking),
        "block_size": None if block_size is None else int(block_size),
        "generation_mode": str(generation_mode),
        "nax_enabled": bool(nax_enabled),
        "verify_strategy": str(verify_strategy),
        "compiled_verify": str(compiled_verify),
        "runtime_contract": {
            str(key): value for key, value in sorted((runtime_contract or {}).items())
        },
        "runtime_switches": {
            str(key): str(value) for key, value in sorted(runtime_switches.items())
        },
        "aggregation": WEIGHTED_TOK_S_AGGREGATION,
    }


def weighted_tok_s(rows: list[Mapping[str, Any]]) -> float:
    generated_tokens = 0
    decode_seconds = 0.0
    for row in rows:
        tokens = int(row.get("generated_tokens") or 0)
        tok_s = float(row.get("tok_s") or 0.0)
        if tokens <= 0 or tok_s <= 0:
            continue
        generated_tokens += tokens
        decode_seconds += tokens / tok_s
    return generated_tokens / decode_seconds if decode_seconds > 0 else 0.0


def summarize_external_draft_contract(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_sampler: Mapping[str, Any],
    expected_draft_revision: str | None = None,
) -> dict[str, Any]:
    """Summarize the proposal law and verifier lane actually observed.

    This consumes generation events rather than command-line intent.  A
    stochastic DFlash receipt is contract-valid only when every observed
    proposal declares sampled ``q``, reports the expected sampler, and enters
    probability-ratio acceptance with residual correction available.
    """

    declarations: Counter[str] = Counter()
    acceptances: Counter[str] = Counter()
    draft_q_kinds: Counter[str] = Counter()
    sampler_mismatches: list[dict[str, Any]] = []
    draft_revision_mismatches: list[dict[str, Any]] = []
    engine_q_mismatches: list[dict[str, Any]] = []
    proposal_count = 0
    rejected_with_correction = 0
    for event_index, event in enumerate(events):
        block = event.get("block_draft_source")
        if not isinstance(block, Mapping):
            continue
        proposal_count += 1
        metadata = block.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        declarations[str(metadata.get("declaration") or "<missing>")] += 1
        acceptances[str(block.get("acceptance") or "<missing>")] += 1
        draft_q_kinds[str(block.get("draft_q") or "<missing>")] += 1
        observed_sampler = metadata.get("sampler")
        observed_sampler = (
            observed_sampler if isinstance(observed_sampler, Mapping) else {}
        )
        normalized_observed = {
            "temperature": float(observed_sampler.get("temperature", 0.0)),
            "top_p": float(observed_sampler.get("top_p", 0.0)),
            "top_k": int(observed_sampler.get("top_k", 0)),
        }
        normalized_expected = {
            "temperature": float(expected_sampler["temperature"]),
            "top_p": float(expected_sampler["top_p"]),
            "top_k": int(expected_sampler["top_k"]),
        }
        if normalized_observed != normalized_expected:
            sampler_mismatches.append(
                {
                    "event_index": event_index,
                    "observed": normalized_observed,
                    "expected": normalized_expected,
                }
            )
        artifact = metadata.get("draft_artifact")
        artifact = artifact if isinstance(artifact, Mapping) else {}
        observed_revision = artifact.get("resolved_revision")
        if (
            expected_draft_revision is not None
            and str(observed_revision or "") != str(expected_draft_revision)
        ):
            draft_revision_mismatches.append(
                {
                    "event_index": event_index,
                    "observed": observed_revision,
                    "expected": str(expected_draft_revision),
                }
            )
        proposal_tokens = int(block.get("proposal_tokens") or 0)
        support_sizes = block.get("engine_q_support_sizes")
        sampled_probabilities = block.get("engine_q_sampled_probabilities")
        if (
            not isinstance(support_sizes, Sequence)
            or isinstance(support_sizes, (str, bytes))
            or not isinstance(sampled_probabilities, Sequence)
            or isinstance(sampled_probabilities, (str, bytes))
            or len(support_sizes) != proposal_tokens
            or len(sampled_probabilities) != proposal_tokens
            or any(int(size) <= 0 for size in support_sizes)
            or any(
                not 0.0 < float(probability) <= 1.0
                for probability in sampled_probabilities
            )
        ):
            engine_q_mismatches.append(
                {
                    "event_index": event_index,
                    "proposal_tokens": proposal_tokens,
                    "support_sizes": list(support_sizes or []),
                    "sampled_probabilities": list(sampled_probabilities or []),
                }
            )
        for draft in event.get("drafts", ()):
            if not isinstance(draft, Mapping):
                continue
            if (
                draft.get("accepted") is False
                and draft.get("correction") is not None
                and draft.get("correction_origin") == "residual_p_minus_q"
            ):
                rejected_with_correction += 1

    contract_matches = bool(
        proposal_count > 0
        and declarations == Counter({"sampled_q": proposal_count})
        and acceptances == Counter({"probability_ratio_residual": proposal_count})
        and draft_q_kinds == Counter({"soft": proposal_count})
        and not sampler_mismatches
        and not draft_revision_mismatches
        and not engine_q_mismatches
    )
    return {
        "proposal_count": proposal_count,
        "declarations": dict(sorted(declarations.items())),
        "acceptance_lanes": dict(sorted(acceptances.items())),
        "draft_q_kinds": dict(sorted(draft_q_kinds.items())),
        "soft_q_proposals": int(draft_q_kinds.get("soft", 0)),
        "rejected_with_residual_correction": rejected_with_correction,
        "sampler_mismatches": sampler_mismatches,
        "draft_revision_mismatches": draft_revision_mismatches,
        "engine_q_mismatches": engine_q_mismatches,
        "contract_matches": contract_matches,
        "residual_path_exercised": rejected_with_correction > 0,
    }


def compare_token_position_distributions(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
    *,
    max_positions: int = 16,
    permutations: int = 2000,
    alpha: float = 0.01,
    min_samples_per_arm: int = 128,
    seed: int = 0,
) -> dict[str, Any]:
    """Permutation-test unconditional token marginals at generated positions.

    The test is deliberately described as ``no_detected_divergence`` rather
    than proof of equality.  Non-rejection cannot turn this diagnostic green:
    it lacks a pre-registered equivalence bound/power calibration, and position
    marginals do not test the joint sequence law.  Missing positions use a stop
    sentinel, so early termination is part of the sampled output law.
    Before sampling permutations, the diagnostic also verifies that its
    smallest attainable Monte Carlo p-value can cross the Bonferroni-adjusted
    threshold.  An under-resolved test is reported as inconclusive rather than
    being allowed to manufacture a permanent non-rejection.
    """

    left_rows = [tuple(int(token) for token in row) for row in left]
    right_rows = [tuple(int(token) for token in row) for row in right]
    if len(left_rows) < min_samples_per_arm or len(right_rows) < min_samples_per_arm:
        return {
            "status": "deferred_insufficient_samples",
            "release_gate_pass": False,
            "inference": "inconclusive_insufficient_samples",
            "equivalence_bound_available": False,
            "power_calibration_available": False,
            "joint_sequence_law_tested": False,
            "left_samples": len(left_rows),
            "right_samples": len(right_rows),
            "min_samples_per_arm": int(min_samples_per_arm),
            "alpha_familywise": float(alpha),
            "permutations": int(permutations),
            "positions": [],
        }
    if max_positions < 1:
        raise ValueError("max_positions must be >= 1")
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    observed_positions = min(
        int(max_positions),
        max((len(row) for row in [*left_rows, *right_rows]), default=0),
    )
    if observed_positions == 0:
        return {
            "status": "deferred_no_generated_tokens",
            "release_gate_pass": False,
            "inference": "inconclusive_no_generated_tokens",
            "equivalence_bound_available": False,
            "power_calibration_available": False,
            "joint_sequence_law_tested": False,
            "left_samples": len(left_rows),
            "right_samples": len(right_rows),
            "min_samples_per_arm": int(min_samples_per_arm),
            "alpha_familywise": float(alpha),
            "permutations": int(permutations),
            "positions": [],
        }

    per_position_alpha = float(alpha) / observed_positions
    min_permutation_p = 1.0 / (permutations + 1.0)
    if min_permutation_p >= per_position_alpha:
        return {
            "status": "insufficient_permutation_resolution",
            "release_gate_pass": False,
            "inference": "inconclusive_insufficient_permutation_resolution",
            "equivalence_bound_available": False,
            "power_calibration_available": False,
            "joint_sequence_law_tested": False,
            "left_samples": len(left_rows),
            "right_samples": len(right_rows),
            "min_samples_per_arm": int(min_samples_per_arm),
            "alpha_familywise": float(alpha),
            "bonferroni_alpha": per_position_alpha,
            "permutations": int(permutations),
            "min_permutation_p": min_permutation_p,
            "observed_positions": observed_positions,
            "method": "position_marginal_total_variation_permutation_bonferroni",
            "positions": [],
        }

    sentinel = object()
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    for position in range(observed_positions):
        left_values = [
            row[position] if position < len(row) else sentinel for row in left_rows
        ]
        right_values = [
            row[position] if position < len(row) else sentinel for row in right_rows
        ]
        observed_tv = _empirical_total_variation(left_values, right_values)
        pooled = [*left_values, *right_values]
        left_size = len(left_values)
        exceedances = 0
        for _ in range(permutations):
            rng.shuffle(pooled)
            permuted_tv = _empirical_total_variation(
                pooled[:left_size], pooled[left_size:]
            )
            if permuted_tv >= observed_tv - 1e-15:
                exceedances += 1
        p_value = (exceedances + 1.0) / (permutations + 1.0)
        results.append(
            {
                "position": position,
                "empirical_total_variation": observed_tv,
                "permutation_p_value": p_value,
                "bonferroni_alpha": per_position_alpha,
                "divergence_detected": p_value < per_position_alpha,
            }
        )

    divergence = any(row["divergence_detected"] for row in results)
    return {
        "status": ("divergence_detected" if divergence else "no_detected_divergence"),
        "release_gate_pass": False,
        "inference": (
            "divergence_detected" if divergence else "inconclusive_non_rejection"
        ),
        "equivalence_bound_available": False,
        "power_calibration_available": False,
        "joint_sequence_law_tested": False,
        "required_for_green": (
            "pre-register and pass an equivalence tolerance or synthetic-falsifier "
            "power target, plus a joint-sequence-law test"
        ),
        "left_samples": len(left_rows),
        "right_samples": len(right_rows),
        "min_samples_per_arm": int(min_samples_per_arm),
        "alpha_familywise": float(alpha),
        "permutations": int(permutations),
        "method": "position_marginal_total_variation_permutation_bonferroni",
        "positions": results,
    }


def _empirical_total_variation(left: Sequence[Any], right: Sequence[Any]) -> float:
    left_counts = Counter(left)
    right_counts = Counter(right)
    left_total = float(len(left))
    right_total = float(len(right))
    if left_total <= 0 or right_total <= 0:
        raise ValueError("distribution comparison requires two non-empty samples")
    support = set(left_counts) | set(right_counts)
    return 0.5 * sum(
        abs(left_counts[value] / left_total - right_counts[value] / right_total)
        for value in support
    )


def protocol_mismatches(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return fields that make two arms protocol-incomparable.

    Backend, model/draft identity, block size, generation mode, and NAX are arm
    dimensions.  The fields below are the shared protocol and must match.
    """

    mismatches: dict[str, tuple[Any, Any]] = {}
    for field in _COMPARISON_FIELDS:
        if left.get(field) != right.get(field):
            mismatches[field] = (left.get(field), right.get(field))
    return mismatches


def assert_protocol_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> None:
    mismatches = protocol_mismatches(left, right)
    if not mismatches:
        return
    details = ", ".join(
        f"{field}={before!r} != {after!r}"
        for field, (before, after) in mismatches.items()
    )
    raise ValueError(f"benchmark protocol mismatch: {details}")


def _required_revision(value: str | None, field: str) -> str:
    revision = str(value or "").strip()
    if not revision:
        raise ValueError(f"{field} must be pinned for an evidence-grade run")
    return revision
