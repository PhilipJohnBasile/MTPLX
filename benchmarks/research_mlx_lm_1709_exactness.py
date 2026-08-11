#!/usr/bin/env python3
"""CPU-only, public-safe reproducer for MLX-LM PR #1709 integration claims.

This file is intentionally self-contained.  It does not import MLX, a model,
or a checkout of MLX-LM, so it neither exercises a GPU nor claims live
integration coverage.  Instead, the negative cases copy the narrow semantics
visible at the immutable PR head named below and the positive case is an
independent rational-arithmetic oracle for the narrow, valid K=1 contract.

Run::

    python benchmarks/research_mlx_lm_1709_exactness.py \
      --json-out /tmp/mlx-lm-1709-exactness.json

The process exits nonzero when a claimed failure does not reproduce or when
the positive oracle fails.  JSON records the evidence mode for every case.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import struct
from typing import Callable, Sequence


PR_HEAD = "a318a32c44c1fe7666160122d26e11b319248fd3"
PR_URL = f"https://github.com/ml-explore/mlx-lm/commit/{PR_HEAD}"
EVIDENCE_COPIED = "copied_pr_semantics_not_imported"
EVIDENCE_ORACLE = "independent_rational_oracle_not_pr_code"
VOCABULARY = (0, 1)


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _bfloat16(value: float) -> float:
    """Round a Python float to IEEE bfloat16, returned as a float32 value."""
    bits = struct.unpack("!I", struct.pack("!f", _float32(value)))[0]
    upper, lower = bits >> 16, bits & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    return struct.unpack("!f", struct.pack("!I", upper << 16))[0]


def _float16(value: float) -> float:
    return struct.unpack("!e", struct.pack("!e", value))[0]


def _softmax(logits: Sequence[float]) -> list[float]:
    maximum = max(logits)
    terms = [math.exp(value - maximum) for value in logits]
    normalizer = sum(terms)
    return [term / normalizer for term in terms]


def _total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right, strict=True))


def _positive_part_residual(
    target: Sequence[Fraction], proposal: Sequence[Fraction]
) -> tuple[Fraction, ...]:
    raw = tuple(max(p - q, Fraction(0)) for p, q in zip(target, proposal, strict=True))
    total = sum(raw, Fraction(0))
    if not total:
        raise ValueError("the valid-contract fixture requires nonzero residual mass")
    return tuple(value / total for value in raw)


def _add(
    law: dict[tuple[int, ...], Fraction], key: tuple[int, ...], mass: Fraction
) -> None:
    if mass:
        law[key] = law.get(key, Fraction(0)) + mass


def _direct_target_law(
    target_rows: Sequence[Sequence[Fraction]], length: int
) -> dict[tuple[int, ...], Fraction]:
    law: dict[tuple[int, ...], Fraction] = {(): Fraction(1)}
    for position in range(length):
        next_law: dict[tuple[int, ...], Fraction] = {}
        for prefix, prefix_mass in law.items():
            for token, probability in enumerate(target_rows[position]):
                _add(next_law, prefix + (token,), prefix_mass * probability)
        law = next_law
    return law


def _k1_speculative_law(
    target_rows: Sequence[Sequence[Fraction]],
    proposal_rows: Sequence[Sequence[Fraction]],
    *,
    rule: str,
    length: int,
) -> dict[tuple[int, ...], Fraction]:
    """Enumerate the K=1 residual/block law through a fixed emitted horizon.

    This is intentionally independent of PR code.  At K=1 the PR's block
    threshold is ``min(p(x)/q(x), 1)`` and its scaled residual is the ordinary
    residual, so both valid rules reduce to this transition.  A full accepted
    K=1 round emits the target bonus token; a rejection begins the next round.
    """
    if rule not in {"residual", "block_k1"}:
        raise ValueError(f"unsupported oracle rule: {rule}")

    law: dict[tuple[int, ...], Fraction] = {}

    def emit_cycle(position: int, prefix: tuple[int, ...], mass: Fraction) -> None:
        if len(prefix) == length:
            _add(law, prefix, mass)
            return
        target, proposal = target_rows[position], proposal_rows[position]
        residual = _positive_part_residual(target, proposal)
        for draft_token, q in enumerate(proposal):
            if not q:
                continue
            acceptance = min(Fraction(1), target[draft_token] / q)
            accepted_prefix = prefix + (draft_token,)
            accepted_mass = mass * q * acceptance
            if len(accepted_prefix) == length:
                _add(law, accepted_prefix, accepted_mass)
            else:
                # The PR's full-accept path emits its already sampled target
                # bonus row before beginning another draft round.
                for bonus_token, p_bonus in enumerate(target_rows[position + 1]):
                    _add(law, accepted_prefix + (bonus_token,), accepted_mass * p_bonus)

            rejected_mass = mass * q * (Fraction(1) - acceptance)
            if rejected_mass:
                for correction, correction_probability in enumerate(residual):
                    emit_cycle(
                        position + 1,
                        prefix + (correction,),
                        rejected_mass * correction_probability,
                    )

    emit_cycle(0, (), Fraction(1))
    return law


def _law_max_error(
    actual: dict[tuple[int, ...], Fraction], expected: dict[tuple[int, ...], Fraction]
) -> Fraction:
    keys = set(actual) | set(expected)
    return max(
        (abs(actual.get(key, 0) - expected.get(key, 0)) for key in keys),
        default=Fraction(0),
    )


def _dtype_order_case() -> dict[str, object]:
    """Copy the two PR scaling orders without importing the PR or MLX.

    ``sample_utils.categorical_sampling`` multiplies the current dtype before
    sampling; ``generate._sampling_logprobs`` converts to float32 first.  The
    fixture is four already-bfloat16 log-probability values at temperature 0.7.
    """
    source_logits = (-5.8125, -2.234375, -2.28125, -3.9375)
    temperature = 0.7
    inverse_temperature = _float32(1.0 / _float32(temperature))

    bfloat_logits = [_bfloat16(value) for value in source_logits]
    bfloat_sampler = _softmax(
        [_bfloat16(value * inverse_temperature) for value in bfloat_logits]
    )
    bfloat_verifier = _softmax(
        [_float32(value) * inverse_temperature for value in bfloat_logits]
    )
    bfloat_tv = _total_variation(bfloat_sampler, bfloat_verifier)

    float16_logits = [_float16(value) for value in source_logits]
    float16_sampler = _softmax(
        [_float16(value * inverse_temperature) for value in float16_logits]
    )
    float16_verifier = _softmax(
        [_float32(value) * inverse_temperature for value in float16_logits]
    )
    float16_tv = _total_variation(float16_sampler, float16_verifier)

    expected_reported_bfloat_tv = 0.0026887
    passed = (
        abs(bfloat_tv - expected_reported_bfloat_tv) < 1e-7
        and bfloat_tv > 0.0
        and float16_tv > 0.0
    )
    return {
        "name": "temperature_dtype_order_mismatch",
        "evidence_mode": EVIDENCE_COPIED,
        "pr_source": [
            "mlx_lm/sample_utils.py:290-292 (scale in current dtype)",
            "mlx_lm/generate.py:475-481 (cast to float32 before scaling)",
        ],
        "fixture": {"dtype_input_values": source_logits, "temperature": temperature},
        "bfloat16": {
            "sampler_distribution": bfloat_sampler,
            "verifier_distribution": bfloat_verifier,
            "total_variation": bfloat_tv,
            "reported_total_variation": expected_reported_bfloat_tv,
        },
        "float16": {
            "sampler_distribution": float16_sampler,
            "verifier_distribution": float16_verifier,
            "total_variation": float16_tv,
        },
        "expected_failure_reproduced": passed,
        "limitation": "Emulates dtype rounding in Python; it does not execute MLX kernels or the PR.",
    }


def _history_bug_pair(
    draft_first: int, target_first: int, *, rule: str
) -> tuple[int, int]:
    """Copy the relevant row-conditioning mismatch for a two-token fixture.

    Draft row 2 is processed after the draft's first token, so it repeats that
    draft token.  PR #1709 processes target verification row 2 after a newly
    sampled target row 1, despite the target forward having already been
    computed along the draft prefix.  Both residual and K=2 block verification
    then return the pair below.
    """
    if rule not in {"residual", "block"}:
        raise ValueError(f"unsupported copied rule: {rule}")
    # First row is uniform for p and q, so the first draft is always accepted.
    # Second q is point(draft_first); second p after the processor is
    # point(target_first).  If they differ, the residual correction is exactly
    # target_first.  The PR's K=2 block thresholds make the same choice here.
    return draft_first, draft_first if draft_first == target_first else target_first


def _history_processor_case() -> dict[str, object]:
    target_law = {(0, 0): Fraction(1, 2), (1, 1): Fraction(1, 2)}
    rule_laws: dict[str, dict[tuple[int, int], Fraction]] = {}
    for rule in ("residual", "block"):
        copied_law: dict[tuple[int, int], Fraction] = {}
        for draft_first in VOCABULARY:
            for target_first in VOCABULARY:
                pair = _history_bug_pair(draft_first, target_first, rule=rule)
                copied_law[pair] = copied_law.get(pair, Fraction(0)) + Fraction(1, 4)
        rule_laws[rule] = copied_law

    # This fixed realization is explicitly only a readable realization of the
    # analytic 50% law.  It uses Python's PRNG, not MLX's PRNG; seed 256 gives
    # the audit's observed 1,034 / 2,000 = 51.7% repeat count.
    rng = random.Random(256)
    trial_pairs = [
        _history_bug_pair(
            int(rng.random() >= 0.5), int(rng.random() >= 0.5), rule="residual"
        )
        for _ in range(2000)
    ]
    repeat_count = sum(first == second for first, second in trial_pairs)
    repeat_rate = repeat_count / len(trial_pairs)

    passed = (
        all(set(law) == {(0, 0), (0, 1), (1, 0), (1, 1)} for law in rule_laws.values())
        and all(
            sum(law[pair] for pair in ((0, 0), (1, 1))) == Fraction(1, 2)
            for law in rule_laws.values()
        )
        and target_law != rule_laws["residual"]
        and repeat_count == 1034
    )
    return {
        "name": "history_dependent_processor_conditioning_mismatch",
        "evidence_mode": EVIDENCE_COPIED,
        "pr_source": [
            "mlx_lm/generate.py:695-719 (processor samples target rows in a loop)",
            "mlx_lm/generate.py:782-787 (target forward is computed along draft tokens)",
        ],
        "target_ar_joint_law": {"00": "1/2", "11": "1/2"},
        "copied_integration_joint_laws": {
            rule: {
                "".join(map(str, pair)): str(mass) for pair, mass in sorted(law.items())
            }
            for rule, law in rule_laws.items()
        },
        "analytic_repeat_probability": 0.5,
        "deterministic_2000_trial_realization": {
            "prng": "python.random.Random",
            "seed": 256,
            "repeat_count": repeat_count,
            "trial_count": len(trial_pairs),
            "repeat_rate": repeat_rate,
            "reported_audit_rate": 0.517,
        },
        "expected_failure_reproduced": passed,
        "limitation": "Copied control flow only; the 51.7% realization is not MLX PRNG or a live target forward.",
    }


def _top_one_sampler(logprobs: Sequence[float]) -> int:
    return max(range(len(logprobs)), key=logprobs.__getitem__)


_top_one_sampler.temp = 1.0  # type: ignore[attr-defined]


def _accept_rule_guard(sampler: Callable[[Sequence[float]], int]) -> bool:
    """The PR guard from generate.py, represented without importing it."""
    return getattr(sampler, "temp", None) is not None


def _forged_sampler_case() -> dict[str, object]:
    """Show why a public ``.temp`` marker cannot certify a sampling law."""
    target = (Fraction(313, 500), Fraction(187, 500))  # top-1 must emit token 0
    proposal = (Fraction(0), Fraction(1))  # draft top-1 emits token 1
    accepted_probability = target[1] / proposal[1]
    residual = _positive_part_residual(target, proposal)
    uniforms = tuple(Fraction(2 * index + 1, 4000) for index in range(2000))

    rule_counts: dict[str, int] = {}
    for rule in ("residual", "block_k1"):
        impossible = 0
        for uniform in uniforms:
            # At K=1 block's threshold equals the residual acceptance ratio.
            committed = 1 if uniform <= accepted_probability else 0
            if committed == 1:
                impossible += 1
            else:
                assert residual[0] == 1 and residual[1] == 0
        rule_counts[rule] = impossible

    target_token = _top_one_sampler([float(value) for value in target])
    passed = (
        _accept_rule_guard(_top_one_sampler)
        and target_token == 0
        and rule_counts == {"residual": 748, "block_k1": 748}
    )
    return {
        "name": "forgeable_custom_sampler_temp_marker",
        "evidence_mode": EVIDENCE_COPIED,
        "pr_source": [
            "mlx_lm/generate.py:646-658 (acceptance capability inferred from sampler.temp)",
            "mlx_lm/generate.py:750-770 (reconstructed soft distributions drive verification)",
        ],
        "sampler": {"kind": "custom_top1", "forged_temp": 1.0, "guard_accepts": True},
        "target_sampler_token": target_token,
        "proposal_token": 1,
        "soft_target_probability_of_impossible_token": str(target[1]),
        "accepted_probability": str(accepted_probability),
        "deterministic_trials": 2000,
        "impossible_target_sampler_tokens": rule_counts,
        "reported_audit_count": 748,
        "expected_failure_reproduced": passed,
        "limitation": "No model or sampler from MLX-LM is imported; this isolates the forgeable capability contract.",
    }


def _rational_oracle_case() -> dict[str, object]:
    """A positive exact joint-law check for the advertised narrow K=1 contract."""
    target_rows = (
        (Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 8)),
        (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)),
        (Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 8)),
    )
    proposal_rows = (
        (Fraction(1, 8), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2)),
        (Fraction(7, 10), Fraction(1, 10), Fraction(1, 10), Fraction(1, 10)),
    )
    expected = _direct_target_law(target_rows, length=2)
    actual = {
        rule: _k1_speculative_law(target_rows, proposal_rows, rule=rule, length=2)
        for rule in ("residual", "block_k1")
    }
    errors = {rule: _law_max_error(law, expected) for rule, law in actual.items()}
    totals = {rule: sum(law.values(), Fraction(0)) for rule, law in actual.items()}
    passed = all(error == 0 for error in errors.values()) and all(
        total == 1 for total in totals.values()
    )
    return {
        "name": "narrow_valid_contract_rational_joint_law",
        "evidence_mode": EVIDENCE_ORACLE,
        "contract": {
            "draft_depth": 1,
            "one_sequence": True,
            "float32_or_exact_probabilities": True,
            "plain_categorical_sampler": True,
            "no_logits_processors": True,
            "prefix_independent_fixture": True,
        },
        "rules_checked": ["residual", "block_k1"],
        "joint_horizon_tokens": 2,
        "law_total": {rule: str(total) for rule, total in totals.items()},
        "max_absolute_joint_probability_error": {
            rule: str(error) for rule, error in errors.items()
        },
        "positive_oracle_passed": passed,
        "limitation": "This proves only independently re-derived K=1 arithmetic on a finite fixture, not the PR integration.",
    }


def _result_passed(result: dict[str, object]) -> bool:
    return bool(
        result.get(
            "expected_failure_reproduced", result.get("positive_oracle_passed", False)
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        required=True,
        type=Path,
        help="JSON receipt path; parent must exist",
    )
    args = parser.parse_args(argv)
    if not args.json_out.parent.is_dir():
        parser.error(f"JSON output parent does not exist: {args.json_out.parent}")

    results = [
        _dtype_order_case(),
        _history_processor_case(),
        _forged_sampler_case(),
        _rational_oracle_case(),
    ]
    passed = all(_result_passed(result) for result in results)
    receipt = {
        "schema_version": 1,
        "purpose": "CPU-only reproducer for the MLX-LM #1709 audit claims",
        "pr": {"head": PR_HEAD, "url": PR_URL},
        "imports_exact_pr_code": False,
        "uses_external_models": False,
        "uses_gpu": False,
        "all_required_results_passed": passed,
        "results": results,
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.json_out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
