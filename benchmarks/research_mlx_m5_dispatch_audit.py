#!/usr/bin/env python3
"""Generate the CPU-only static MLX M5 dispatch matrix.

This audit intentionally does not import MLX, load a model, initialize Metal,
or collect environment variables.  It turns the pinned source predicates and
model-config snapshots recorded in the staged v1 receipt into a materialized,
canonical matrix that another reviewer can inspect and hash.

The previous receipt recorded a 288-row digest but not its matrix or the
canonicalization rule.  Therefore this script preserves the 4 targets x 8
source predicates x 9 verify-width envelope while honestly treating its
canonical SHA256 as a v2 replacement rather than claiming to reproduce the
legacy digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


SCHEMA = "mtplx-m5-static-dispatch-audit-v5"
VERIFY_WIDTHS = tuple(range(8, 17))
MLX_LM_SWITCH_GLU_REVISION = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
MLX_LM_SWITCH_GLU_SORT_THRESHOLD = 64
MLX_LM_SWITCH_GLU_SOURCE = {
    "revision": MLX_LM_SWITCH_GLU_REVISION,
    "router": "mlx_lm/models/qwen3_next.py:308-344",
    "router_file_sha256": (
        "3c572fe3fbb36721efab4d80d1bb6af11beb4ad1caae18deefc9fc84cbcd9b79"
    ),
    "switch_glu": "mlx_lm/models/switch_layers.py:176-199",
    "switch_glu_file_sha256": (
        "073a6a808d5c90bb699a2ecca0e559b06727ae96dbc1f0253e4c7e77e4ee1ef2"
    ),
    "geometry": (
        "qwen3_next selects top_k experts per token; SwitchGLU sorts when "
        "indices.size >= 64, expands one token row per routed assignment, and "
        "calls GatherQMM with inner M=1, B=verify_rows*top_k"
    ),
}
SOURCE_ORDER = (
    "mlx_pr_3838",
    "mlx_pr_3842",
    "mlx_pr_4020",
    "mlx_pr_4023",
    "mlx_pr_4077",
    "mlx_pr_4171",
    "mlx_lm_pr_1559",
    "mtplx_private_nax",
)

PINNED_REVISIONS = {
    "mlx_pr_3838": "3df41a6ade727ffd1c84336fd556146a75a4737b",
    "mlx_pr_3842": "31f1555a80d529a0ab27441733293b37ada1223b",
    "mlx_pr_4020": "4ad8724334f942dc95bdc0c962fb7bda9b59b77c",
    "mlx_pr_4023": "dae83dff6f02b657d78e5557216e9f9d47004fb5",
    "mlx_pr_4077": "c55982c0972c8ecee6f8bfcc5e71d933c662be5f",
    "mlx_pr_4171": "46c9b8593fbefcbf8e789710655ce71a5ff50d8d",
    "mlx_lm_pr_1559": "e1cc9c51442aef48ab206d23400720a1e2022448",
    "mtplx_private_nax": "5262443fd4b4b04faab5690346d1225ce6ee5b0e",
}

SOURCE_LOGIC = {
    "mlx_pr_3838": {
        "availability": "closed_unmerged_historical_code_only",
        "logic": (
            "8 < verify_rows <= 16; supported D in {64,96,128,256}; "
            "GQA * ceil(verify_rows / (4 if D <= 128 else 2)) <= 32; "
            "verify_rows <= K; K >= 1024 is required only where the ordinary "
            "full kernel supports the head dimension"
        ),
    },
    "mlx_pr_3842": {
        "availability": "pinned_candidate_source",
        "logic": (
            "full-attention prefill only: D=256, causal, no array mask, "
            "NAX available, non-float32-or-TF32, and query_rows >= 1024"
        ),
    },
    "mlx_pr_4020": {
        "availability": "pinned_candidate_source",
        "logic": (
            "Gated DeltaNet primitive is a prefill alternative; this matrix "
            "audits verify rows, so no verify-row dispatch is claimed"
        ),
    },
    "mlx_pr_4023": {
        "availability": "merged_source_with_runtime_conditions",
        "logic": (
            "GatherQMM sorted-RHS route requires generic routed MoE, M == 1, "
            "B >= 16, B / E >= 4, right_sorted, and NAX. MLX-LM SwitchGLU "
            "flattens verify_rows*top_k routed assignments, so GatherQMM sees "
            "M=1 and B=verify_rows*top_k; it sorts when B >= 64. Qwen35 and "
            "Laguna verify rows 8..16 satisfy M/B/sorting but fail B/E >= 4"
        ),
    },
    "mlx_pr_4077": {
        "availability": "pinned_candidate_source",
        "logic": "decode only: verify_rows=1, GQA=8, and D in {64,128}",
    },
    "mlx_pr_4171": {
        "availability": "pinned_candidate_source",
        "logic": (
            "transposed affine qmm_t_nax supports 2/3/4/5/6/8-bit modes; "
            "the recorded applegpu_g17s large-projection boundary is "
            "verify_rows >= 13"
        ),
    },
    "mlx_lm_pr_1559": {
        "availability": "pinned_candidate_source",
        "logic": (
            "packed Gated DeltaNet is a prefill candidate; it is not a "
            "verify-row dispatch"
        ),
    },
    "mtplx_private_nax": {
        "availability": "private_source_not_generic_mlx_lm",
        "logic": (
            "installed MTPLX QuantizedLinear patch: generic 4-bit calls with "
            "4 <= M <= 16 and phase != prefill. M=4 and M=5..6 can take "
            "earlier m4/m6 lanes; the m16 NAX tile is otherwise eligible for "
            "1 <= M <= 16 when group size is 32/64/128, dtype is bf16/fp16, "
            "K % 256 == 0, N % 32 == 0, NAX is available, and the patch and "
            "m16 lane are enabled"
        ),
    },
}


def canonical_json(value: Any) -> str:
    """Serialize JSON in the sole format used by the matrix digest."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def no_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=no_duplicate_object_pairs)
    if not isinstance(value, dict):
        raise ValueError("reference receipt must contain a JSON object")
    return value


def public_path(path: Path, repository_root: Path) -> str:
    """Return a stable repository-relative identifier without leaking paths."""

    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return "external_input"


def quantization_bits(snapshot: dict[str, Any]) -> int | None:
    quantization = snapshot.get("quantization")
    if not isinstance(quantization, str):
        return None
    prefix = quantization.split("-", maxsplit=1)[0]
    return int(prefix) if prefix.isdigit() else None


def quantization_group_size(snapshot: dict[str, Any]) -> int | None:
    """Extract the declared affine quantization group size from a snapshot."""

    quantization = snapshot.get("quantization")
    if not isinstance(quantization, str):
        return None
    marker = "group-"
    _, separator, suffix = quantization.rpartition(marker)
    if separator and suffix.isdigit():
        return int(suffix)
    return None


def has_gdn(snapshot: dict[str, Any]) -> bool:
    return bool(snapshot.get("gdn_layers", 0))


def is_generic_mlx_lm_target(target: str) -> bool:
    return target != "deepseek_v4_flash_0731"


def source_3838(target: str, snapshot: dict[str, Any], rows: int) -> dict[str, Any]:
    head_dim = int(snapshot["head_dim"])
    gqa = int(snapshot["gqa"])
    max_nq = 4 if head_dim <= 128 else 2
    full_kernel_supports_head_dim = head_dim in {64, 80, 128}
    static_terms = {
        "head_dim_supported": head_dim in {64, 96, 128, 256},
        "row_band_9_through_16": 8 < rows <= 16,
        "threadgroup_simdgroups_at_most_32": gqa * math.ceil(rows / max_nq) <= 32,
    }
    terms = {
        **static_terms,
        # The upstream route accepts both mask and causal forms. The matrix has
        # no key-length coordinate, however: every call requires Q <= K and a
        # standard full-kernel head dimension needs K >= 1024 for this branch.
        "mask_and_causal_mode_do_not_gate_multirow_route": True,
        "key_rows_at_least_query_rows": "unrepresented_runtime_condition",
        "long_key_rows_at_least_1024": (
            "unrepresented_runtime_condition"
            if full_kernel_supports_head_dim
            else "not_required_for_this_head_dim"
        ),
    }
    static_preconditions = all(static_terms.values())
    return cell_result(
        target,
        "mlx_pr_3838",
        rows,
        static_preconditions,
        terms,
        (
            "static source terms pass; runtime key-length relation remains required"
            if static_preconditions
            else "one or more source predicate terms fail"
        ),
        conditional=static_preconditions,
    )


def source_3842(target: str, snapshot: dict[str, Any], rows: int) -> dict[str, Any]:
    terms = {
        "head_dim_256": int(snapshot["head_dim"]) == 256,
        "named_shape_is_full_attention_prefill": False,
        "query_rows_at_least_1024": rows >= 1024,
        "runtime_nax_and_call_contract_unverified": False,
    }
    return cell_result(
        target,
        "mlx_pr_3842",
        rows,
        False,
        terms,
        "this verify-width matrix cannot establish the prefill-only dispatch",
    )


def source_4020(target: str, snapshot: dict[str, Any], rows: int) -> dict[str, Any]:
    terms = {
        "target_has_gated_deltanet": has_gdn(snapshot),
        "named_shape_is_prefill": False,
        "runtime_dtype_and_mask_contract_unverified": False,
    }
    return cell_result(
        target,
        "mlx_pr_4020",
        rows,
        False,
        terms,
        "a prefill primitive is not a verify-row dispatch claim",
    )


def source_4023(
    target: str,
    snapshot: dict[str, Any],
    rows: int,
    *,
    nax_available: bool | None = None,
) -> dict[str, Any]:
    """Model the full sorted-RHS GatherQMM predicate.

    ``rows`` is the target-verify token-row count, not GatherQMM's inner M.
    At the pinned MLX-LM revision, Qwen3Next selects ``top_k`` experts per
    token and SwitchGLU flattens those routed assignments before GatherQMM.
    Each gathered expert matrix therefore receives M=1 while B is the total
    routed-assignment count.  NAX availability remains a runtime condition.
    """

    expert_count = int(snapshot.get("experts", 0))
    top_k = int(snapshot.get("top_k", 0))
    routed_assignments = rows * top_k if top_k > 0 else 0
    gatherqmm_inner_rows_m = 1 if expert_count > 0 and top_k > 0 else None
    right_sorted = routed_assignments >= MLX_LM_SWITCH_GLU_SORT_THRESHOLD
    terms = {
        "generic_mlx_lm_moe_path": is_generic_mlx_lm_target(target),
        "target_has_routed_experts": expert_count > 0,
        "target_top_k_exposed": top_k > 0,
        "target_verify_token_rows": rows,
        "target_top_k": top_k,
        "routed_assignment_count": routed_assignments,
        "switchglu_sort_threshold_assignments": MLX_LM_SWITCH_GLU_SORT_THRESHOLD,
        "switchglu_sorts_routed_assignments": right_sorted,
        "gatherqmm_inner_rows_m": gatherqmm_inner_rows_m,
        "gatherqmm_inner_rows_m_is_one": gatherqmm_inner_rows_m == 1,
        "batch_rows_b": routed_assignments,
        "batch_rows_b_at_least_16": routed_assignments >= 16,
        "expert_count_e": expert_count,
        "batch_rows_per_expert_b_over_e_at_least_4": (
            expert_count > 0 and routed_assignments / expert_count >= 4
        ),
        "right_sorted": right_sorted,
        "nax_available": (
            nax_available
            if nax_available is not None
            else "unrepresented_runtime_condition"
        ),
    }
    static_preconditions = all(
        terms[key]
        for key in (
            "generic_mlx_lm_moe_path",
            "target_has_routed_experts",
            "target_top_k_exposed",
            "gatherqmm_inner_rows_m_is_one",
            "batch_rows_b_at_least_16",
            "batch_rows_per_expert_b_over_e_at_least_4",
            "right_sorted",
        )
    )
    eligible = static_preconditions and nax_available is True
    conditional = static_preconditions and nax_available is None
    return cell_result(
        target,
        "mlx_pr_4023",
        rows,
        eligible,
        terms,
        (
            "all sorted-RHS GatherQMM predicate terms hold"
            if eligible
            else (
                "static sorted-RHS terms hold; NAX availability is unrepresented"
                if conditional
                else "one or more sorted-RHS GatherQMM predicate terms fail"
            )
        ),
        conditional=conditional,
    )


def source_4077(target: str, snapshot: dict[str, Any], rows: int) -> dict[str, Any]:
    terms = {
        "decode_single_row": rows == 1,
        "gqa_factor_8": int(snapshot["gqa"]) == 8,
        "head_dim_64_or_128": int(snapshot["head_dim"]) in {64, 128},
    }
    return cell_result(
        target,
        "mlx_pr_4077",
        rows,
        all(terms.values()),
        terms,
        "verify rows 8 through 16 are not decode single-row calls",
    )


def source_4171(target: str, snapshot: dict[str, Any], rows: int) -> dict[str, Any]:
    bits = quantization_bits(snapshot)
    terms = {
        "generic_mlx_qmm_path": is_generic_mlx_lm_target(target),
        "quantization_declares_affine_group_size": quantization_group_size(snapshot)
        is not None,
        "affine_bits_supported_by_transposed_qmm_t_nax": bits in {2, 3, 4, 5, 6, 8},
        "m5_large_projection_rows_at_least_13": rows >= 13,
        "runtime_call_is_transposed_qmm_unverified": False,
    }
    static_preconditions = all(
        terms[key]
        for key in (
            "generic_mlx_qmm_path",
            "quantization_declares_affine_group_size",
            "affine_bits_supported_by_transposed_qmm_t_nax",
            "m5_large_projection_rows_at_least_13",
        )
    )
    return cell_result(
        target,
        "mlx_pr_4171",
        rows,
        static_preconditions,
        terms,
        (
            "static row and supported-affine-quantization preconditions hold; actual QMM call remains required"
            if static_preconditions
            else "one or more static QMM preconditions fail"
        ),
        conditional=static_preconditions,
    )


def source_1559(target: str, snapshot: dict[str, Any], rows: int) -> dict[str, Any]:
    terms = {
        "target_has_gated_deltanet": has_gdn(snapshot),
        "named_shape_is_prefill": False,
        "packed_cache_lifecycle_unverified": False,
    }
    return cell_result(
        target,
        "mlx_lm_pr_1559",
        rows,
        False,
        terms,
        "a packed prefill proposal is not a verify-row dispatch claim",
    )


def private_nax_precedence(rows: int) -> str:
    """Describe the actual 4-bit branch order before the m16 NAX lane."""

    if rows == 4:
        return "qmm_m4_precedes_m16_when_enabled_and_eligible"
    if 5 <= rows <= 6:
        return "qmm_m6_precedes_m16_when_enabled_and_eligible"
    if 7 <= rows <= 16:
        return "m16_not_shadowed_by_earlier_4bit_lanes"
    return "outside_installed_4bit_verify_band"


def source_private_nax(
    target: str, snapshot: dict[str, Any], rows: int
) -> dict[str, Any]:
    group_size = quantization_group_size(snapshot)
    precedence = private_nax_precedence(rows)
    static_terms = {
        "target_uses_generic_mlx_quantizedlinear_path": is_generic_mlx_lm_target(
            target
        ),
        "quantization_is_4_bit": quantization_bits(snapshot) == 4,
        "group_size_is_32_64_or_128": group_size in {32, 64, 128},
        "verify_rows_in_installed_4bit_band": 4 <= rows <= 16,
        "earlier_lane_precedence": precedence,
    }
    terms: dict[str, Any] = {
        **static_terms,
        "activation_dtype_is_bf16_or_fp16": "unrepresented_runtime_condition",
        "input_k_is_multiple_of_256": "unrepresented_runtime_condition",
        "output_n_is_multiple_of_32": "unrepresented_runtime_condition",
        "mtplx_nax_patch_is_installed": "unrepresented_runtime_condition",
        "attention_phase_is_not_prefill": "unrepresented_runtime_condition",
        "nax_hardware_and_os_are_available": "unrepresented_runtime_condition",
        "m16_lane_not_disabled_by_selfcheck": "unrepresented_runtime_condition",
    }
    static_preconditions = (
        all(
            value is True
            for key, value in static_terms.items()
            if key != "earlier_lane_precedence"
        )
        and precedence == "m16_not_shadowed_by_earlier_4bit_lanes"
    )
    return cell_result(
        target,
        "mtplx_private_nax",
        rows,
        static_preconditions,
        terms,
        (
            "static m16 NAX terms pass; runtime QuantizedLinear call conditions remain"
            if static_preconditions
            else "one or more static m16 NAX terms fail"
        ),
        conditional=static_preconditions,
    )


def cell_result(
    target: str,
    source: str,
    rows: int,
    eligible: bool,
    terms: dict[str, Any],
    reason: str,
    *,
    conditional: bool = False,
) -> dict[str, Any]:
    return {
        "candidate": source,
        "eligibility": "conditional"
        if conditional
        else "true"
        if eligible
        else "false",
        "predicate_terms": terms,
        "reason": reason,
        "source_availability": SOURCE_LOGIC[source]["availability"],
        "target": target,
        "verify_rows": rows,
    }


GENERATORS: dict[str, Callable[[str, dict[str, Any], int], dict[str, Any]]] = {
    "mlx_pr_3838": source_3838,
    "mlx_pr_3842": source_3842,
    "mlx_pr_4020": source_4020,
    "mlx_pr_4023": source_4023,
    "mlx_pr_4077": source_4077,
    "mlx_pr_4171": source_4171,
    "mlx_lm_pr_1559": source_1559,
    "mtplx_private_nax": source_private_nax,
}


def build_matrix(snapshots: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for target in sorted(snapshots):
        snapshot = snapshots[target]
        for source in SOURCE_ORDER:
            generator = GENERATORS[source]
            matrix.extend(generator(target, snapshot, rows) for rows in VERIFY_WIDTHS)
    return matrix


def find_cell(
    matrix: list[dict[str, Any]], target: str, source: str, rows: int
) -> dict[str, Any]:
    matches = [
        cell
        for cell in matrix
        if cell["target"] == target
        and cell["candidate"] == source
        and cell["verify_rows"] == rows
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one matrix cell, found {len(matches)}")
    return matches[0]


def run_self_checks(
    reference: dict[str, Any], matrix: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run independent contract checks; each assertion can fail."""

    results: list[dict[str, Any]] = []

    def check(name: str, condition: bool) -> None:
        results.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    method = reference.get("method", {})
    sources = reference.get("sources", {})
    snapshots = reference.get("target_snapshots", {})
    expected_rows = len(snapshots) * len(SOURCE_ORDER) * len(VERIFY_WIDTHS)

    check(
        "reference_schema_is_v1",
        reference.get("schema") == "mtplx-m5-static-dispatch-audit-v1",
    )
    check("reference_claimed_288_rows", method.get("rows_evaluated") == 288)
    check("reference_has_all_pinned_sources", set(sources) == set(SOURCE_ORDER))
    check(
        "reference_revisions_match_pinned_predicates",
        all(
            sources[key].get("revision") == PINNED_REVISIONS[key]
            for key in SOURCE_ORDER
        ),
    )
    check("matrix_has_expected_288_coordinates", len(matrix) == expected_rows == 288)
    coordinates = {
        (cell["target"], cell["candidate"], cell["verify_rows"]) for cell in matrix
    }
    check("matrix_coordinates_are_unique", len(coordinates) == len(matrix))
    check(
        "qwen27_3838_m9_m10_require_runtime_key_relation",
        [
            find_cell(matrix, "qwen36_27b_8bit", "mlx_pr_3838", rows)["eligibility"]
            for rows in VERIFY_WIDTHS
        ]
        == [
            "false",
            "conditional",
            "conditional",
            "false",
            "false",
            "false",
            "false",
            "false",
            "false",
        ],
    )
    check(
        "qwen35_3838_never_eligible",
        all(
            find_cell(matrix, "qwen36_35b_a3b_4bit", "mlx_pr_3838", rows)["eligibility"]
            == "false"
            for rows in VERIFY_WIDTHS
        ),
    )
    check(
        "laguna_3838_m9_through_m16_require_long_key_context",
        [
            find_cell(matrix, "laguna_s21_4bit", "mlx_pr_3838", rows)["eligibility"]
            for rows in VERIFY_WIDTHS
        ]
        == [
            "false",
            "conditional",
            "conditional",
            "conditional",
            "conditional",
            "conditional",
            "conditional",
            "conditional",
            "conditional",
        ],
    )
    check(
        "qwen35_qmm_boundary",
        find_cell(matrix, "qwen36_35b_a3b_4bit", "mlx_pr_4171", 12)["eligibility"]
        == "false"
        and find_cell(matrix, "qwen36_35b_a3b_4bit", "mlx_pr_4171", 13)["eligibility"]
        == "conditional",
    )
    check(
        "qwen27_8bit_qmm_t_nax_supports_the_pinned_affine_mode_family",
        find_cell(matrix, "qwen36_27b_8bit", "mlx_pr_4171", 12)["eligibility"]
        == "false"
        and find_cell(matrix, "qwen36_27b_8bit", "mlx_pr_4171", 13)["eligibility"]
        == "conditional",
    )
    check(
        "qwen35_and_laguna_verify_geometry_is_sorted_but_below_density_gate",
        all(
            (
                (cell := find_cell(matrix, target, "mlx_pr_4023", rows))["eligibility"]
                == "false"
                and cell["predicate_terms"]["gatherqmm_inner_rows_m"] == 1
                and cell["predicate_terms"]["batch_rows_b"]
                == rows * int(snapshots[target]["top_k"])
                and cell["predicate_terms"]["right_sorted"] is True
                and cell["predicate_terms"]["batch_rows_per_expert_b_over_e_at_least_4"]
                is False
            )
            for target in ("qwen36_35b_a3b_4bit", "laguna_s21_4bit")
            for rows in VERIFY_WIDTHS
        ),
    )
    check(
        "private_nax_general_4bit_m8_through_m16_is_conditional",
        all(
            find_cell(matrix, "qwen36_35b_a3b_4bit", "mtplx_private_nax", rows)[
                "eligibility"
            ]
            == "conditional"
            for rows in VERIFY_WIDTHS
        ),
    )
    check(
        "private_nax_is_not_qwen_only",
        all(
            find_cell(matrix, "laguna_s21_4bit", "mtplx_private_nax", rows)[
                "eligibility"
            ]
            == "conditional"
            for rows in VERIFY_WIDTHS
        ),
    )
    check(
        "private_nax_8bit_is_excluded",
        all(
            find_cell(matrix, "qwen36_27b_8bit", "mtplx_private_nax", rows)[
                "eligibility"
            ]
            == "false"
            for rows in VERIFY_WIDTHS
        ),
    )
    check(
        "private_nax_precedence_is_not_m16_only",
        private_nax_precedence(4) == "qmm_m4_precedes_m16_when_enabled_and_eligible"
        and private_nax_precedence(5) == "qmm_m6_precedes_m16_when_enabled_and_eligible"
        and private_nax_precedence(6) == "qmm_m6_precedes_m16_when_enabled_and_eligible"
        and private_nax_precedence(7) == "m16_not_shadowed_by_earlier_4bit_lanes"
        and private_nax_precedence(16) == "m16_not_shadowed_by_earlier_4bit_lanes",
    )
    if len(results) != 16:
        raise AssertionError(f"expected sixteen self-checks, found {len(results)}")
    return results


def read_reference(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    reference = load_json(path)
    snapshots = reference.get("target_snapshots")
    if not isinstance(snapshots, dict):
        raise ValueError("reference receipt lacks target_snapshots")
    normalized: dict[str, dict[str, Any]] = {}
    for target, snapshot in snapshots.items():
        if not isinstance(target, str) or not isinstance(snapshot, dict):
            raise ValueError("target snapshots must map names to objects")
        required = {"heads", "kv_heads", "head_dim", "gqa"}
        missing = sorted(required - set(snapshot))
        if missing:
            raise ValueError(f"snapshot {target!r} lacks: {', '.join(missing)}")
        if int(snapshot["heads"]) // int(snapshot["kv_heads"]) != int(snapshot["gqa"]):
            raise ValueError(f"snapshot {target!r} has inconsistent GQA fields")
        if int(snapshot.get("experts", 0)) > 0 and int(snapshot.get("top_k", 0)) <= 0:
            raise ValueError(f"routed-MoE snapshot {target!r} lacks a positive top_k")
        normalized[target] = snapshot
    return reference, normalized


def build_report(reference_path: Path, repository_root: Path) -> dict[str, Any]:
    reference, snapshots = read_reference(reference_path)
    matrix = build_matrix(snapshots)
    tests = run_self_checks(reference, matrix)
    legacy_method = reference["method"]
    legacy_hash = legacy_method.get("deterministic_full_matrix_sha256")
    matrix_hash = digest_json(matrix)
    script_path = Path(__file__).resolve()
    return {
        "canonical_matrix_sha256": matrix_hash,
        "canonicalization": {
            "encoding": "utf-8",
            "json": "sorted object keys, compact separators, ASCII escaping",
            "hashed_value": "matrix",
        },
        "generator": {
            "model_loading": False,
            "metal_calls": False,
            "path": public_path(script_path, repository_root),
            "script_sha256": digest_file(script_path),
        },
        "legacy_comparison": {
            "changed_cell_count": None,
            "changed_cell_count_status": (
                "not_computable: v1 published only a digest, not the matrix or its canonicalization"
            ),
            "legacy_canonical_sha256": legacy_hash,
            "legacy_claimed_rows": legacy_method.get("rows_evaluated"),
            "legacy_claimed_tests": legacy_method.get("unit_tests_passed"),
            "legacy_full_matrix_present": False,
            "new_matrix_rows": len(matrix),
            "new_matrix_tests": len(tests),
            "row_count_preserved": legacy_method.get("rows_evaluated") == len(matrix),
            "sha256_reproduced": False,
            "sha256_status": "corrected transparent v5 canonicalization; legacy digest cannot be reproduced",
        },
        "matrix": matrix,
        "method": {
            "matrix_dimensions": {
                "sources": len(SOURCE_ORDER),
                "targets": len(snapshots),
                "verify_widths": list(VERIFY_WIDTHS),
            },
            "model_loading": False,
            "metal_calls": False,
            "qualification": (
                "true means all source terms represented by the matrix pass; conditional "
                "means static terms pass but a required call-shape or runtime condition is "
                "unrepresented; false means at least one static term fails"
            ),
        },
        "schema": SCHEMA,
        "consumer_call_geometry_source": MLX_LM_SWITCH_GLU_SOURCE,
        "source_predicates": {
            source: {
                "availability": SOURCE_LOGIC[source]["availability"],
                "logic": SOURCE_LOGIC[source]["logic"],
                "predicate_location": reference["sources"][source].get("predicate"),
                "revision": reference["sources"][source]["revision"],
            }
            for source in SOURCE_ORDER
        },
        "source_receipt": {
            "path": public_path(reference_path, repository_root),
            "sha256": digest_file(reference_path),
        },
        "target_snapshots": snapshots,
        "tests": tests,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=repository_root
        / "benchmarks/results/mlx-m5-research-20260810/dispatch_truth_table.json",
        help="v1 staged dispatch receipt to audit (default: repository receipt)",
    )
    parser.add_argument(
        "--json-out", type=Path, required=True, help="output JSON report"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repository_root = Path(__file__).resolve().parents[1]
    report = build_report(args.reference, repository_root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "canonical_matrix_sha256": report["canonical_matrix_sha256"],
                "row_count": len(report["matrix"]),
                "test_count": len(report["tests"]),
                "legacy_discrepancy": report["legacy_comparison"][
                    "changed_cell_count_status"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
