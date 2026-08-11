"""Focused CPU-only tests for the transparent M5 dispatch audit."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "benchmarks/research_mlx_m5_dispatch_audit.py"
)
SPEC = importlib.util.spec_from_file_location("research_mlx_m5_dispatch_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)

REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/results/mlx-m5-research-20260810/dispatch_truth_table.json"
)


def test_private_nax_uses_the_actual_generic_m8_through_m16_contract() -> None:
    _, snapshots = audit.read_reference(REFERENCE)
    matrix = audit.build_matrix(snapshots)

    for target in ("qwen36_35b_a3b_4bit", "laguna_s21_4bit"):
        for rows in audit.VERIFY_WIDTHS:
            cell = audit.find_cell(matrix, target, "mtplx_private_nax", rows)
            assert cell["eligibility"] == "conditional"
            assert cell["predicate_terms"]["group_size_is_32_64_or_128"] is True
            assert (
                cell["predicate_terms"]["earlier_lane_precedence"]
                == "m16_not_shadowed_by_earlier_4bit_lanes"
            )

    for rows in audit.VERIFY_WIDTHS:
        assert (
            audit.find_cell(matrix, "qwen36_27b_8bit", "mtplx_private_nax", rows)[
                "eligibility"
            ]
            == "false"
        )


def test_private_nax_precedence_records_the_earlier_lanes() -> None:
    assert (
        audit.private_nax_precedence(4)
        == "qmm_m4_precedes_m16_when_enabled_and_eligible"
    )
    assert (
        audit.private_nax_precedence(5)
        == "qmm_m6_precedes_m16_when_enabled_and_eligible"
    )
    assert (
        audit.private_nax_precedence(6)
        == "qmm_m6_precedes_m16_when_enabled_and_eligible"
    )
    assert audit.private_nax_precedence(7) == "m16_not_shadowed_by_earlier_4bit_lanes"
    assert audit.private_nax_precedence(16) == "m16_not_shadowed_by_earlier_4bit_lanes"
    assert audit.private_nax_precedence(17) == "outside_installed_4bit_verify_band"


def test_3838_key_length_is_conditional_only_where_the_source_requires_it() -> None:
    _, snapshots = audit.read_reference(REFERENCE)
    matrix = audit.build_matrix(snapshots)

    qwen = audit.find_cell(matrix, "qwen36_27b_8bit", "mlx_pr_3838", 9)
    assert qwen["eligibility"] == "conditional"
    assert (
        qwen["predicate_terms"]["long_key_rows_at_least_1024"]
        == "not_required_for_this_head_dim"
    )
    assert (
        qwen["predicate_terms"]["key_rows_at_least_query_rows"]
        == "unrepresented_runtime_condition"
    )

    laguna = audit.find_cell(matrix, "laguna_s21_4bit", "mlx_pr_3838", 9)
    assert laguna["eligibility"] == "conditional"
    assert (
        laguna["predicate_terms"]["long_key_rows_at_least_1024"]
        == "unrepresented_runtime_condition"
    )
    assert (
        laguna["predicate_terms"]["mask_and_causal_mode_do_not_gate_multirow_route"]
        is True
    )


def test_4171_keeps_qwen27_8bit_in_the_supported_affine_mode_family() -> None:
    _, snapshots = audit.read_reference(REFERENCE)
    matrix = audit.build_matrix(snapshots)

    before_boundary = audit.find_cell(matrix, "qwen36_27b_8bit", "mlx_pr_4171", 12)
    at_boundary = audit.find_cell(matrix, "qwen36_27b_8bit", "mlx_pr_4171", 13)
    assert before_boundary["eligibility"] == "false"
    assert at_boundary["eligibility"] == "conditional"
    assert (
        at_boundary["predicate_terms"]["affine_bits_supported_by_transposed_qmm_t_nax"]
        is True
    )


def test_4023_derives_sorted_verify_geometry_but_rejects_density() -> None:
    _, snapshots = audit.read_reference(REFERENCE)
    expected_top_k = {
        "qwen36_35b_a3b_4bit": 8,
        "laguna_s21_4bit": 10,
    }
    for target, top_k in expected_top_k.items():
        assert snapshots[target]["top_k"] == top_k
        for rows in audit.VERIFY_WIDTHS:
            cell = audit.source_4023(target, snapshots[target], rows)
            terms = cell["predicate_terms"]
            assert cell["eligibility"] == "false"
            assert terms["target_verify_token_rows"] == rows
            assert terms["gatherqmm_inner_rows_m"] == 1
            assert terms["batch_rows_b"] == rows * top_k
            assert terms["batch_rows_b_at_least_16"] is True
            assert terms["right_sorted"] is True
            assert terms["batch_rows_per_expert_b_over_e_at_least_4"] is False
            assert terms["nax_available"] == "unrepresented_runtime_condition"


def test_4023_has_positive_and_runtime_conditional_density_cases() -> None:
    _, snapshots = audit.read_reference(REFERENCE)
    conditional = audit.source_4023(
        "qwen36_35b_a3b_4bit",
        snapshots["qwen36_35b_a3b_4bit"],
        rows=128,
    )
    assert conditional["eligibility"] == "conditional"
    assert conditional["predicate_terms"]["gatherqmm_inner_rows_m"] == 1
    assert conditional["predicate_terms"]["batch_rows_b"] == 1024
    assert conditional["predicate_terms"]["right_sorted"] is True

    eligible = audit.source_4023(
        "qwen36_35b_a3b_4bit",
        snapshots["qwen36_35b_a3b_4bit"],
        rows=128,
        nax_available=True,
    )
    assert eligible["eligibility"] == "true"
    assert eligible["predicate_terms"]["batch_rows_b_at_least_16"] is True
    assert (
        eligible["predicate_terms"]["batch_rows_per_expert_b_over_e_at_least_4"] is True
    )
    assert eligible["predicate_terms"]["nax_available"] is True


def test_report_is_cpu_only_and_regenerates_a_v5_matrix(tmp_path: Path) -> None:
    report = audit.build_report(REFERENCE, SCRIPT.parents[1])
    assert report["schema"] == "mtplx-m5-static-dispatch-audit-v5"
    assert len(report["matrix"]) == 288
    assert len(report["tests"]) == 16
    assert (
        report["consumer_call_geometry_source"]["revision"]
        == audit.MLX_LM_SWITCH_GLU_REVISION
    )

    destination = tmp_path / "dispatch.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["canonical_matrix_sha256"] == audit.digest_json(saved["matrix"])
