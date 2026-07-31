from __future__ import annotations

from pathlib import Path

import pytest

from mtplx.benchmarks.protocol import (
    WEIGHTED_TOK_S_AGGREGATION,
    assert_protocol_match,
    build_effective_run_record,
    compare_token_position_distributions,
    protocol_mismatches,
    summarize_external_draft_contract,
    weighted_tok_s,
)


def _record(tmp_path: Path, **overrides):
    suite = tmp_path / "suite.jsonl"
    suite.write_text('{"id":"one","prompt":"hello"}\n', encoding="utf-8")
    kwargs = {
        "backend": "dflash_mlx_official",
        "model_ref": "target",
        "model_revision": "target-sha",
        "draft_ref": "draft",
        "draft_revision": "draft-sha",
        "prompt_suite": suite,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 256,
        "seed": 0,
        "enable_thinking": False,
        "block_size": 8,
        "generation_mode": "dflash",
        "nax_enabled": False,
        "verify_strategy": "official_dflash",
        "compiled_verify": "not_applicable",
        "runtime_switches": {"EXAMPLE_SWITCH": "off"},
        "runtime_contract": {
            "base_hidden_variant": "pre_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
            "mtp_position_mode": "local",
        },
    }
    kwargs.update(overrides)
    return build_effective_run_record(**kwargs)


def test_effective_record_pins_the_executed_protocol(tmp_path: Path) -> None:
    record = _record(tmp_path)

    assert record["prompt_suite_sha256"]
    assert record["enable_thinking"] is False
    assert record["model"]["revision"] == "target-sha"
    assert record["draft"]["revision"] == "draft-sha"
    assert record["verify_strategy"] == "official_dflash"
    assert record["compiled_verify"] == "not_applicable"
    assert record["runtime_switches"] == {"EXAMPLE_SWITCH": "off"}
    assert record["runtime_contract"]["base_hidden_variant"] == "pre_norm"
    assert record["aggregation"] == WEIGHTED_TOK_S_AGGREGATION


def test_effective_record_requires_revisions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_revision"):
        _record(tmp_path, model_revision=None)
    with pytest.raises(ValueError, match="draft_revision"):
        _record(tmp_path, draft_revision=None)


def test_protocol_gate_catches_thinking_mismatch(tmp_path: Path) -> None:
    thinking_off = _record(tmp_path, enable_thinking=False)
    thinking_on = _record(tmp_path, enable_thinking=True)

    assert protocol_mismatches(thinking_off, thinking_on) == {
        "enable_thinking": (False, True)
    }
    with pytest.raises(ValueError, match="enable_thinking"):
        assert_protocol_match(thinking_off, thinking_on)


def test_protocol_gate_catches_runtime_contract_mismatch(tmp_path: Path) -> None:
    pre_norm = _record(tmp_path)
    post_norm = _record(
        tmp_path,
        runtime_contract={
            **pre_norm["runtime_contract"],
            "base_hidden_variant": "post_norm",
        },
    )

    assert protocol_mismatches(pre_norm, post_norm) == {
        "runtime_contract": (
            pre_norm["runtime_contract"],
            post_norm["runtime_contract"],
        )
    }


def test_arm_dimensions_do_not_make_protocol_incomparable(tmp_path: Path) -> None:
    dflash = _record(tmp_path, block_size=8, generation_mode="dflash")
    mtp = _record(
        tmp_path,
        backend="mtplx",
        draft_ref=None,
        draft_revision=None,
        block_size=1,
        generation_mode="mtp",
        nax_enabled=True,
    )

    assert protocol_mismatches(dflash, mtp) == {}


def test_weighted_tok_s_aggregates_tokens_over_decode_seconds() -> None:
    rows = [
        {"generated_tokens": 100, "tok_s": 100.0},
        {"generated_tokens": 100, "tok_s": 50.0},
    ]
    assert weighted_tok_s(rows) == pytest.approx(200.0 / 3.0)


def test_external_draft_contract_uses_observed_events() -> None:
    summary = summarize_external_draft_contract(
        [
            {
                "block_draft_source": {
                    "metadata": {
                        "declaration": "sampled_q",
                        "sampler": {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "top_k": 20,
                        },
                        "draft_artifact": {"resolved_revision": "abc123"},
                    },
                    "draft_q": "soft",
                    "acceptance": "probability_ratio_residual",
                    "proposal_tokens": 2,
                    "engine_q_support_sizes": [20, 20],
                    "engine_q_sampled_probabilities": [0.2, 0.1],
                },
                "drafts": [
                    {
                        "accepted": False,
                        "correction": 17,
                        "correction_origin": "residual_p_minus_q",
                    },
                    {
                        "accepted": True,
                        "correction": 23,
                        "correction_origin": "accepted_draft",
                    },
                ],
            }
        ],
        expected_sampler={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        expected_draft_revision="abc123",
    )

    assert summary["contract_matches"] is True
    assert summary["soft_q_proposals"] == 1
    assert summary["residual_path_exercised"] is True


def test_external_draft_contract_rejects_intent_observation_mismatch() -> None:
    summary = summarize_external_draft_contract(
        [
            {
                "block_draft_source": {
                    "metadata": {
                        "declaration": "one_hot",
                        "sampler": {
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "top_k": 0,
                        },
                    },
                    "draft_q": "point_mass",
                    "acceptance": "greedy_match",
                    "proposal_tokens": 1,
                    "engine_q_support_sizes": [1],
                    "engine_q_sampled_probabilities": [1.0],
                },
                "drafts": [],
            }
        ],
        expected_sampler={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    )

    assert summary["contract_matches"] is False
    assert summary["sampler_mismatches"]


def test_external_draft_contract_rejects_point_mass_with_sampled_q_metadata() -> None:
    summary = summarize_external_draft_contract(
        [
            {
                "block_draft_source": {
                    "metadata": {
                        "declaration": "sampled_q",
                        "sampler": {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "top_k": 20,
                        },
                    },
                    "draft_q": "point_mass",
                    "acceptance": "probability_ratio_residual",
                    "proposal_tokens": 1,
                    "engine_q_support_sizes": [1],
                    "engine_q_sampled_probabilities": [1.0],
                },
                "drafts": [],
            }
        ],
        expected_sampler={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    )

    assert summary["declarations"] == {"sampled_q": 1}
    assert summary["draft_q_kinds"] == {"point_mass": 1}
    assert summary["contract_matches"] is False


def test_distribution_gate_defers_without_real_stochastic_receipts() -> None:
    result = compare_token_position_distributions(
        [[1, 2]],
        [[1, 2]],
        min_samples_per_arm=128,
    )

    assert result["status"] == "deferred_insufficient_samples"
    assert result["release_gate_pass"] is False


def test_distribution_non_rejection_does_not_mint_equivalence() -> None:
    samples = [[1, 2, 3] for _ in range(16)]
    result = compare_token_position_distributions(
        samples,
        samples,
        min_samples_per_arm=16,
        permutations=399,
    )

    assert result["status"] == "no_detected_divergence"
    assert result["inference"] == "inconclusive_non_rejection"
    assert result["release_gate_pass"] is False
    assert result["joint_sequence_law_tested"] is False


def test_distribution_gate_defers_when_permutations_cannot_cross_alpha() -> None:
    result = compare_token_position_distributions(
        [[1, 1, 1, 1] for _ in range(32)],
        [[2, 2, 2, 2] for _ in range(32)],
        max_positions=4,
        min_samples_per_arm=32,
        permutations=79,
        alpha=0.05,
        seed=7,
    )

    assert result["status"] == "insufficient_permutation_resolution"
    assert result["inference"] == (
        "inconclusive_insufficient_permutation_resolution"
    )
    assert result["min_permutation_p"] == pytest.approx(1 / 80)
    assert result["bonferroni_alpha"] == pytest.approx(0.05 / 4)
    assert result["positions"] == []
    assert result["release_gate_pass"] is False


def test_distribution_gate_detects_clear_divergence_at_configured_resolution() -> None:
    result = compare_token_position_distributions(
        [[1, 1, 1, 1] for _ in range(32)],
        [[2, 2, 2, 2] for _ in range(32)],
        max_positions=4,
        min_samples_per_arm=32,
        permutations=99,
        alpha=0.05,
        seed=7,
    )

    assert result["status"] == "divergence_detected"
    assert result["permutations"] == 99
    assert len(result["positions"]) == 4
    assert all(position["divergence_detected"] for position in result["positions"])
    assert all(
        position["permutation_p_value"] == pytest.approx(1 / 100)
        for position in result["positions"]
    )
    assert all(
        position["bonferroni_alpha"] == pytest.approx(0.05 / 4)
        for position in result["positions"]
    )
    assert result["release_gate_pass"] is False


def test_dflash_cli_requires_an_explicit_thinking_mode() -> None:
    from mtplx.cli import build_parser

    parser = build_parser()
    common = [
        "dflash-mlx-baseline",
        "--model-revision",
        "target-sha",
        "--draft-revision",
        "draft-sha",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(common)
    assert parser.parse_args([*common, "--disable-thinking"]).enable_thinking is False
    assert parser.parse_args([*common, "--enable-thinking"]).enable_thinking is True
