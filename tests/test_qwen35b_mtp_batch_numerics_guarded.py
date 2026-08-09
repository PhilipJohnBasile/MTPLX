from __future__ import annotations


def test_round_summary_uses_completed_tokens_and_requires_unique_ids():
    from scripts.qwen35b_mtp_batch_numerics_guarded import summarize_round

    summary = summarize_round(
        [
            {"id": "one", "usage": {"completion_tokens": 40}},
            {"id": "two", "usage": {"completion_tokens": 60}},
        ],
        wall_s=0.25,
    )

    assert summary == {
        "requests": 2,
        "completion_tokens": 100,
        "wall_s": 0.25,
        "aggregate_output_tps": 400.0,
        "unique_response_ids": 2,
    }


def test_benchmark_rejects_a_server_that_installed_the_wrong_profile():
    import pytest

    from scripts.qwen35b_mtp_batch_numerics_guarded import validate_health_profile

    with pytest.raises(RuntimeError, match="requested balanced.*installed throughput"):
        validate_health_profile(
            {"scheduler": {"mtp_batch_numerics": "throughput"}},
            expected="balanced",
        )


def test_benchmark_requires_the_selected_route_and_real_width_eight():
    import pytest

    from scripts.qwen35b_mtp_batch_numerics_guarded import (
        validate_b8_benchmark_health,
    )

    health = {
        "scheduler": {
            "mtp_batch_numerics": "balanced",
            "mtp_batch_route_id": (
                "qwen35b_a3b_mtp_batch_b8_t2_l0_b1_qkv_z_b_balanced"
            ),
            "mtp_batch": {
                "batch_histogram": {"1": 24, "8": 4},
                "fixed_width_histogram": {"8": 400},
                "last_route_id": ("qwen35b_a3b_mtp_batch_b8_t2_l0_b1_qkv_z_b_balanced"),
            },
        }
    }

    validate_b8_benchmark_health(health, expected="balanced")

    health["scheduler"]["mtp_batch"]["batch_histogram"]["7"] = 1
    with pytest.raises(RuntimeError, match="real cohort widths.*7"):
        validate_b8_benchmark_health(health, expected="balanced")


def test_benchmark_parser_accepts_the_serial_b1_exact_control():
    from scripts.qwen35b_mtp_batch_numerics_guarded import _parse_args

    args = _parse_args(
        [
            "--model",
            "/model",
            "--mtplx",
            "/mtplx",
            "--chat-template",
            "/template",
            "--numerics",
            "b1-exact",
            "--mode",
            "greedy",
            "--output",
            "/receipt.json",
        ]
    )

    assert args.numerics == "b1-exact"


def test_legacy_workload_matches_the_original_pr_benchmark_contract():
    from scripts.qwen35b_mtp_batch_numerics_guarded import completion_payload

    payload, marker = completion_payload(
        row=3,
        mode="greedy",
        max_tokens=256,
        workload="legacy",
    )

    assert marker == "ROW_3_ONLY"
    assert payload["seed"] == 4203
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["top_k"] == 20
    assert payload["messages"] == [
        {
            "role": "user",
            "content": (
                "Begin with the exact marker ROW_3_ONLY. Explain why deterministic "
                "concurrent request ownership matters in a model server. Do not "
                "mention any other marker."
            ),
        }
    ]


def test_paired_round_summary_reports_serial_and_b8_using_actual_tokens():
    from scripts.qwen35b_mtp_batch_numerics_guarded import summarize_paired_round

    summary = summarize_paired_round(
        serial={"completion_tokens": 1200, "wall_s": 10.0},
        b8={"completion_tokens": 1100, "wall_s": 4.0},
    )

    assert summary == {
        "serial_completion_tokens": 1200,
        "serial_wall_s": 10.0,
        "serial_aggregate_output_tps": 120.0,
        "b8_completion_tokens": 1100,
        "b8_wall_s": 4.0,
        "b8_aggregate_output_tps": 275.0,
        "speedup": 275.0 / 120.0,
    }


def test_legacy_warmup_can_synchronize_without_changing_timed_rounds(
    monkeypatch,
):
    import scripts.qwen35b_mtp_batch_numerics_guarded as module

    seen_barriers = []

    def fake_completion(_base_url, *, row, barrier, **_kwargs):
        seen_barriers.append(barrier)
        return {
            "id": f"row-{row}",
            "usage": {"completion_tokens": 1},
            "_client_elapsed_s": 0.01,
            "_output_sha256": f"sha-{row}",
            "_marker_isolated": True,
        }

    monkeypatch.setattr(module, "_completion", fake_completion)

    module._run_round(
        "http://unused",
        mode="greedy",
        max_tokens=32,
        workload="legacy",
        cohort=True,
        synchronize_cohort=True,
    )
    assert len(seen_barriers) == 8
    assert all(barrier is not None for barrier in seen_barriers)

    seen_barriers.clear()
    module._run_round(
        "http://unused",
        mode="greedy",
        max_tokens=256,
        workload="legacy",
        cohort=True,
    )
    assert seen_barriers == [None] * 8
