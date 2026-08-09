from __future__ import annotations


def test_attribution_names_first_divergence_and_real_shapes():
    from scripts.qwen35b_mtp_batch_numerics_attribution import build_report

    report = build_report(
        {
            "geometry": {"target": [8, 2], "draft": [8, 1]},
            "boundaries": [
                {
                    "operator": "target.layers.0.q_proj",
                    "layer": 0,
                    "phase": "decode_verify",
                    "b1_shape": [1, 2, 2048],
                    "b8_shape": [8, 2, 2048],
                    "bitwise": False,
                    "max_abs": 0.03125,
                    "max_ulp": 2,
                    "argmax_equal": True,
                },
                {
                    "operator": "target.layers.0.k_proj",
                    "layer": 0,
                    "phase": "decode_verify",
                    "b1_shape": [1, 2, 512],
                    "b8_shape": [8, 2, 512],
                    "bitwise": True,
                    "max_abs": 0.0,
                    "max_ulp": 0,
                    "argmax_equal": True,
                },
            ],
            "row_isolation_parity": True,
        },
        model="/models/qwen35b",
        route_id="qwen35b_a3b_mtp_batch_b8_t2_m16_throughput",
        config_fingerprint="config:throughput:route",
    )

    assert report["geometry"] == {"target": [8, 2], "draft": [8, 1]}
    assert report["first_material_divergence"]["operator"] == ("target.layers.0.q_proj")
    assert report["first_material_divergence"]["b1_shape"] == [1, 2, 2048]
    assert report["first_material_divergence"]["b8_shape"] == [8, 2, 2048]
    assert report["row_isolation_parity"] is True


def test_attribution_report_requires_exact_boundary_schema():
    import pytest

    from scripts.qwen35b_mtp_batch_numerics_attribution import build_report

    with pytest.raises(ValueError, match="max_ulp"):
        build_report(
            {
                "geometry": {"target": [8, 2], "draft": [8, 1]},
                "boundaries": [{"operator": "target.layers.0.q_proj"}],
            },
            model="qwen",
            route_id="route",
            config_fingerprint="fingerprint",
        )


def test_attribution_prepends_refined_boundaries_before_capture_receipt():
    from scripts.qwen35b_mtp_batch_numerics_attribution import build_report

    capture_boundary = {
        "operator": "target.layers.0.gdn_postconv.conv_state",
        "layer": 0,
        "phase": "decode_verify",
        "b1_shape": [1, 2, 32, 128],
        "b8_shape": [8, 2, 32, 128],
        "bitwise": False,
        "max_abs": 0.125,
        "max_ulp": -1,
        "argmax_equal": True,
    }
    qkv_boundary = {
        "operator": "target.layers.0.linear_attn.in_proj_qkv",
        "layer": 0,
        "phase": "decode_verify",
        "b1_shape": [1, 2, 8192],
        "b8_shape": [8, 2, 8192],
        "bitwise": False,
        "max_abs": 0.03125,
        "max_ulp": -1,
        "argmax_equal": True,
    }

    report = build_report(
        {
            "geometry": {"target": [8, 2], "draft": [8, 1]},
            "boundaries": [capture_boundary],
            "row_isolation_parity": True,
        },
        model="qwen",
        route_id="throughput",
        config_fingerprint="fingerprint",
        refined_boundaries=[qkv_boundary],
    )

    assert report["boundaries"] == [qkv_boundary, capture_boundary]
    assert report["first_material_divergence"] == qkv_boundary
