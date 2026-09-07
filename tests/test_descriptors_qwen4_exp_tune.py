"""Tune is enabled for the Qwen 3.8 Flash-Next family (qwen4_exp).

qwen4-next ships backend qwen4_exp with can_run_verified=True and the forged
packs record mtp_depth_max 3, yet the family never reached tune: the allowlist
stopped at qwen3_8, so `mtplx tune` reported the model unsupported and no
depth could be measured or chosen for the geometry that 2.11 made its headline
decode lane.
"""

from __future__ import annotations

from mtplx.backends.descriptors import (
    model_family_from_inspection,
    tune_policy_for_model,
)

FLASH_NEXT_INSPECTION = {
    "model_type": "qwen4_exp_text",
    "architecture": "Qwen4ExpForConditionalGeneration",
    "mtp_arch": "qwen4-next",
    "num_hidden_layers": 48,
    "mtp_num_hidden_layers": 1,
}


def test_flash_next_resolves_to_the_qwen4_exp_family():
    assert model_family_from_inspection(FLASH_NEXT_INSPECTION) == "qwen4_exp"


def test_tune_is_supported_for_flash_next():
    assert tune_policy_for_model(inspection=FLASH_NEXT_INSPECTION).supported is True


def test_flash_next_offers_depths_one_to_three():
    policy = tune_policy_for_model(inspection=FLASH_NEXT_INSPECTION)
    assert policy.candidates == ("AR", "D1", "D2", "D3")


def test_the_family_does_not_ride_the_dense_qwen3_8_contract():
    # Flash-Next carries "3.8" in its public name; it must resolve to its own
    # family, not the dense-27B qwen3_8 behaviour contract.
    assert model_family_from_inspection({"model_type": "qwen4_exp_text"}) != "qwen3_8"
