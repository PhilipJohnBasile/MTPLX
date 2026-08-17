"""Tail sweep: profile reports/stamps equal per-model launch resolution.

Historic bug class: surfaces reporting "sustained" for artifacts the
engine's default resolution actually launches on turbo (flagship
promotion in ``mtplx.commands.public``). Three surfaces are pinned here:
the args-free resolver core itself, the cache listing
(``CachedModel.to_dict``), and the forge runtime stamp.
``mtplx doctor``'s support matrix is pinned for consistency with the
resolver so a future default-model flip cannot desynchronize them.
"""

from __future__ import annotations

import json
from pathlib import Path

import mtplx.commands.forge as forge
from mtplx.commands.public import resolved_default_profile_name_for_ref
from mtplx.diagnostics import SUPPORT_MATRIX
from mtplx.hf_loader import CachedModel
from mtplx.profiles import DEFAULT_HF_MODEL_ID, DEFAULT_PROFILE_NAME


FLAGSHIP_HF_ID = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
FLAGSHIP_CACHE_DIR = "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_mtp_config() -> dict:
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "vocab_size": 128,
        },
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
    }


def _speed_win_rows() -> list[dict]:
    return [
        {
            "depth": 0,
            "tok_s": 22.0,
            "multiplier_vs_ar": 1.0,
            "acceptance_by_position": [],
            "verify_time_s": 0.1,
        },
        {
            "depth": 1,
            "tok_s": 34.0,
            "multiplier_vs_ar": 1.5454545455,
            "acceptance_by_position": [0.9],
            "verify_time_s": 0.2,
        },
        {
            "depth": 2,
            "tok_s": 40.0,
            "multiplier_vs_ar": 1.8181818182,
            "acceptance_by_position": [0.88, 0.55],
            "verify_time_s": 0.3,
        },
        {
            "depth": 3,
            "tok_s": 44.0,
            "multiplier_vs_ar": 2.0,
            "acceptance_by_position": [0.9, 0.7, 0.5],
            "verify_time_s": 0.4,
        },
    ]


# ---------------------------------------------------------------------------
# Resolver core
# ---------------------------------------------------------------------------


def test_resolver_promotes_flagships_and_keeps_others_on_default():
    assert resolved_default_profile_name_for_ref(FLAGSHIP_HF_ID) == "turbo"
    # The shipped default model itself is a promoted flagship today.
    assert resolved_default_profile_name_for_ref(DEFAULT_HF_MODEL_ID) == "turbo"
    assert (
        resolved_default_profile_name_for_ref("someorg/derivative-model")
        == DEFAULT_PROFILE_NAME
    )


# ---------------------------------------------------------------------------
# Cache listing (mtplx list)
# ---------------------------------------------------------------------------


def _cached_model(path: Path, *, ok: bool) -> CachedModel:
    return CachedModel(
        repo_id=path.name.replace("--", "/"),
        path=path,
        size_bytes=1,
        has_runtime_contract=False,
        has_config=True,
        validation={"ok": ok},
    )


def test_cached_flagship_reports_turbo(tmp_path):
    flagship = tmp_path / FLAGSHIP_CACHE_DIR
    flagship.mkdir()
    assert _cached_model(flagship, ok=True).to_dict()["recommended_profile"] == "turbo"


def test_cached_third_party_reports_default_profile(tmp_path):
    third_party = tmp_path / "someorg--custom-model"
    third_party.mkdir()
    assert (
        _cached_model(third_party, ok=True).to_dict()["recommended_profile"]
        == DEFAULT_PROFILE_NAME
    )


def test_cached_invalid_artifact_reports_none(tmp_path):
    broken = tmp_path / FLAGSHIP_CACHE_DIR
    broken.mkdir()
    assert _cached_model(broken, ok=False).to_dict()["recommended_profile"] is None


# ---------------------------------------------------------------------------
# Forge runtime stamp
# ---------------------------------------------------------------------------


def _stamp(model_path: Path, rows: list[dict]) -> dict:
    return forge._stamp_runtime_metadata(
        model_path,
        branded_name=model_path.name,
        source_repo="owner/source",
        source_sha="abc123",
        source_format=forge.SOURCE_COMPRESSED_TENSORS_AWQ,
        recipe={"mtp_policy": "keep_bf16"},
        forge_inputs={
            "trunk_path": str(model_path),
            "mtp_source_path": str(model_path),
        },
        rows=rows,
        mtp_contract={
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
        },
        existing=None,
    )


def test_forge_stamps_turbo_for_flagship_identified_artifact(tmp_path):
    model_path = tmp_path / "Qwen3.8-27B-MTPLX-Optimized-Speed"
    _write_json(model_path / "config.json", _minimal_mtp_config())
    rows = forge._annotate_verify_rows(_speed_win_rows())
    runtime = _stamp(model_path, rows)
    # Serve-time resolution maps this branded dir to the flagship public id
    # and launches turbo; the stamp must agree or the profile-scoped runtime
    # contract is hidden on the profile the artifact actually runs under.
    assert runtime["recommended_profile"] == "turbo"


def test_forge_keeps_default_profile_stamp_for_unrecognized_artifact(tmp_path):
    model_path = tmp_path / "Fixture-MTPLX-Neutral"
    _write_json(model_path / "config.json", _minimal_mtp_config())
    rows = forge._annotate_verify_rows(_speed_win_rows())
    runtime = _stamp(model_path, rows)
    assert runtime["recommended_profile"] == DEFAULT_PROFILE_NAME


def test_forge_keeps_stable_stamp_without_mtp_win(tmp_path):
    model_path = tmp_path / "Qwen3.8-27B-MTPLX-Optimized-Speed"
    _write_json(model_path / "config.json", _minimal_mtp_config())
    rows = forge._annotate_verify_rows(
        [
            {"depth": 0, "tok_s": 42.0, "acceptance_by_position": []},
            {"depth": 1, "tok_s": 40.0, "acceptance_by_position": [0.3]},
            {"depth": 2, "tok_s": 41.0, "acceptance_by_position": [0.3, 0.1]},
            {"depth": 3, "tok_s": 39.0, "acceptance_by_position": [0.3, 0.1, 0.0]},
        ]
    )
    runtime = _stamp(model_path, rows)
    # No MTP depth beat AR: even a flagship-named artifact keeps the
    # no-MTP "stable" recommendation.
    assert runtime["recommended_profile"] == "stable"


def test_forge_existing_stamp_survives(tmp_path):
    model_path = tmp_path / "Qwen3.8-27B-MTPLX-Optimized-Speed"
    _write_json(model_path / "config.json", _minimal_mtp_config())
    rows = forge._annotate_verify_rows(_speed_win_rows())
    runtime = forge._stamp_runtime_metadata(
        model_path,
        branded_name=model_path.name,
        source_repo="owner/source",
        source_sha="abc123",
        source_format=forge.SOURCE_COMPRESSED_TENSORS_AWQ,
        recipe={"mtp_policy": "keep_bf16"},
        forge_inputs={
            "trunk_path": str(model_path),
            "mtp_source_path": str(model_path),
        },
        rows=rows,
        mtp_contract={
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
        },
        existing={"recommended_profile": "sustained"},
    )
    # setdefault semantics are deliberate: an existing artifact stamp is
    # prior evidence and is never rewritten by a re-stamp.
    assert runtime["recommended_profile"] == "sustained"


# ---------------------------------------------------------------------------
# Doctor support matrix
# ---------------------------------------------------------------------------


def test_support_matrix_default_profile_matches_resolution():
    assert SUPPORT_MATRIX["supported"]["default_model"] == DEFAULT_HF_MODEL_ID
    assert SUPPORT_MATRIX["supported"][
        "default_profile"
    ] == resolved_default_profile_name_for_ref(DEFAULT_HF_MODEL_ID)
