from __future__ import annotations

import argparse

from mtplx.config import apply_user_config, load_user_config
from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR
from mtplx.profiles import DEFAULT_HF_MODEL_ID, DEFAULT_PROFILE_NAME


def test_load_user_config_reads_runtime_defaults(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{tmp_path / "models"}"\n'
        'profile = "exact"\n'
        'thermal_control = "none"\n',
        encoding="utf-8",
    )

    loaded = load_user_config(config)

    assert loaded.exists is True
    assert loaded.model == "mtplx/example"
    assert loaded.model_dir == str(tmp_path / "models")
    assert loaded.profile == "exact"
    assert loaded.thermal_control == "none"


def test_load_user_config_reads_mtp_batch_numerics(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'mtp_batch_numerics = "balanced"\n',
        encoding="utf-8",
    )

    loaded = load_user_config(config)

    assert loaded.mtp_batch_numerics == "balanced"


def test_apply_user_config_respects_explicit_mtp_batch_numerics(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'mtp_batch_numerics = "balanced"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="serve",
        mtp_batch_numerics="throughput",
        _cli_flags={"mtp-batch-numerics"},
    )

    apply_user_config(args, config_path=config)

    assert args.mtp_batch_numerics == "throughput"


def test_apply_user_config_fills_mtp_batch_numerics_default(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'mtp_batch_numerics = "balanced"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="serve",
        mtp_batch_numerics="throughput",
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.mtp_batch_numerics == "balanced"


def test_apply_user_config_fills_runtime_defaults(tmp_path):
    config = tmp_path / "config.toml"
    model_dir = tmp_path / "models"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{model_dir}"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="run",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        _cli_flags=set(),
    )

    loaded = apply_user_config(args, config_path=config)

    assert loaded.exists is True
    assert args.model == "mtplx/example"
    assert args.cache_dir == str(model_dir)
    assert args.profile == "exact"
    assert args.mtplx_config["path"] == str(config)


def test_apply_user_config_ignores_legacy_optimized_speed_default(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "models/Qwen3.6-27B-MTPLX-Optimized-Speed"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="quickstart",
        model=DEFAULT_HF_MODEL_ID,
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.model == DEFAULT_HF_MODEL_ID


def test_apply_user_config_preserves_explicit_runtime_values(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "mtplx/example"\n'
        'model_dir = "/tmp/mtplx-models"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="serve",
        model="models/local",
        cache_dir="/tmp/explicit-cache",
        profile="performance-cold",
        _cli_flags={"model", "cache-dir", "profile"},
    )

    apply_user_config(args, config_path=config)

    assert args.model == "models/local"
    assert args.cache_dir == "/tmp/explicit-cache"
    assert args.profile == "performance-cold"


def test_apply_user_config_preserves_explicit_default_like_model(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "mtplx/example"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    explicit_model = "/Users/example/models/Qwen3.6-27B-MTPLX-Optimized-Speed"
    args = argparse.Namespace(
        command="serve",
        model=explicit_model,
        cache_dir=None,
        profile=DEFAULT_PROFILE_NAME,
        _cli_flags={"model"},
    )

    apply_user_config(args, config_path=config)

    assert args.model == explicit_model


def test_apply_user_config_fills_tune_model_defaults(tmp_path):
    config = tmp_path / "config.toml"
    model_dir = tmp_path / "models"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{model_dir}"\n'
        'profile = "exact"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="tune",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile="performance-cold",
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.model == "mtplx/example"
    assert args.cache_dir == str(model_dir)
    assert args.profile == "performance-cold"


def test_apply_user_config_fills_bench_tune_model_defaults(tmp_path):
    config = tmp_path / "config.toml"
    model_dir = tmp_path / "models"
    config.write_text(
        'model = "mtplx/example"\n'
        f'model_dir = "{model_dir}"\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        command="bench",
        bench_action="tune",
        model=str(DEFAULT_RUNTIME_MODEL_DIR),
        cache_dir=None,
        profile=None,
        _cli_flags=set(),
    )

    apply_user_config(args, config_path=config)

    assert args.model == "mtplx/example"
    assert args.cache_dir == str(model_dir)
