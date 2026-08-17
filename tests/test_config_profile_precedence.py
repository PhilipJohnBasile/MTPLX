from __future__ import annotations

import argparse

from mtplx.config import apply_user_config
from mtplx.profiles import DEFAULT_PROFILE_NAME


def _args(*, command: str, flags: set[str], profile: str = DEFAULT_PROFILE_NAME):
    return argparse.Namespace(
        command=command,
        model=None,
        cache_dir=None,
        profile=profile,
        max="max" in flags,
        _cli_flags=flags,
    )


def test_quickstart_max_keeps_sustained_over_stale_config(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('profile = "performance-cold"\n', encoding="utf-8")
    args = _args(command="quickstart", flags={"max"})

    apply_user_config(args, config_path=config)

    assert args.profile == "sustained"


def test_serve_max_keeps_sustained_over_stale_config(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('profile = "performance-cold"\n', encoding="utf-8")
    args = _args(command="serve", flags={"max"})

    apply_user_config(args, config_path=config)

    assert args.profile == "sustained"


def test_start_max_keeps_sustained_over_stale_config(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('profile = "performance-cold"\n', encoding="utf-8")
    args = _args(command="start", flags={"max"})

    apply_user_config(args, config_path=config)

    assert args.profile == "sustained"


def test_explicit_profile_still_beats_config_when_using_max(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('profile = "performance-cold"\n', encoding="utf-8")
    args = _args(command="quickstart", flags={"profile", "max"}, profile="sustained")

    apply_user_config(args, config_path=config)

    assert args.profile == "sustained"


def test_config_profile_still_applies_without_max_flag(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('profile = "performance-cold"\n', encoding="utf-8")
    args = _args(command="quickstart", flags=set())

    apply_user_config(args, config_path=config)

    assert args.profile == "performance-cold"


def _flagship_public_id() -> str:
    from mtplx.profiles import QWEN38_BARE_SPEED_PUBLIC_MODEL_ID

    return QWEN38_BARE_SPEED_PUBLIC_MODEL_ID


def test_config_sustained_is_a_pin_not_a_promotion_target(tmp_path, capsys):
    """F12: config.toml "sustained" equals the parser default, so it used to
    be silently promoted to turbo — the user's pin was ignored."""

    from mtplx.commands.public import (
        _apply_model_default_profile,
        _resolved_default_profile_name,
    )

    config = tmp_path / "config.toml"
    config.write_text('profile = "sustained"\n', encoding="utf-8")
    args = _args(command="serve", flags=set())

    apply_user_config(args, config_path=config)

    assert args.profile == "sustained"
    assert args._profile_from_config == str(config)
    out = capsys.readouterr().out
    assert out.count("profile: sustained (from config.toml)") == 1

    flagship = _flagship_public_id()
    assert _apply_model_default_profile(args, flagship) is False
    assert args.profile == "sustained"
    args.model = flagship
    assert _resolved_default_profile_name(args) == "sustained"


def test_config_stable_sticks_and_prints(tmp_path, capsys):
    """F12: a non-default config profile always stuck, but silently."""

    from mtplx.commands.public import _resolved_default_profile_name

    config = tmp_path / "config.toml"
    config.write_text('profile = "stable"\n', encoding="utf-8")
    args = _args(command="serve", flags=set())

    apply_user_config(args, config_path=config)

    assert args.profile == "stable"
    out = capsys.readouterr().out
    assert out.count("profile: stable (from config.toml)") == 1
    args.model = _flagship_public_id()
    assert _resolved_default_profile_name(args) == "stable"


def test_no_config_keeps_per_model_promotion(tmp_path, capsys):
    from mtplx.commands.public import _resolved_default_profile_name

    args = _args(command="serve", flags=set())
    apply_user_config(args, config_path=tmp_path / "missing.toml")

    assert getattr(args, "_profile_from_config", None) is None
    assert capsys.readouterr().out == ""
    args.model = _flagship_public_id()
    assert _resolved_default_profile_name(args) == "turbo"


def test_config_profile_line_respects_json_mode(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text('profile = "stable"\n', encoding="utf-8")
    args = _args(command="serve", flags=set())
    args.json = True

    apply_user_config(args, config_path=config)

    assert args.profile == "stable"
    assert args._profile_from_config == str(config)
    assert capsys.readouterr().out == ""


def test_explicit_cli_profile_beats_config_without_pin_marker(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text('profile = "stable"\n', encoding="utf-8")
    args = _args(command="serve", flags={"profile"}, profile="turbo")

    apply_user_config(args, config_path=config)

    assert args.profile == "turbo"
    assert getattr(args, "_profile_from_config", None) is None
    assert capsys.readouterr().out == ""
