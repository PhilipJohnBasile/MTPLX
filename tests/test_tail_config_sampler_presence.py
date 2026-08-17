"""config.toml sampler pins survive family-default injection.

The launch sampler trio (temperature/top_p/top_k) used value-sentinels
(``in (None, 0.6)``) to detect "unset": a config.toml value that happens
to EQUAL the parser default was read as unset and silently replaced by
the family sampler. Presence in the parsed config (``args.mtplx_config``)
now pins the value; injected family defaults are recorded in
``_injected_default_flags`` (wave-2 provenance contract) so telemetry can
tell an injected default from a user pin.
"""

from __future__ import annotations

from mtplx.cli import build_parser
from mtplx.commands.public import _apply_backend_serve_defaults
from mtplx.config import apply_user_config


def _inspection(model_dir) -> dict:
    # qwen3_8 family (from the model_dir marker): family sampler is the
    # official thinking sampler with temperature 1.0 — distinct from the
    # 0.6 parser default, so overwrites are observable.
    return {
        "model_dir": str(model_dir),
        "recommended_backend": "qwen3_next",
        "runtime_compatibility": "native-contract-gated",
        "compatibility": {
            "can_run": True,
            "exit_code": 0,
            "runtime_contract": {
                "arch_id": "qwen3-next-mtp",
                "mtp_depth_max": 3,
                "recommended_profile": "sustained",
            },
        },
    }


def _serve_args(extra=()):
    return build_parser().parse_args(["serve", "--model", "m", *extra])


def _qwen38_model_dir(tmp_path):
    model_dir = tmp_path / "Qwen3.8-27B-Custom"
    model_dir.mkdir()
    return model_dir


def test_config_value_equal_to_parser_default_is_honored(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("temperature = 0.6\ntop_p = 0.95\ntop_k = 20\n")
    model_dir = _qwen38_model_dir(tmp_path)

    args = _serve_args()
    apply_user_config(args, config_path=config_path)
    _apply_backend_serve_defaults(args, _inspection(model_dir))

    injected = set(getattr(args, "_injected_default_flags", set()) or set())
    # Honored: the standing config pin survives even though it equals the
    # old sentinel values (0.6/0.95/20 would previously become 1.0/...).
    assert args.temperature == 0.6
    assert args.top_p == 0.95
    assert args.top_k == 20
    # Marked explicit: config provenance is visible and the values are NOT
    # recorded as injected defaults.
    assert args.mtplx_config["temperature"] == 0.6
    assert args.mtplx_config["top_p"] == 0.95
    assert args.mtplx_config["top_k"] == 20
    assert "temperature" not in injected
    assert "top-p" not in injected
    assert "top-k" not in injected


def test_config_non_default_value_still_honored(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("temperature = 0.3\n")
    model_dir = _qwen38_model_dir(tmp_path)

    args = _serve_args()
    apply_user_config(args, config_path=config_path)
    _apply_backend_serve_defaults(args, _inspection(model_dir))

    injected = set(getattr(args, "_injected_default_flags", set()) or set())
    assert args.temperature == 0.3
    assert "temperature" not in injected


def test_absent_config_key_injects_family_default_and_marks_it(tmp_path):
    config_path = tmp_path / "config.toml"  # never written: no config file
    model_dir = _qwen38_model_dir(tmp_path)

    args = _serve_args()
    apply_user_config(args, config_path=config_path)
    _apply_backend_serve_defaults(args, _inspection(model_dir))

    injected = set(getattr(args, "_injected_default_flags", set()) or set())
    # Injected: the family sampler replaces the untouched parser default.
    assert args.temperature == 1.0
    # Marked injected — the whole trio is recorded as provenance.
    assert "temperature" in injected
    assert "top-p" in injected
    assert "top-k" in injected


def test_partial_config_pins_only_present_keys(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("top_k = 20\n")
    model_dir = _qwen38_model_dir(tmp_path)

    args = _serve_args()
    apply_user_config(args, config_path=config_path)
    _apply_backend_serve_defaults(args, _inspection(model_dir))

    injected = set(getattr(args, "_injected_default_flags", set()) or set())
    assert args.top_k == 20
    assert "top-k" not in injected
    # Keys absent from the config still receive the family default.
    assert args.temperature == 1.0
    assert "temperature" in injected


def test_cli_flag_still_wins_and_is_not_marked_injected(tmp_path):
    model_dir = _qwen38_model_dir(tmp_path)

    args = _serve_args(("--temperature", "0.6"))
    apply_user_config(args, config_path=tmp_path / "config.toml")
    _apply_backend_serve_defaults(args, _inspection(model_dir))

    injected = set(getattr(args, "_injected_default_flags", set()) or set())
    assert args.temperature == 0.6
    assert "temperature" not in injected


def test_config_pin_survives_quickstart_serve_handoff(tmp_path):
    """_with_server_policy_args forwards raw sampler VALUES onto a fresh
    namespace; without the parsed config riding along, a config 0.6 pin is
    indistinguishable from the parser default in the child and dies there —
    the same handoff class the profile pin is already fenced against."""

    from types import SimpleNamespace

    from mtplx.commands.public import _with_server_policy_args

    config_path = tmp_path / "config.toml"
    config_path.write_text("temperature = 0.6\n")
    model_dir = _qwen38_model_dir(tmp_path)

    parent = _serve_args()
    apply_user_config(parent, config_path=config_path)
    child = SimpleNamespace(temperature=0.6, top_p=0.95, top_k=20)
    _with_server_policy_args(child, parent)
    _apply_backend_serve_defaults(child, _inspection(model_dir))

    injected = set(getattr(child, "_injected_default_flags", set()) or set())
    assert child.temperature == 0.6
    assert "temperature" not in injected
