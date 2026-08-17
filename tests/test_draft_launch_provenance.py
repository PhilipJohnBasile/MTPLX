"""Launch-side draft-sampler provenance: the plumbing is honest (F25 + F2).

- The artifact's recommended_draft_sampler survives a launch-profile
  mismatch by falling back to the top-level mtplx_runtime.json metadata,
  exactly like the measured depth default already does.
- Family defaults injected by _apply_backend_serve_defaults are RECORDED
  (injected-default provenance flags) so they reach the daemon when no
  contract/profile spec exists — and NEVER clobber an artifact stamp or
  pin the per-family curve.
- A daemon started directly with --draft-temperature and no
  --draft-sampler-source gets explicit-flag provenance (pins); launcher
  handoffs always stamp the source and injected defaults stay unpinned.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from mtplx.commands import public
from mtplx.commands.public import (
    _apply_backend_serve_defaults,
    _explicit_draft_sampler_override,
    _model_draft_sampler_spec,
    _profile_draft_sampler_spec,
    get_profile,
)
from mtplx.cli import build_parser
from mtplx.server import openai as server_openai

from test_public_cli import _serve_dry_run_payload_for_model  # noqa: E402


def _inspection(model_dir, *, recommended_profile="sustained", extra=None):
    contract = {
        "arch_id": "qwen3-next-mtp",
        "mtp_depth_max": 3,
        "recommended_profile": recommended_profile,
    }
    contract.update(extra or {})
    return {
        "model_dir": str(model_dir),
        "recommended_backend": "qwen3_next",
        "runtime_compatibility": "native-contract-gated",
        "compatibility": {
            "can_run": True,
            "exit_code": 0,
            "runtime_contract": contract,
        },
    }


# ---------------------------------------------------------------------------
# F25 — artifact metadata fallback on profile mismatch (mirror of depth)
# ---------------------------------------------------------------------------


def test_profile_mismatch_keeps_artifact_draft_sampler_stamp(tmp_path):
    model_dir = tmp_path / "Qwen3.6-27B-Custom"
    model_dir.mkdir()
    (model_dir / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "recommended_profile": "sustained",
                "recommended_draft_sampler": {
                    "temperature": 0.55,
                    "top_p": 0.9,
                    "top_k": 30,
                },
            }
        )
    )
    inspection = _inspection(model_dir)
    # Launch profile != recommended_profile hides the typed contract; the
    # stamp is a property of the ARTIFACT and must survive.
    spec = _model_draft_sampler_spec(inspection, get_profile("turbo"))
    assert spec == {"temperature": 0.55, "top_p": 0.9, "top_k": 30}


def test_profile_mismatch_without_metadata_falls_back_to_profile(tmp_path):
    model_dir = tmp_path / "Qwen3.6-27B-Custom"
    model_dir.mkdir()
    profile = get_profile("turbo")
    inspection = _inspection(model_dir)
    assert _model_draft_sampler_spec(inspection, profile) == (
        _profile_draft_sampler_spec(profile)
    )


def test_profile_mismatch_with_malformed_metadata_degrades_to_profile(tmp_path):
    model_dir = tmp_path / "Qwen3.6-27B-Custom"
    model_dir.mkdir()
    (model_dir / "mtplx_runtime.json").write_text(
        json.dumps({"recommended_draft_sampler": {"temperature": -1.0}})
    )
    profile = get_profile("turbo")
    inspection = _inspection(model_dir)
    # Artifact metadata is fail-safe: malformed stamps degrade, not crash.
    assert _model_draft_sampler_spec(inspection, profile) == (
        _profile_draft_sampler_spec(profile)
    )


# ---------------------------------------------------------------------------
# F2 companion — injected family defaults are recorded, subordinate, unpinned
# ---------------------------------------------------------------------------


def _serve_args(extra=()):
    args = build_parser().parse_args(["serve", "--model", "m", *extra])
    return args


def test_backend_serve_defaults_record_injected_draft_flags(tmp_path):
    model_dir = tmp_path / "Qwen3.6-27B-Custom"
    model_dir.mkdir()
    args = _serve_args()
    _apply_backend_serve_defaults(args, _inspection(model_dir))
    injected = set(getattr(args, "_injected_default_flags", set()) or set())
    assert {"draft-temperature", "draft-top-p", "draft-top-k"} <= injected
    assert args.draft_temperature is not None


def test_injected_defaults_fill_the_gap_but_never_clobber_a_stamp(tmp_path):
    model_dir = tmp_path / "Qwen3.6-27B-Custom"
    model_dir.mkdir()
    args = _serve_args()
    _apply_backend_serve_defaults(args, _inspection(model_dir))

    # No contract/profile spec: the recorded injected defaults flow.
    filled = _explicit_draft_sampler_override(args, None)
    assert filled is not None
    assert filled["temperature"] == args.draft_temperature

    # A stamped spec exists: injected defaults defer to it entirely.
    stamp = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}
    assert _explicit_draft_sampler_override(args, stamp) is None

    # A user-typed flag still beats the stamp.
    typed = _serve_args(["--draft-temperature", "0.33"])
    _apply_backend_serve_defaults(typed, _inspection(model_dir))
    override = _explicit_draft_sampler_override(typed, stamp)
    assert override is not None
    assert override["temperature"] == 0.33


def test_family_default_launch_ships_draft_flags_unpinned(
    monkeypatch, tmp_path, capsys
):
    """A model with no recommended_draft_sampler used to launch the daemon
    with NO draft flags at all (the injected launch defaults died at the
    argv gate) — the daemon then silently target-mirrored while launch
    config said 0.6. The injected family defaults now reach the daemon as
    an unpinned curve anchor."""

    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    model_dir = tmp_path / "Qwen3.6-27B-Custom"
    model_dir.mkdir()
    payload = _serve_dry_run_payload_for_model(monkeypatch, capsys, model_dir)
    command = payload["server_command"]
    assert "--draft-temperature 0.6" in command
    assert "--draft-top-p 0.95" in command
    assert "--draft-top-k 20" in command
    assert "--draft-sampler-source default" in command


# ---------------------------------------------------------------------------
# F17 — direct daemon --draft-temperature pins like any explicit flag
# ---------------------------------------------------------------------------


def _daemon_args(argv):
    return server_openai.parse_args(["--warmup-tokens", "0", *argv])


def test_direct_daemon_draft_flag_pins_without_source():
    args = _daemon_args(["--draft-temperature", "0.5"])
    draft = SimpleNamespace(temperature=0.5)
    assert server_openai._launch_draft_sampler_pinned(args, draft) is True


def test_launcher_stamped_default_source_does_not_pin():
    args = _daemon_args(
        ["--draft-temperature", "0.5", "--draft-sampler-source", "default"]
    )
    draft = SimpleNamespace(temperature=0.5)
    assert server_openai._launch_draft_sampler_pinned(args, draft) is False


def test_explicit_source_pins_and_no_draft_sampler_never_pins():
    explicit = _daemon_args(
        ["--draft-temperature", "0.5", "--draft-sampler-source", "explicit"]
    )
    assert (
        server_openai._launch_draft_sampler_pinned(
            explicit, SimpleNamespace(temperature=0.5)
        )
        is True
    )
    bare = _daemon_args([])
    assert server_openai._launch_draft_sampler_pinned(bare, None) is False
