"""Runtime observability: turbo warmup ladder env (F6b) + operator-override
visibility in the profile env applier/status (F23c)."""

from __future__ import annotations

import mtplx.profiles as profiles
from mtplx.profiles import (
    PROFILE_ENV_USER_OVERRIDE_KEYS,
    apply_profile_env,
    get_profile,
    profile_env_status,
    restore_profile_env,
)

TURBO_LADDER = "512,1024,2048,2560,4096,8192,16384,32768"


# ---------------------------------------------------------------------------
# F6b: the turbo profile carries the warmup ladder the benchmark needs.
# ---------------------------------------------------------------------------


def test_turbo_profile_carries_warmup_ladder() -> None:
    env = get_profile("turbo").env_dict()
    assert env["MTPLX_WARMUP_LADDER"] == TURBO_LADDER
    # Rungs must parse exactly like the server consumer
    # (mtplx.server.openai._warmup_ladder_contexts): positive ints, comma
    # separated, deduped, ordered here so operators can read them.
    rungs = [int(part) for part in env["MTPLX_WARMUP_LADDER"].split(",")]
    assert rungs == sorted(rungs)
    assert len(set(rungs)) == len(rungs)
    assert all(r > 0 for r in rungs)
    # The deepest rung reaches the turbo compiled-verify router fence, so
    # every pow2 KV bucket a compiled benchmark row can touch is walked
    # during warmup, not inside a measured row.
    assert rungs[-1] == int(env["MTPLX_COMPILED_VERIFY_MAX_CONTEXT"])


def test_warmup_ladder_is_operator_overridable() -> None:
    assert "MTPLX_WARMUP_LADDER" in PROFILE_ENV_USER_OVERRIDE_KEYS
    environ = {"MTPLX_WARMUP_LADDER": "512"}
    previous = apply_profile_env("turbo", environ=environ)
    assert environ["MTPLX_WARMUP_LADDER"] == "512"  # operator env wins
    restore_profile_env(previous, environ=environ)
    assert environ["MTPLX_WARMUP_LADDER"] == "512"


def test_other_profiles_do_not_force_the_ladder() -> None:
    # F6 scopes the deep ladder to turbo launches; sustained keeps the
    # server default ("512,2560") by leaving the env unset.
    for name in ("sustained", "stable", "performance-cold", "exact"):
        assert "MTPLX_WARMUP_LADDER" not in get_profile(name).env_dict(), name


# ---------------------------------------------------------------------------
# F23c: operator envs that beat the profile are visible, not silent.
# ---------------------------------------------------------------------------


def test_apply_records_and_prints_operator_overrides(capsys) -> None:
    environ = {"MTPLX_GQA_PACKED_SDPA_THRESHOLD": "4096"}
    apply_profile_env("turbo", environ=environ)
    assert environ["MTPLX_GQA_PACKED_SDPA_THRESHOLD"] == "4096"
    assert profiles.profile_env_overridden == [
        {
            "var": "MTPLX_GQA_PACKED_SDPA_THRESHOLD",
            "profile_value": "8192",
            "actual_value": "4096",
        }
    ]
    out = capsys.readouterr().out
    assert out.count("profile env override:") == 1
    assert "MTPLX_GQA_PACKED_SDPA_THRESHOLD=4096" in out
    assert "operator env wins" in out


def test_equal_value_operator_pin_is_not_an_override(capsys) -> None:
    environ = {"MTPLX_GQA_PACKED_SDPA": "1"}  # same as the turbo value
    apply_profile_env("turbo", environ=environ)
    assert profiles.profile_env_overridden == []
    assert "profile env override:" not in capsys.readouterr().out


def test_override_list_is_rebuilt_per_apply() -> None:
    environ = {"MTPLX_COMPILED_VERIFY_MAX_CONTEXT": "6144"}
    apply_profile_env("turbo", environ=environ)
    assert [entry["var"] for entry in profiles.profile_env_overridden] == [
        "MTPLX_COMPILED_VERIFY_MAX_CONTEXT"
    ]
    apply_profile_env("turbo", environ={})
    assert profiles.profile_env_overridden == []


def test_status_flags_overridden_but_keeps_ok_true() -> None:
    environ = {"MTPLX_COMPILED_VERIFY_MAX_CONTEXT": "6144"}
    apply_profile_env("turbo", environ=environ)
    status = profile_env_status("turbo", environ=environ)
    entry = status["MTPLX_COMPILED_VERIFY_MAX_CONTEXT"]
    assert entry["ok"] is True  # strict startup must keep passing
    assert entry["overridden"] is True
    assert entry["expected"] == "32768"
    assert entry["observed"] == "6144"
    # Non-overridden keys carry the flag as False.
    assert status["MTPLX_NAX_VERIFY"]["overridden"] is False
    assert status["MTPLX_NAX_VERIFY"]["ok"] is True
    # Every entry stays ok — an operator override never fails the launch.
    assert all(value["ok"] for value in status.values())


def test_non_overridable_env_is_stomped_and_not_listed(capsys) -> None:
    # MTPLX_NAX_VERIFY is not in PROFILE_ENV_USER_OVERRIDE_KEYS: the
    # profile stomps it (historical behavior) and the override list stays
    # empty — no false positives.
    environ = {"MTPLX_NAX_VERIFY": "0"}
    apply_profile_env("turbo", environ=environ)
    assert environ["MTPLX_NAX_VERIFY"] == "1"
    assert profiles.profile_env_overridden == []
    assert "profile env override:" not in capsys.readouterr().out
