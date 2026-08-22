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

# 2.8.3: the PRODUCT ladder is the two rungs interactive chat touches
# early. The F6 deep walk (…,16384,32768) burned 30-60+ s of max GPU on
# every user boot for benchmark-row cosmetics (2026-08-17 field
# regression); harnesses that want deep buckets pre-warmed set
# MTPLX_WARMUP_LADDER themselves (operator env wins).
TURBO_LADDER = "512,2560"


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
    # 2.8.3: the product ladder stays SHALLOW — every rung must sit well
    # under the compiled-verify router fence, because walking the fence's
    # every pow2 bucket at boot is benchmark-harness work, not something
    # to bill every user's GPU for (2026-08-17 field regression).
    assert rungs[-1] <= 2560
    assert rungs[-1] < int(env["MTPLX_COMPILED_VERIFY_MAX_CONTEXT"])


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
    # profile stomps it and the override list stays empty — no false
    # positives. Since the turbo-truth audit the stomp itself is LOUD
    # (one line), never silent.
    environ = {"MTPLX_NAX_VERIFY": "0"}
    apply_profile_env("turbo", environ=environ)
    assert environ["MTPLX_NAX_VERIFY"] == "1"
    assert profiles.profile_env_overridden == []
    out = capsys.readouterr().out
    assert "profile env override:" not in out
    assert out.count("profile env stomp:") == 1
    assert "MTPLX_NAX_VERIFY=0 replaced by profile turbo value 1" in out


# ---------------------------------------------------------------------------
# Turbo-truth audit (2026-08-21): the batched/lazy target-distribution pair.
# ---------------------------------------------------------------------------


def test_profiles_do_not_claim_the_batched_lane_the_lazy_gate_kills() -> None:
    # generation.py only builds batched target distributions when the lazy
    # strategy is off; a profile setting both to "1" is self-contradictory
    # (shipped that way 1.0.0 -> 2.9.0, PR #314). The profile must express
    # the strategy that actually runs: lazy on, batched off.
    for name in ("turbo", "sustained", "performance-cold"):
        env = get_profile(name).env_dict()
        assert env["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "1", name
        assert env["MTPLX_BATCH_TARGET_ARRAYS"] == "0", name


def test_batched_lane_ab_arm_is_operator_launchable(capsys) -> None:
    # PR #314's measured arm: lazy off + batched on, exported before launch.
    # Both keys must survive the profile applier and be announced.
    assert "MTPLX_LAZY_TARGET_DISTRIBUTIONS" in PROFILE_ENV_USER_OVERRIDE_KEYS
    assert "MTPLX_BATCH_TARGET_ARRAYS" in PROFILE_ENV_USER_OVERRIDE_KEYS
    environ = {
        "MTPLX_LAZY_TARGET_DISTRIBUTIONS": "0",
        "MTPLX_BATCH_TARGET_ARRAYS": "1",
    }
    apply_profile_env("turbo", environ=environ)
    assert environ["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "0"
    assert environ["MTPLX_BATCH_TARGET_ARRAYS"] == "1"
    assert sorted(entry["var"] for entry in profiles.profile_env_overridden) == [
        "MTPLX_BATCH_TARGET_ARRAYS",
        "MTPLX_LAZY_TARGET_DISTRIBUTIONS",
    ]
    out = capsys.readouterr().out
    assert out.count("profile env override:") == 2
    # lazy=0 means the batched flag is live, not gated: no dead-flag line.
    assert "env gated at runtime:" not in out
    status = profile_env_status("turbo", environ=environ)
    assert status["MTPLX_LAZY_TARGET_DISTRIBUTIONS"]["ok"] is True
    assert status["MTPLX_BATCH_TARGET_ARRAYS"]["ok"] is True
    assert status["MTPLX_LAZY_TARGET_DISTRIBUTIONS"]["overridden"] is True
    assert status["MTPLX_BATCH_TARGET_ARRAYS"]["overridden"] is True


def test_runtime_gated_env_combo_is_announced_loudly(capsys) -> None:
    # An operator (or stale launcher config) re-creating the dead pair gets
    # one loud line naming the dead flag and its gate — never silence.
    environ = {"MTPLX_BATCH_TARGET_ARRAYS": "1"}
    apply_profile_env("turbo", environ=environ)  # profile keeps lazy=1
    assert environ["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "1"
    assert environ["MTPLX_BATCH_TARGET_ARRAYS"] == "1"  # override honored
    out = capsys.readouterr().out
    assert out.count("env gated at runtime:") == 1
    assert "MTPLX_BATCH_TARGET_ARRAYS=1 has no effect" in out
    assert "MTPLX_LAZY_TARGET_DISTRIBUTIONS=1" in out


def test_coding_agent_lane_bonus_verify_pin_is_announced(capsys) -> None:
    # The app/CLI coding-agent lanes inject MTPLX_LAZY_BONUS_VERIFY=1 into
    # the daemon env while every product profile runs the lazy-distribution
    # strategy that disables it (generation.py records
    # disabled_by=lazy_target_distributions per event, which nobody reads).
    # Serve startup must say it out loud instead.
    environ = {"MTPLX_LAZY_BONUS_VERIFY": "1"}
    apply_profile_env("sustained", environ=environ)
    out = capsys.readouterr().out
    assert out.count("env gated at runtime:") == 1
    assert "MTPLX_LAZY_BONUS_VERIFY=1 has no effect" in out


def test_profile_defaults_emit_no_gated_env_lines(capsys) -> None:
    # The shipped profiles alone must be contradiction-free.
    for name in ("turbo", "sustained", "performance-cold", "stable", "exact"):
        apply_profile_env(name, environ={})
    assert "env gated at runtime:" not in capsys.readouterr().out
