"""Qwen3.8 family contract: official sampler, reasoning effort, preserved thinking.

Qwen3.8-27B shares the qwen3_next lane with Qwen3.6/3.5 but ships its own
inference contract (model card, 2026-08-14): thinking-mode sampler
temperature=1.0/top_p=0.95/top_k=20, reasoning_effort levels
xhigh (default)/medium/low, and preserve_thinking on by default for all
workloads. These tests pin the family-scoped resolution added for the drop
and — just as deliberately — that the qwen3_5/qwen3_6 behavior is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.backends.descriptors import (
    QWEN3_NEXT_DESCRIPTOR,
    draft_semantics_for_model,
    model_controls_for_descriptor,
    model_family_from_inspection,
    reasoning_policy_for_model,
    sampler_defaults_for_model,
    tune_policy_for_model,
)
from mtplx.default_models import public_model_id_for_ref
from mtplx.profiles import (
    QWEN38_BARE_SPEED_HF_MODEL_ID,
    QWEN38_BARE_SPEED_PUBLIC_MODEL_ID,
    QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID,
    QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
)

BARE_SPEED = QWEN38_BARE_SPEED_HF_MODEL_ID
OFFICIAL = "Qwen/Qwen3.8-27B"
V2_36 = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2"


# ---------------------------------------------------------------- family sniff


@pytest.mark.parametrize(
    "ref",
    [
        BARE_SPEED,
        QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID,
        QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID,
        OFFICIAL,
        "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed",
        "mtplx-qwen38-27b-bare-speed",
    ],
)
def test_qwen38_family_detected(ref: str) -> None:
    assert (
        model_family_from_inspection(
            model_ref=ref, descriptor=QWEN3_NEXT_DESCRIPTOR
        )
        == "qwen3_8"
    )


def test_qwen36_family_unchanged() -> None:
    assert (
        model_family_from_inspection(
            model_ref=V2_36, descriptor=QWEN3_NEXT_DESCRIPTOR
        )
        == "qwen3_6"
    )


# ------------------------------------------------------------- family policies


def test_qwen38_official_thinking_sampler() -> None:
    sampler = sampler_defaults_for_model(BARE_SPEED, None, QWEN3_NEXT_DESCRIPTOR)
    assert sampler.temperature == 1.0
    assert sampler.top_p == 0.95
    assert sampler.top_k == 20


def test_qwen36_sampler_unchanged() -> None:
    sampler = sampler_defaults_for_model(V2_36, None, QWEN3_NEXT_DESCRIPTOR)
    assert (sampler.temperature, sampler.top_p, sampler.top_k) == (0.6, 0.95, 20)


def test_qwen38_reasoning_effort_levels() -> None:
    codec = reasoning_policy_for_model(BARE_SPEED, None, QWEN3_NEXT_DESCRIPTOR)
    assert codec.effort_levels == ("xhigh", "medium", "low")
    assert codec.default_effort == "xhigh"
    assert codec.parser == "qwen3"


def test_qwen36_reasoning_codec_unchanged() -> None:
    codec = reasoning_policy_for_model(V2_36, None, QWEN3_NEXT_DESCRIPTOR)
    assert codec.effort_levels == ()
    assert codec.default_effort is None


def test_qwen38_draft_range_extends_to_d6() -> None:
    semantics = draft_semantics_for_model(BARE_SPEED, None, QWEN3_NEXT_DESCRIPTOR)
    assert semantics.default == 3
    assert semantics.maximum == 6
    tune = tune_policy_for_model(BARE_SPEED, None, QWEN3_NEXT_DESCRIPTOR)
    assert tune.supported
    assert tune.candidates == ("AR", "D1", "D2", "D3", "D4", "D5", "D6")


def test_qwen36_draft_range_unchanged() -> None:
    assert draft_semantics_for_model(V2_36, None, QWEN3_NEXT_DESCRIPTOR).maximum == 3


def test_qwen38_model_controls_payload() -> None:
    controls = model_controls_for_descriptor(
        QWEN3_NEXT_DESCRIPTOR, model_ref=BARE_SPEED
    )
    assert controls["model_family"] == "qwen3_8"
    assert controls["sampling"]["temperature"] == 1.0
    assert controls["reasoning"]["effort_levels"] == ["xhigh", "medium", "low"]
    assert controls["reasoning"]["default_effort"] == "xhigh"
    assert controls["draft_control"]["maximum"] == 6


# ------------------------------------------------------- public id resolution


@pytest.mark.parametrize(
    ("ref", "public_id"),
    [
        (QWEN38_BARE_SPEED_HF_MODEL_ID, QWEN38_BARE_SPEED_PUBLIC_MODEL_ID),
        (QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID, QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID),
        (
            QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID,
            QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID,
        ),
        (QWEN38_BARE_SPEED_PUBLIC_MODEL_ID, QWEN38_BARE_SPEED_PUBLIC_MODEL_ID),
        (
            "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed",
            QWEN38_BARE_SPEED_PUBLIC_MODEL_ID,
        ),
    ],
)
def test_qwen38_public_model_id_resolution(ref: str, public_id: str) -> None:
    assert public_model_id_for_ref(ref) == public_id


def test_qwen38_derivative_names_fall_through() -> None:
    # The V3-RC lesson: name extensions must NOT inherit a first-party id.
    assert (
        public_model_id_for_ref("Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed-RC1")
        != QWEN38_BARE_SPEED_PUBLIC_MODEL_ID
    )


def test_qwen38_turbo_default_promotion() -> None:
    from mtplx.commands.public import (
        _TURBO_DEFAULT_PUBLIC_MODEL_IDS,
        _apply_model_default_profile,
    )

    for public_id in (
        QWEN38_BARE_SPEED_PUBLIC_MODEL_ID,
        QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
        QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID,
    ):
        assert public_id in _TURBO_DEFAULT_PUBLIC_MODEL_IDS
    args = SimpleNamespace(profile="sustained", _cli_flags=set())
    assert _apply_model_default_profile(args, QWEN38_BARE_SPEED_PUBLIC_MODEL_ID)
    assert args.profile == "turbo"
    # An explicit --profile flag still wins.
    pinned = SimpleNamespace(profile="sustained", _cli_flags={"profile"})
    assert not _apply_model_default_profile(pinned, QWEN38_BARE_SPEED_PUBLIC_MODEL_ID)
    assert pinned.profile == "sustained"


# ------------------------------------------------------------- server behavior


def _state(model_ref: str, **arg_overrides: object) -> SimpleNamespace:
    args = SimpleNamespace(
        model=model_ref,
        reasoning_effort=None,
        preserve_thinking="auto",
        strip_assistant_reasoning_history=False,
        enable_thinking=True,
        reasoning_parser="qwen3",
    )
    for key, value in arg_overrides.items():
        setattr(args, key, value)
    return SimpleNamespace(
        args=args,
        backend_descriptor=QWEN3_NEXT_DESCRIPTOR,
        model_id=model_ref,
        reasoning_history_scoped_capable=True,
    )


def test_reasoning_effort_resolves_for_qwen38_state() -> None:
    from mtplx.server import openai as srv

    state = _state(BARE_SPEED)
    assert (
        srv._reasoning_effort_for_state(state, thinking_enabled=True) == "xhigh"
    )
    assert (
        srv._reasoning_effort_for_state(
            state, thinking_enabled=True, request_effort="low"
        )
        == "low"
    )
    assert (
        srv._reasoning_effort_for_state(state, thinking_enabled=False) is None
    )


def test_reasoning_effort_still_none_for_qwen36_state() -> None:
    from mtplx.server import openai as srv

    state = _state(V2_36)
    assert srv._reasoning_effort_for_state(state, thinking_enabled=True) is None


def test_normalize_reasoning_effort_accepts_xhigh() -> None:
    from mtplx.server import openai as srv

    assert srv._normalize_reasoning_effort("xhigh") == "xhigh"
    with pytest.raises(ValueError):
        srv._normalize_reasoning_effort("ultra")


def test_reasoning_history_auto_preserves_for_qwen38() -> None:
    from mtplx.server import openai as srv

    assert srv._reasoning_history_mode(_state(BARE_SPEED)) == "preserve"
    # 3.6 keeps its scoped rolling-checkpoint resolution.
    assert srv._reasoning_history_mode(_state(V2_36)) == "scoped"
    # An operator's explicit choice always wins.
    assert (
        srv._reasoning_history_mode(_state(BARE_SPEED, preserve_thinking="scoped"))
        == "scoped"
    )
    assert (
        srv._reasoning_history_mode(_state(BARE_SPEED, preserve_thinking="off"))
        == "strip"
    )


def test_chat_template_kwargs_enable_thinking_shim() -> None:
    from mtplx.server import openai as srv

    state = _state(BARE_SPEED)
    card_style = srv.ChatCompletionRequest(
        model="m", messages=[], chat_template_kwargs={"enable_thinking": False}
    )
    assert srv._thinking_enabled_for_request(state, card_style) is False
    plain = srv.ChatCompletionRequest(model="m", messages=[])
    assert srv._thinking_enabled_for_request(state, plain) is True
    # Top-level field wins over the template-kwargs spelling.
    both = srv.ChatCompletionRequest(
        model="m",
        messages=[],
        enable_thinking=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert srv._thinking_enabled_for_request(state, both) is True


def test_anthropic_translation_carries_chat_template_kwargs() -> None:
    from mtplx.server import openai as srv

    request = srv.AnthropicMessagesRequest(
        model="m",
        max_tokens=64,
        messages=[{"role": "user", "content": "hi"}],
        chat_template_kwargs={"enable_thinking": False},
    )
    translated = srv._anthropic_to_chat_request(request)
    assert srv._request_chat_template_kwargs(translated) == {
        "enable_thinking": False
    }
    state = _state(BARE_SPEED)
    assert srv._thinking_enabled_for_request(state, translated) is False
