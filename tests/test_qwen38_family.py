"""Qwen3.8 family contract: official sampler, reasoning effort, preserved thinking.

Qwen3.8-27B shares the qwen3_next lane with Qwen3.6/3.5 but ships its own
inference contract (model card, 2026-08-14): thinking-mode sampler
temperature=1.0/top_p=0.95/top_k=20, reasoning_effort levels
xhigh/medium/low, and preserve_thinking on by default for all workloads.
Upstream defaults to xhigh; MTPLX's measured coding default is medium. These
tests pin that family-scoped resolution and — just as deliberately — that the
qwen3_5/qwen3_6 behavior is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.backends.descriptors import (
    QWEN3_NEXT_DESCRIPTOR,
    descriptor_for_model,
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


def test_qwen38_reasoning_effort_levels_and_product_default() -> None:
    codec = reasoning_policy_for_model(BARE_SPEED, None, QWEN3_NEXT_DESCRIPTOR)
    assert codec.effort_levels == ("xhigh", "medium", "low")
    assert codec.default_effort == "medium"
    assert codec.parser == "qwen3"


def test_qwen36_reasoning_codec_unchanged() -> None:
    codec = reasoning_policy_for_model(V2_36, None, QWEN3_NEXT_DESCRIPTOR)
    assert codec.effort_levels == ()
    assert codec.default_effort is None


def test_qwen38_draft_range_capped_at_d3_dropday() -> None:
    semantics = draft_semantics_for_model(BARE_SPEED, None, QWEN3_NEXT_DESCRIPTOR)
    assert semantics.default == 3
    assert semantics.maximum == 3  # capped drop-day: D4 live lane crash (see QWEN3_8_DRAFT_SEMANTICS)
    tune = tune_policy_for_model(BARE_SPEED, None, QWEN3_NEXT_DESCRIPTOR)
    assert tune.supported
    assert tune.candidates == ("AR", "D1", "D2", "D3")  # drop-day cap


def test_qwen38_public_depth_and_tune_validation_follow_model_controls() -> None:
    from mtplx.commands import public

    qwen38_args = SimpleNamespace(model=BARE_SPEED, model_id=None)
    qwen38_support = public._tune_support_payload(BARE_SPEED)
    assert public._public_depth_ceiling(qwen38_args) == 3  # drop-day cap, D4 crash receipt
    assert public._parse_tune_candidate_values(
        "1,3", support_payload=qwen38_support
    ) == [1, 3]
    with pytest.raises(ValueError, match="one of 1,2,3"):
        public._parse_tune_candidate_values(
            "6", support_payload=qwen38_support
        )

    qwen36_args = SimpleNamespace(model=V2_36, model_id=None)
    qwen36_support = public._tune_support_payload(V2_36)
    assert public._public_depth_ceiling(qwen36_args) == 3
    with pytest.raises(ValueError, match="one of 1,2,3"):
        public._parse_tune_candidate_values(
            "4", support_payload=qwen36_support
        )


def test_qwen38_tune_sampling_uses_family_defaults_unless_explicit() -> None:
    from mtplx.commands import public

    support = public._tune_support_payload(BARE_SPEED)
    args = SimpleNamespace(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        _cli_flags=set(),
    )
    public._apply_tune_sampling_defaults(args, support)
    assert (args.temperature, args.top_p, args.top_k) == (1.0, 0.95, 20)

    explicit = SimpleNamespace(
        temperature=0.7,
        top_p=0.8,
        top_k=10,
        _cli_flags={"temperature", "top-p", "top-k"},
    )
    public._apply_tune_sampling_defaults(explicit, support)
    assert (explicit.temperature, explicit.top_p, explicit.top_k) == (
        0.7,
        0.8,
        10,
    )


def test_qwen38_serve_defaults_use_official_template_and_sampler() -> None:
    from mtplx.commands import public

    args = SimpleNamespace(
        model=BARE_SPEED,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        draft_temperature=0.6,
        draft_top_p=None,
        draft_top_k=20,
        depth=3,
        reasoning=None,
        reasoning_parser="qwen3",
        reasoning_effort=None,
        tool_prompt_mode="hybrid",
        chat_template_profile="local_qwen36",
        chat_template_path=None,
        adaptive_policy="none",
        _cli_flags=set(),
    )
    inspection = {
        "model_dir": BARE_SPEED,
        "recommended_backend": "qwen3_next",
    }

    public._apply_backend_serve_defaults(args, inspection)

    assert (args.temperature, args.top_p, args.top_k) == (1.0, 0.95, 20)
    assert (args.draft_temperature, args.draft_top_p, args.draft_top_k) == (
        1.0,  # QWEN3_8_DRAFT_TEMPERATURE, strict max-fan A/B winner
        0.95,
        20,
    )
    assert args.reasoning_effort == "medium"
    assert args.chat_template_profile == "tokenizer"


def test_qwen36_draft_range_unchanged() -> None:
    assert draft_semantics_for_model(V2_36, None, QWEN3_NEXT_DESCRIPTOR).maximum == 3


def test_qwen38_model_controls_payload() -> None:
    controls = model_controls_for_descriptor(
        QWEN3_NEXT_DESCRIPTOR, model_ref=BARE_SPEED
    )
    assert controls["model_family"] == "qwen3_8"
    assert controls["sampling"]["temperature"] == 1.0
    assert controls["reasoning"]["effort_levels"] == ["xhigh", "medium", "low"]
    assert controls["reasoning"]["default_effort"] == "medium"
    assert controls["draft_control"]["maximum"] == 3  # drop-day cap


def test_qwen38_resolved_descriptor_matches_model_controls() -> None:
    descriptor = descriptor_for_model(QWEN3_NEXT_DESCRIPTOR, model_ref=BARE_SPEED)
    assert descriptor.sampler_defaults.temperature == 1.0
    assert descriptor.reasoning_codec.default_effort == "medium"
    assert descriptor.draft_semantics.maximum == 3  # drop-day cap
    assert descriptor.tune_policy.candidates[-1] == "D3"  # drop-day cap

    legacy = descriptor_for_model(QWEN3_NEXT_DESCRIPTOR, model_ref=V2_36)
    assert legacy == QWEN3_NEXT_DESCRIPTOR


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
        srv._reasoning_effort_for_state(state, thinking_enabled=True) == "medium"
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


def test_qwen38_server_descriptor_and_request_validation_reach_d3_cap() -> None:
    from mtplx.server import openai as srv

    state = _state(BARE_SPEED, depth=3, generation_mode="mtp")
    descriptor = srv._backend_descriptor(state)
    assert descriptor.draft_semantics.maximum == 3  # drop-day cap
    assert descriptor.sampler_defaults.temperature == 1.0
    request = srv.ChatCompletionRequest(model="m", messages=[], depth=3)
    assert srv._request_depth_for_generation(
        state, request, generation_mode="mtp"
    ) == 3
    # Depth 4+ live serving killed the daemon on drop day (memory-kill
    # signature); the family cap rejects it until the deep lane is fixed.
    rejected = srv.ChatCompletionRequest(model="m", messages=[], depth=4)
    with pytest.raises(srv.HTTPException, match="between 1 and 3"):
        srv._request_depth_for_generation(
            state, rejected, generation_mode="mtp"
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
