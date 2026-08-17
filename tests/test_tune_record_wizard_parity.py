"""The quickstart wizard must find the tune record `mtplx tune` saves.

Issue #280: the 2.8.0/2.8.1 wizard rebuilt the tune-state settings dict by
hand (stale profile name, unjoined depths, missing keys), so its lookup hash
never matched the record the tune it had just run saved — users were offered
tuning again on every start. The wizard additionally lost the picked model:
`_tune_requested_model` re-resolved the hardware default because the wizard's
namespace never declared the model explicit, so picking one model tuned a
different one.
"""

from types import SimpleNamespace

import pytest

import mtplx.commands.public as public


FAKE_HW = {"chip": "TestChip", "chip_family": "test", "hw_model": "Test1,1", "machine": "arm64"}
FAKE_SW = {"mtplx_version": "0.0-test", "mlx_version": "0", "mlx_lm_version": "0"}
FAKE_BACKEND = {"mlx_core_path": "/dev/null", "stock_mlx_likely": True}
MODEL = "/models/fake-model"


@pytest.fixture()
def tune_env(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "_apple_hardware_context", lambda: dict(FAKE_HW))
    monkeypatch.setattr(public, "_software_context", lambda: dict(FAKE_SW))
    monkeypatch.setattr(public, "_mlx_backend_context", lambda: dict(FAKE_BACKEND))
    monkeypatch.setattr(
        public, "_resolve_runtime_model_path", lambda model, cache_dir=None: (model, None)
    )
    monkeypatch.setattr(
        public, "_tune_support_payload", lambda model, **kwargs: {"tune_supported": True}
    )
    monkeypatch.setattr(public, "_tune_state_path", lambda: tmp_path / "tuning.json")

    def _no_default(*args, **kwargs):
        raise AssertionError(
            "select_default_model() must not be consulted for an explicit "
            "wizard model (picked-model identity leak)"
        )

    monkeypatch.setattr(public, "select_default_model", _no_default)
    return tmp_path


def test_wizard_lookup_finds_saved_tune_record(tune_env):
    """A record saved under the shared state key is found by the wizard."""
    tune_args = SimpleNamespace(
        command="tune",
        model=MODEL,
        _cli_flags={"model"},
        cache_dir=None,
        profile=None,
        depths=public.TUNE_DEFAULT_DEPTHS,
        max_tokens=public.TUNE_DEFAULT_MAX_TOKENS,
        limit=public.TUNE_DEFAULT_LIMIT,
        seed=public.TUNE_DEFAULT_SEED,
        run_id=None,
        output_dir=None,
        output=None,
        json=False,
        verbose=False,
        dry_run=False,
        no_save=False,
        retune=False,
        unsafe_force_unverified=False,
        yes=True,
    )
    context = public._tune_state_context_for_args(tune_args)
    assert context is not None
    _hw, _sw, _backend, state_key, key_material = context
    public._save_tune_record(
        state_key,
        key_material=key_material,
        payload={"best": {"depth": 2}},
    )

    args = SimpleNamespace(
        _explicit_depth=False,
        cache_dir=None,
        profile=None,
        unsafe_force_unverified=False,
        depth=None,
    )
    public._quickstart_apply_tuned_depth(
        args,
        runtime_model=MODEL,
        target="openwebui",
        can_prompt=False,
    )
    assert args.depth == 2, (
        "wizard lookup key does not match the key the tune save used — "
        "users would be re-offered tuning on every start"
    )


def test_tune_requested_model_honors_explicit_namespace_model(tune_env):
    ns = SimpleNamespace(model=MODEL, _cli_flags={"model"})
    assert public._tune_requested_model(ns) == MODEL
