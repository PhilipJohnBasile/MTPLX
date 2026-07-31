from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.dflash_source as source_module
from mtplx.dflash_source import load_dflash_draft


@pytest.fixture(autouse=True)
def _force_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _FakeCache:
    pass


class _FakePortDraft:
    target_layer_ids = (0, 2)
    block_size = 4
    mask_token_id = 7
    layers = (object(),)

    def __call__(self, *, noise_embedding, target_hidden, cache):
        del target_hidden
        assert len(cache) == 1
        return noise_embedding + 1.0


class _Embed:
    def __call__(self, tokens):
        return tokens.astype(mx.float32)[..., None]


class _Head:
    def __call__(self, hidden):
        return mx.concatenate([hidden, -hidden], axis=-1)


def _install_fake_port(monkeypatch):
    original_import = importlib.import_module

    def fake_import(name):
        if name == "dflash_mlx.runtime":
            return SimpleNamespace(
                load_draft_bundle=lambda path: (
                    _FakePortDraft(),
                    {"resolved_model_ref": str(path)},
                )
            )
        if name == "dflash_mlx.model":
            return SimpleNamespace(ContextOnlyDraftKVCache=_FakeCache)
        return original_import(name)

    monkeypatch.setattr(source_module.importlib, "import_module", fake_import)


def test_competitors_extra_pins_the_supported_dflash_mlx_port() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirements = project["project"]["optional-dependencies"]["competitors"]

    assert "dflash-mlx==0.1.0" in requirements


def test_installed_competitors_extra_exposes_adapter_api() -> None:
    try:
        runtime = importlib.import_module("dflash_mlx.runtime")
        model = importlib.import_module("dflash_mlx.model")
    except ImportError:
        pytest.skip("install mtplx[competitors] to exercise the optional API")

    assert callable(runtime.load_draft_bundle)
    assert callable(model.ContextOnlyDraftKVCache)


def test_port_adapter_loads_local_artifact_and_uses_shared_target_head(
    monkeypatch, tmp_path
) -> None:
    _install_fake_port(monkeypatch)
    (tmp_path / "config.json").write_text('{"model_type":"dflash"}\n')

    draft, artifact = load_dflash_draft(str(tmp_path))
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=_Embed()),
        lm_head=_Head(),
    )
    draft.bind(target)
    cache = draft.make_cache()
    logits = draft(
        mx.array([[1, 2, 3]]),
        mx.zeros((1, 1, 4)),
        cache,
        logits_start=1,
    )
    mx.eval(logits)

    assert tuple(logits.shape) == (1, 2, 2)
    assert artifact["source_kind"] == "local"
    assert artifact["immutable_revision"] is False
    assert artifact["artifact_layout_sha256"].startswith("layout-sha256:")


def test_hub_revision_is_resolved_and_loaded_by_immutable_sha(
    monkeypatch, tmp_path
) -> None:
    _install_fake_port(monkeypatch)
    (tmp_path / "config.json").write_text('{"model_type":"dflash"}\n')
    resolved_sha = "a" * 40
    calls = {}

    import huggingface_hub

    class _Api:
        def model_info(self, model_ref, revision=None):
            calls["model_info"] = (model_ref, revision)
            return SimpleNamespace(sha=resolved_sha)

    def snapshot_download(model_ref, *, revision, allow_patterns):
        calls["snapshot"] = (model_ref, revision, tuple(allow_patterns))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    _draft, artifact = load_dflash_draft(
        "org/draft",
        revision="candidate-tag",
    )

    assert calls["model_info"] == ("org/draft", "candidate-tag")
    assert calls["snapshot"][1] == resolved_sha
    assert artifact["requested_revision"] == "candidate-tag"
    assert artifact["resolved_revision"] == resolved_sha
    assert artifact["immutable_revision"] is True
