from __future__ import annotations

import inspect
from types import SimpleNamespace

import mlx.core as mx
from mtplx.models.qwen4_exp import Qwen4ExpMTP, TextModel
from mtplx.profiles import MODEL_RUNTIME_ENV_OVERRIDE_KEYS


class _TinyPrepare:
    def __init__(self) -> None:
        self._hc = 2
        self.pre_fc_norm_hidden = SimpleNamespace(weight=mx.zeros((8,), dtype=mx.bfloat16))
        self.pre_fc_norm_embedding = SimpleNamespace(weight=mx.zeros((4,), dtype=mx.bfloat16))
        self._mtp_prepare_inputs_impl = self._prepare_inputs_eager

    def _prepare_inputs_eager(self, widened, token_embedding):
        return widened + mx.concatenate([token_embedding, token_embedding], axis=-1)


def test_compiled_prepare_installs_only_after_exact_shape_parity():
    prepare = _TinyPrepare()

    report = Qwen4ExpMTP.install_compiled_prepare(prepare)
    widened = mx.arange(8).reshape(1, 1, 8).astype(mx.bfloat16)
    embedding = mx.arange(4).reshape(1, 1, 4).astype(mx.bfloat16)
    expected = prepare._prepare_inputs_eager(widened, embedding)
    actual = prepare._mtp_prepare_inputs_impl(widened, embedding)
    mx.eval(expected, actual)

    assert report["installed"] is True
    assert report["shape"] == [1, 1, 8]
    assert bool(mx.array_equal(actual, expected).item())


def test_history_updates_keep_the_uncompiled_prepare_route():
    source = inspect.getsource(TextModel.mtp_update_cache)

    assert "fuse_and_run_history" in source
    assert "fuse_and_run(" not in source


def test_compiled_prepare_is_a_construction_environment_override():
    assert "MTPLX_QWEN4_COMPILED_MTP_PREPARE" in MODEL_RUNTIME_ENV_OVERRIDE_KEYS
