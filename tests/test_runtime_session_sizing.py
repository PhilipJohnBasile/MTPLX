"""Session-bank sizing: nested safetensors layouts and floor visibility (F16-S).

``model_weights_bytes`` feeds the model-aware auto budget (half the post-model
RAM surplus). A non-recursive scan missed nested layouts — the shipped
``mtp/weights.safetensors`` sidecar (mtplx/artifacts.py) and wrapper dirs
whose shards live below a subdirectory — silently disabling or skewing the
budget the founder ruling 2026-07-05 depends on. And when the 1 GiB floor
engages, it must announce itself with the computed numbers instead of leaving
users to read a starved warm cache as "the cache broke".
"""

from __future__ import annotations

from mtplx import engine_session
from mtplx.engine_session import model_weights_bytes


def _write(path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


def test_model_weights_bytes_counts_nested_shards(tmp_path):
    root = tmp_path / "model"
    _write(root / "model-00001-of-00002.safetensors", 1000)
    _write(root / "model-00002-of-00002.safetensors", 500)
    # The shipped nested sidecar layout (artifacts.py: "mtp/weights.safetensors").
    _write(root / "mtp" / "weights.safetensors", 250)
    # Deeper nesting must count too.
    _write(root / "snapshots" / "ab12" / "model.safetensors", 125)
    (root / "config.json").write_text("{}")

    assert model_weights_bytes(root) == 1875


def test_model_weights_bytes_sees_nested_only_layout(tmp_path):
    # Wrapper dir with no top-level shards: the old scan returned None here,
    # silently swapping the RAM-aware auto budget for the legacy flat default.
    root = tmp_path / "wrapper"
    _write(root / "snapshots" / "rev" / "model.safetensors", 4096)

    assert model_weights_bytes(root) == 4096


def test_model_weights_bytes_unknown_cases_stay_none(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert model_weights_bytes(empty) is None
    assert model_weights_bytes(tmp_path / "missing") is None
    file_path = tmp_path / "plain.txt"
    file_path.write_text("not a dir")
    assert model_weights_bytes(file_path) is None


def test_auto_budget_floor_announces_computed_numbers(monkeypatch, capsys):
    gib = 1024**3
    monkeypatch.setattr(
        engine_session,
        "_detect_total_ram_bytes_for_session_bank",
        lambda: 32 * gib,
    )
    monkeypatch.setattr(engine_session, "_auto_floor_announced", False)

    # Model appears larger than RAM: surplus <= 0 engages the 1 GiB floor.
    budget = engine_session._auto_session_bank_max_bytes(33 * gib)

    assert budget == engine_session._AUTO_BUDGET_FLOOR_BYTES
    out = capsys.readouterr().out
    assert "session-bank auto budget floored" in out
    assert "total_ram=32.0G" in out
    assert "model_weights=33.0G" in out
    assert "MTPLX_SESSION_BANK_MAX_BYTES" in out

    # Once per process: a second floor engagement stays quiet.
    engine_session._auto_session_bank_max_bytes(40 * gib)
    assert "session-bank auto budget floored" not in capsys.readouterr().out


def test_auto_budget_floor_announces_on_sub_floor_surplus(monkeypatch, capsys):
    gib = 1024**3
    monkeypatch.setattr(
        engine_session,
        "_detect_total_ram_bytes_for_session_bank",
        lambda: 32 * gib,
    )
    monkeypatch.setattr(engine_session, "_auto_floor_announced", False)

    # Positive surplus whose half is below 1 GiB also lands on the floor.
    budget = engine_session._auto_session_bank_max_bytes(int(31.5 * gib))

    assert budget == engine_session._AUTO_BUDGET_FLOOR_BYTES
    assert "session-bank auto budget floored" in capsys.readouterr().out


def test_budget_line_distinguishes_legacy_fallback_from_explicit(
    monkeypatch, capsys
):
    # No env override and no model size: auto sizing cannot engage; the
    # console line must say the legacy default applied, not claim the user
    # set an explicit budget.
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    manager = engine_session.EngineSessionManager(model_weights_bytes=None)
    out = capsys.readouterr().out
    assert "session-bank budget" in out
    assert "legacy default" in out
    assert "explicit" not in out

    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "2G")
    manager = engine_session.EngineSessionManager(model_weights_bytes=None)
    out = capsys.readouterr().out
    assert "explicit" in out
    del manager
