"""Tests for the --max crash safety machinery.

When a `mtplx start --max` session pins the fans, we drop a marker file
recording the pid. If the session exits via a recoverable signal, our
`atexit` / signal-handler chain runs `thermalforge auto` and clears the
marker. If it dies hard (kill -9, terminal slammed shut), the marker stays
behind and the *next* MTPLX invocation must detect it, see the pid is gone,
restore fans, and clear the marker.

These tests pin that contract without touching real fan hardware.
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_marker(tmp_path, monkeypatch):
    from mtplx import thermal

    monkeypatch.setattr(thermal, "MAX_MARKER_FILE", tmp_path / "max-active.json")
    yield


def test_check_and_recover_with_no_marker_is_noop(monkeypatch):
    from mtplx import thermal

    called: list[str] = []
    monkeypatch.setattr(
        "mtplx.thermal.restore_thermal_profile_verified",
        lambda **kw: called.append("restore") or {"ok": True},
    )

    result = thermal.check_and_recover_stale_max()

    assert result == {"recovered": False, "stale_pid": None, "still_running": False}
    assert called == []


def test_check_and_recover_when_marker_pid_is_alive(monkeypatch):
    from mtplx import thermal

    # Use the current pid as "still alive".
    thermal._write_max_marker(pid=os.getpid())

    called: list[str] = []
    monkeypatch.setattr(
        "mtplx.thermal.restore_thermal_profile_verified",
        lambda **kw: called.append("restore") or {"ok": True},
    )

    result = thermal.check_and_recover_stale_max()

    assert result["recovered"] is False
    assert result["still_running"] is True
    assert result["stale_pid"] == os.getpid()
    # Marker survives because owner is alive.
    assert thermal.MAX_MARKER_FILE.exists()
    # We must NOT touch fans when the previous owner is still around.
    assert called == []


def test_check_and_recover_when_marker_pid_is_gone(monkeypatch):
    from mtplx import thermal

    # PID 1 is launchd; we definitely don't own it. Use a clearly-dead
    # pid instead — pick a very large one that's almost certainly free.
    fake_pid = 999_999_999
    thermal._write_max_marker(pid=fake_pid)

    called: list[str] = []
    monkeypatch.setattr(
        "mtplx.thermal.restore_thermal_profile_verified",
        lambda **kw: called.append("restore") or {"ok": True},
    )

    result = thermal.check_and_recover_stale_max()

    assert result["recovered"] is True
    assert result["still_running"] is False
    assert result["stale_pid"] == fake_pid
    # Marker cleared.
    assert not thermal.MAX_MARKER_FILE.exists()
    # Fans were restored to silent.
    assert called == ["restore"]


def test_check_and_recover_keeps_marker_when_restore_fails(monkeypatch):
    from mtplx import thermal

    fake_pid = 999_999_998
    thermal._write_max_marker(pid=fake_pid)

    monkeypatch.setattr(
        "mtplx.thermal.restore_thermal_profile_verified",
        lambda **kw: {"ok": False, "message": "still manual"},
    )

    result = thermal.check_and_recover_stale_max()

    assert result["recovered"] is False
    assert result["marker_cleared"] is False
    assert thermal.MAX_MARKER_FILE.exists()


def test_install_max_lifecycle_hooks_writes_marker_and_returns_cleanup(monkeypatch):
    from mtplx import thermal

    restore_calls: list[str] = []
    monkeypatch.setattr(
        "mtplx.thermal.restore_thermal_profile_verified",
        lambda **kw: restore_calls.append("restore") or {"ok": True},
    )
    monkeypatch.setattr(
        "mtplx.thermal._spawn_thermal_sidecar", lambda owner_token=None: None
    )
    # Bypass real signal/atexit registration in the test thread.
    monkeypatch.setattr("mtplx.thermal.signal.signal", lambda *a, **kw: None)
    monkeypatch.setattr("mtplx.thermal.atexit.register", lambda *a, **kw: None)

    cleanup = thermal.install_max_lifecycle_hooks()

    # Marker is written with the current pid.
    assert thermal.MAX_MARKER_FILE.exists()
    marker = json.loads(thermal.MAX_MARKER_FILE.read_text())
    assert marker["pid"] == os.getpid()

    # Cleanup runs `thermalforge auto` and clears the marker.
    assert cleanup()["ok"] is True
    assert restore_calls == ["restore"]
    assert not thermal.MAX_MARKER_FILE.exists()

    # Cleanup is idempotent — calling twice does not re-set the profile.
    cleanup()
    assert restore_calls == ["restore"]


def test_old_cleanup_does_not_restore_over_new_max_owner(monkeypatch):
    """A previous daemon may exit after its replacement has pinned fans."""

    from mtplx import thermal

    restore_calls: list[str] = []
    monkeypatch.setattr(
        "mtplx.thermal.restore_thermal_profile_verified",
        lambda **kw: restore_calls.append("restore") or {"ok": True},
    )
    monkeypatch.setattr(
        "mtplx.thermal._spawn_thermal_sidecar", lambda owner_token=None: None
    )
    monkeypatch.setattr("mtplx.thermal.signal.signal", lambda *a, **kw: None)
    monkeypatch.setattr("mtplx.thermal.atexit.register", lambda *a, **kw: None)

    old_cleanup = thermal.install_max_lifecycle_hooks()
    thermal._write_max_marker(pid=999_999_999)  # replacement daemon's lease

    result = old_cleanup()

    assert result == {
        "ok": True,
        "skipped": True,
        "reason": "max_owner_changed",
    }
    assert restore_calls == []
    assert thermal._read_max_marker()["pid"] == 999_999_999


def test_max_off_clears_marker(monkeypatch, tmp_path):
    """`mtplx max --off` (action=silent) must clear the marker so a future
    --status doesn't report a stale max state."""
    from mtplx import thermal
    from mtplx.commands.public import cmd_max_public
    from types import SimpleNamespace

    thermal._write_max_marker(pid=os.getpid())
    monkeypatch.setattr(
        "mtplx.thermal.set_thermal_profile",
        lambda profile, **kw: {"ok": True, "profile": profile, "detection": {"available": True}},
    )

    args = SimpleNamespace(max_action="silent", dry_run=False, json=False, no_daemon=False)
    rc = cmd_max_public(args)

    assert rc == 0
    assert not thermal.MAX_MARKER_FILE.exists()
