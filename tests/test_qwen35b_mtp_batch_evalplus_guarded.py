from pathlib import Path
from types import SimpleNamespace

import pytest


def test_codegen_command_pins_the_evalplus_quality_contract():
    from scripts.qwen35b_mtp_batch_evalplus_guarded import build_codegen_command

    command = build_codegen_command(
        python=Path("/evalplus/python"),
        generator=Path("/bench/evalplus_paired_codegen.py"),
        root=Path("/receipts/balanced"),
        port=18080,
    )

    assert command == [
        "/evalplus/python",
        "-u",
        "/bench/evalplus_paired_codegen.py",
        "--arm",
        "b8",
        "--root",
        "/receipts/balanced",
        "--endpoint",
        "http://127.0.0.1:18080/v1",
        "--model",
        "qwen35b-mtp-b8-numerics",
        "--datasets",
        "humaneval",
        "mbpp",
        "--max-tokens",
        "768",
        "--no-resume",
    ]


def test_private_evalplus_server_uses_auditable_gather_window():
    from scripts.qwen35b_mtp_batch_evalplus_guarded import _server_command

    command = _server_command(
        SimpleNamespace(
            mtplx=Path("/mtplx"),
            model=Path("/model"),
            port=18080,
            numerics="balanced",
            chat_template=Path("/template"),
        )
    )

    index = command.index("--batch-wait-ms")
    assert command[index + 1] == "2000"


def test_evalplus_preflight_pins_discovered_site_packages(monkeypatch):
    from scripts import qwen35b_mtp_batch_evalplus_guarded as module

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="/evalplus-venv/lib/python3.12/site-packages\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    env = {"PYTHONPATH": "/existing"}

    module.pin_evalplus_site_packages(
        python=Path("/evalplus-venv/bin/python"),
        cwd=Path("/bench"),
        env=env,
    )

    assert captured["command"][0] == "/evalplus-venv/bin/python"
    assert captured["kwargs"]["cwd"] == Path("/bench")
    assert env["PYTHONPATH"] == (
        "/evalplus-venv/lib/python3.12/site-packages:/existing"
    )


def test_launcher_path_keeps_virtualenv_symlink(tmp_path):
    from scripts.qwen35b_mtp_batch_evalplus_guarded import absolute_launcher_path

    base = tmp_path / "base-python"
    base.touch()
    launcher = tmp_path / "evalplus-python"
    launcher.symlink_to(base)

    normalized = absolute_launcher_path(launcher)

    assert normalized == launcher.absolute()
    assert normalized != launcher.resolve()


def test_evalplus_receipt_requires_real_fixed_b8_route():
    from scripts.qwen35b_mtp_batch_evalplus_guarded import (
        validate_evalplus_b8_receipt,
    )

    scheduler = {
        "mtp_batch_route_id": "balanced-route",
        "mtp_batch": {
            "batch_histogram": {"8": 69},
            "fixed_width_histogram": {"8": 42},
            "last_route_id": "balanced-route",
            "last_error": None,
        },
    }
    validate_evalplus_b8_receipt(scheduler)

    scheduler["mtp_batch"]["fixed_width_histogram"] = {}
    with pytest.raises(RuntimeError, match="physical B8"):
        validate_evalplus_b8_receipt(scheduler)

    scheduler["mtp_batch"]["fixed_width_histogram"] = {"8": 42}
    scheduler["mtp_batch"]["batch_histogram"] = {"1": 1, "8": 68}
    with pytest.raises(RuntimeError, match="69 real-width-eight"):
        validate_evalplus_b8_receipt(scheduler)
