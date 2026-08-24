"""bench --harness depth-sweep must honor or loudly refuse every flag (#285).

Four flags used to be accepted by argparse and silently discarded
(depths/seed hardcoded, --stock-ar and --generation-mode never read), so
"different" A/B configs produced byte-identical runs.
"""

from __future__ import annotations

import pytest

from mtplx.cli import _cmd_bench_profile, build_parser


def _bench_args(*extra: str):
    parser = build_parser()
    return parser.parse_args(
        ["bench", "--profile", "performance-cold", "--harness", "depth-sweep", *extra]
    )


def test_stock_ar_is_refused_loudly():
    args = _bench_args("--stock-ar")
    with pytest.raises(SystemExit, match="stock-ar is not available"):
        _cmd_bench_profile(args)


def test_depths_seed_and_ar_mode_reach_the_runner(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_sweep(model, prompts, **kwargs):
        captured.update(kwargs)
        return {"depths": [], "seed": kwargs.get("seed")}

    import mtplx.benchmarks.runners.mtp_depth_sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "run_mtp_depth_sweep", fake_sweep)
    monkeypatch.setattr(
        "mtplx.benchmarks.runners.preflight.run_preflight",
        lambda *a, **k: {"clean": True},
    )
    args = _bench_args(
        "--depths",
        "1,2",
        "--seed",
        "1234",
        "--generation-mode",
        "ar",
        "--output",
        str(tmp_path / "sweep.json"),
    )
    _cmd_bench_profile(args)
    assert captured["depths"] == "1,2"
    assert captured["seed"] == 1234
    assert captured["compare_ar"] is True and captured["ar_only"] is True


def test_defaults_preserve_the_profile_contract(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_sweep(model, prompts, **kwargs):
        captured.update(kwargs)
        return {"depths": []}

    import mtplx.benchmarks.runners.mtp_depth_sweep as sweep_mod

    monkeypatch.setattr(sweep_mod, "run_mtp_depth_sweep", fake_sweep)
    args = _bench_args("--output", str(tmp_path / "sweep.json"))
    _cmd_bench_profile(args)
    assert captured["depths"] == "3"
    assert captured["seed"] == 0
    assert captured["compare_ar"] is False and captured["ar_only"] is False
    assert captured["verify_strategy"] == "capture_commit"
