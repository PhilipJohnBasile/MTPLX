"""#427: --allow-swap / MTPLX_ALLOW_SWAP lifts the memory-fit cap.

2.10 introduced a machine memory plan whose fit shapes the default context
window and refuses prompts past it with a structured 507. Operators who
ran 2.9.x past the fit on purpose (32 GB Macs, swap accepted) lost that
option. The override restores it explicitly: the default window is the
model's own maximum again and the fit refusal is skipped; the plan still
reports the overcommit and the pressure guard keeps shedding.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import mtplx.server.openai as srv


class _Plan:
    def __init__(self, fit: int, resolved: int, overcommitted: bool) -> None:
        self.context_window_fit = fit
        self.context_window_resolved = resolved
        self.context_overcommitted = overcommitted


class _State:
    def __init__(
        self, window: int, plan: _Plan | None = None, *, allow_swap: bool = False
    ) -> None:
        self.context_window = window
        self.memory_plan = plan
        self.allow_swap = allow_swap


class TestAllowSwapEnabled:
    def test_flag_wins(self, monkeypatch):
        monkeypatch.delenv("MTPLX_ALLOW_SWAP", raising=False)
        assert srv._allow_swap_enabled(SimpleNamespace(allow_swap=True)) is True
        assert srv._allow_swap_enabled(SimpleNamespace(allow_swap=False)) is False
        assert srv._allow_swap_enabled(SimpleNamespace()) is False

    @pytest.mark.parametrize("raw", ["1", "true", "YES", " on "])
    def test_env_form_for_launchers_without_flags(self, monkeypatch, raw):
        monkeypatch.setenv("MTPLX_ALLOW_SWAP", raw)
        assert srv._allow_swap_enabled(SimpleNamespace(allow_swap=False)) is True

    @pytest.mark.parametrize("raw", ["0", "false", "off", ""])
    def test_env_off_values_stay_off(self, monkeypatch, raw):
        monkeypatch.setenv("MTPLX_ALLOW_SWAP", raw)
        assert srv._allow_swap_enabled(SimpleNamespace(allow_swap=False)) is False


class TestDefaultWindow:
    """The server passes machine_fit=0 under --allow-swap, so the default is
    the model maximum again; an explicit request still wins either way."""

    def _backend(self):
        return SimpleNamespace(
            backend_id="native_mtp",
            context_window_policy=SimpleNamespace(default=262144),
        )

    def test_fit_shapes_the_default_without_the_override(self):
        window = srv._select_backend_context_window(
            self._backend(), model_max=262144, requested=None, machine_fit=40960
        )
        assert window == 40960

    def test_override_restores_the_model_maximum(self):
        window = srv._select_backend_context_window(
            self._backend(), model_max=262144, requested=None, machine_fit=0
        )
        assert window == 262144

    def test_explicit_request_wins_either_way(self):
        for fit in (40960, 0):
            window = srv._select_backend_context_window(
                self._backend(), model_max=262144, requested=131072, machine_fit=fit
            )
            assert window == 131072


class TestPromptAdmission:
    def test_prompt_past_fit_is_admitted(self):
        state = _State(131072, _Plan(40960, 131072, True), allow_swap=True)
        assert srv._reject_prompt_over_context(state, 54772) is None

    def test_same_prompt_is_507_without_the_override(self):
        state = _State(131072, _Plan(40960, 131072, True), allow_swap=False)
        with pytest.raises(HTTPException) as e:
            srv._reject_prompt_over_context(state, 54772)
        assert e.value.status_code == 507
        assert e.value.detail["code"] == "insufficient_memory"

    def test_window_overflow_is_still_400(self):
        # The override accepts swap, not prompts the window cannot hold.
        state = _State(131072, _Plan(40960, 131072, True), allow_swap=True)
        with pytest.raises(HTTPException) as e:
            srv._reject_prompt_over_context(state, 131072)
        assert e.value.status_code == 400
        assert e.value.detail["code"] == "context_length_exceeded"


class TestAllocationAdvice:
    def test_advice_names_the_override_when_it_admitted_the_prompt(self, monkeypatch):
        monkeypatch.setattr(srv, "_shed_after_allocation_failure", lambda state: None)
        state = _State(131072, _Plan(40960, 131072, True), allow_swap=True)
        exc = srv._allocation_failure_http_exception(state, RuntimeError("oom"))
        assert exc.status_code == 507
        assert "--allow-swap" in exc.detail
        assert "40960" in exc.detail

    def test_advice_without_the_override_points_at_the_fit(self, monkeypatch):
        monkeypatch.setattr(srv, "_shed_after_allocation_failure", lambda state: None)
        state = _State(131072, _Plan(40960, 131072, True), allow_swap=False)
        exc = srv._allocation_failure_http_exception(state, RuntimeError("oom"))
        assert "--allow-swap" not in exc.detail
        assert "fit of 40960" in exc.detail


class TestServerArgparse:
    def test_server_flag_parses_off_by_default(self):
        assert srv.parse_args(["--model", "m"]).allow_swap is False
        assert srv.parse_args(["--model", "m", "--allow-swap"]).allow_swap is True

    def test_public_cli_serve_accepts_the_flag(self):
        from mtplx.cli import build_parser

        parser = build_parser()
        assert parser.parse_args(["serve"]).allow_swap is False
        assert parser.parse_args(["serve", "--allow-swap"]).allow_swap is True


def test_public_serve_passes_the_flag_to_the_daemon(monkeypatch):
    from mtplx.commands import public

    calls: dict = {}
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)

    def run(allow_swap: bool) -> list[str]:
        args = SimpleNamespace(
            command="serve",
            model="models/example",
            model_id="mtplx-example",
            cache_dir=None,
            profile="sustained",
            unsafe_force_unverified=False,
            yes=True,
            host="127.0.0.1",
            port=8000,
            depth=3,
            no_mtp=False,
            stock_ar=False,
            api_key="mtplx-local",
            rate_limit=0,
            stream_interval=1,
            context_window=131072,
            allow_swap=allow_swap,
            max_response_tokens=None,
            temperature=0.6,
            top_p=0.95,
            reasoning_parser="qwen3",
            stats_footer=False,
            warmup_tokens=0,
            strict_warmup=False,
            strict_fast_path=False,
            quickstart_pi=False,
            max=False,
            _cli_flags=set(),
        )
        try:
            public.cmd_serve_public(args)
        except SystemExit as exc:
            assert exc.code == 0
        return list(calls["cmd"])

    with_flag = run(True)
    assert "--allow-swap" in with_flag
    assert with_flag[with_flag.index("--context-window") + 1] == "131072"
    assert "--allow-swap" not in run(False)
