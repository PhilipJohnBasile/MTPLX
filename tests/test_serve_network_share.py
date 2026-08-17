"""Network-sharing UX: key-file generation, LAN URL, and the refusal's recovery path.

Covers the failure chain a Parallels user hit publicly (2026-08-16): the
non-localhost refusal suggests `--api-key-file ~/.mtplx/api-key`, which then
died on FileNotFoundError because nothing created the file. Server entrypoints
now generate a missing key file (printed once); wildcard binds print a
dialable network URL.
"""

from __future__ import annotations

import argparse
import stat

import pytest

from mtplx import server_urls
from mtplx.commands import public
from mtplx.runtime_options import generate_api_key_file, resolve_api_key


def _args(**kwargs) -> argparse.Namespace:
    defaults = {
        "api_key": None,
        "api_key_file": None,
        "paged_kv_quantization": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestGenerateApiKeyFile:
    def test_creates_file_with_prefixed_key_and_0600(self, tmp_path):
        path = tmp_path / "api-key"
        key = generate_api_key_file(path)
        assert key.startswith("mtplx-")
        assert len(key) > 20
        assert path.read_text(encoding="utf-8").strip() == key
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_creates_missing_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "api-key"
        key = generate_api_key_file(path)
        assert path.read_text(encoding="utf-8").strip() == key

    def test_refuses_to_overwrite_existing_file(self, tmp_path):
        path = tmp_path / "api-key"
        path.write_text("existing-secret\n", encoding="utf-8")
        with pytest.raises(OSError):
            generate_api_key_file(path)
        assert path.read_text(encoding="utf-8") == "existing-secret\n"


class TestResolveRuntimeOptionsGeneration:
    def test_missing_key_file_is_generated_and_printed_once(self, tmp_path):
        key_path = tmp_path / "api-key"
        args = _args(api_key_file=str(key_path))
        lines: list[str] = []
        assert public._resolve_runtime_options_on_args(args, printer=lines.append) is None
        assert key_path.is_file()
        stored = key_path.read_text(encoding="utf-8").strip()
        assert args.api_key == stored
        assert args.api_key_source == "file"
        assert any("Generated a new API key" in line for line in lines)
        assert any(stored in line for line in lines)

    def test_existing_key_file_is_read_without_generation_lines(self, tmp_path):
        key_path = tmp_path / "api-key"
        key_path.write_text("mtplx-preexisting\n", encoding="utf-8")
        args = _args(api_key_file=str(key_path))
        lines: list[str] = []
        assert public._resolve_runtime_options_on_args(args, printer=lines.append) is None
        assert args.api_key == "mtplx-preexisting"
        assert not any("Generated" in line for line in lines)

    def test_empty_existing_key_file_still_errors(self, tmp_path):
        key_path = tmp_path / "api-key"
        key_path.write_text("\n", encoding="utf-8")
        args = _args(api_key_file=str(key_path))
        lines: list[str] = []
        assert public._resolve_runtime_options_on_args(args, printer=lines.append) == 2
        assert any("empty" in line for line in lines)

    def test_explicit_api_key_skips_file_generation(self, tmp_path):
        key_path = tmp_path / "api-key"
        args = _args(api_key="mtplx-explicit", api_key_file=str(key_path))
        lines: list[str] = []
        assert public._resolve_runtime_options_on_args(args, printer=lines.append) is None
        assert args.api_key == "mtplx-explicit"
        assert not key_path.exists()

    def test_resolver_itself_stays_strict_on_missing_file(self, tmp_path):
        with pytest.raises(OSError):
            resolve_api_key(api_key_file=str(tmp_path / "absent"))


class _FakeSocket:
    def __init__(self, sockname: str | None, fail: bool = False):
        self._sockname = sockname
        self._fail = fail

    def __call__(self, *a, **k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def connect(self, target):
        if self._fail:
            raise OSError("network unreachable")

    def getsockname(self):
        return (self._sockname, 0)


class TestNetworkUrl:
    def test_primary_lan_ip_returns_route_address(self, monkeypatch):
        import socket as socket_module

        monkeypatch.setattr(socket_module, "socket", _FakeSocket("192.168.1.20"))
        assert server_urls.primary_lan_ip() == "192.168.1.20"

    def test_primary_lan_ip_none_on_loopback_or_failure(self, monkeypatch):
        import socket as socket_module

        monkeypatch.setattr(socket_module, "socket", _FakeSocket("127.0.0.1"))
        assert server_urls.primary_lan_ip() is None
        monkeypatch.setattr(socket_module, "socket", _FakeSocket(None, fail=True))
        assert server_urls.primary_lan_ip() is None

    def test_network_url_only_for_wildcard_binds(self, monkeypatch):
        monkeypatch.setattr(server_urls, "primary_lan_ip", lambda: "192.168.1.20")
        assert (
            server_urls.network_url_for_bind("0.0.0.0", 8000, path="/v1")
            == "http://192.168.1.20:8000/v1"
        )
        assert server_urls.network_url_for_bind("::", 8000, path="/v1") is not None
        assert server_urls.network_url_for_bind("127.0.0.1", 8000, path="/v1") is None

    def test_network_url_none_when_ip_undetectable(self, monkeypatch):
        monkeypatch.setattr(server_urls, "primary_lan_ip", lambda: None)
        assert server_urls.network_url_for_bind("0.0.0.0", 8000, path="/v1") is None


class TestServeHandoffNetworkLine:
    def _handoff_lines(self, monkeypatch, host: str) -> list[str]:
        lines: list[str] = []
        monkeypatch.setattr(
            public, "_print_serve_start_line", lambda *a: lines.append(a[0] if a else "")
        )
        monkeypatch.setattr(public, "network_url_for_bind", lambda *a, **k: (
            "http://192.168.1.20:8000/v1" if server_urls.is_wildcard_bind(host) else None
        ))
        args = argparse.Namespace(host=host, port=8000)
        public._print_serve_handoff(args, "model-x", "turbo")
        return lines

    def test_wildcard_handoff_prints_network_base_url(self, monkeypatch):
        lines = self._handoff_lines(monkeypatch, "0.0.0.0")
        assert any("Network API Base URL: http://192.168.1.20:8000/v1" in l for l in lines)

    def test_localhost_handoff_has_no_network_line(self, monkeypatch):
        lines = self._handoff_lines(monkeypatch, "127.0.0.1")
        assert not any("Network API Base URL" in l for l in lines)
