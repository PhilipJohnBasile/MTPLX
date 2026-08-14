from __future__ import annotations

import base64
import json

from mtplx.opencode import (
    build_opencode_provider_config,
    ensure_opencode_reasoning_summaries_visible,
    merge_opencode_config,
    opencode_model_ref,
    opencode_session_headers_plugin_path,
    repair_opencode_desktop_state,
    write_opencode_config,
)


def test_opencode_model_ref_uses_provider_namespace():
    assert (
        opencode_model_ref("mtplx-qwen36-27b-optimized-quality")
        == "mtplx/mtplx-qwen36-27b-optimized-quality"
    )


def test_build_opencode_config_keeps_policy_server_side():
    payload = build_opencode_provider_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
        api_key="1234",
        context_window=262144,
    )

    provider = payload["provider"]["mtplx"]
    model = provider["models"]["mtplx-qwen36-27b-optimized-quality"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:18083/v1"
    assert provider["options"]["timeout"] is False
    assert provider["options"]["chunkTimeout"] == 900000
    assert provider["options"]["headers"]["x-mtplx-client"] == "opencode"
    assert provider["options"]["apiKey"] == "1234"
    assert model["reasoning"] is False
    assert model["tool_call"] is True
    assert model["temperature"] is False
    assert model["limit"] == {"context": 262144, "output": 262144}
    assert "interleaved" not in model
    assert "options" not in model
    assert "maxTokens" not in json.dumps(payload)


def test_build_opencode_config_keeps_gemma_policy_server_side():
    payload = build_opencode_provider_config(
        base_url="http://127.0.0.1:18108/v1",
        model_id="gemma4-mtplx-optimized-speed",
        context_window=262144,
    )

    provider = payload["provider"]["mtplx"]
    assert "apiKey" not in provider["options"]
    model = payload["provider"]["mtplx"]["models"]["gemma4-mtplx-optimized-speed"]
    assert model["reasoning"] is False
    assert model["temperature"] is False
    assert "interleaved" not in model
    assert "options" not in model


def test_ensure_opencode_reasoning_summaries_visible_enables_desktop_store(tmp_path):
    store = tmp_path / "default.dat"
    store.write_text(
        json.dumps(
            {
                "settings.v3": json.dumps(
                    {
                        "general": {
                            "autoSave": True,
                            "showReasoningSummaries": False,
                        },
                        "appearance": {"fontSize": 14},
                    }
                ),
                "highlights.v1": json.dumps({"version": "1.15.7"}),
            }
        ),
        encoding="utf-8",
    )

    result = ensure_opencode_reasoning_summaries_visible(store)

    assert result["status"] == "enabled"
    assert result["did_change"] is True
    assert result["backup_path"]
    root = json.loads(store.read_text(encoding="utf-8"))
    settings = json.loads(root["settings.v3"])
    assert settings["general"]["showReasoningSummaries"] is True
    assert settings["general"]["autoSave"] is True
    assert root["highlights.v1"] == json.dumps({"version": "1.15.7"})

    second = ensure_opencode_reasoning_summaries_visible(store)
    assert second["status"] == "already_visible"
    assert second["did_change"] is False


def test_merge_opencode_config_preserves_other_providers():
    fragment = build_opencode_provider_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
    )

    merged = merge_opencode_config(
        {
            "provider": {"lmstudio": {"name": "LM Studio"}},
            "model": "lmstudio/foo",
        },
        config_fragment=fragment,
    )

    assert merged["provider"]["lmstudio"] == {"name": "LM Studio"}
    assert merged["provider"]["mtplx"]["models"]
    assert merged["model"] == "mtplx/mtplx-qwen36-27b-optimized-quality"
    assert merged["small_model"] == "mtplx/mtplx-qwen36-27b-optimized-quality"


def test_merge_opencode_config_preserves_existing_plugins_and_injects_session_headers():
    fragment = build_opencode_provider_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
    )

    merged = merge_opencode_config(
        {
            "plugin": ["/existing/plugin.js"],
            "provider": {"lmstudio": {"name": "LM Studio"}},
        },
        config_fragment=fragment,
        session_headers_plugin_path="/tmp/mtplx-session-headers.js",
    )

    assert merged["plugin"] == ["/existing/plugin.js", "/tmp/mtplx-session-headers.js"]


def test_write_opencode_config_backs_up_invalid_json(tmp_path, monkeypatch):
    path = tmp_path / "opencode.json"
    settings_store = tmp_path / "default.dat"
    path.write_text("{bad json", encoding="utf-8")
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(path))
    monkeypatch.setenv("MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(settings_store))

    result = write_opencode_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
        api_key="1234",
    )

    assert result["written"] is True
    assert result["backup_path"]
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provider"]["mtplx"]["options"]["baseURL"] == "http://127.0.0.1:18083/v1"
    assert result["reasoning_visibility"]["status"] == "enabled"
    assert settings_store.exists()


def test_write_opencode_config_installs_session_headers_plugin(tmp_path, monkeypatch):
    path = tmp_path / "opencode.json"
    settings_store = tmp_path / "default.dat"
    path.write_text(
        json.dumps({"plugin": [str(path.parent / "mtplx-session-headers.js")]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(path))
    monkeypatch.setenv("MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(settings_store))

    result = write_opencode_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
        api_key="1234",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provider"]["mtplx"]["options"]["apiKey"] == "1234"
    assert payload["provider"]["mtplx"]["options"]["headers"]["x-mtplx-client"] == "opencode"
    plugin_path = str(opencode_session_headers_plugin_path(path))
    assert plugin_path in payload["plugin"]
    assert (path.parent / "mtplx-session-headers.js").exists()
    assert result["reasoning_visibility"]["path"] == str(settings_store)
    assert result["reasoning_visibility"]["did_change"] is True
    assert payload["provider"]["mtplx"]["models"]
    assert result["session_headers_plugin_path"] == plugin_path
    assert not (path.parent / "package.json").exists()
    plugin_source = (path.parent / "mtplx-session-headers.js").read_text(
        encoding="utf-8"
    )
    assert 'output.headers["x-mtplx-session-id"]' in plugin_source
    assert '"chat.params"' in plugin_source
    assert "output.maxOutputTokens = undefined" in plugin_source
    assert "process.stdout.write" not in plugin_source
    assert "message.updated" not in plugin_source


def test_repair_opencode_desktop_state_prunes_missing_workspace(tmp_path, monkeypatch):
    app_support = tmp_path / "OpenCodeSupport"
    app_support.mkdir()
    present = tmp_path / "present-project"
    present.mkdir()
    missing_path = "/private/tmp/mtplx-opencode-desktop-qa"
    missing_key = base64.urlsafe_b64encode(missing_path.encode()).decode().rstrip("=")
    present_key = base64.urlsafe_b64encode(str(present).encode()).decode().rstrip("=")
    store = app_support / "opencode.global.dat"
    store.write_text(
        json.dumps(
            {
                "layout": json.dumps(
                    {
                        "sessionTabs": {
                            f"{missing_key}/ses_dead": {"all": []},
                            f"{present_key}/ses_live": {"all": ["context"]},
                        },
                        "sessionView": {
                            f"{missing_key}/ses_dead": {"scroll": {}},
                            f"{present_key}/ses_live": {"scroll": {}},
                        },
                    }
                ),
                "layout.page": json.dumps(
                    {
                        "lastProjectSession": {
                            missing_path: {
                                "directory": missing_path,
                                "id": "ses_dead",
                            },
                            str(present): {
                                "directory": str(present),
                                "id": "ses_live",
                            },
                        },
                        "workspaceExpanded": {
                            missing_path: True,
                            str(present): True,
                        },
                    }
                ),
                "server": json.dumps(
                    {
                        "projects": {
                            "local": [
                                {
                                    "worktree": missing_path,
                                    "expanded": True,
                                },
                                {"worktree": str(present), "expanded": True},
                            ]
                        },
                        "lastProject": {"local": str(present)},
                    }
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MTPLX_OPENCODE_DESKTOP_APP_SUPPORT", str(app_support))

    result = repair_opencode_desktop_state()

    assert result["status"] == "repaired"
    assert result["did_change"] is True
    assert result["backup_path"]
    repaired = json.loads(store.read_text(encoding="utf-8"))
    assert "mtplx-opencode-desktop-qa" not in repaired["layout"]
    assert "mtplx-opencode-desktop-qa" not in repaired["layout.page"]
    assert "mtplx-opencode-desktop-qa" not in repaired["server"]
    assert str(present) in repaired["layout.page"]
