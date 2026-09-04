"""Browser origin policy, browser sign-in, and the public dashboard bundle.

The server used to run ``CORSMiddleware(allow_origins=["*"],
allow_credentials=True)``: any web page the user visited could POST to the
local model, read the answer on a keyless localhost bind, list and clear
sessions, and keep the GPU busy. These tests pin the replacement:

- same-origin browser requests are allowed with credentials;
- a foreign ``Origin`` gets a 403 and no ``Access-Control-Allow-*`` headers;
- ``--cors-origin`` allowlists an origin for the API but never for ``/admin``
  or the browser sign-in routes;
- requests without ``Origin`` (native app, OpenCode, curl) are untouched;
- browser sign-in works without the key in a URL (POST form, single-use
  ticket), the cookie flags are right, and the dashboard bundle loads
  without a cookie while every JSON route keeps the key gate.
"""

from __future__ import annotations

import sys
import urllib.parse

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.dashboard import has_static_bundle
from mtplx.server import openai
from mtplx.server.openai import (
    _BROWSER_AUTH_COOKIE,
    _BROWSER_AUTH_PATH,
    _BROWSER_AUTH_TICKET_PATH,
    _BrowserAuthTickets,
    _parse_cors_origins,
    _request_origin_is_own,
    create_app,
    parse_args,
)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from test_server_openai import _fake_state  # noqa: E402

OWN = "http://127.0.0.1:8000"
FOREIGN = "http://evil.example"
ALLOWLISTED = "http://localhost:5173"


def _client(*, api_key: str | None = None, cors_origins=(), base_url: str = OWN) -> TestClient:
    state = _fake_state(api_key=api_key)
    state.args.cors_origins = tuple(cors_origins)
    return TestClient(create_app(state), base_url=base_url)


def _cors_headers(response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower().startswith("access-control-")
    }


# ---- same-origin computation -------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "scheme", "host", "expected"),
    [
        ("http://127.0.0.1:8000", "http", "127.0.0.1:8000", True),
        ("http://localhost:8000", "http", "localhost:8000", True),
        ("http://LOCALHOST:8000", "http", "localhost:8000", True),
        ("http://[::1]:8000", "http", "[::1]:8000", True),
        ("http://192.168.1.20:8000", "http", "192.168.1.20:8000", True),
        ("http://mtplx.local", "http", "mtplx.local:80", True),
        ("https://mtplx.example", "https", "mtplx.example", True),
        ("http://localhost:8000", "http", "127.0.0.1:8000", False),
        ("http://127.0.0.1:9000", "http", "127.0.0.1:8000", False),
        ("https://127.0.0.1:8000", "http", "127.0.0.1:8000", False),
        ("http://evil.example", "http", "127.0.0.1:8000", False),
        ("null", "http", "127.0.0.1:8000", False),
        ("http://127.0.0.1:8000", "http", None, False),
    ],
)
def test_request_origin_is_own_compares_scheme_host_and_port(origin, scheme, host, expected):
    assert _request_origin_is_own(origin, scheme=scheme, host_header=host) is expected


def test_parse_cors_origins_normalizes_and_merges_env():
    origins = _parse_cors_origins(
        ["http://Localhost:5173/", "https://app.example:443", "http://a.example,http://b.example:80"],
        "http://c.example, http://localhost:5173",
    )
    assert origins == (
        "http://localhost:5173",
        "https://app.example",
        "http://a.example",
        "http://b.example",
        "http://c.example",
    )


@pytest.mark.parametrize("bad", ["localhost:5173", "*", "http://", "http://a.example/path", "ftp://a.example"])
def test_parse_cors_origins_refuses_anything_that_is_not_an_origin(bad):
    with pytest.raises(ValueError):
        _parse_cors_origins([bad])


def test_cors_origin_flag_is_repeatable_and_refuses_bad_values(monkeypatch):
    monkeypatch.delenv("MTPLX_CORS_ORIGINS", raising=False)
    args = parse_args(["--warmup-tokens", "0", "--cors-origin", ALLOWLISTED, "--cors-origin", "https://x.example"])
    assert args.cors_origins == (ALLOWLISTED, "https://x.example")

    monkeypatch.setenv("MTPLX_CORS_ORIGINS", "http://env.example:8080")
    assert parse_args(["--warmup-tokens", "0"]).cors_origins == ("http://env.example:8080",)

    with pytest.raises(SystemExit):
        parse_args(["--warmup-tokens", "0", "--cors-origin", "*"])


# ---- origin gate ---------------------------------------------------------------


def test_request_without_origin_is_untouched():
    client = _client()

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert _cors_headers(response) == {}


def test_same_origin_browser_request_is_allowed_including_admin():
    client = _client()

    models = client.get("/v1/models", headers={"Origin": OWN})
    cleared = client.post("/admin/cache/clear", headers={"Origin": OWN})

    assert models.status_code == 200
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] is True


def test_foreign_origin_is_refused_with_no_allow_headers_even_without_an_api_key():
    client = _client()

    for origin in (FOREIGN, "null", "http://127.0.0.1:9000", "http://localhost:8000"):
        response = client.post(
            "/v1/chat/completions",
            headers={"Origin": origin},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 403, origin
        assert response.json()["error"]["type"] == "origin_error"
        assert _cors_headers(response) == {}, origin

    listing = client.get("/admin/sessions", headers={"Origin": FOREIGN})
    assert listing.status_code == 403
    assert _cors_headers(listing) == {}


def test_foreign_preflight_gets_403_and_no_private_network_grant():
    client = _client()

    response = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": FOREIGN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )

    assert response.status_code == 403
    assert _cors_headers(response) == {}


def test_allowlisted_origin_reaches_the_api_with_credentials_and_private_network_grant():
    client = _client(cors_origins=[ALLOWLISTED])

    preflight = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": ALLOWLISTED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == ALLOWLISTED
    assert preflight.headers["access-control-allow-credentials"] == "true"
    assert preflight.headers["access-control-allow-methods"] == "POST"
    assert preflight.headers["access-control-allow-headers"] == "authorization, content-type"
    assert preflight.headers["access-control-allow-private-network"] == "true"
    assert "Origin" in preflight.headers.get("vary", "")

    models = client.get("/v1/models", headers={"Origin": ALLOWLISTED})
    assert models.status_code == 200
    assert models.headers["access-control-allow-origin"] == ALLOWLISTED
    assert models.headers["access-control-allow-credentials"] == "true"
    assert "Origin" in models.headers.get("vary", "")

    # Case and trailing-slash differences in the page's Origin still match.
    assert client.get("/v1/models", headers={"Origin": "http://LOCALHOST:5173"}).status_code == 200
    # A second origin is still foreign.
    assert client.get("/v1/models", headers={"Origin": FOREIGN}).status_code == 403


def test_allowlisted_origin_never_reaches_admin_or_browser_sign_in():
    client = _client(api_key="test-key", cors_origins=[ALLOWLISTED])
    auth = {"Origin": ALLOWLISTED, "Authorization": "Bearer test-key"}

    assert client.get("/v1/models", headers=auth).status_code == 200
    for method, path in (
        ("GET", "/admin/sessions"),
        ("POST", "/admin/sessions/abc/clear"),
        ("POST", "/admin/cache/clear"),
        ("GET", "/admin/cache/ssd"),
        ("POST", "/admin/cache/ssd/archive"),
        ("POST", _BROWSER_AUTH_PATH),
        ("POST", _BROWSER_AUTH_TICKET_PATH),
    ):
        response = client.request(method, path, headers=auth, json={})
        assert response.status_code == 403, (method, path)
        assert _cors_headers(response) == {}, (method, path)
        preflight = client.options(
            path, headers={"Origin": ALLOWLISTED, "Access-Control-Request-Method": method}
        )
        assert preflight.status_code == 403, (method, path)


def test_origin_gate_runs_before_the_key_gate():
    # A foreign page must not be able to probe whether a key is configured,
    # and a wrong-key foreign request must not spend a rate-limit slot.
    client = _client(api_key="test-key")

    response = client.get("/v1/models", headers={"Origin": FOREIGN})

    assert response.status_code == 403
    assert "www-authenticate" not in response.headers


# ---- browser sign-in ------------------------------------------------------------


def test_post_browser_auth_sets_the_cookie_for_the_right_key_only():
    client = _client(api_key="test-key")

    wrong = client.post(_BROWSER_AUTH_PATH, headers={"Origin": OWN}, json={"api_key": "nope", "next": "/dashboard/"})
    assert wrong.status_code == 401
    assert wrong.json() == {"error": {"message": "missing or invalid API key", "type": "auth"}}
    assert "set-cookie" not in wrong.headers
    assert client.get("/health").status_code == 401

    missing = client.post(_BROWSER_AUTH_PATH, headers={"Origin": OWN}, json={"next": "/dashboard/"})
    assert missing.status_code == 401

    signed_in = client.post(
        _BROWSER_AUTH_PATH, headers={"Origin": OWN}, json={"api_key": "test-key", "next": "/dashboard/"}
    )
    assert signed_in.status_code == 200
    assert signed_in.json() == {"ok": True, "next": "/dashboard/"}
    cookie = signed_in.headers["set-cookie"]
    assert f"{_BROWSER_AUTH_COOKIE}=test-key" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Secure" not in cookie  # plain http request
    assert "Path=/" in cookie
    assert "Max-Age=43200" in cookie

    # The cookie now unlocks the JSON routes for this browser.
    assert client.get("/health").status_code == 200
    assert client.get("/v1/mtplx/snapshot").status_code == 200


def test_post_browser_auth_sanitises_next_like_the_get_form():
    client = _client(api_key="test-key")

    response = client.post(
        _BROWSER_AUTH_PATH,
        headers={"Origin": OWN},
        json={"api_key": "test-key", "next": "//evil.example/steal"},
    )

    assert response.status_code == 200
    assert response.json()["next"] == "/"


def test_post_browser_auth_without_a_configured_key_is_a_no_op_success():
    client = _client()

    response = client.post(_BROWSER_AUTH_PATH, headers={"Origin": OWN}, json={"api_key": "anything"})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "next": "/"}
    assert "set-cookie" not in response.headers


def test_browser_auth_cookie_is_secure_over_https():
    client = _client(api_key="test-key", base_url="https://mtplx.example")

    response = client.post(
        _BROWSER_AUTH_PATH,
        headers={"Origin": "https://mtplx.example"},
        json={"api_key": "test-key"},
    )

    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie

    redirect = client.get(f"{_BROWSER_AUTH_PATH}?mtplx_api_key=test-key&next=/", follow_redirects=False)
    assert redirect.status_code == 303
    assert "Secure" in redirect.headers["set-cookie"]


def test_ticket_mint_consume_single_use_and_unknown():
    client = _client(api_key="test-key")

    unauthenticated = client.post(_BROWSER_AUTH_TICKET_PATH, json={"next": "/dashboard/"})
    assert unauthenticated.status_code == 401

    minted = client.post(
        _BROWSER_AUTH_TICKET_PATH,
        headers={"Authorization": "Bearer test-key"},
        json={"next": "/dashboard/"},
    )
    assert minted.status_code == 200
    payload = minted.json()
    assert payload["expires_in"] == 60
    url = payload["url"]
    assert url.startswith(f"{OWN}{_BROWSER_AUTH_PATH}?")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert query["next"] == ["/dashboard/"]
    assert len(query["ticket"][0]) >= 32
    assert "test-key" not in url

    # A fresh browser (no cookie) redeems the ticket once.
    browser = TestClient(client.app, base_url=OWN)
    redeemed = browser.get(url[len(OWN) :], follow_redirects=False)
    assert redeemed.status_code == 303
    assert redeemed.headers["location"] == "/dashboard/"
    assert f"{_BROWSER_AUTH_COOKIE}=test-key" in redeemed.headers["set-cookie"]
    assert browser.get("/health").status_code == 200

    # Single use: the same ticket is dead, and made-up tickets never work.
    again = TestClient(client.app, base_url=OWN).get(url[len(OWN) :], follow_redirects=False)
    assert again.status_code == 401
    unknown = TestClient(client.app, base_url=OWN).get(
        f"{_BROWSER_AUTH_PATH}?ticket=not-a-ticket&next=/dashboard/", follow_redirects=False
    )
    assert unknown.status_code == 401
    assert "set-cookie" not in unknown.headers


def test_tickets_expire_after_their_ttl():
    tickets = _BrowserAuthTickets(ttl_s=60)

    fresh = tickets.mint(now=1000.0)
    assert tickets.consume(fresh, now=1059.9) is True

    stale = tickets.mint(now=1000.0)
    assert tickets.consume(stale, now=1060.0) is False
    assert tickets.consume(None) is False
    assert tickets.consume("") is False


def test_ticket_endpoint_without_a_configured_key_returns_a_plain_url():
    client = _client()

    response = client.post(_BROWSER_AUTH_TICKET_PATH, json={"next": "/dashboard/"})

    assert response.status_code == 200
    assert response.json() == {"url": f"{OWN}/dashboard/", "expires_in": None}

    # And the GET form behaves as before: a plain redirect, no cookie.
    redirect = client.get(f"{_BROWSER_AUTH_PATH}?ticket=whatever&next=/dashboard/", follow_redirects=False)
    assert redirect.status_code == 303
    assert redirect.headers["location"] == "/dashboard/"
    assert "set-cookie" not in redirect.headers


def test_query_key_bootstrap_still_works_and_wrong_key_still_401s():
    client = _client(api_key="test-key")

    assert client.get(f"{_BROWSER_AUTH_PATH}?mtplx_api_key=wrong", follow_redirects=False).status_code == 401
    ok = client.get(f"{_BROWSER_AUTH_PATH}?mtplx_api_key=test-key&next=/dashboard/", follow_redirects=False)
    assert ok.status_code == 303
    assert ok.headers["location"] == "/dashboard/"
    assert f"{_BROWSER_AUTH_COOKIE}=test-key" in ok.headers["set-cookie"]


# ---- public dashboard bundle ------------------------------------------------------


def test_dashboard_bundle_loads_without_a_cookie_while_json_routes_stay_gated():
    client = _client(api_key="test-key")

    index = client.get("/dashboard/")
    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]
    assert client.get("/dashboard").status_code in {200, 307}

    if has_static_bundle():
        from mtplx.dashboard import DASHBOARD_STATIC_DIR

        assets = sorted(p.name for p in (DASHBOARD_STATIC_DIR / "assets").iterdir())
        assert assets
        asset = client.get(f"/dashboard/assets/{assets[0]}")
        assert asset.status_code == 200
        # The client normalises `..` before sending, so this leaves the bundle
        # prefix and meets the key gate; either way it must never be served.
        assert client.get("/dashboard/assets/../../../etc/passwd").status_code in {400, 401, 404}

    for path in (
        "/",
        "/health",
        "/metrics",
        "/mtplx/settings",
        "/v1/mtplx/settings",
        "/v1/mtplx/snapshot",
        "/v1/mtplx/metrics/stream",
        "/v1/models",
        "/admin/sessions",
    ):
        response = client.get(path)
        assert response.status_code == 401, path
    assert client.post("/dashboard/", json={}).status_code in {401, 405}


def test_dashboard_bundle_does_not_spend_the_rate_limit():
    state = _fake_state(api_key="test-key", rate_limit=1)
    state.args.cors_origins = ()
    client = TestClient(create_app(state), base_url=OWN)

    for _ in range(3):
        assert client.get("/dashboard/").status_code == 200
    assert client.get("/health", headers={"Authorization": "Bearer test-key"}).status_code == 200
    assert client.get("/health", headers={"Authorization": "Bearer test-key"}).status_code == 429


def test_openai_module_no_longer_ships_the_wildcard_cors_middleware():
    assert not hasattr(openai, "CORSMiddleware")
