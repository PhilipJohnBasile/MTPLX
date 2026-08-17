"""Helpers for separating server bind addresses from client URLs."""

from __future__ import annotations


def _clean_host(host: str | None) -> str:
    return str(host or "").strip()


def _unbracket_host(host: str | None) -> str:
    cleaned = _clean_host(host)
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned[1:-1]
    return cleaned


def is_wildcard_bind(host: str | None) -> bool:
    return _unbracket_host(host).lower() in {"0.0.0.0", "::"}


def connect_host_for_bind(host: str | None) -> str:
    normalized = _unbracket_host(host).lower()
    if normalized in {"", "0.0.0.0", "::", "localhost"}:
        return "127.0.0.1"
    return _clean_host(host)


def url_host(host: str | None) -> str:
    cleaned = _clean_host(host)
    raw = _unbracket_host(cleaned)
    if ":" in raw and not (cleaned.startswith("[") and cleaned.endswith("]")):
        return f"[{raw}]"
    return cleaned


def local_url_for_bind(host: str | None, port: int, *, path: str = "") -> str:
    suffix = path if path.startswith("/") or not path else f"/{path}"
    return f"http://{url_host(connect_host_for_bind(host))}:{int(port)}{suffix}"


def bind_label(host: str | None, port: int) -> str:
    raw = _unbracket_host(host)
    display = "[::]" if raw == "::" else (_clean_host(host) or "127.0.0.1")
    label = f"{display}:{int(port)}"
    if is_wildcard_bind(host):
        label += " (all interfaces)"
    return label


def primary_lan_ip() -> str | None:
    """This Mac's default-route IPv4, or None when it can't be determined.

    Wildcard binds print this so the user knows what address OTHER machines
    (LAN devices, Parallels/VM guests) should dial — 0.0.0.0 is not dialable.
    The UDP connect never sends a packet; the target is a TEST-NET-3 literal
    so no DNS lookup and no real host is involved. Loopback answers mean no
    usable route, reported as None.
    """
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("203.0.113.1", 9))
            address = str(probe.getsockname()[0])
    except OSError:
        return None
    if not address or address.startswith("127.") or address == "0.0.0.0":
        return None
    return address


def network_url_for_bind(host: str | None, port: int, *, path: str = "") -> str | None:
    """Dialable URL for other machines, or None (non-wildcard/undetectable)."""
    if not is_wildcard_bind(host):
        return None
    address = primary_lan_ip()
    if not address:
        return None
    suffix = path if path.startswith("/") or not path else f"/{path}"
    return f"http://{address}:{int(port)}{suffix}"
