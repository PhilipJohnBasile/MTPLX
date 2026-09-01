from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


KV_QUANT_MODES = ("off", "q8", "q4")

#: The one boolean vocabulary for MTPLX env flags.
#:
#: Every reader of a boolean ``MTPLX_*`` var should go through
#: :func:`env_bool` so a spelling means the same thing everywhere. Values
#: outside these sets raise rather than being silently read as "off" by one
#: reader and "on" by another — the failure mode catalogued in
#: docs/AUDIT_2026-07-18.md, where ``=enabled`` disabled a feature in the
#: server while enabling it in the generation loop.
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enable", "enabled"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disable", "disabled"})


def env_bool(
    name: str,
    *,
    default: bool,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Parse a boolean ``MTPLX_*`` env var, or raise on an unknown spelling.

    Unset (and set-but-empty) yields ``default``. Anything that is neither
    a recognized true nor a recognized false value is a configuration
    error: guessing is what let one variable mean three things.
    """

    source = os.environ if env is None else env
    raw = source.get(name)
    if raw is None:
        return bool(default)
    token = str(raw).strip().lower()
    if not token:
        return bool(default)
    if token in ENV_TRUE_VALUES:
        return True
    if token in ENV_FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(ENV_TRUE_VALUES | ENV_FALSE_VALUES))
    raise ValueError(
        f"{name}={raw!r} is not a boolean; expected one of: {accepted}"
    )


#: Exact-preserving op diet for the compiled fixed-M4 verify graph.
#:
#: Read ONCE at import so the hot path never touches ``os.environ`` and so a
#: mid-run env change cannot make two traces of the same graph disagree. With
#: the flag off every gated site executes the pre-diet expression verbatim;
#: with it on the rewritten sites are value-identical by construction (see
#: tests/test_fable_opdiet.py, which proves each rewrite against its original
#: on random inputs).
_FABLE_OPDIET = env_bool("MTPLX_FABLE_OPDIET", default=False)

#: The independently selectable rewrites behind the master switch.
#:
#: ``bank``  QSA fixed pooled-bank conditional write (_extend_pooled_fixed)
#: ``rope``  half-width RoPE tables, shared per forward, split-half rotation
#: ``resid`` hyper-connection residual write fused into one kernel
#: ``k20``   eager K20 target/draft support (fused deterministic+ordered pair)
#:
#: Removing dispatches is not the same as removing GPU time: a rewrite can
#: trade contiguous vectorized kernels for broadcast/general ones and lose.
#: ``MTPLX_FABLE_OPDIET_ITEMS`` exists so a measured regression can be
#: attributed to ONE item instead of the whole flag (2026-09-01: the four
#: together measured -2.9% tok/s against control on seed 20260829).
FABLE_OPDIET_ITEMS = ("bank", "rope", "resid", "k20")


def parse_opdiet_items(
    raw: str | None,
    *,
    known: tuple[str, ...] = FABLE_OPDIET_ITEMS,
) -> frozenset[str]:
    """Parse ``MTPLX_FABLE_OPDIET_ITEMS``; unset/empty selects everything.

    An unknown name raises rather than being dropped: a typo that silently
    disables the item under test would make the A/B measure the wrong thing.
    """

    if raw is None:
        return frozenset(known)
    tokens = {token.strip().lower() for token in str(raw).split(",")}
    tokens.discard("")
    if not tokens:
        return frozenset(known)
    if tokens == {"all"}:
        return frozenset(known)
    unknown = sorted(tokens - set(known))
    if unknown:
        raise ValueError(
            f"MTPLX_FABLE_OPDIET_ITEMS={raw!r} has unknown item(s) "
            f"{', '.join(unknown)}; expected a comma list from: "
            f"{', '.join(known)}"
        )
    return frozenset(tokens)


_FABLE_OPDIET_SELECTED = parse_opdiet_items(
    os.environ.get("MTPLX_FABLE_OPDIET_ITEMS")
)


def fable_opdiet_enabled(item: str | None = None) -> bool:
    """True when the op diet is armed, and this item is selected.

    ``item=None`` answers only the master switch. Every gated call site names
    its item so ``MTPLX_FABLE_OPDIET_ITEMS`` can isolate one rewrite.
    """

    if not _FABLE_OPDIET:
        return False
    if item is None:
        return True
    if item not in FABLE_OPDIET_ITEMS:
        raise ValueError(f"unknown op-diet item {item!r}")
    return item in _FABLE_OPDIET_SELECTED


@dataclass(frozen=True)
class ResolvedAPIKey:
    value: str | None
    source: str

    @property
    def required(self) -> bool:
        return bool(self.value)


def parser_option_names(parser: object, namespace: object = None) -> set[str]:
    """Every option name reachable in the parse context, without dashes.

    Walks the root parser plus whichever subparsers the parse actually
    descended into (using ``namespace`` to pick the branch), which is the
    same scope argparse resolves abbreviations against.
    """

    names: set[str] = set()
    seen: set[int] = set()
    pending = [parser]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        for action in getattr(current, "_actions", ()):
            for option in getattr(action, "option_strings", ()) or ():
                names.add(str(option).lstrip("-"))
            choices = getattr(action, "choices", None)
            dest = getattr(action, "dest", None)
            if not isinstance(choices, dict) or not dest:
                continue
            # A subparsers action: follow only the branch that was taken.
            picked = getattr(namespace, str(dest), None) if namespace else None
            if picked is not None and picked in choices:
                pending.append(choices[picked])
            elif namespace is None:
                pending.extend(choices.values())
    return names


def canonicalize_flag_tokens(
    tokens: set[str],
    parser: object,
    namespace: object = None,
) -> set[str]:
    """Expand argparse abbreviations to the flag names they resolved to.

    ``--temp 0.9`` sets ``args.temperature`` but was recorded as ``temp``,
    so every ``"temperature" in cli_flags`` check read it as *not typed*
    and the config file happily overwrote the user's value. Resolving here
    (rather than setting ``allow_abbrev=False``) keeps abbreviations
    working while making the explicit-flag signal true.

    The raw token is kept alongside the expansion, so checks written
    against either spelling keep working. Ambiguous prefixes expand to
    nothing — argparse would have rejected the command anyway.
    """

    known = parser_option_names(parser, namespace)
    resolved = set(tokens)
    for token in tokens:
        if token in known:
            continue
        matches = {name for name in known if name.startswith(token)}
        if len(matches) == 1:
            resolved |= matches
    return resolved


def block_prefix_restore_enabled() -> bool:
    """The single parse of ``MTPLX_SESSION_BLOCK_PREFIX_RESTORE``.

    Default ON. Every reader — the decode loop, the engine session, the
    session bank's cold tier, and the server's settings view — goes through
    this, so one spelling cannot mean ON in one and OFF in another. Unset
    used to mean OFF in the cold tier and ON everywhere else, which
    silently disabled cold-tier block-prefix restore for library embedders
    (the CLI path masks it by force-setting "1"); and the server's
    allowlist-only read reported "off" for spellings the runtime honours as
    on, e.g. ``=enabled``.

    Lives here rather than in :mod:`mtplx.session_bank` because this module
    has no heavy imports: the server reads the setting on paths where the
    mlx-backed runtime may be unavailable.
    """

    return env_bool("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", default=True)


def normalize_paged_kv_quantization(value: object | None, *, allow_none: bool = False) -> str | None:
    if value is None:
        if allow_none:
            return None
        return "off"
    raw = str(value).strip().lower().replace("-", "_")
    if raw in ("", "none", "false", "0", "disabled", "disable"):
        return "off"
    if raw in ("off", "q8", "q4"):
        return raw
    if raw in ("8", "8bit", "int8", "uint8", "q8_0"):
        return "q8"
    if raw in ("4", "4bit", "int4", "uint4", "q4_0"):
        return "q4"
    choices = ", ".join(KV_QUANT_MODES)
    raise ValueError(f"unsupported paged KV quantization mode {value!r}; expected one of: {choices}")


def paged_kv_quantization_env(mode: object | None) -> dict[str, str]:
    canonical = normalize_paged_kv_quantization(mode)
    return {
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": canonical,
        "MTPLX_PAGED_KV_QUANT": canonical,
    }


def apply_paged_kv_quantization_env(mode: object | None, env: dict[str, str] | None = None) -> str:
    canonical = normalize_paged_kv_quantization(mode)
    target = os.environ if env is None else env
    target.update(paged_kv_quantization_env(canonical))
    return canonical


def generate_api_key_file(api_key_file: str | os.PathLike[str]) -> str:
    """Create ``api_key_file`` holding a fresh random key and return the key.

    Server entrypoints call this when the user passed ``--api-key-file`` for a
    path that does not exist yet, so the recovery command our own non-localhost
    refusal prints is runnable as-is instead of dying on FileNotFoundError.
    Read-only consumers of key files (doctor, connect) must NOT call this — a
    missing file is a real error there. The file is created 0600.
    """
    import secrets

    path = Path(api_key_file).expanduser()
    key = "mtplx-" + secrets.token_hex(24)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    return key


def resolve_api_key(
    *,
    explicit_api_key: str | None = None,
    api_key_file: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedAPIKey:
    explicit = _clean_secret(explicit_api_key)
    if explicit:
        return ResolvedAPIKey(explicit, "flag")

    if api_key_file:
        path = Path(api_key_file).expanduser()
        secret = _clean_secret(path.read_text(encoding="utf-8"))
        if not secret:
            raise ValueError(f"API key file is empty: {path}")
        return ResolvedAPIKey(secret, "file")

    source_env = os.environ if env is None else env
    api_key = _clean_secret(source_env.get("MTPLX_API_KEY"))
    if api_key:
        return ResolvedAPIKey(api_key, "env:MTPLX_API_KEY")

    legacy = _clean_secret(source_env.get("MTPLX_AUTH"))
    if legacy:
        return ResolvedAPIKey(legacy, "env:MTPLX_AUTH")

    return ResolvedAPIKey(None, "none")


def _clean_secret(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
