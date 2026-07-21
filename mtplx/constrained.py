"""Grammar-constrained decoding (structured output) for the serial AR path.

Phase 1 of the plan in upstream issue #186: ``response_format`` of type
``json_object`` / ``json_schema`` is enforced with llguidance token bitmasks
applied to target logits before sampling, on the serial AR lane only.
Constrained requests never ride the batched AR pump or the MTP lanes; the
server pins them to ``generation_mode="ar"`` and bypasses the batch scheduler.

llguidance is an optional dependency: requests that do not use
``response_format`` never touch it, and requests that do get a clear 400 when
it is missing instead of silent non-enforcement (which is what shipped before
this module existed).

The mask must hit the logits row before any shaping (temperature, top-p/k,
penalties) so that both the greedy argmax branch and the sampled branch of
``_sample_from_logits`` operate on the constrained distribution. Illegal
tokens are set to -inf, which survives every downstream shaping step.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - exercised via LLGUIDANCE_AVAILABLE branches
    import llguidance as _llg
    import llguidance.hf as _llg_hf
    import llguidance.mlx as _llg_mlx

    LLGUIDANCE_AVAILABLE = True
    LLGUIDANCE_VERSION = str(_llg.get_version())
except Exception:  # pragma: no cover
    _llg = None
    _llg_hf = None
    _llg_mlx = None
    LLGUIDANCE_AVAILABLE = False
    LLGUIDANCE_VERSION = None

SUPPORTED_RESPONSE_FORMAT_TYPES = ("text", "json_object", "json_schema")

# ``json_object`` promises a JSON object (OpenAI semantics), not merely any
# JSON value, so the generic grammar pins the top-level type.
_JSON_OBJECT_SCHEMA = '{"type": "object"}'


class ResponseFormatError(ValueError):
    """Invalid or unsupported ``response_format``; message is client-safe."""


@dataclass(frozen=True)
class ConstraintSpec:
    """A validated, tokenizer-independent grammar for one request.

    Built once at request-validation time (so bad schemas 400 before any
    model work) and bound to the runtime tokenizer lazily via ``build`` —
    once per generation attempt, because matcher state is consumed by a
    generation and blank-retry attempts must start fresh.
    """

    grammar: str
    source_type: str

    def build(self, tokenizer: Any) -> "GrammarConstraint":
        return GrammarConstraint(self.grammar, tokenizer)


def constraint_spec_from_response_format(
    response_format: Any,
) -> ConstraintSpec | None:
    """Parse/validate a request's ``response_format`` into a ConstraintSpec.

    Returns None when no constraint applies (absent or ``type: text``).
    Raises ResponseFormatError for anything the server cannot honestly
    enforce — the caller turns that into a 400.
    """
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise ResponseFormatError(
            "response_format must be an object with a 'type' field"
        )
    format_type = response_format.get("type")
    if format_type not in SUPPORTED_RESPONSE_FORMAT_TYPES:
        raise ResponseFormatError(
            "unsupported response_format type "
            f"{format_type!r}; supported: {', '.join(SUPPORTED_RESPONSE_FORMAT_TYPES)}"
        )
    if format_type == "text":
        return None
    if not LLGUIDANCE_AVAILABLE:
        raise ResponseFormatError(
            f"response_format type {format_type!r} requires the optional "
            "llguidance dependency (pip install llguidance); refusing to "
            "silently return unconstrained output"
        )
    if format_type == "json_object":
        schema_json = _JSON_OBJECT_SCHEMA
    else:
        wrapper = response_format.get("json_schema")
        if wrapper is None and isinstance(response_format.get("schema"), dict):
            # Lenient shape some clients send: {"type": "json_schema",
            # "schema": {...}} without the OpenAI wrapper object.
            schema = response_format["schema"]
        elif isinstance(wrapper, dict):
            schema = wrapper.get("schema")
        else:
            schema = None
        if not isinstance(schema, dict):
            raise ResponseFormatError(
                "response_format type 'json_schema' requires json_schema.schema "
                "to be a JSON Schema object"
            )
        schema_json = _canonical_schema_json(schema)
    grammar = _cached_grammar_for_schema(schema_json)
    return ConstraintSpec(grammar=grammar, source_type=str(format_type))


class GrammarConstraint:
    """Per-generation matcher state: mask logits rows, advance per token.

    The llguidance tokenizer wrap needs the model's logits width (which can
    exceed the tokenizer vocab on padded lm_heads), so binding is deferred to
    the first ``mask_logits_row`` call, where the row's shape provides it.
    Tokens beyond the tokenizer vocab are always masked out.
    """

    def __init__(self, grammar: str, tokenizer: Any):
        self._grammar = grammar
        self._tokenizer = tokenizer
        self._matcher: Any | None = None
        self._bitmask: Any | None = None
        self.masked_steps = 0
        self.mask_time_s = 0.0

    def _bind(self, n_vocab: int) -> None:
        ll_tokenizer = _cached_ll_tokenizer(self._tokenizer, n_vocab)
        matcher = _llg.LLMatcher(ll_tokenizer, self._grammar)
        err = matcher.get_error()
        if err:
            raise ResponseFormatError(f"response_format grammar rejected: {err}")
        self._matcher = matcher
        self._bitmask = _llg_mlx.allocate_token_bitmask(1, n_vocab)

    def mask_logits_row(self, row: Any) -> Any:
        """Apply the current-step token mask to a 1-D logits row (mx.array)."""
        if self._matcher is None:
            self._bind(int(row.shape[-1]))
        if self._matcher.is_stopped():
            return row
        started = time.perf_counter()
        _llg_mlx.fill_next_token_bitmask(self._matcher, self._bitmask)
        masked = _llg_mlx.apply_token_bitmask(row.reshape(1, -1), self._bitmask)
        self.mask_time_s += time.perf_counter() - started
        self.masked_steps += 1
        return masked.reshape(row.shape)

    def advance(self, token_id: int) -> None:
        if self._matcher is not None and not self._matcher.is_stopped():
            self._matcher.consume_token(int(token_id))

    @property
    def stopped(self) -> bool:
        return self._matcher is not None and bool(self._matcher.is_stopped())

    @property
    def completed(self) -> bool:
        """True when the emitted text is a complete document per the grammar."""
        if self._matcher is None:
            return False
        return bool(self._matcher.is_accepting() or self._matcher.is_stopped())


# --- caches ---------------------------------------------------------------
#
# A compiled grammar is schema- and engine-version-specific; the LLTokenizer
# wrap is tokenizer-object- and vocab-width-specific. Both caches hold strong
# references (the server keeps one tokenizer for its lifetime) and are
# bounded, so id() reuse after GC cannot alias a live entry.

_GRAMMAR_CACHE: OrderedDict[str, str] = OrderedDict()
_GRAMMAR_CACHE_MAX = 64
_TOKENIZER_CACHE: dict[tuple[int, int], tuple[Any, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _canonical_schema_json(schema: dict[str, Any]) -> str:
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def _cached_grammar_for_schema(schema_json: str) -> str:
    key = f"{LLGUIDANCE_VERSION}:{schema_json}"
    with _CACHE_LOCK:
        cached = _GRAMMAR_CACHE.get(key)
        if cached is not None:
            _GRAMMAR_CACHE.move_to_end(key)
            return cached
    try:
        grammar = _llg.LLMatcher.grammar_from_json_schema(schema_json)
    except Exception as exc:
        raise ResponseFormatError(f"unsupported JSON Schema: {exc}") from exc
    err = _llg.LLMatcher.validate_grammar(grammar)
    if err:
        raise ResponseFormatError(f"unsupported JSON Schema: {err}")
    with _CACHE_LOCK:
        _GRAMMAR_CACHE[key] = grammar
        while len(_GRAMMAR_CACHE) > _GRAMMAR_CACHE_MAX:
            _GRAMMAR_CACHE.popitem(last=False)
    return grammar


def _unwrap_hf_tokenizer(tokenizer: Any) -> Any:
    """Return the underlying fast tokenizer llguidance requires.

    The runtime hands us mlx_lm's TokenizerWrapper, which delegates
    attribute access to the fast tokenizer it holds but fails llguidance's
    strict isinstance check; unwrap it when present.
    """
    import transformers

    if isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
        return tokenizer
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None and isinstance(inner, transformers.PreTrainedTokenizerFast):
        return inner
    return tokenizer


def _cached_ll_tokenizer(tokenizer: Any, n_vocab: int) -> Any:
    tokenizer = _unwrap_hf_tokenizer(tokenizer)
    key = (id(tokenizer), int(n_vocab))
    with _CACHE_LOCK:
        entry = _TOKENIZER_CACHE.get(key)
        if entry is not None and entry[0] is tokenizer:
            return entry[1]
    ll_tokenizer = _llg_hf.from_tokenizer(tokenizer, n_vocab=int(n_vocab))
    with _CACHE_LOCK:
        _TOKENIZER_CACHE[key] = (tokenizer, ll_tokenizer)
    return ll_tokenizer
