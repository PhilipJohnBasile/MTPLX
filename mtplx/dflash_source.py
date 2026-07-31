"""Commit-only experimental DFlash draft-source foundation.

The lane supports greedy proposals declared as point masses and independently
sampled proposals carrying their exact sparse q distributions.  Sampling stays
outside compiled MLX code and uses the request RNG.  Exact output sampling then
depends on the engine applying probability-ratio acceptance plus residual
correction against those q rows; byte equality still depends on authoritative
target-verifier numerics and remains an external release gate.

This module owns only the companion drafter and target-layer taps.  Target
cache commit/rollback remains in :func:`mtplx.generation.generate_mtpk`.
"""

from __future__ import annotations

import hashlib
import importlib
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BlockDraftProposal:
    """One independently verified block proposal."""

    tokens: tuple[int, ...]
    source: str
    elapsed_s: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Exact proposal distributions. DFlash emits explicit point masses on the
    # greedy lane and sparse q rows on the stochastic lane. ``None`` remains
    # available for legacy third-party sources, which the engine must reject
    # when exact probability-ratio acceptance requires q.
    draft_qs: tuple[Any, ...] | None = None


class _TappedLayer:
    def __init__(self, layer: Any, slot: int, storage: list[list[Any]]) -> None:
        self._mtplx_dflash_original = layer
        self._slot = int(slot)
        self._storage = storage

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._mtplx_dflash_original(*args, **kwargs)
        captured = (
            output[0] if isinstance(output, tuple) else output
        )
        self._storage[self._slot].append(captured)
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mtplx_dflash_original, name)


class TargetTapCapture:
    """Capture selected target-layer outputs without global monkey patches."""

    def __init__(self, target_model: Any, layer_ids: Sequence[int]) -> None:
        ids = tuple(int(layer_id) for layer_id in layer_ids)
        if not ids:
            raise ValueError("DFlash target_layer_ids cannot be empty")
        layers = _target_layers(target_model)
        if len(set(ids)) != len(ids):
            raise ValueError("DFlash target_layer_ids must be unique")
        if min(ids) < 0 or max(ids) >= len(layers):
            raise ValueError(
                f"DFlash target layer outside [0, {len(layers) - 1}]: {ids}"
            )
        self._layers = layers
        self._ids = ids
        self._storage: list[list[Any]] = [[] for _ in ids]
        self._originals: list[Any] = []
        self._closed = False
        for slot, layer_id in enumerate(ids):
            original = layers[layer_id]
            if hasattr(original, "_mtplx_dflash_original"):
                raise RuntimeError(
                    f"target layer {layer_id} already has a DFlash tap"
                )
            self._originals.append(original)
            layers[layer_id] = _TappedLayer(original, slot, self._storage)

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self._ids

    def current(self) -> Any:
        missing = [
            layer_id
            for layer_id, values in zip(self._ids, self._storage)
            if not values
        ]
        if missing:
            raise RuntimeError(
                "target forward did not populate DFlash taps for layers "
                f"{missing}; compiled target paths must be disabled"
            )
        import mlx.core as mx

        per_layer = [
            values[0] if len(values) == 1 else mx.concatenate(values, axis=1)
            for values in self._storage
        ]
        sequence_lengths = {int(value.shape[1]) for value in per_layer}
        if len(sequence_lengths) != 1:
            raise RuntimeError(
                "DFlash target taps captured inconsistent sequence lengths: "
                f"{sorted(sequence_lengths)}"
            )
        return mx.concatenate(per_layer, axis=-1)

    def reset(self) -> None:
        for values in self._storage:
            values.clear()

    def close(self) -> None:
        if self._closed:
            return
        for layer_id, original in zip(self._ids, self._originals):
            current = self._layers[layer_id]
            if isinstance(current, _TappedLayer):
                self._layers[layer_id] = original
        self._closed = True


@dataclass(frozen=True)
class _PortDraftConfig:
    target_layer_ids: tuple[int, ...]
    block_size: int
    mask_token_id: int


class _DFlashMLXPortAdapter:
    """Adapt the pinned ``dflash-mlx`` package to the in-engine source API.

    The PyPI port exposes a context-only cache and returns draft hidden states;
    MTPLX supplies the target embedding and LM head that DFlash intentionally
    shares.  Keeping this adapter local avoids depending on z-lab's unpublished
    ``dflash`` package from published project metadata.
    """

    def __init__(self, model: Any, cache_type: type[Any]) -> None:
        self._model = model
        self._cache_type = cache_type
        self._embed_tokens: Any | None = None
        self._lm_head: Any | None = None
        self._embed_scale: Any = 1.0
        self.config = _PortDraftConfig(
            target_layer_ids=tuple(int(value) for value in model.target_layer_ids),
            block_size=int(model.block_size),
            mask_token_id=int(model.mask_token_id),
        )

    def bind(self, target_model: Any) -> "_DFlashMLXPortAdapter":
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        else:
            raise AttributeError(
                f"cannot find target embed_tokens in {type(target_model).__name__}"
            )
        language_model = getattr(target_model, "language_model", target_model)
        lm_head = getattr(target_model, "lm_head", None) or getattr(
            language_model, "lm_head", None
        )
        if lm_head is None:
            lm_head = getattr(inner.embed_tokens, "as_linear", None)
        if not callable(lm_head):
            raise AttributeError(
                f"cannot find target lm_head in {type(target_model).__name__}"
            )
        self._embed_tokens = inner.embed_tokens
        self._embed_scale = getattr(
            self._embed_tokens,
            "embed_scale",
            getattr(inner, "embed_scale", 1.0),
        )
        self._lm_head = lm_head
        return self

    def make_cache(self) -> list[Any]:
        return [self._cache_type() for _ in self._model.layers]

    def __call__(
        self,
        inputs: Any,
        target_hidden: Any,
        cache: list[Any],
        *,
        logits_start: int = 0,
    ) -> Any:
        if self._embed_tokens is None or self._lm_head is None:
            raise RuntimeError("DFlash port adapter must be bound to the target")
        noise_embedding = self._embed_tokens(inputs) * self._embed_scale
        hidden = self._model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=cache,
        )
        if logits_start:
            hidden = hidden[:, int(logits_start) :]
        return self._lm_head(hidden)


def load_dflash_draft(
    model_ref: str,
    *,
    revision: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve an immutable Hub revision and load the pinned DFlash port."""

    try:
        runtime_module = importlib.import_module("dflash_mlx.runtime")
        model_module = importlib.import_module("dflash_mlx.model")
    except Exception as exc:
        raise RuntimeError(
            "DFlash support requires the optional 'competitors' dependency "
            "(pip install 'mtplx[competitors]')"
        ) from exc
    load_bundle = getattr(runtime_module, "load_draft_bundle", None)
    cache_type = getattr(model_module, "ContextOnlyDraftKVCache", None)
    if not callable(load_bundle) or not callable(cache_type):
        raise RuntimeError(
            "installed dflash-mlx does not expose load_draft_bundle and "
            "ContextOnlyDraftKVCache"
        )
    requested_ref = str(model_ref)
    local_candidate = Path(requested_ref).expanduser()
    if local_candidate.exists():
        load_ref = str(local_candidate.resolve())
        resolved_revision = None
        source_kind = "local"
    else:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(requested_ref, revision=revision)
        resolved_revision = str(info.sha or "")
        if len(resolved_revision) != 40:
            raise RuntimeError(
                f"Hugging Face did not resolve an immutable SHA for {requested_ref!r}"
            )
        load_ref = snapshot_download(
            requested_ref,
            revision=resolved_revision,
            allow_patterns=["*.safetensors", "*.json"],
        )
        source_kind = "huggingface"

    loaded = load_bundle(load_ref)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("dflash-mlx load_draft_bundle returned an invalid result")
    model, loader_metadata = loaded
    adapter = _DFlashMLXPortAdapter(model, cache_type)
    try:
        package_version = version("dflash-mlx")
    except PackageNotFoundError:
        package_version = "unknown"
    metadata = {
        "loader": "dflash_mlx.runtime.load_draft_bundle",
        "loader_package": "dflash-mlx",
        "loader_package_version": package_version,
        "model_ref": requested_ref,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "immutable_revision": resolved_revision is not None,
        "source_kind": source_kind,
        "resolved_model_ref": str(
            dict(loader_metadata or {}).get("resolved_model_ref") or load_ref
        ),
        "artifact_layout_sha256": _artifact_layout_sha256(Path(load_ref)),
        "context_cache": "dflash_mlx.model.ContextOnlyDraftKVCache",
    }
    return adapter, metadata


def _artifact_layout_sha256(path: Path) -> str:
    root = path
    digest = hashlib.sha256()
    for candidate in sorted(root.glob("*")):
        if not candidate.is_file():
            continue
        if candidate.name.endswith(".safetensors"):
            digest.update(f"{candidate.name}:{candidate.stat().st_size}\n".encode())
        elif candidate.name in {"config.json", "model.safetensors.index.json"}:
            digest.update(candidate.name.encode())
            digest.update(candidate.read_bytes())
    return f"layout-sha256:{digest.hexdigest()}"


class DFlashDraftSource:
    """A fixed-width DFlash source with committed-only context."""

    name = "dflash-one-hot"

    def __init__(
        self,
        draft_model_ref: str,
        *,
        block_size: int = 8,
        staged_k1: bool = False,
        draft_model: Any | None = None,
        draft_artifact: Mapping[str, Any] | None = None,
        draft_revision: str | None = None,
    ) -> None:
        if not 2 <= int(block_size) <= 16:
            raise ValueError("DFlash block_size must be in [2, 16]")
        self.draft_model_ref = str(draft_model_ref)
        self.block_size = int(block_size)
        self.staged_k1 = bool(staged_k1)
        self.draft_revision = None if draft_revision is None else str(draft_revision)
        self.supports_target_prefix = self.staged_k1
        self.requires_compiled_a3b_target_prefix = self.staged_k1
        self._draft = draft_model
        self._draft_artifact = dict(draft_artifact or {})
        self._draft_cache: list[Any] | None = None
        self._tap_capture: TargetTapCapture | None = None
        self._target_model: Any | None = None
        self._pending_target_hidden: Any | None = None
        self._prepared = False
        self._outstanding = False
        self._proposal_count = 0
        self._committed_context_rows = 0
        self._queued_tokens: list[int] = []
        self._queued_distributions: list[Any] = []
        self._queued_soft_q = False

    @property
    def identity(self) -> str:
        revision = (
            self._draft_artifact.get("resolved_revision")
            or self.draft_revision
            or "mutable"
        )
        return (
            f"dflash:{self.draft_model_ref}@{revision}:"
            f"b{self.block_size}:host-q-v1"
        )

    @property
    def target_model(self) -> Any | None:
        return self._target_model

    @property
    def target_layer_count(self) -> int:
        return 0 if self._tap_capture is None else len(self._tap_capture.layer_ids)

    def prepare(self, runtime: Any) -> None:
        if self._prepared:
            return
        draft = self._draft
        if draft is None:
            try:
                draft, artifact = load_dflash_draft(
                    self.draft_model_ref,
                    revision=self.draft_revision,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load DFlash drafter {self.draft_model_ref!r}"
                ) from exc
            self._draft_artifact = dict(artifact)
        config = getattr(draft, "config", None)
        layer_ids = tuple(getattr(config, "target_layer_ids", ()) or ())
        configured_block = int(getattr(config, "block_size", 0) or 0)
        if configured_block and self.block_size > configured_block:
            raise ValueError(
                f"requested block {self.block_size} exceeds drafter block "
                f"{configured_block}"
            )
        draft.bind(runtime.model)
        runtime.model._mtplx_dflash_target_layer_ids = layer_ids
        runtime.model._mtplx_dflash_captured_hidden = None
        self._target_model = runtime.model
        self._tap_capture = TargetTapCapture(runtime.model, layer_ids)
        self._draft_cache = list(draft.make_cache())
        self._draft = draft
        self._prepared = True

    def begin_request(self) -> None:
        """Reset all companion state before a fresh prompt prefill."""

        if not self._prepared or self._draft is None or self._tap_capture is None:
            raise RuntimeError("DFlashDraftSource.prepare(runtime) must run first")
        if self._outstanding:
            raise RuntimeError("cannot reset DFlash with an outstanding proposal")
        self._draft_cache = list(self._draft.make_cache())
        self._pending_target_hidden = None
        self._proposal_count = 0
        self._committed_context_rows = 0
        self._queued_tokens.clear()
        self._queued_distributions.clear()
        self._queued_soft_q = False
        self._tap_capture.reset()
        if self._target_model is not None:
            self._target_model._mtplx_dflash_captured_hidden = None

    def propose(
        self,
        *,
        primary_token: int,
        max_draft_tokens: int,
        committed_tokens: Sequence[int],
        target_hidden: Any,
        draft_sampler: Any | None = None,
        rng: Any | None = None,
    ) -> BlockDraftProposal:
        del target_hidden  # DFlash consumes the selected multi-layer tap tensor.
        if not self._prepared or self._draft is None or self._tap_capture is None:
            raise RuntimeError("DFlashDraftSource.prepare(runtime) must run first")
        if self._draft_cache is None:
            raise RuntimeError("DFlash draft cache is unavailable")
        if self._outstanding:
            raise RuntimeError("previous DFlash proposal was not committed")

        requested_drafts = max(0, int(max_draft_tokens))
        draft_count = min(requested_drafts, self.block_size - 1)
        if draft_count <= 0:
            return BlockDraftProposal(tokens=(), source=self.name)

        stochastic = bool(
            draft_sampler is not None
            and float(getattr(draft_sampler, "temperature", 0.0)) > 0.0
        )
        if stochastic:
            if int(getattr(draft_sampler, "top_k", 0)) <= 0:
                raise ValueError(
                    "stochastic DFlash requires draft_sampler.top_k > 0 "
                    "so its exact q distribution stays bounded"
                )
            if (
                float(getattr(draft_sampler, "presence_penalty", 0.0)) != 0.0
                or float(getattr(draft_sampler, "frequency_penalty", 0.0)) != 0.0
            ):
                raise ValueError(
                    "stochastic DFlash does not yet support draft penalties; "
                    "refusing to misdeclare q"
                )
            if rng is None:
                raise ValueError(
                    "stochastic DFlash requires the request RNG for host sampling"
                )

        import mlx.core as mx

        if self.staged_k1 and requested_drafts != 1:
            raise RuntimeError(
                "staged-K1 DFlash requires speculative_depth=1"
            )
        if self.staged_k1 and self._queued_tokens:
            if len(self._queued_distributions) != len(self._queued_tokens):
                self._queued_tokens.clear()
                self._queued_distributions.clear()
                self._queued_soft_q = False
                raise RuntimeError("staged DFlash token/q queues lost alignment")
            expected_primary = self._queued_tokens.pop(0)
            self._queued_distributions.pop(0)
            if int(primary_token) != int(expected_primary):
                self._queued_tokens.clear()
                self._queued_distributions.clear()
                self._queued_soft_q = False
            elif self._queued_tokens:
                staged_token = int(self._queued_tokens.pop(0))
                staged_distribution = self._queued_distributions.pop(0)
                staged_soft_q = self._queued_soft_q
                if not self._queued_tokens:
                    self._queued_soft_q = False
                self._proposal_count += 1
                self._outstanding = True
                return BlockDraftProposal(
                    tokens=(staged_token,),
                    source=(
                        "dflash-soft-q-staged"
                        if staged_soft_q
                        else f"{self.name}-staged"
                    ),
                    draft_qs=(staged_distribution,),
                    metadata={
                        "block_size": self.block_size,
                        "proposal": self._proposal_count,
                        "target_context_rows": 0,
                        "committed_context_rows": self._committed_context_rows,
                        "completion_tokens_before_cycle": len(committed_tokens),
                        "declaration": (
                            "sampled_q"
                            if staged_soft_q
                            else "one_hot"
                        ),
                        "sampler": _sampler_metadata(draft_sampler),
                        "sampling_policy": (
                            "host_request_rng_independent_rows"
                            if staged_soft_q
                            else "host_argmax"
                        ),
                        "draft_forward_calls": 0,
                        "proposal_block_from_single_forward": True,
                        "q_support_sizes": [
                            int(len(staged_distribution.token_ids))
                        ],
                        "draft_artifact": dict(self._draft_artifact),
                        "staged_k1": True,
                        "queued_tokens_remaining": len(self._queued_tokens),
                    },
                )

        target_context = (
            self._pending_target_hidden
            if self._pending_target_hidden is not None
            else self._tap_capture.current()
        )
        context_rows = int(target_context.shape[1])
        if context_rows <= 0:
            raise RuntimeError("DFlash target context is empty")
        self._pending_target_hidden = None

        mask_id = int(getattr(self._draft.config, "mask_token_id"))
        if self.staged_k1:
            draft_count = self.block_size - 1
        block = mx.array(
            [[int(primary_token), *([mask_id] * draft_count)]],
            dtype=mx.int32,
        )
        started = time.perf_counter()
        logits = self._draft(
            block,
            target_context,
            self._draft_cache,
            logits_start=1,
        )
        distributions: tuple[Any, ...] | None
        if stochastic:
            # Compile may own the drafter forward and its logits, never the
            # random draw.  Materialize the exact sparse q rows, then sample
            # independently on the host with the request RNG.
            from .fast_sampling import sparse_distributions_from_mlx_logits
            from .sampling import sample_from_distribution

            q_rows = sparse_distributions_from_mlx_logits(logits, draft_sampler)
            if q_rows is None or len(q_rows) != draft_count:
                found = 0 if q_rows is None else len(q_rows)
                raise RuntimeError(
                    f"DFlash produced {found} q rows for {draft_count} masks"
                )
            sampled_tokens = tuple(
                sample_from_distribution(distribution, rng)
                for distribution in q_rows
            )
            for token, distribution in zip(
                sampled_tokens, q_rows, strict=True
            ):
                if distribution.probability(token) <= 0.0:
                    raise RuntimeError(
                        "DFlash sampled a token outside its declared q support"
                    )
            tokens = sampled_tokens
            distributions = tuple(q_rows)
        else:
            # Greedy host extraction remains an exact point-mass declaration.
            from .sampling import SparseDistribution

            token_array = mx.argmax(logits, axis=-1)
            mx.eval(token_array)
            tokens = tuple(
                int(token) for token in token_array.reshape(-1).tolist()
            )
            vocab_size = int(logits.shape[-1])
            distributions = tuple(
                SparseDistribution.one_hot(token, vocab_size) for token in tokens
            )
        elapsed = time.perf_counter() - started
        if len(tokens) != draft_count:
            raise RuntimeError(
                f"DFlash returned {len(tokens)} tokens for {draft_count} masks"
        )
        if self.staged_k1:
            self._queued_tokens = [int(token) for token in tokens[1:]]
            self._queued_distributions = list(distributions[1:])
            self._queued_soft_q = stochastic
            tokens = tokens[:1]
            distributions = distributions[:1]

        self._committed_context_rows += context_rows
        self._proposal_count += 1
        self._outstanding = True
        return BlockDraftProposal(
            tokens=tokens,
            source="dflash-soft-q" if stochastic else self.name,
            draft_qs=distributions,
            elapsed_s=elapsed,
            metadata={
                "block_size": self.block_size,
                "proposal": self._proposal_count,
                "target_context_rows": context_rows,
                "committed_context_rows": self._committed_context_rows,
                "completion_tokens_before_cycle": len(committed_tokens),
                "declaration": (
                    "sampled_q" if stochastic else "one_hot"
                ),
                "sampler": _sampler_metadata(draft_sampler),
                "sampling_policy": (
                    "host_request_rng_independent_rows"
                    if stochastic
                    else "host_argmax"
                ),
                "draft_forward_calls": 1,
                "proposal_block_from_single_forward": True,
                "q_support_sizes": [
                    int(len(distribution.token_ids))
                    for distribution in distributions
                ],
                "draft_artifact": dict(self._draft_artifact),
                "staged_k1": self.staged_k1,
                "queued_tokens_remaining": len(self._queued_tokens),
            },
        )

    def begin_target_verify(self) -> None:
        """Start a new tap epoch for the target verify forward."""

        if not self._outstanding or self._tap_capture is None:
            raise RuntimeError("DFlash proposal must exist before target verify")
        self._tap_capture.reset()
        if self._target_model is not None:
            self._target_model._mtplx_dflash_captured_hidden = None

    def commit_target_prefix(
        self,
        accepted_draft_tokens: int,
        *,
        committed_target_rows: int | None = None,
        residual_correction_rows: int = 0,
    ) -> None:
        """Stage only target rows belonging to the committed prefix.

        Target verify rows are ``[primary, *drafts]``.  Therefore an
        acceptance count of ``a`` commits exactly ``a + 1`` tapped target rows
        for the companion's next context.  Rejected/uncommitted rows never
        enter the draft cache and need no rollback.
        """

        if not self._outstanding or self._tap_capture is None:
            raise RuntimeError("no outstanding DFlash proposal to commit")
        import mlx.core as mx

        accepted_draft_tokens = int(accepted_draft_tokens)
        residual_correction_rows = int(residual_correction_rows)
        keep_rows = (
            accepted_draft_tokens + 1
            if committed_target_rows is None
            else int(committed_target_rows)
        )
        if keep_rows != accepted_draft_tokens + 1 + residual_correction_rows:
            raise RuntimeError(
                "committed DFlash tap rows do not match primary + accepted "
                "drafts + residual corrections"
            )
        captured = self._captured_target_hidden()
        if keep_rows < 1 or keep_rows > int(captured.shape[1]):
            raise RuntimeError(
                f"cannot commit {keep_rows} rows from capture length "
                f"{int(captured.shape[1])}"
            )
        committed_hidden = captured[:, :keep_rows, :]
        if self._pending_target_hidden is None:
            self._pending_target_hidden = committed_hidden
        else:
            self._pending_target_hidden = mx.concatenate(
                [self._pending_target_hidden, committed_hidden],
                axis=1,
            )
        mx.eval(self._pending_target_hidden)
        if self.staged_k1 and (
            accepted_draft_tokens != 1 or residual_correction_rows != 0
        ):
            self._queued_tokens.clear()
            self._queued_distributions.clear()
            self._queued_soft_q = False
        self._outstanding = False

    def close(self) -> None:
        if self._tap_capture is not None:
            self._tap_capture.close()
        if self._target_model is not None:
            for name in (
                "_mtplx_dflash_target_layer_ids",
                "_mtplx_dflash_captured_hidden",
            ):
                if hasattr(self._target_model, name):
                    delattr(self._target_model, name)
        self._tap_capture = None
        self._target_model = None
        self._draft_cache = None
        self._pending_target_hidden = None
        self._queued_tokens.clear()
        self._queued_distributions.clear()
        self._queued_soft_q = False
        self._prepared = False
        self._outstanding = False

    def _captured_target_hidden(self) -> Any:
        if self._draft is None or self._tap_capture is None:
            raise RuntimeError("DFlash source is not prepared")
        captured = (
            getattr(self._target_model, "_mtplx_dflash_captured_hidden", None)
            if self._target_model is not None
            else None
        )
        if captured is not None:
            import mlx.core as mx

            return mx.concatenate(list(captured), axis=-1)
        return self._tap_capture.current()


def _sampler_metadata(config: Any | None) -> dict[str, Any]:
    return {
        "temperature": float(getattr(config, "temperature", 0.0) or 0.0),
        "top_p": float(getattr(config, "top_p", 1.0) or 0.0),
        "top_k": int(getattr(config, "top_k", 0) or 0),
        "presence_penalty": float(
            getattr(config, "presence_penalty", 0.0) or 0.0
        ),
        "frequency_penalty": float(
            getattr(config, "frequency_penalty", 0.0) or 0.0
        ),
    }


def _target_layers(model: Any) -> Any:
    candidates = (
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "language_model", None), "model", None),
        model,
    )
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    raise AttributeError(f"cannot find target layers in {type(model).__name__}")
