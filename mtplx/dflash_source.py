"""Commit-only experimental DFlash draft-source foundation.

The lane deliberately starts with greedy proposals declared as point masses.
That declaration preserves the rejection-sampling law for any dependence
between DFlash positions, but byte equality still depends on authoritative
target-verifier numerics and remains an external release gate.

This module owns only the companion drafter and target-layer taps.  Target
cache commit/rollback remains in :func:`mtplx.generation.generate_mtpk`.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BlockDraftProposal:
    """One independently verified block proposal."""

    tokens: tuple[int, ...]
    source: str
    elapsed_s: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


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


class DFlashDraftSource:
    """A fixed-width greedy DFlash source with committed-only context."""

    name = "dflash-one-hot"

    def __init__(
        self,
        draft_model_ref: str,
        *,
        block_size: int = 8,
        staged_k1: bool = False,
        draft_model: Any | None = None,
    ) -> None:
        if not 2 <= int(block_size) <= 16:
            raise ValueError("DFlash block_size must be in [2, 16]")
        self.draft_model_ref = str(draft_model_ref)
        self.block_size = int(block_size)
        self.staged_k1 = bool(staged_k1)
        self.supports_target_prefix = self.staged_k1
        self.requires_compiled_a3b_target_prefix = self.staged_k1
        self._draft = draft_model
        self._draft_cache: list[Any] | None = None
        self._tap_capture: TargetTapCapture | None = None
        self._target_model: Any | None = None
        self._pending_target_hidden: Any | None = None
        self._prepared = False
        self._outstanding = False
        self._proposal_count = 0
        self._committed_context_rows = 0
        self._queued_tokens: list[int] = []

    @property
    def identity(self) -> str:
        return f"dflash:{self.draft_model_ref}:b{self.block_size}:one-hot"

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
                module = importlib.import_module("dflash.model_mlx")
            except Exception as exc:
                raise RuntimeError(
                    "DFlash support requires the optional 'competitors' "
                    "dependency (pip install 'mtplx[competitors]')"
                ) from exc
            try:
                draft = module.load_draft(self.draft_model_ref)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load DFlash drafter {self.draft_model_ref!r}"
                ) from exc
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

        import mlx.core as mx

        if self.staged_k1 and requested_drafts != 1:
            raise RuntimeError(
                "staged-K1 DFlash requires speculative_depth=1"
            )
        if self.staged_k1 and self._queued_tokens:
            expected_primary = self._queued_tokens.pop(0)
            if int(primary_token) != int(expected_primary):
                self._queued_tokens.clear()
            elif self._queued_tokens:
                staged_token = int(self._queued_tokens.pop(0))
                self._proposal_count += 1
                self._outstanding = True
                return BlockDraftProposal(
                    tokens=(staged_token,),
                    source=f"{self.name}-staged",
                    metadata={
                        "block_size": self.block_size,
                        "proposal": self._proposal_count,
                        "target_context_rows": 0,
                        "committed_context_rows": self._committed_context_rows,
                        "completion_tokens_before_cycle": len(committed_tokens),
                        "declaration": "one_hot",
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
        # Day-one invariant: compile may own logits, never sampling.  Greedy
        # host extraction is declared as a point mass by the engine.
        token_array = mx.argmax(logits, axis=-1)
        mx.eval(token_array)
        elapsed = time.perf_counter() - started
        tokens = tuple(int(token) for token in token_array.reshape(-1).tolist())
        if len(tokens) != draft_count:
            raise RuntimeError(
                f"DFlash returned {len(tokens)} tokens for {draft_count} masks"
            )
        if self.staged_k1:
            self._queued_tokens = [int(token) for token in tokens[1:]]
            tokens = tokens[:1]

        self._committed_context_rows += context_rows
        self._proposal_count += 1
        self._outstanding = True
        return BlockDraftProposal(
            tokens=tokens,
            source=self.name,
            elapsed_s=elapsed,
            metadata={
                "block_size": self.block_size,
                "proposal": self._proposal_count,
                "target_context_rows": context_rows,
                "committed_context_rows": self._committed_context_rows,
                "completion_tokens_before_cycle": len(committed_tokens),
                "declaration": "one_hot",
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

    def commit_target_prefix(self, accepted_draft_tokens: int) -> None:
        """Stage only target rows belonging to the committed prefix.

        Target verify rows are ``[primary, *drafts]``.  Therefore an
        acceptance count of ``a`` commits exactly ``a + 1`` tapped target rows
        for the companion's next context.  Rejected/uncommitted rows never
        enter the draft cache and need no rollback.
        """

        if not self._outstanding or self._tap_capture is None:
            raise RuntimeError("no outstanding DFlash proposal to commit")
        import mlx.core as mx

        keep_rows = int(accepted_draft_tokens) + 1
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
        if self.staged_k1 and int(accepted_draft_tokens) != 1:
            self._queued_tokens.clear()
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
