"""Hy3 (hy_v3) native MTP support injection.

Unlike families whose MTP layers must be grafted on at load time, the mlx-lm
hy_v3 model class (MTP revision) already owns its MTP head: with
``num_nextn_predict_layers > 0`` the checkpoint's mtp.* weights load onto an
``MTPBlock`` submodule natively. Injection therefore installs the MTPLX
runtime surface (``mtp_forward`` / ``mtp_update_cache`` / ``make_mtp_cache``,
plus ``return_hidden`` on the trunk forward) as a subclass wrapper over the
already-loaded model — no weight rewriting.

Architecture contract (must match the checkpoint):
- one appended NextN layer (depth 1) with its own MoE MLP;
- draft input is concat[enorm(next-token embedding), hnorm(trunk hidden)]
  with the trunk hidden taken PRE-final-norm ("embedding_hidden" order);
- shared embeddings and lm_head.

Status: experimental — pending hy_v3 landing in the pinned mlx-lm
(ml-explore/mlx-lm#1211 + MTP follow-up) and a hardware-measured runtime
contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def is_hy_v3_mtp_config(config: dict[str, Any]) -> bool:
    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(a) for a in config.get("architectures") or []]
    if model_type != "hy_v3" and "HyV3ForCausalLM" not in architectures:
        return False
    return int(config.get("num_nextn_predict_layers") or 0) > 0


def inject_hy_v3_mtp_support(
    model: Any,
    path: Path,
    config: dict[str, Any],
    contract: Any,
) -> bool:
    """Install the MTPLX draft surface on an already-loaded hy_v3 model.

    Returns True when the model exposes a usable MTP head. Raises if the
    config promises MTP but the loaded model cannot draft (an AR-only export
    with the sidecar stripped, or an mlx-lm predating hy_v3 MTP support).
    """
    if not is_hy_v3_mtp_config(config):
        return False

    if getattr(model, "num_nextn_predict_layers", 0) <= 0 or not hasattr(model, "mtp"):
        raise RuntimeError(
            f"{path}: config declares num_nextn_predict_layers="
            f"{config.get('num_nextn_predict_layers')} but the loaded model has "
            "no MTP submodule. The checkpoint is likely an AR-only export "
            "(model-mtp.safetensors absent) or the installed mlx-lm predates "
            "hy_v3 MTP support."
        )

    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache

    # validate_mtp_support (mtp_patch.py) requires model.mtp.layers to be a
    # truthy container; the native mlx-lm MTPBlock exposes its single decoder
    # as `.layer`. Expose a `.layers` alias (assigning a list to an nn.Module
    # attribute registers it as a child-module container — an alias, not a
    # weight copy) so the validator and any depth-indexing caller see a
    # 1-element list.
    if getattr(model.mtp, "layers", None) is None:
        model.mtp.layers = [model.mtp.layer]

    original_outer_class = model.__class__

    # Hy3 has exactly one native draft wiring (pre-final-norm trunk hidden,
    # [enorm(embedding), hnorm(hidden)] concat). Like the sibling appended-layer
    # backends (glm_mtp defaults post_norm, step3p5 pre_norm), we ACCEPT and
    # IGNORE whatever hidden_variant / concat_order the runtime resolves from
    # the contract default — raising would break every draft call, since
    # runtime.draft_mtp resolves None/auto/contract to contract.hidden_variant
    # (== 'post_norm' for a bare MTPContract).
    class _MTPLXHyV3Model(original_outer_class):
        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
            input_embeddings=None,
            hidden_variant: str | None = None,
            **kwargs,
        ):
            if input_embeddings is not None:
                raise ValueError("Hy3 MTP backend does not support input_embeddings")
            if return_hidden:
                # hy_v3's native forward already returns (logits, pre-norm h)
                return super().__call__(
                    inputs, cache=cache, return_hidden_states=True
                )
            return super().__call__(inputs, cache=cache)

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
            concat_order=None,
            return_hidden: bool = False,
            mtp_hidden_variant: str = "pre_norm",
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            # depth is informational for a single-layer head — the one NextN
            # layer is reused regardless (mirrors GLM's modulo-into-layers).
            layer_cache = mtp_cache if mtp_cache is not None else cache
            if isinstance(layer_cache, list):
                layer_cache = layer_cache[0] if layer_cache else None
            # replicate Model.predict_next_tokens, keeping the draft hidden
            e_next = self.model.embed_tokens(next_token_ids)
            mask = create_attention_mask(e_next, layer_cache)
            h_mtp = self.mtp(hidden_states, e_next, mask, layer_cache)
            logits = self._logits(h_mtp)
            if not return_hidden:
                return logits
            return logits, h_mtp

        def mtp_update_cache(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache=None,
            concat_order=None,
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            _logits, hidden = self.mtp_forward(
                hidden_states,
                next_token_ids,
                mtp_cache=mtp_cache,
                concat_order=concat_order,
                return_hidden=True,
                mtp_depth=mtp_depth,
            )
            return hidden

        def make_mtp_cache(self):
            return [KVCache()]

    model.__class__ = _MTPLXHyV3Model
    logger.info("[Hy3 MTP inject] native head bound (depth 1) for %s", path)
    return True
