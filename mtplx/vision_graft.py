"""Vision-tower graft: restore official vision weights into forged artifacts.

The ``mlx_lm.convert`` forge lane serializes only the text model, so
multimodal sources lose their vision tower, ``vision_config``, and the
preprocessor sidecars (issue #263). This module grafts them back from the
original source checkpoint without touching language or MTP tensors:

- copies the vision tensors byte-for-byte (no dequant round-trip) into a
  ``model-vision.safetensors`` shard, renaming ``model.visual.*`` to
  ``vision_tower.*`` on the way out,
- registers the tensors in ``model.safetensors.index.json`` (the runtime
  discovers vision weights only through the index weight_map),
- restores ``vision_config`` into ``config.json``,
- copies / synthesizes the preprocessor sidecars.

Shared by the one-off repair script ``scripts/graft_vision_tower.py`` and
the forge pipeline (``_ensure_vision_tower`` in ``mtplx/commands/forge.py``).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from mtplx.vision import resolve_vision_prefix, vision_spec_for_model_dir

VISION_FILE = "model-vision.safetensors"

_VISION_SIDECAR_FILES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
)

# Top-level config.json token ids the vision splice relies on. Copied from
# the source only when the destination does not already carry them.
_VISION_TOKEN_ID_KEYS = (
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)

_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "I64": 64,
    "U64": 64,
    "F64": 64,
}


class VisionGraftError(RuntimeError):
    """Raised when a vision graft cannot be performed consistently."""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        header_size_raw = handle.read(8)
        if len(header_size_raw) != 8:
            raise VisionGraftError(f"{path.name} is not a valid safetensors file")
        header_size = int.from_bytes(header_size_raw, "little")
        try:
            header = json.loads(handle.read(header_size).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise VisionGraftError(
                f"{path.name} has invalid safetensors metadata: {exc}"
            ) from exc
    if not isinstance(header, dict):
        raise VisionGraftError(f"{path.name} has invalid safetensors metadata")
    return header_size, header


def _tensor_parameters(info: dict[str, Any]) -> int:
    count = 1
    for dim in info.get("shape") or []:
        count *= int(dim)
    return count


def _rename_vision_key(key: str) -> str:
    if key.startswith("model.visual."):
        return "vision_tower." + key[len("model.visual.") :]
    return key


def _normalize_vision_config(vision_config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(vision_config)
    if normalized.get("model_type") == "qwen3_5_vision":
        normalized["model_type"] = "qwen3_5"
    normalized.pop("dtype", None)
    return normalized


def _collect_vision_tensors(
    source: Path, weight_map: dict[str, str], prefix: str
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Return ``(source_key, output_key, shard_name, tensor_info)`` tuples,
    sorted by output key for a deterministic sidecar layout."""
    by_file: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            by_file.setdefault(str(shard), []).append(str(key))

    tensors: list[tuple[str, str, str, dict[str, Any]]] = []
    for shard_name, keys in by_file.items():
        shard = source / shard_name
        if not shard.is_file():
            raise VisionGraftError(
                f"source shard {shard_name} referenced by the index is missing"
            )
        _, header = _read_safetensors_header(shard)
        for key in keys:
            info = header.get(key)
            if not isinstance(info, dict):
                raise VisionGraftError(f"vision tensor {key} missing from {shard_name}")
            offsets = info.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(item, int) for item in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                raise VisionGraftError(
                    f"vision tensor {key} has invalid safetensors offsets in {shard_name}"
                )
            if not isinstance(info.get("dtype"), str) or not isinstance(
                info.get("shape"), list
            ):
                raise VisionGraftError(
                    f"vision tensor {key} has invalid safetensors metadata in {shard_name}"
                )
            tensors.append((key, _rename_vision_key(key), shard_name, info))
    tensors.sort(key=lambda item: item[1])
    return tensors


def _write_vision_sidecar(
    source: Path,
    tensors: list[tuple[str, str, str, dict[str, Any]]],
    target: Path,
) -> None:
    header: dict[str, Any] = {"__metadata__": {"format": "mlx"}}
    offset = 0
    for _, output_key, _, info in tensors:
        start, end = info["data_offsets"]
        length = end - start
        header[output_key] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + length],
        }
        offset += length
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")

    tmp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    handles: dict[str, Any] = {}
    header_sizes: dict[str, int] = {}
    try:
        with tmp.open("wb") as out:
            out.write(len(encoded).to_bytes(8, "little"))
            out.write(encoded)
            for source_key, _, shard_name, info in tensors:
                handle = handles.get(shard_name)
                if handle is None:
                    handle = (source / shard_name).open("rb")
                    handles[shard_name] = handle
                    header_sizes[shard_name], _ = _read_safetensors_header(
                        source / shard_name
                    )
                header_size = header_sizes[shard_name]
                start, end = info["data_offsets"]
                handle.seek(8 + header_size + start)
                payload = handle.read(end - start)
                if len(payload) != end - start:
                    raise VisionGraftError(
                        f"vision tensor {source_key} payload is truncated in {shard_name}"
                    )
                out.write(payload)
        os.replace(tmp, target)
    finally:
        for handle in handles.values():
            handle.close()
        if tmp.exists():
            tmp.unlink()


def _copy_vision_sidecars(source: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for name in _VISION_SIDECAR_FILES:
        src = source / name
        if src.is_file():
            shutil.copy2(src, destination / name)
            copied.append(name)
    if "processor_config.json" not in copied:
        from mtplx.compressed_tensors import _processor_config_fallback

        _processor_config_fallback(destination)
        if (destination / "processor_config.json").exists():
            copied.append("processor_config.json")
    return copied


def _verify_strict_load(destination: Path) -> None:
    # Bypass mtplx.vision.load_vision_tower so the module-level cache never
    # serves a pre-graft tower for the same path.
    from mtplx.vision.qwen3_vl_tower import Qwen3VLVisionTower

    Qwen3VLVisionTower.from_model_dir(str(destination))


def graft_vision_tower(
    source: Path | str,
    destination: Path | str,
    *,
    dry_run: bool = False,
    verify_load: bool = True,
) -> dict[str, Any]:
    """Graft the vision tower from ``source`` into ``destination``.

    Idempotent: a destination whose index already resolves a vision prefix
    is left untouched, and a text-only source is a no-op. Returns a report
    dict whose ``status`` is one of ``grafted``, ``already-present``,
    ``no-vision-in-source``, or ``dry-run``.
    """
    source = Path(source)
    destination = Path(destination)
    report: dict[str, Any] = {
        "source": str(source),
        "destination": str(destination),
        "vision_file": VISION_FILE,
    }

    # A destination without a (well-formed) weight index cannot register
    # vision tensors; artifact completeness is enforced elsewhere (forge
    # verification, _validate_vision_payload), so this is a no-op here.
    dest_index_path = destination / "model.safetensors.index.json"
    if not dest_index_path.is_file():
        report["status"] = "no-destination-index"
        return report
    dest_index = _load_json(dest_index_path)
    dest_weight_map = dest_index.get("weight_map")
    if not isinstance(dest_weight_map, dict):
        report["status"] = "no-destination-index"
        return report
    if resolve_vision_prefix(dest_weight_map) is not None:
        report["status"] = "already-present"
        return report

    source_index_path = source / "model.safetensors.index.json"
    if not source_index_path.is_file():
        report["status"] = "no-vision-in-source"
        return report
    source_index = _load_json(source_index_path)
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, dict):
        report["status"] = "no-vision-in-source"
        return report
    prefix = resolve_vision_prefix(source_weight_map)
    if prefix is None:
        report["status"] = "no-vision-in-source"
        return report

    source_config = _load_json(source / "config.json")
    source_vision_config = source_config.get("vision_config")
    if not isinstance(source_vision_config, dict):
        raise VisionGraftError(
            f"{source} carries vision tensors but no vision_config; refusing an "
            "inconsistent graft"
        )
    if not (source / "preprocessor_config.json").is_file():
        raise VisionGraftError(
            f"{source} has no preprocessor_config.json; the runtime cannot decode "
            "images without it"
        )

    tensors = _collect_vision_tensors(source, source_weight_map, prefix)
    collisions = [key for _, key, _, _ in tensors if key in dest_weight_map]
    if collisions:
        raise VisionGraftError(
            f"destination index already maps {len(collisions)} vision keys "
            f"(e.g. {collisions[0]})"
        )
    vision_bytes = sum(
        info["data_offsets"][1] - info["data_offsets"][0] for _, _, _, info in tensors
    )
    vision_parameters = sum(_tensor_parameters(info) for _, _, _, info in tensors)
    report.update(
        {
            "prefix": prefix,
            "tensors": len(tensors),
            "bytes": vision_bytes,
            "parameters": vision_parameters,
        }
    )

    if dry_run:
        report["status"] = "dry-run"
        return report

    _write_vision_sidecar(source, tensors, destination / VISION_FILE)

    for _, output_key, _, _ in tensors:
        dest_weight_map[output_key] = VISION_FILE
    metadata = dest_index.get("metadata")
    if isinstance(metadata, dict):
        if isinstance(metadata.get("total_size"), (int, float)):
            metadata["total_size"] = int(metadata["total_size"]) + vision_bytes
        if isinstance(metadata.get("total_parameters"), (int, float)):
            metadata["total_parameters"] = (
                int(metadata["total_parameters"]) + vision_parameters
            )
    _atomic_write_json(dest_index_path, dest_index)

    config_path = destination / "config.json"
    config = _load_json(config_path)
    config["vision_config"] = _normalize_vision_config(source_vision_config)
    for key in _VISION_TOKEN_ID_KEYS:
        if key not in config and key in source_config:
            config[key] = source_config[key]
    _atomic_write_json(config_path, config)

    report["sidecars"] = _copy_vision_sidecars(source, destination)

    spec = vision_spec_for_model_dir(destination)
    if spec is None:
        raise VisionGraftError(
            f"graft finished but {destination} still resolves no vision spec"
        )
    grafted_index = _load_json(dest_index_path)
    grafted_keys = [
        key
        for key, shard in grafted_index["weight_map"].items()
        if shard == VISION_FILE
    ]
    if len(grafted_keys) != len(tensors):
        raise VisionGraftError(
            f"index registers {len(grafted_keys)} vision tensors, expected {len(tensors)}"
        )
    if verify_load:
        _verify_strict_load(destination)
    report["verified"] = {"spec": True, "strict_load": bool(verify_load)}
    report["status"] = "grafted"
    return report
