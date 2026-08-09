"""Construction-time numerics profiles for the fixed Qwen 35B MTP batch lane."""

from __future__ import annotations

from enum import Enum


class MTPBatchNumerics(str, Enum):
    """Closed public names for construction-installed B8 arithmetic routes."""

    THROUGHPUT = "throughput"
    BALANCED = "balanced"
    B1_EXACT = "b1-exact"


MTP_BATCH_NUMERICS_CHOICES = tuple(item.value for item in MTPBatchNumerics)


def normalize_mtp_batch_numerics(
    value: object | None,
) -> MTPBatchNumerics:
    """Normalize one startup value without importing MLX."""

    raw = str(value or MTPBatchNumerics.THROUGHPUT.value).strip().lower()
    try:
        return MTPBatchNumerics(raw)
    except ValueError as exc:
        choices = ", ".join(MTP_BATCH_NUMERICS_CHOICES)
        raise ValueError(
            f"unknown mtp_batch numerics profile {raw!r}; "
            f"expected one of: {choices}"
        ) from exc

