from __future__ import annotations

import numpy as np
import pytest

from mtplx.generation import _validated_external_draft_qs
from mtplx.sampling import SparseDistribution


def test_external_draft_q_validation_preserves_exact_declarations() -> None:
    sparse = SparseDistribution(
        np.array([1, 3], dtype=np.int64),
        np.array([0.25, 0.75], dtype=np.float64),
        vocab_size=4,
    )
    dense = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    validated = _validated_external_draft_qs(
        (sparse, dense),
        (3, 2),
        vocab_size=4,
    )

    assert validated[0] is sparse
    assert validated[1] is dense


@pytest.mark.parametrize(
    ("draft_qs", "tokens", "message"),
    [
        (None, (1,), "requires exact draft_qs"),
        ((), (1,), "length must match"),
        ((np.array([0.5, 0.5]),), (2,), "outside the target vocab"),
        ((np.array([0.5, 0.4]),), (1,), "exact normalized dense"),
        ((np.array([1.0, 0.0]),), (1,), "assigns no mass"),
    ],
)
def test_external_draft_q_validation_fails_closed(
    draft_qs: object,
    tokens: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _validated_external_draft_qs(draft_qs, tokens, vocab_size=2)


def test_external_sparse_draft_q_rejects_duplicate_or_wrong_vocab_ids() -> None:
    duplicate_ids = SparseDistribution(
        np.array([1, 1], dtype=np.int64),
        np.array([0.5, 0.5], dtype=np.float64),
        vocab_size=3,
    )
    with pytest.raises(RuntimeError, match="exact normalized sparse"):
        _validated_external_draft_qs((duplicate_ids,), (1,), vocab_size=3)

    wrong_vocab = SparseDistribution.one_hot(1, vocab_size=4)
    with pytest.raises(RuntimeError, match="vocab mismatch"):
        _validated_external_draft_qs((wrong_vocab,), (1,), vocab_size=3)
