"""Issue #404 regression gates: macOS 27 MPP cooperative-tensor compatibility.

The macOS 27 MetalPerformancePrimitives SDK gates
``get_destination_cooperative_tensor`` on ``__is_tensor_type_v /
__is_cooperative_tensor_type_v``, which an address-space-qualified
``decltype(thread_local_ct)`` operand fails.  2.10.1 therefore armed the QSA
sparse prefill lane at startup and answered the first 33K+ prompt with a
mid-request HTTP 500 (issues #404/#405/#407).  Fix credit: mrmurphy (first
patch) and sunnybluesea (three-site sweep + receipts), matching mlx's own
``steel/gemm/nax.h`` pattern.

Two layers of protection are gated here:
1. Source law (MLX-free): every ``get_destination_cooperative_tensor`` call
   built from ``decltype`` operands must wrap them in
   ``metal::remove_addrspace_t``.
2. Startup probe: the QSA prefill enable path must route through the
   one-shot real-pipeline compile probe so an SDK that still refuses the
   pattern degrades to dense prefill with a diagnostic instead of a 500.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The three sites named in issue #404, plus any future kernel that adopts the
# same construction.
MPP_KERNEL_SOURCES = [
    ROOT / "mtplx/kernels/qsa_indexer_prefill.py",
    ROOT / "mtplx/kernels/qsa_prefill_flash.py",
    ROOT / "mtplx/kernels/sdpa_nax_tile.py",
]

_CALL = re.compile(
    r"get_destination_cooperative_tensor<(?P<args>[^;]*?)>\s*\(\)",
    re.DOTALL,
)


def _destination_calls(text: str) -> list[str]:
    return [match.group("args") for match in _CALL.finditer(text)]


def test_all_decltype_destination_operands_are_addrspace_stripped() -> None:
    """No raw decltype operand may reach get_destination_cooperative_tensor."""

    checked = 0
    for path in MPP_KERNEL_SOURCES:
        for args in _destination_calls(path.read_text()):
            for operand in args.split(","):
                operand = operand.strip()
                if "decltype" not in operand:
                    continue
                checked += 1
                assert "remove_addrspace_t" in operand, (
                    f"{path.name}: decltype operand passed to "
                    "get_destination_cooperative_tensor without "
                    f"metal::remove_addrspace_t (issue #404): {operand!r}"
                )
    assert checked >= 6, (
        "expected at least six decltype operands across the three #404 "
        f"kernel sites, found {checked}; site inventory changed — update "
        "this gate rather than deleting it"
    )


def test_probe_exists_and_is_cached_once() -> None:
    text = (ROOT / "mtplx/kernels/qsa_prefill_probe.py").read_text()
    probe = text.find("def qsa_prefill_mpp_compile_supported")
    assert probe != -1, "issue-#404 compile probe missing"
    window = text[max(0, probe - 120) : probe]
    assert "@lru_cache(maxsize=1)" in window, (
        "the #404 probe must cache its verdict for the process lifetime "
        "(issue-#400 precedent)"
    )
    backend = (ROOT / "mtplx/kernels/qsa_indexer_prefill.py").read_text()
    assert "mx.eval" not in backend, (
        "the prefill backend module is contractually sync-free; the probe "
        "must stay in qsa_prefill_probe.py"
    )


def test_enable_path_routes_through_probe_both_ways() -> None:
    """Explicit ON and AUTO must both fail closed through the compile probe."""

    text = (ROOT / "mtplx/models/qwen4_exp.py").read_text()
    start = text.find("def _qsa_prefill_enabled")
    assert start != -1
    end = text.find("\ndef ", start + 1)
    body = text[start:end]
    assert body.count("_qsa_prefill_mpp_compile_ok()") >= 2, (
        "_qsa_prefill_enabled must consult the #404 compile probe on both "
        "the explicit-on and AUTO paths; honoring MTPLX_QSA_PREFILL=1 "
        "verbatim on a refusing SDK is a guaranteed mid-request 500"
    )
    assert "return True" not in body, (
        "no unconditional True path may remain in _qsa_prefill_enabled"
    )
