"""Page-residency-aware row gathers for the PLE n-gram sidecar.

Pure Python + NumPy + ctypes: no MLX import, nothing here runs on the GPU,
and every function is safe on a worker thread (the memmaps are read-only and
``mincore``/``madvise`` go through ``ctypes.CDLL``, which releases the GIL).

Why this module exists
----------------------
``_SidecarGather.prepare_rows_np`` opens every big gather with a threaded
``os.pread`` warm pass, then does the real read as one NumPy fancy index over
the memmaps.  On a page-cache-warm table the warm pass is the whole cost:

===============================  ==========  =========
formulation (32,768 rows)        pread warm  memmap fx
===============================  ==========  =========
shipped (step<=64, 512 tasks)      164.8 ms    0.62 ms
**no warm, memmap only**           **0 ms**    0.44 ms
===============================  ==========  =========

That is ~5 us of GIL-contended Python per ``os.pread``, times three maps times
the chunk's unique rows -- the syscall count IS the cost, so no batching or
page-dedup formulation moves it (measured J/W4).

Dropping the warm pass unconditionally is not safe.  Demand-faulting an mmap
is FLAT at 1.40 GiB/s however many threads touch it, while pooled ``pread``
saturates at 12.9 GiB/s (measured 2026-07-11, `mmap-willneed-unwired`): on a
COLD table the fancy index degenerates into serial faults *with the GIL held*,
which stalls the generation thread as well.  The 2026-08-26 receipt puts that
at ~36 s serial vs ~7.5 s pooled for a cold 100k-token prefill.

So the choice is measured, not assumed: :func:`warm_decision` samples the rows
this gather will actually read and asks ``mincore(2)`` whether their pages are
already in core.  A false "cold" costs nothing (it takes the shipped pread
path); a false "warm" is the expensive mistake, so the threshold is high and
an unavailable/erroring probe answers "pread".
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from functools import lru_cache

__all__ = [
    "ENV_FLAG",
    "MADVISE_ENV",
    "PREWARM_AT_LOAD_ENV",
    "PREWARM_CHUNK_BYTES",
    "MINCORE_INCORE",
    "PAGE_SIZE",
    "RESIDENT_FRACTION_THRESHOLD",
    "SAMPLE_ROWS",
    "enabled",
    "gather_matrices",
    "madvise_choice",
    "prewarm_at_load_enabled",
    "prewarm_file",
    "resident_fraction",
    "touch_rows",
    "warm_decision",
]

ENV_FLAG = "MTPLX_FABLE_PLE_FIRST_GATHER_EARLY"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})

PAGE_SIZE = os.sysconf("SC_PAGESIZE")
MINCORE_INCORE = 0x1

#: How many of the gather's rows to probe.  ~1.9 us per probe on the M5 Max
#: (measured 2026-09-02), so 256 probes is ~0.5 ms against the 165 ms warm
#: pass it decides -- and 256 draws put the sampling error well under the
#: margin between the two regimes this has to tell apart (a prewarmed table
#: probes at 1.00, a cold one at 0.00; there is no realistic middle).
SAMPLE_ROWS = 256

#: Accept the vectorised path only at essentially full residency.  At 1% cold
#: the fancy index pays ~650 serial faults (~40 ms) with the GIL held; below
#: that the pread pool is strictly better, and taking it is never wrong.
RESIDENT_FRACTION_THRESHOLD = 0.99


@lru_cache(maxsize=1)
def enabled() -> bool:
    """Resolve :data:`ENV_FLAG` once.  Unparseable values raise."""

    raw = (os.environ.get(ENV_FLAG) or "").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    accepted = sorted((_TRUE | _FALSE) - {""})
    raise ValueError(
        f"{ENV_FLAG} must be one of {accepted}, got {os.environ.get(ENV_FLAG)!r}"
    )


class _Libc:
    """``mincore(2)`` through ctypes, resolved once and never re-raised."""

    _instance: "_Libc | None" = None
    _resolved = False

    def __init__(self) -> None:
        name = ctypes.util.find_library("c")
        if name is None:
            raise OSError("cannot locate libc")
        lib = ctypes.CDLL(name, use_errno=True)
        lib.mincore.restype = ctypes.c_int
        lib.mincore.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
        ]
        self.lib = lib

    @classmethod
    def get(cls) -> "_Libc | None":
        if not cls._resolved:
            cls._resolved = True
            try:
                cls._instance = cls()
            except Exception:
                cls._instance = None
        return cls._instance


def _row_address(memmap, row: int) -> tuple[int, int]:
    """(byte address, byte length) of one row of a 2-D numpy memmap."""

    stride = int(memmap.strides[0])
    return int(memmap.ctypes.data) + int(row) * stride, stride


def resident_fraction(memmap, rows, *, sample: int = SAMPLE_ROWS):
    """Fraction of the sampled rows' pages already in core, or None.

    ``rows`` is any 1-D integer array of row indices.  The sample is a fixed
    stride through it rather than a random draw: the answer must not move
    run to run for the same gather, or an A/B could not attribute a delta to
    the code instead of to the sampler.
    """

    libc = _Libc.get()
    if libc is None:
        return None
    total = int(len(rows))
    if total <= 0:
        return None
    step = max(1, total // max(1, int(sample)))
    pages = 0
    resident = 0
    buffer = ctypes.create_string_buffer(8)
    try:
        for i in range(0, total, step):
            address, length = _row_address(memmap, int(rows[i]))
            start = (address // PAGE_SIZE) * PAGE_SIZE
            end = -(-(address + length) // PAGE_SIZE) * PAGE_SIZE
            span = end - start
            count = span // PAGE_SIZE
            if count > len(buffer):
                buffer = ctypes.create_string_buffer(count)
            if libc.lib.mincore(ctypes.c_void_p(start), span, buffer) != 0:
                return None
            pages += count
            resident += sum(
                1 for byte in buffer.raw[:count] if byte & MINCORE_INCORE
            )
    except Exception:
        return None
    if pages <= 0:
        return None
    return resident / pages


def warm_decision(memmaps, rows, *, sample: int = SAMPLE_ROWS):
    """``("vectorized" | "pread", fraction_or_None)`` for this gather.

    Every map is probed, and the WORST answers: the gather reads all three,
    so one cold map is a cold gather.
    """

    worst = None
    for memmap in memmaps:
        fraction = resident_fraction(memmap, rows, sample=sample)
        if fraction is None:
            return "pread", None
        worst = fraction if worst is None else min(worst, fraction)
    if worst is None:
        return "pread", None
    if worst >= RESIDENT_FRACTION_THRESHOLD:
        return "vectorized", worst
    return "pread", worst


def gather_matrices(memmaps: dict, uniq, inverse, names) -> dict:
    """The row read itself -- one NumPy fancy index per map, then un-unique.

    This is the SAME expression the shipped gather runs (`_rows_matrices` and
    `prepare_rows_np` both call it), so "vectorised" and "pread" differ only
    in whether the pages were warmed first: the bytes are identical by
    construction, not by comparison.
    """

    import numpy as np

    return {
        name: np.ascontiguousarray(memmaps[name][uniq])[inverse]
        for name in names
    }


def touch_rows(memmaps, rows, *, block: int = 1 << 18) -> int:
    """Fault ``rows`` of every map into the page cache, ascending, in blocks.

    The madvise(WILLNEED) equivalent for a hash-scattered row set: WILLNEED
    over the whole mapping would be readahead on rows this prompt never
    touches, and per-row advice is the syscall count the pread loop already
    proved is the cost.  Reading the rows themselves in ASCENDING order is
    what the kernel can actually coalesce, and it leaves exactly the pages a
    later chunk's fancy index wants.

    Returns the number of rows touched.  Blocked so the temporary copy stays
    bounded (a 262k-token prompt hashes to 4.2M rows = ~420 MB if read whole).
    """

    import numpy as np

    ordered = np.asarray(rows, dtype=np.int64).reshape(-1)
    if ordered.size == 0:
        return 0
    ordered = np.sort(ordered)
    step = max(1, int(block))
    for memmap in memmaps:
        for start in range(0, ordered.size, step):
            chunk = ordered[start : start + step]
            # `.sum()` keeps the read from being optimised away and costs one
            # pass over bytes already in L2; the point is the fault, not the
            # value.
            int(np.asarray(memmap[chunk]).view(np.uint8).sum(dtype=np.int64))
    return int(ordered.size)


# ---------------------------------------------------------------------------
# Mapping advice, and the load-time sequential prewarm
# ---------------------------------------------------------------------------

MADVISE_ENV = "MTPLX_FABLE_NGRAM_MADVISE"
PREWARM_AT_LOAD_ENV = "MTPLX_FABLE_NGRAM_PREWARM_AT_LOAD"

#: 64 MiB, the same unit the fable driver's ``--prewarm-ngram-table`` uses, so
#: the two receipts' GiB/s are directly comparable.
PREWARM_CHUNK_BYTES = 64 * 1024 * 1024

MADV_NORMAL = 0
MADV_RANDOM = 1
MADV_SEQUENTIAL = 2

_ADVICE = {
    "normal": MADV_NORMAL,
    "random": MADV_RANDOM,
    "sequential": MADV_SEQUENTIAL,
}


def madvise_choice() -> tuple[str, int]:
    """Which ``madvise`` advice the n-gram maps should carry, and why.

    The shipped advice is ``MADV_RANDOM``, on the argument that the rows are
    hash-scattered so readahead around a fault is wasted IO.  That argument is
    about MAPPING FAULTS only -- ``pread(2)`` does not consult it at all, so it
    never governed the pread pool the comment credits it to -- and under
    MTPLX_FABLE_PLE_FIRST_GATHER_EARLY the mapping faults are exactly the two
    cases readahead helps: the ascending-order pre-touch of the whole prompt's
    rows, and the vectorised gather's residual faults on a partly-cold table.
    So the flag flips it to the kernel default.

    ``MTPLX_FABLE_NGRAM_MADVISE=random|normal|sequential`` overrides either
    way, so the choice is A/B-able without a code change.  It is applied once,
    at construction, and recorded on the sidecar.
    """

    raw = (os.environ.get(MADVISE_ENV) or "").strip().lower()
    if raw:
        if raw not in _ADVICE:
            raise ValueError(
                f"{MADVISE_ENV} must be one of {sorted(_ADVICE)}, got {raw!r}"
            )
        return raw, _ADVICE[raw]
    name = "normal" if enabled() else "random"
    return name, _ADVICE[name]


def prewarm_at_load_enabled() -> bool:
    """Whether to read the n-gram table sequentially at model load.

    Independent of :func:`enabled` on purpose: the as-found page-cache state
    is what production actually serves at, and a benchmark harness that reads
    the table itself (the fable driver's ``--prewarm-ngram-table``) hides that
    from every receipt it writes.  Off by default -- it costs the read.
    """

    raw = (os.environ.get(PREWARM_AT_LOAD_ENV) or "").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    accepted = sorted((_TRUE | _FALSE) - {""})
    raise ValueError(
        f"{PREWARM_AT_LOAD_ENV} must be one of {accepted}, "
        f"got {os.environ.get(PREWARM_AT_LOAD_ENV)!r}"
    )


def prewarm_file(path, *, chunk_bytes: int = PREWARM_CHUNK_BYTES) -> dict:
    """Read a whole file sequentially into the page cache; return the receipt.

    Sequential ``read(2)`` into ONE reusable buffer: the unified buffer cache
    is shared between the vnode's read path and every mapping of it, so pages
    landed this way are the pages ``mincore`` then reports resident through
    the memmaps -- which is what makes the vectorised gather engage at all.
    """

    import time

    size = os.path.getsize(path)
    buffer = memoryview(bytearray(int(chunk_bytes)))
    started = time.perf_counter()
    total = 0
    with open(path, "rb", buffering=0) as handle:
        while True:
            read = handle.readinto(buffer)
            if not read:
                break
            total += read
    elapsed = time.perf_counter() - started
    return {
        "path": str(path),
        "bytes": int(total),
        "file_bytes": int(size),
        "complete": bool(total == size),
        "seconds": float(elapsed),
        "chunk_bytes": int(chunk_bytes),
        "gib_per_s": (total / 1024**3) / elapsed if elapsed > 0 else None,
    }
