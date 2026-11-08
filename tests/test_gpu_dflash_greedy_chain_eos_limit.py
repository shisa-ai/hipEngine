"""The device EOS bound matches the host greedy-chain reference exactly.

``dflash_greedy_chain_eos_limit_i32`` replaces the default MTP route's
pre-accept candidate/target readback. It must return the same accept budget as
:func:`hipengine.speculative.streaming.greedy_chain_eos_limit` for every
request in the root-prefixed accept layout, including padding rows after the
last request, EOS at the root, EOS inside and beyond the accepted prefix, EOS
past the remaining-decode bound, and requests with no EOS.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

try:  # pragma: no cover - platform guard
    ctypes.CDLL("libamdhip64.so")
    _HIP_AVAILABLE = True
except OSError:  # pragma: no cover - no ROCm on this runner
    _HIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _HIP_AVAILABLE, reason="requires the HIP runtime (libamdhip64.so)"
)

from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.speculative.dflash_accept import (  # noqa: E402
    dflash_greedy_chain_eos_limit_i32,
)
from hipengine.speculative.streaming import greedy_chain_eos_limit  # noqa: E402


def _upload(array: np.ndarray):
    array = np.ascontiguousarray(array, dtype=np.int32)
    buffer = malloc(max(array.nbytes, 4))
    copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
    return buffer


def _device_limits(candidates, targets, eos_ids, remaining, pad_rows):
    """Lay requests out as the accept buffers do and run the device bound."""

    count = len(candidates)
    counts = np.asarray([len(c) for c in candidates], dtype=np.int32)
    first = count + np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int32)
    tokens = np.concatenate(
        [np.full((count,), 7, dtype=np.int32)]
        + [np.asarray(c, dtype=np.int32) for c in candidates]
        + [np.zeros((pad_rows,), dtype=np.int32)]
    )
    top1 = np.concatenate(
        [np.asarray([t[0] for t in targets], dtype=np.int32)]
        + [np.asarray(t[1:], dtype=np.int32) for t in targets]
        + [np.full((pad_rows,), 5, dtype=np.int32)]
    )
    meta = np.concatenate(
        (first, counts, np.asarray([-1 if e is None else e for e in eos_ids], np.int32))
    )
    buffers = [_upload(tokens), _upload(top1), _upload(meta), _upload(np.asarray(remaining))]
    try:
        dflash_greedy_chain_eos_limit_i32(
            buffers[0].ptr, buffers[1].ptr, buffers[2].ptr, buffers[3].ptr, count
        )
        out = np.empty((count,), dtype=np.int32)
        copy_device_to_host(host_array_ptr(out), buffers[3])
        return tuple(int(v) for v in out)
    finally:
        for buffer in buffers:
            free(buffer)


def _host_limits(candidates, targets, eos_ids, remaining):
    return tuple(
        greedy_chain_eos_limit(c, t, remaining_decode=r, eos_token_id=e)
        for c, t, e, r in zip(candidates, targets, eos_ids, remaining)
    )


def test_named_boundaries_match_the_host_reference() -> None:
    eos = 99
    candidates = [
        (11, 12, 13),  # EOS predicted at the root
        (21, 22, 23),  # EOS inside the accepted prefix
        (31, 32, 33),  # EOS after a rejected edge: unreachable
        (41, 42, 43),  # EOS as the full-accept bonus
        (51, 52, 53),  # EOS past the remaining-decode bound
        (61, 62, 63),  # no EOS active
        (),            # no candidates, EOS at the root
    ]
    targets = [
        (99, 1, 2, 3),
        (21, 22, 99, 4),
        (31, 0, 99, 5),
        (41, 42, 43, 99),
        (51, 52, 53, 99),
        (99, 62, 63, 6),
        (99,),
    ]
    eos_ids = [eos, eos, eos, eos, eos, None, eos]
    remaining = [8, 8, 8, 8, 2, 8, 8]
    expected = _host_limits(candidates, targets, eos_ids, remaining)
    assert expected == (1, 3, 8, 4, 2, 8, 1)
    assert _device_limits(candidates, targets, eos_ids, remaining, pad_rows=3) == expected


@pytest.mark.parametrize("seed", range(6))
def test_random_chains_match_the_host_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    count = int(rng.integers(1, 9))
    vocab = 6  # small vocabulary so matches, EOS hits and rejections all occur
    candidates, targets, eos_ids, remaining = [], [], [], []
    for _ in range(count):
        width = int(rng.integers(0, 8))
        chain = tuple(int(v) for v in rng.integers(0, vocab, size=width))
        top1 = list(int(v) for v in rng.integers(0, vocab, size=width + 1))
        # Bias towards accepted edges so deep prefixes are exercised.
        for index in range(width):
            if rng.random() < 0.6:
                top1[index] = chain[index]
        candidates.append(chain)
        targets.append(tuple(top1))
        eos_ids.append(None if rng.random() < 0.2 else int(rng.integers(0, vocab)))
        remaining.append(int(rng.integers(0, width + 3)))
    expected = _host_limits(candidates, targets, eos_ids, remaining)
    pad_rows = int(rng.integers(0, 4))
    assert _device_limits(candidates, targets, eos_ids, remaining, pad_rows) == expected
