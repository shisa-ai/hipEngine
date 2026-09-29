"""Live test for M2: ``Gemma4Runner.forward`` exporting every row's ``h``.

M2 asks for the post-``output_norm`` state of each verified row (llama.cpp's
``t_h_nextn``) so a multi-row lm_head can consume states already computed. The
contract has two halves, both asserted here:

1. ``return_hidden=True`` really keeps all ``rows`` states and reports so.
2. The extension changes nothing the existing projection reads: rmsnorm is
   per-row, so the multi-row path's last row must equal the default path's
   only row byte-for-byte, and the logits must be identical. If that failed,
   the new states would not be safe to expose.

**One test function on purpose.** ``tests/conftest.py`` restores the kernel
registry to a collection-time snapshot after every test, which drops kernels
the model load registered lazily; a second load inside the same module then
fails to resolve them (observed as ``MissingKernelError`` for
``linear/gguf_q5_1/gemv_bf16_bf16_out`` when these checks were split across
two tests). Production never restores the registry, so the interaction is a
harness artefact -- but keeping the whole sequence in one test avoids relying
on it either way.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, host_array_ptr

_MODEL_DIR = Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF")
_TARGET = _MODEL_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"

_BF16_BYTES = 2


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available() or not _TARGET.exists(),
    reason="needs ROCm (libamdhip64.so) and the gemma4 fixture artifact",
)


def test_live_gemma4_return_hidden_states_and_single_row_equivalence():
    import hipengine

    llm = hipengine.LLM(model=str(_TARGET))
    runner = llm._get_text_generator()._ensure_runner()
    config = runner.weights.config
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size or 0)
    assert vocab > 0
    tokens = [pid % vocab for pid in (11, 22, 33, 44)]
    rows = len(tokens)
    row_bytes = hidden * _BF16_BYTES

    # The buffer must hold the whole block, not one row.
    assert runner._normalized.nbytes >= rows * row_bytes, (
        f"_normalized is {runner._normalized.nbytes} bytes, "
        f"needs at least {rows * row_bytes} for {rows} rows"
    )

    # --- Default path: normalizes exactly one row. ---
    logits_a = runner.forward(tokens)
    single = np.empty(row_bytes, dtype=np.uint8)
    copy_device_to_host(host_array_ptr(single), runner._normalized, single.nbytes)
    assert runner.normalized_hidden_rows == 0, "default path must not claim hidden rows"

    runner.reset()
    assert runner.position == 0
    assert runner.normalized_hidden_rows == 0, "reset must invalidate exposed rows"

    # --- return_hidden: the whole block's post-output_norm states. ---
    logits_b = runner.forward(tokens, return_hidden=True)
    assert runner.normalized_hidden_rows == rows
    assert runner._normalized.nbytes >= rows * row_bytes

    stacked = np.empty(rows * row_bytes, dtype=np.uint8)
    copy_device_to_host(host_array_ptr(stacked), runner._normalized, stacked.nbytes)

    # The oracle: rmsnorm is independent per row, so the last row here must be
    # bit-identical to the only row the default path produced. Anything else
    # would mean exposing h changed what the projection reads.
    np.testing.assert_array_equal(
        stacked[-row_bytes:],
        single,
        err_msg="multi-row normalization did not reproduce the single-row state",
    )
    np.testing.assert_array_equal(
        logits_a, logits_b, err_msg="return_hidden altered the logits"
    )

    # --- The count is per-block: a one-token step reports one row. ---
    runner.forward([pid % vocab for pid in (9,)], return_hidden=True)
    assert runner.normalized_hidden_rows == 1

    # Not asking clears the claim even though the buffer still holds bytes.
    runner.forward([pid % vocab for pid in (10,)])
    assert runner.normalized_hidden_rows == 0
    assert runner.normalized_hidden.nbytes >= row_bytes