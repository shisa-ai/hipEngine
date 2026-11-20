"""Live tier: launch_gguf_linear_exact_rows is bit-identical to AR decode.

The batched-invariant speculative block verify projects every draft row through
launch_gguf_linear_exact_rows. Its contract is that each output row equals the
rows=1 decode GEMV bit-for-bit (so verify-written KV equals true AR). This gate
compares the helper against per-row decode launches on the resident Glimmer
weights across the row counts the verify uses, including the >8 rowtile chunk
boundary and the row-independent Q5/Q6 GEMV. Requires ROCm + the local artifact.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP_AVAILABLE = True
except OSError:
    _HIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _HIP_AVAILABLE, reason="ROCm runtime not available")

MODEL = Path(
    "/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
)


@pytest.fixture(scope="module")
def session():
    if not MODEL.exists():
        pytest.skip(f"local Muse Glimmer GGUF not found: {MODEL}")
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.muse_glimmer_gguf import materialize_muse_glimmer_weights
    from hipengine.runtime.muse_glimmer import MuseGlimmerResidentSession

    reader = GGUFReader(MODEL)
    loaded = materialize_muse_glimmer_weights(reader, backend="hip_gfx1151")
    sess = MuseGlimmerResidentSession(loaded, max_context=2048)
    yield sess
    sess.close()
    loaded.free()


@pytest.mark.parametrize(
    "name,in_features,out_features",
    [
        ("blk.0.ffn_gate.weight", 6656, 19968),   # Q4_K T16 -> rowtile
        ("blk.0.attn_q.weight", 6656, 4096),      # Q4_K T16 -> rowtile
        ("blk.0.attn_output.weight", 4096, 6656), # Q4_K T16 -> rowtile
        ("blk.0.ffn_down.weight", 19968, 6656),   # Q6_K T16 -> row-independent
        ("blk.0.attn_v.weight", 6656, 256),       # Q6_K T16 -> row-independent
    ],
)
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 9, 15])
def test_exact_rows_bit_identical_to_decode(session, name, in_features, out_features, rows) -> None:
    from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.runtime.gguf_linear import launch_gguf_linear, launch_gguf_linear_exact_rows

    rt = session.runtime
    weight = session.loaded.weight(name)
    rng = np.random.default_rng(rows)
    x_f32 = (rng.standard_normal((rows, in_features)) * 0.1).astype(np.float32)
    x_bf16 = float_array_to_bf16_bits(x_f32.reshape(-1)).reshape(rows, in_features)
    xbuf = malloc(rows * in_features * 2, runtime=rt)
    copy_host_array_to_device(xbuf, np.ascontiguousarray(x_bf16), runtime=rt)

    # Reference: the rows=1 decode GEMV applied to each row independently.
    reference = np.empty((rows, out_features), dtype=np.uint16)
    rowbuf = malloc(out_features * 2, runtime=rt)
    for r in range(rows):
        launch_gguf_linear(
            weight, xbuf.ptr + r * in_features * 2, rowbuf.ptr, 1,
            in_features, out_features, output_dtype="bf16",
            backend=session.backend, runtime=rt,
        )
        row = np.empty(out_features, dtype=np.uint16)
        copy_device_to_host(row.ctypes.data, rowbuf, runtime=rt)
        reference[r] = row
    free(rowbuf, runtime=rt)

    # Batched exact-row owner.
    batched = np.empty((rows, out_features), dtype=np.uint16)
    outbuf = malloc(rows * out_features * 2, runtime=rt)
    launch_gguf_linear_exact_rows(
        weight, xbuf.ptr, outbuf.ptr, rows, in_features, out_features,
        output_dtype="bf16", backend=session.backend, runtime=rt,
    )
    copy_device_to_host(batched.ctypes.data, outbuf, runtime=rt)
    free(outbuf, runtime=rt)
    free(xbuf, runtime=rt)

    assert np.array_equal(reference, batched), (
        f"{name} rows={rows}: exact-row owner diverged from per-row decode "
        f"({np.count_nonzero(reference != batched)}/{reference.size} elements)"
    )