"""AOTriton flash prefill against the exact Gemma 4 attention kernel.

The flash path is a different association -- online softmax over key tiles
instead of a staged logit row and two ascending-key passes -- so it is compared
numerically, not bitwise. What has to hold exactly is the *geometry*: the causal
mask AOTriton applies for a query block that is a suffix of the attended range
must be the mask the exact kernel reads.
"""

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

_HEADS = 16
_KV_HEADS = 8
_HEAD_DIM = 256
# Output is BF16, so agreement is bounded by representation, not by the
# softmax association: the probe measured ~2 ULP on the largest magnitudes.
_TOLERANCE = {"rtol": 2.0e-2, "atol": 1.0e-2}


def _bf16(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    )


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float32)


def _causal_mask(rows: int, keys: int) -> np.ndarray:
    """``key <= query`` with the query block ending at the last key."""
    queries = np.arange(rows, dtype=np.int64)[:, None] + (keys - rows)
    key_positions = np.arange(keys, dtype=np.int64)[None, :]
    return np.ascontiguousarray((key_positions <= queries).astype(np.uint8))


def _run_both(rows: int, keys: int, *, window: int | None = None, seed: int = 5):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        Gemma4AttentionScratch,
        gemma4_attention_prefill_aotriton,
        gemma4_attention_prefill_bf16,
    )

    rng = np.random.default_rng(seed)
    query = _bf16(rng.normal(0.0, 0.5, (rows, _HEADS, _HEAD_DIM)))
    key = _bf16(rng.normal(0.0, 0.5, (keys, _KV_HEADS, _HEAD_DIM)))
    value = _bf16(rng.normal(0.0, 0.5, (keys, _KV_HEADS, _HEAD_DIM)))
    mask = _causal_mask(rows, keys)
    if window is not None:
        queries = np.arange(rows, dtype=np.int64)[:, None] + (keys - rows)
        key_positions = np.arange(keys, dtype=np.int64)[None, :]
        mask = np.ascontiguousarray(
            (mask.astype(bool) & ((queries - key_positions) < window)).astype(np.uint8)
        )

    runtime = get_hip_runtime()
    buffers = []
    scratch = Gemma4AttentionScratch()
    try:

        def upload(array: np.ndarray):
            buffer = malloc(array.nbytes, runtime=runtime)
            buffers.append(buffer)
            copy_host_to_device(
                buffer, host_array_ptr(array), array.nbytes, runtime=runtime
            )
            return buffer

        q_buf = upload(query)
        k_buf = upload(key)
        v_buf = upload(value)
        m_buf = upload(mask)
        exact_host = np.zeros((rows, _HEADS, _HEAD_DIM), dtype=np.uint16)
        exact_buf = upload(exact_host)
        flash_host = np.zeros((rows, _HEADS, _HEAD_DIM), dtype=np.uint16)
        flash_buf = upload(flash_host)

        gemma4_attention_prefill_bf16(
            q_buf.ptr,
            k_buf.ptr,
            v_buf.ptr,
            m_buf.ptr,
            exact_buf.ptr,
            tokens=rows,
            keys=keys,
            num_heads=_HEADS,
            num_kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM,
            scale=1.0,
            scratch=scratch,
            runtime=runtime,
        )
        gemma4_attention_prefill_aotriton(
            q_buf.ptr,
            k_buf.ptr,
            v_buf.ptr,
            flash_buf.ptr,
            tokens=rows,
            keys=keys,
            num_heads=_HEADS,
            num_kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM,
            scale=1.0,
            scratch=scratch,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(exact_host), exact_buf, exact_host.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(flash_host), flash_buf, flash_host.nbytes, runtime=runtime
        )
        return _bf16_to_f32(exact_host), _bf16_to_f32(flash_host)
    finally:
        scratch.close()
        for buffer in buffers:
            free(buffer, runtime=runtime)


@pytest.mark.parametrize(
    ("rows", "keys"),
    [
        pytest.param(64, 64, id="dense-block-keys-equal-rows"),
        pytest.param(64, 128, id="suffix-block-bottom-right-aligned"),
        pytest.param(128, 128, id="dense-block-128"),
        pytest.param(32, 96, id="short-query-long-key-range"),
    ],
)
def test_aotriton_prefill_matches_the_exact_kernel(rows: int, keys: int) -> None:
    exact, flash = _run_both(rows, keys)
    # A fixture that produced a constant would compare equal for the wrong
    # reason, so require the reference to carry real spread first.
    assert float(np.std(exact)) > 1.0e-2
    np.testing.assert_allclose(flash, exact, **_TOLERANCE)


def test_the_exact_kernel_is_sensitive_to_the_window_the_flash_path_ignores() -> None:
    """Why admission requires a vacuous window: the flash path reads no mask.

    With a binding window the two disagree, which is the failure the admission
    policy exists to prevent -- so the policy, not the tolerance, is what keeps
    a windowed block on the exact kernel.
    """

    rows, keys, window = 64, 128, 64
    exact_windowed, flash_unmasked = _run_both(rows, keys, window=window)
    exact_causal, _ = _run_both(rows, keys)
    # The window changes the exact kernel's answer...
    assert not np.allclose(exact_windowed, exact_causal, **_TOLERANCE)
    # ...and the flash path, which cannot see it, reproduces the causal answer.
    np.testing.assert_allclose(flash_unmasked, exact_causal, **_TOLERANCE)
