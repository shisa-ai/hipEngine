"""GPU RED tests: staged DFlash GQA attention must match the parent byte for byte.

``dflash_gqa_attention_f32_bf16_staged`` runs the same arithmetic as
``dflash_gqa_attention_f32_bf16`` split across three kernels (score, softmax,
value) so the score pass stops serialising ``kv_len`` tree reductions inside one
block.  The contract is exact: identical BF16 output bits for identical inputs.

Both kernels live in one translation unit and are built with one flag set, so
``expf``, division, and the reduction tree are compiled the same way; the tests
here are what prove the staged passes did not perturb any of them.  Coverage
targets the failure modes that matter for this split:

* contexts on both sides of the 128-lane width and far beyond it
  (1, 2, 17, 31, 63, 129, 257, 512, 4096, 8191);
* ``head_dim`` below, at, and above the lane count (64, 96, 128, 160), which is
  where the parent's zero-partial lanes and multi-iteration lanes appear;
* GQA head grouping, ``batch_size > 1``, and ``query_len > 1``;
* a non-default scale and input magnitudes that spread the softmax widely;
* a workspace pre-filled with NaN plus output and workspace canary tails, which
  catches a pass that reads a row the score pass never wrote, or writes past the
  declared workspace or output extent.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
    dflash_gqa_attention_f32_bf16,
    dflash_gqa_attention_f32_bf16_staged,
    dflash_gqa_attention_scores_workspace_elements,
)


def _has_gpu() -> bool:
    try:  # noqa: SIM105
        from hipengine.core.hip import get_hip_runtime
    except Exception:
        return False
    try:
        get_hip_runtime()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="HIP runtime not available")


def _to_bf16_bits(x_f32: np.ndarray) -> np.ndarray:
    bits = x_f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


@pytest.fixture(scope="module")
def _drafter_lib(hip_test_target_arch):
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import build_dflash_drafter

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = (
        pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    )
    with hip_target_arch_environment(hip_test_target_arch):
        return build_dflash_drafter(load=True, compiler_version=compiler_version)


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


def _upload(runtime, bufs, array, dtype=None):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    arr = np.ascontiguousarray(array, dtype=dtype if dtype is not None else array.dtype)
    buf = malloc(arr.nbytes, runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(arr), runtime=runtime)
    return buf


def _download(runtime, buf, shape, dtype):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    arr = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(arr), buf, runtime=runtime)
    return arr


def _free_all(runtime, bufs):
    from hipengine.core.memory import free

    for buf in reversed(bufs):
        free(buf, runtime=runtime)


def _assert_bits_equal(got: np.ndarray, expected: np.ndarray, label: str) -> None:
    if np.array_equal(got, expected):
        return
    flat_got = got.reshape(-1)
    flat_expected = expected.reshape(-1)
    bad = np.flatnonzero(flat_got != flat_expected)
    first = int(bad[0])
    raise AssertionError(
        f"{label}: staged output differs from the parent at {bad.size} of "
        f"{flat_got.size} BF16 elements; first index {first} "
        f"staged=0x{int(flat_got[first]):04x} parent=0x{int(flat_expected[first]):04x}"
    )


def _attention_case(
    runtime,
    library,
    *,
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seed: int,
    scale: float | None,
    threads: int = 128,
    q_scale: float = 3.0,
    k_scale: float = 2.5,
    special: str | None = None,
    cpu_check: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the parent and the staged route on one shape; return their outputs.

    Returns ``(parent_out, staged_out, staged_out_canary, workspace, workspace_canary)``
    where the canaries are the untouched tails of the two buffers.
    """

    rng = np.random.default_rng(seed)
    query = rng.standard_normal(
        (batch_size, query_len, num_q_heads, head_dim), dtype=np.float32
    ) * np.float32(q_scale)
    key = rng.standard_normal(
        (batch_size, kv_len, num_kv_heads, head_dim), dtype=np.float32
    ) * np.float32(k_scale)
    value_f32 = (
        rng.standard_normal((batch_size, kv_len, num_kv_heads, head_dim), dtype=np.float32) * 0.75
    )
    value_bf16 = _to_bf16_bits(value_f32)
    if cpu_check:
        query = (_to_bf16_bits(query).astype(np.uint32) << 16).view(np.float32)
        key = (_to_bf16_bits(key).astype(np.uint32) << 16).view(np.float32)
    if special == "zeros":
        query.fill(0)
        key.fill(0)
    elif special == "nonfinite":
        query.reshape(-1)[0] = np.nan
        key.reshape(-1)[-1] = np.inf
        value_bf16.reshape(-1)[3] = np.uint16(0x7FC1)

    out_elems = batch_size * query_len * num_q_heads * head_dim
    workspace_elems = dflash_gqa_attention_scores_workspace_elements(
        batch_size, query_len, kv_len, num_q_heads
    )
    workspace_slack = 16
    nan_sentinel = np.float32(np.frombuffer(np.uint32(0x7FC00001).tobytes(), dtype=np.float32)[0])
    tail_sentinel = np.float32(-12345.5)
    workspace = np.full(workspace_elems + workspace_slack, nan_sentinel, dtype=np.float32)
    workspace[workspace_elems:] = tail_sentinel
    # One canary slot past the last output element, to catch an overrun.
    out_canary_bits = np.uint16(0xDEAD)
    out_parent = np.full(out_elems + 1, out_canary_bits, dtype=np.uint16)
    out_staged = np.full(out_elems + 1, out_canary_bits, dtype=np.uint16)

    bufs: list = []
    try:
        q_dev = _upload(runtime, bufs, query)
        k_dev = _upload(runtime, bufs, key)
        v_dev = _upload(runtime, bufs, value_bf16)
        out_parent_dev = _upload(runtime, bufs, out_parent)
        out_staged_dev = _upload(runtime, bufs, out_staged)
        ws_dev = _upload(runtime, bufs, workspace)

        dflash_gqa_attention_f32_bf16(
            q_dev.ptr,
            k_dev.ptr,
            v_dev.ptr,
            out_parent_dev.ptr,
            batch_size,
            query_len,
            kv_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale=scale,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        dflash_gqa_attention_f32_bf16_staged(
            q_dev.ptr,
            k_dev.ptr,
            v_dev.ptr,
            out_staged_dev.ptr,
            ws_dev.ptr,
            batch_size,
            query_len,
            kv_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scores_workspace_elements=workspace_elems,
            scale=scale,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        parent_bits = _download(runtime, out_parent_dev, out_elems + 1, np.uint16)
        staged_bits = _download(runtime, out_staged_dev, out_elems + 1, np.uint16)
        got_workspace = _download(runtime, ws_dev, workspace_elems + workspace_slack, np.float32)
    finally:
        _free_all(runtime, bufs)

    if cpu_check:
        from hipengine.kernels.cpu_reference.yue2 import gqa_attention
        value = (value_bf16.astype(np.uint32) << 16).view(np.float32)
        reference = np.stack([gqa_attention(query[b], key[b], value[b], scale=scale) for b in range(batch_size)])
        got = (staged_bits[:out_elems].astype(np.uint32) << 16).view(np.float32).reshape(reference.shape)
        np.testing.assert_allclose(got, reference, atol=0.008, rtol=0.008)
        def probs(x):
            x = x.reshape(-1, head_dim).astype(np.float64)
            x = np.exp(x - x.max(axis=-1, keepdims=True))
            return x / x.sum(axis=-1, keepdims=True)
        teacher, candidate = probs(reference), probs(got)
        assert float(np.max(np.sum(teacher * np.log(teacher / candidate), axis=-1))) <= 0.05
        assert float(np.mean(reference.argmax(axis=-1) == got.argmax(axis=-1))) >= 0.90
    return (
        parent_bits[:out_elems],
        staged_bits[:out_elems],
        staged_bits[out_elems:],
        got_workspace[:workspace_elems],
        got_workspace[workspace_elems:],
    )


# name, batch_size, query_len, num_q_heads, num_kv_heads, head_dim
_SHAPES = (
    ("mha_d128", 1, 1, 8, 8, 128),
    ("gqa4_d128", 1, 1, 32, 8, 128),
    ("batch_query_gqa2_d128", 2, 3, 6, 3, 128),
    ("gqa2_d96_below_lane_count", 1, 2, 4, 2, 96),
    ("mha_d160_above_lane_count", 1, 2, 4, 4, 160),
    ("paro_gqa8_d256", 1, 1, 16, 2, 256),
)

_CONTEXTS = (1, 2, 17, 31, 63, 127, 128, 129, 257, 512, 4096, 8191)


@pytest.mark.parametrize("context_len", _CONTEXTS)
@pytest.mark.parametrize("shape", _SHAPES, ids=[shape[0] for shape in _SHAPES])
def test_staged_attention_matches_parent_bits(
    _drafter_lib,
    _runtime,
    shape,
    context_len,
) -> None:
    _, batch_size, query_len, num_q_heads, num_kv_heads, head_dim = shape
    seed = 0x5A6E0000 + context_len * 31 + head_dim
    parent_bits, staged_bits, out_canary, workspace, workspace_canary = _attention_case(
        _runtime,
        _drafter_lib,
        batch_size=batch_size,
        query_len=query_len,
        kv_len=context_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seed=seed,
        scale=0.0731,
    )
    _assert_bits_equal(staged_bits, parent_bits, f"{shape[0]} kv_len={context_len}")

    # The parent's output tail canary and the staged one must both survive.
    assert np.array_equal(out_canary, np.array([0xDEAD], dtype=np.uint16))
    # Pass 1 must have written every workspace element the softmax and value
    # passes read: the buffer was NaN before the call.
    assert np.isfinite(workspace).all()
    assert np.all(workspace >= 0.0) and np.all(workspace <= 1.0)
    # And nothing may be written past the declared workspace size.
    assert np.all(workspace_canary == np.float32(-12345.5))


@pytest.mark.parametrize("threads", (64, 256))
def test_staged_attention_matches_parent_bits_at_other_block_widths(
    _drafter_lib,
    _runtime,
    threads,
) -> None:
    """The reduction tree must track the parent at every allowed lane count."""

    parent_bits, staged_bits, out_canary, workspace, workspace_canary = _attention_case(
        _runtime,
        _drafter_lib,
        batch_size=2,
        query_len=2,
        kv_len=257,
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=128,
        seed=0x5A6E1000 + threads,
        scale=0.0731,
        threads=threads,
    )
    _assert_bits_equal(staged_bits, parent_bits, f"threads={threads}")
    assert np.array_equal(out_canary, np.array([0xDEAD], dtype=np.uint16))
    assert np.isfinite(workspace).all()
    assert np.all(workspace_canary == np.float32(-12345.5))


def test_staged_attention_matches_parent_with_default_scale_and_wide_magnitudes(
    _drafter_lib,
    _runtime,
) -> None:
    """Default ``head_dim ** -0.5`` scale and scores large enough to underflow."""

    parent_bits, staged_bits, out_canary, workspace, workspace_canary = _attention_case(
        _runtime,
        _drafter_lib,
        batch_size=1,
        query_len=1,
        kv_len=1024,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=128,
        seed=0x5A6E2000,
        scale=None,
        q_scale=12.0,
        k_scale=9.0,
    )
    _assert_bits_equal(staged_bits, parent_bits, "default scale, wide magnitudes")
    assert np.array_equal(out_canary, np.array([0xDEAD], dtype=np.uint16))
    assert np.isfinite(workspace).all()
    assert np.all(workspace_canary == np.float32(-12345.5))


@pytest.mark.parametrize("special,scale", [("zeros", 0.0), ("zeros", -0.125), ("nonfinite", 0.0731)])
def test_staged_degenerate_and_nonfinite_parent_bits(_drafter_lib, _runtime, special, scale):
    parent, staged, canary, _workspace, ws_canary = _attention_case(
        _runtime, _drafter_lib, batch_size=2, query_len=2, kv_len=129,
        num_q_heads=4, num_kv_heads=2, head_dim=256, seed=143, scale=scale,
        special=special,
    )
    _assert_bits_equal(staged, parent, special)
    assert np.all(canary == np.uint16(0xDEAD))
    assert np.all(ws_canary == np.float32(-12345.5))


def test_staged_cpu_reference_outer_gate(_drafter_lib, _runtime):
    parent, staged, *_ = _attention_case(
        _runtime, _drafter_lib, batch_size=2, query_len=3, kv_len=31,
        num_q_heads=6, num_kv_heads=2, head_dim=96, seed=911, scale=0.0731,
        q_scale=0.2, k_scale=0.3, cpu_check=True,
    )
    _assert_bits_equal(staged, parent, "CPU outer gate")
