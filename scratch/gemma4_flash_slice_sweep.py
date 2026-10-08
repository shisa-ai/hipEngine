"""Deviation against slice count: what does a float-precision bound protect?

The route's error is not set by any single rounding decision -- it is set by how
many partial sums the reassociation folds. The slice count is the one legitimate
knob that moves it, so the shape of that curve is what decides where a
float-precision bound should sit: inside the shipped policy's own count, and
outside a count coarse enough to mean the policy changed.

Usage:
    PYTHONPATH=. python scratch/gemma4_flash_slice_sweep.py
"""

import numpy as np

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    _SYMBOL_DECODE_BF16,
    _SYMBOL_DECODE_F32,
    build_gemma4_attention,
    flash_workspace_bytes,
)
from tests.test_gpu_gemma4_attention_decode_parity import _raw_launch
from tests.test_gpu_gemma4_flash_decode import _as_float

_SLIDING = dict(num_heads=16, num_kv_heads=8, head_dim=256)
_GLOBAL = dict(num_heads=16, num_kv_heads=2, head_dim=512)


def _inputs(dtype, keys, geom, seed=20260930):
    rng = np.random.default_rng(seed)
    storage = np.float32 if dtype == "f32" else np.uint16

    def cast(array):
        if dtype == "f32":
            return np.ascontiguousarray(array, dtype=np.float32)
        return (
            np.ascontiguousarray(array, dtype=np.float32).view(np.uint32) >> 16
        ).astype(np.uint16)

    query = cast(rng.standard_normal((1, geom["num_heads"], geom["head_dim"])) * 0.7)
    key = cast(rng.standard_normal((keys, geom["num_kv_heads"], geom["head_dim"])) * 0.7)
    value = cast(rng.standard_normal((keys, geom["num_kv_heads"], geom["head_dim"])) * 0.7)
    mask = np.ones((1, keys), dtype=np.uint8)
    return query, key, value, mask, storage


def deviation(library, *, dtype, keys, geom, slices):
    query, key, value, mask, storage = _inputs(dtype, keys, geom)
    out_single = np.zeros((1, geom["num_heads"], geom["head_dim"]), dtype=storage)
    out_flash = np.zeros_like(out_single)
    arrays = [query, key, value, mask, out_single, out_flash]
    symbol = _SYMBOL_DECODE_F32 if dtype == "f32" else _SYMBOL_DECODE_BF16
    shape = dict(
        tokens=1,
        keys=keys,
        num_heads=geom["num_heads"],
        num_kv_heads=geom["num_kv_heads"],
        head_dim=geom["head_dim"],
        scale=1.0,
    )
    buffers = []
    workspace = None
    try:
        for array in arrays:
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            if array is not out_single and array is not out_flash:
                copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        _raw_launch(library, symbol, buffers[:5], split_slices=1, **shape)
        copy_device_to_host(host_array_ptr(out_single), buffers[4], out_single.nbytes)
        workspace = malloc(
            flash_workspace_bytes(1, geom["num_heads"], geom["head_dim"], slices, library=library)
        )
        _raw_launch(
            library,
            symbol,
            [buffers[0], buffers[1], buffers[2], buffers[3], buffers[5]],
            split_workspace=workspace.ptr,
            split_slices=-slices,
            **shape,
        )
        copy_device_to_host(host_array_ptr(out_flash), buffers[5], out_flash.nbytes)
    finally:
        if workspace is not None:
            free(workspace)
        for buffer in buffers:
            free(buffer)
    ref = _as_float(out_single, dtype).astype(np.float64)
    cand = _as_float(out_flash, dtype).astype(np.float64)
    diff = np.abs(cand - ref)
    scale = float(np.max(np.abs(ref))) or 1.0
    return diff.max() / scale, int(np.count_nonzero(cand != ref))


def main() -> None:
    library = build_gemma4_attention(load=True)
    for name, geom in (("sliding hd256", _SLIDING), ("global hd512", _GLOBAL)):
        for dtype in ("f32", "bf16"):
            for keys in (2047, 8191):
                row = []
                for slices in (2, 4, 8, 16, 32, 64, 128, 256):
                    dev, flips = deviation(
                        library, dtype=dtype, keys=keys, geom=geom, slices=slices
                    )
                    row.append(f"{slices}:{dev:.2e}/{flips}")
                print(f"{name} {dtype} k{keys:5d}  " + "  ".join(row))


if __name__ == "__main__":
    main()
