"""Is the flash decode kernel wrong, or is the battery's scale unrealistic?

tests/test_gpu_gemma4_flash_decode.py compares the flash route against the
incumbent single kernel at ``scale=1.0``. At head_dim 512 with the fixture's
N(0, 0.7) entries a logit is a 512-term dot product, so its std is ~15.8 and
the softmax is near-argmax: the two arms then disagree whenever a rounding
difference reorders the top two keys, which is a property of the fixture, not
of the kernel.

This sweeps the scale instead, on the same shapes and data, and reports the
error the battery's bound would have to cover. A kernel that is genuinely
wrong shows a large error at every scale; a fixture that is too peaked shows
a small error at realistic scale and a large one at 1.0.
"""

from __future__ import annotations

import ctypes

import numpy as np

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    _SYMBOL_DECODE_F32,
    build_gemma4_attention,
    flash_slices,
    flash_workspace_bytes,
)
from tests.test_gpu_gemma4_attention_decode_parity import _raw_launch

GEOMS = {
    "sliding": dict(num_heads=16, num_kv_heads=8, head_dim=256),
    "global": dict(num_heads=16, num_kv_heads=2, head_dim=512),
}


def ab(library, *, keys, scale, mask_mode, seed=20260930, **geom):
    slices = flash_slices(keys)
    rng = np.random.default_rng(seed)
    num_heads, num_kv_heads, head_dim = (
        geom["num_heads"], geom["num_kv_heads"], geom["head_dim"]
    )

    def cast(a):
        return np.ascontiguousarray(a, dtype=np.float32)

    query = cast(rng.standard_normal((1, num_heads, head_dim)) * 0.7)
    key = cast(rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.7)
    value = cast(rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.7)
    if mask_mode == "keep":
        mask = np.ones((1, keys), dtype=np.uint8)
    else:
        mask = (rng.random((1, keys)) < 0.7).astype(np.uint8)
        mask[0, 0] = 1

    out_single = np.zeros((1, num_heads, head_dim), dtype=np.float32)
    out_flash = np.zeros_like(out_single)
    arrays = [query, key, value, mask, out_single, out_flash]
    shape = dict(
        tokens=1, keys=keys, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, scale=scale,
    )
    buffers, workspace = [], None
    try:
        for array in arrays:
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            if array is not out_single and array is not out_flash:
                copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)
        _raw_launch(library, _SYMBOL_DECODE_F32, buffers[:5], split_slices=1, **shape)
        copy_device_to_host(host_array_ptr(out_single), buffers[4], out_single.nbytes)
        workspace = malloc(
            flash_workspace_bytes(1, num_heads, head_dim, slices, library=library)
        )
        _raw_launch(
            library, _SYMBOL_DECODE_F32,
            [buffers[0], buffers[1], buffers[2], buffers[3], buffers[5]],
            split_workspace=workspace.ptr, split_slices=-slices, **shape,
        )
        copy_device_to_host(host_array_ptr(out_flash), buffers[5], out_flash.nbytes)
        return out_single, out_flash, slices
    finally:
        for buffer in buffers:
            free(buffer)
        if workspace is not None:
            free(workspace)


def main() -> None:
    library = build_gemma4_attention(load=True)
    print(f"{'geom':>8} {'keys':>5} {'mask':>6} {'scale':>9} {'max_abs':>10} {'max_rel':>10} {'fail@1e-4':>10}")
    for name, geom in GEOMS.items():
        for keys in (1024, 4096):
            for mask_mode in ("keep", "holes"):
                for label, scale in (
                    ("1.0 (battery)", 1.0),
                    ("1/sqrt(d)", 1.0 / np.sqrt(geom["head_dim"])),
                    ("1/16", 0.0625),
                ):
                    single, flash, slices = ab(
                        library, keys=keys, scale=scale, mask_mode=mask_mode, **geom
                    )
                    denom = np.maximum(np.abs(single), 1e-6)
                    max_abs = float(np.max(np.abs(flash - single)))
                    max_rel = float(np.max(np.abs(flash - single) / denom))
                    fails = not np.allclose(flash, single, rtol=1e-4, atol=1e-4)
                    print(
                        f"{name:>8} {keys:>5} {mask_mode:>6} {label:>9} "
                        f"{max_abs:10.3e} {max_rel:10.3e} {str(fails):>10}"
                    )


if __name__ == "__main__":
    main()
