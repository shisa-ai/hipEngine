"""P6 gate: a fused q/k/v projection plus split must equal three projections.

This is the end-to-end contract behind punchlist row P6 (``#37``). The loader
resides q/k/v as one weight and the layer issues one ``gemma4_project`` call
into a transient buffer, then ``gemma4_qkv_split_bf16`` separates the result.
Every consumer downstream still reads three buffers, so the whole optimisation
is sound only if those three buffers are *exactly* what three separate
projections would have written.

Asserted with ``assert_array_equal`` rather than ``allclose``: the fused weight
is the three weights concatenated along axis 0, each output column is its own
reduction over ``in_features``, and the split only moves BF16 bits -- so any
difference at all is a layout or dispatch bug, and a tolerance would hide the
very failure this gate exists to catch. ``tests/test_gpu_gguf_dual_column_independence.py``
covers the other half of the premise (columns do not move when a neighbour's
width changes).

Also pins the two places fusion is deliberately *not* engaged: k_eq_v layers,
where two launches would become projection + split = two and the split would be
bought for nothing, and any artifact whose storage cannot be concatenated.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.gemma4_gguf_device import (
    materialize_gemma4_gguf_device_weight,
    plan_gemma4_gguf_resident_specs,
)
from tests._gemma4_gguf_fixture import (
    default_fixture_tensors,
    fixture_metadata,
    write_fixture_gguf,
)


def _hip_available() -> bool:
    """Explicit HIP guard so no-ROCm CI and publish runners skip, not fail."""
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _require_hip() -> None:
    if not _hip_available():
        pytest.skip("HIP runtime unavailable; skipping gfx1100 kernel tests")


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bit patterns, the way the kernels read them."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def test_fused_projection_plus_split_reproduces_three_projections(tmp_path: Path) -> None:
    """Bitwise parity between the fused path and the path it replaces."""
    _require_hip()
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import gemma4_project
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import gemma4_qkv_split_bf16
    from hipengine.runtime.gemma4 import load_gemma4_device_weights

    reader = GGUFReader(
        write_fixture_gguf(tmp_path / "parity.gguf", default_fixture_tensors(), fixture_metadata())
    )
    weights = load_gemma4_device_weights(reader)
    config = weights.config

    individual = {
        spec.source.name: spec for spec in plan_gemma4_gguf_resident_specs(reader)
    }

    def put(array: np.ndarray):
        host = np.ascontiguousarray(array)
        buffer = malloc(host.nbytes)
        copy_host_to_device(buffer, host_array_ptr(host), host.nbytes)
        return buffer

    def read(buffer, shape) -> np.ndarray:
        out = np.empty(shape, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(out), buffer, out.nbytes)
        return out

    rows = 4
    geometry = config.geometry(0)
    q_spec0 = next(s for n, s in individual.items() if n.endswith("blk.0.attn_q.weight"))
    hidden = int(q_spec0.source.shape[1])
    q_width = geometry.num_heads * geometry.head_dim
    kv_width = geometry.num_kv_heads * geometry.head_dim
    stride = q_width + 2 * kv_width

    rng = np.random.default_rng(20260928)
    x = put(_bf16_bits(rng.standard_normal((rows, hidden))))

    try:
        checked = 0
        for index, layer in enumerate(weights.layers):
            geo = config.geometry(index)
            prefix = f"blk.{index}."
            names = {n for n in individual if n.startswith(prefix)}
            has_v = any(n.endswith("attn_v.weight") for n in names)

            if geo.k_eq_v or not has_v:
                # The loader must not fuse here: two launches would become
                # projection + split = two, so the split buys no reduction.
                assert not layer.qkv_proj, (
                    f"layer {index} is k_eq_v yet carries a fused qkv weight; "
                    "fusing it cannot reduce the launch count and adds the "
                    "split's cost plus the transient buffer"
                )
                continue

            assert layer.qkv_proj, (
                f"layer {index} should have fused q/k/v: its artifact carries "
                "all three tensors in block storage, so fusion is available "
                "and skipping it would silently keep three launches"
            )
            assert layer.q_proj == 0 and layer.k_proj == 0 and layer.v_proj == 0, (
                "a fused layer must not also hold the three separate weights "
                "resident -- that would double the attention weight memory"
            )

            # Reference: three separate projections from the same artifact.
            separate = []
            for suffix, width in (("attn_q", q_width), ("attn_k", kv_width), ("attn_v", kv_width)):
                spec = individual[prefix + suffix + ".weight"]
                weight = materialize_gemma4_gguf_device_weight(
                    reader, spec, backend="hip_gfx1100"
                )
                out = put(np.zeros((rows, width), dtype=np.uint16))
                gemma4_project(x.ptr, weight, out.ptr, rows, hidden, width)
                separate.append(read(out, (rows, width)))
                free(out)

            # The fused path: one projection into a transient, then the split.
            fused_buf = put(np.zeros((rows, stride), dtype=np.uint16))
            q_out = put(np.zeros((rows, q_width), dtype=np.uint16))
            k_out = put(np.zeros((rows, kv_width), dtype=np.uint16))
            v_out = put(np.zeros((rows, kv_width), dtype=np.uint16))
            gemma4_project(x.ptr, layer.qkv_proj, fused_buf.ptr, rows, hidden, stride)
            gemma4_qkv_split_bf16(
                fused_buf.ptr, q_out.ptr, k_out.ptr, v_out.ptr, rows, q_width, kv_width, 3
            )

            for name, ref, got in (
                ("q", separate[0], read(q_out, (rows, q_width))),
                ("k", separate[1], read(k_out, (rows, kv_width))),
                ("v", separate[2], read(v_out, (rows, kv_width))),
            ):
                np.testing.assert_array_equal(
                    ref,
                    got,
                    err_msg=(
                        f"layer {index} {name}: the fused projection plus split "
                        f"diverged from three separate projections (max |delta| "
                        f"{np.abs(ref.astype(np.int32) - got.astype(np.int32)).max()}). "
                        "Either the quantized kernel's out_features handling "
                        "differs at the merged width, or the split's layout is "
                        "wrong -- P6 cannot land until this is bitwise."
                    ),
                )
            checked += 1
            for buffer in (fused_buf, q_out, k_out, v_out):
                free(buffer)

        assert checked, "the fixture exercised no fusable layer, so this proved nothing"
    finally:
        for buffer in (x,):
            free(buffer)