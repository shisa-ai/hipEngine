"""The Q5_1 T16 expert-major compensated WMMA kernel matches the CPU reference.

Down-leg T16 build step 2 (committed spec 4d94cbbb5): the qtype==7 decode
path in ``gguf_k_t16_selected_expert_major_wmma_comp_kernel`` over the
Q5_1 T16 tile layout (``repack_gguf_q5_1_tile16``, bit-lossless).
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    build_gguf_k_t16_selected_prefill,
    gguf_q5_1_t16_selected_expert_major_wmma_comp_bf16_bf16_out,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_t16 import repack_gguf_q5_1_tile16

Q5_1_BLOCK_BYTES = 24


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def make_q5_1_weight(out_features: int, in_features: int, seed: int = 0) -> np.ndarray:
    """Synthetic raw GGUF Q5_1 bytes: d, m (f16), qh (uint32), ql (16B)."""

    if in_features % 32:
        raise ValueError("in_features must be a multiple of 32")
    rng = np.random.default_rng(seed + 1234)
    blocks_per_row = in_features // 32
    data = np.empty((out_features, blocks_per_row * Q5_1_BLOCK_BYTES), dtype=np.uint8)
    d = (0.01 + 0.09 * rng.random((out_features, blocks_per_row))).astype(np.float16)
    m = (rng.random((out_features, blocks_per_row)) - 0.5).astype(np.float16) * 0.1
    d_bits = d.view(np.uint16)
    m_bits = m.view(np.uint16)
    for out_idx in range(out_features):
        for block_idx in range(blocks_per_row):
            start = block_idx * Q5_1_BLOCK_BYTES
            block = data[out_idx, start : start + Q5_1_BLOCK_BYTES]
            dv = int(d_bits[out_idx, block_idx])
            block[0] = dv & 0xFF
            block[1] = (dv >> 8) & 0xFF
            mv = int(m_bits[out_idx, block_idx])
            block[2] = mv & 0xFF
            block[3] = (mv >> 8) & 0xFF
            block[4:8] = rng.integers(0, 256, 4, dtype=np.uint8)
            block[8:24] = rng.integers(0, 256, 16, dtype=np.uint8)
    return data


def _float_array_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = arr.astype(np.float32, copy=False)
    bits = f32.view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _make_q5_1_fixture(
    *, counts: list[int], in_features: int, out_features: int, seed: int = 0
):
    num_experts = len(counts)
    compact_rows = int(sum(counts))
    assert compact_rows > 0
    expert_start_compact = np.zeros(num_experts + 1, dtype=np.int64)
    expert_start_compact[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))

    values = (
        np.arange(compact_rows * in_features, dtype=np.float32).reshape(
            compact_rows, in_features
        )
        + seed
    )
    x_f32 = ((values % 13) - 6) / 64.0
    x_host = np.ascontiguousarray(_float_array_to_bf16_bits(x_f32))
    x_ref = _bf16_bits_to_float32(x_host).astype(np.float16).astype(np.float32)

    qweight = np.ascontiguousarray(
        np.stack(
            [
                np.roll(make_q5_1_weight(out_features, in_features, seed), shift=e, axis=0)
                for e in range(num_experts)
            ],
            axis=0,
        )
    )
    tiles = np.ascontiguousarray(repack_gguf_q5_1_tile16(qweight).tiles)

    reference = np.zeros((compact_rows, out_features), dtype=np.float32)
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        start = int(expert_start_compact[expert])
        stop = start + count
        reference[start:stop] = gguf_quant_gemv(
            x_ref[start:stop], qweight[expert], GGMLQuantizationType.Q5_1
        )

    return {
        "x_host": x_host,
        "expert_start_compact": expert_start_compact,
        "tiles": tiles,
        "reference": reference,
        "compact_rows": compact_rows,
        "in_features": in_features,
        "out_features": out_features,
        "num_experts": num_experts,
    }


def _run_q5_1_expert_major_gpu(fixture: dict) -> np.ndarray:
    runtime = get_hip_runtime()
    library = build_gguf_k_t16_selected_prefill(load=True)
    host_out = np.zeros(
        (fixture["compact_rows"], fixture["out_features"]), dtype=np.uint16
    )
    active = np.flatnonzero(
        np.diff(fixture["expert_start_compact"])
    ).astype(np.int64)
    active_count = np.asarray([active.size], dtype=np.int64)

    bufs = []
    try:
        for arr in (
            fixture["x_host"],
            fixture["expert_start_compact"],
            active,
            active_count,
            fixture["tiles"],
        ):
            dev = malloc(arr.nbytes, runtime=runtime)
            copy_host_to_device(
                dev, host_array_ptr(np.ascontiguousarray(arr)), runtime=runtime
            )
            bufs.append(dev)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.append(out_dev)
        gguf_q5_1_t16_selected_expert_major_wmma_comp_bf16_bf16_out(
            bufs[0].ptr,
            bufs[1].ptr,
            bufs[2].ptr,
            bufs[3].ptr,
            bufs[4].ptr,
            out_dev.ptr,
            fixture["compact_rows"],
            fixture["in_features"],
            fixture["out_features"],
            fixture["num_experts"],
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)
    return _bf16_bits_to_float32(host_out)


def _max_softmax_kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    ref -= ref.max(axis=-1, keepdims=True)
    cand -= cand.max(axis=-1, keepdims=True)
    ref_logp = ref - np.log(np.exp(ref).sum(axis=-1, keepdims=True))
    cand_logp = cand - np.log(np.exp(cand).sum(axis=-1, keepdims=True))
    return float(np.max(np.sum(np.exp(ref_logp) * (ref_logp - cand_logp), axis=-1)))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "counts,in_features,out_features",
    [
        ([1, 0], 256, 16),
        ([17, 33, 0, 16], 256, 32),
        ([0, 1, 2, 15, 16, 17], 512, 48),
        ([8, 24], 1024, 64),
    ],
)
def test_q5_1_t16_expert_major_matches_cpu_quality_gate(
    counts, in_features, out_features
) -> None:
    fixture = _make_q5_1_fixture(
        counts=counts,
        in_features=in_features,
        out_features=out_features,
        seed=17,
    )
    actual = _run_q5_1_expert_major_gpu(fixture)

    assert np.isfinite(actual).all()
    # F16 accumulate vs the F32 CPU reference: the smoke/safety floor.
    assert _max_softmax_kl(fixture["reference"], actual) <= 0.05
    assert np.mean(
        fixture["reference"].argmax(axis=-1) == actual.argmax(axis=-1)
    ) >= 0.9
