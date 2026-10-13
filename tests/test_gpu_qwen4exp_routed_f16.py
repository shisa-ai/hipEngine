"""The wide-tile routed-expert F16 WMMA GEMMs match the CPU reference.

The kernels read raw GGUF Q4_K / Q5_K / Q5_1 / Q8_0 expert tensors, decode to
F16 once per macro tile of grouped rows, and accumulate in F32. They change
arithmetic relative to the exact grouped owners, so the oracle is the F32 CPU
reference on fixture inputs: the outer safety floor plus a relative-error
bound tight enough to catch any layout or tiling defect.
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_routed_f16 import (
    build_qwen4_exp_routed_f16,
    qwen4_exp_q4_k_routed_f16_gate_up_silu_bf16_bf16_out,
    qwen4_exp_q4_k_routed_f16_linear_bf16_bf16_out,
    qwen4_exp_q5_1_routed_f16_linear_bf16_bf16_out,
    qwen4_exp_q5_k_routed_f16_gate_up_silu_bf16_bf16_out,
    qwen4_exp_q5_k_routed_f16_linear_bf16_bf16_out,
    qwen4_exp_q8_0_routed_f16_linear_bf16_bf16_out,
    qwen4_exp_routed_f16_tile_map,
    routed_f16_tile_capacity,
)
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import (
    make_q4_k_weight,
    make_q5_k_weight,
    make_q8_0_weight,
)
from tests.test_gpu_gguf_q5_1_t16_expert_major_wmma import make_q5_1_weight

_SENTINEL = np.uint16(0x7FC1)
# Buckets that are empty, shorter than one 16-row tile, exactly one tile, and
# longer than the widest macro tile, in an order that breaks any tile that
# straddles experts.
_COUNTS = [0, 5, 70, 16, 0, 131, 1, 48]


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


def _to_bf16(arr: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _from_bf16(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _inputs(rows: int, features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    bits = _to_bf16(rng.standard_normal((rows, features)).astype(np.float32) * 0.5)
    return np.ascontiguousarray(bits), _from_bf16(bits)


def _expert_weights(make, experts: int, out_features: int, in_features: int) -> np.ndarray:
    base = make(out_features, in_features)
    return np.ascontiguousarray(
        np.stack([np.roll(base, shift=3 * e + 1, axis=0) for e in range(experts)])
    )


def _reference(x: np.ndarray, weights: np.ndarray, qtype, bounds: np.ndarray) -> np.ndarray:
    out = np.zeros((x.shape[0], weights.shape[1]), dtype=np.float32)
    for expert in range(weights.shape[0]):
        start, stop = int(bounds[expert]), int(bounds[expert + 1])
        if stop > start:
            out[start:stop] = gguf_quant_gemv(x[start:stop], weights[expert], qtype)
    return out


def _silu(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    return (x64 / (1.0 + np.exp(-x64))).astype(np.float32)


class _Device:
    def __init__(self) -> None:
        self.runtime = get_hip_runtime()
        self.library = build_qwen4_exp_routed_f16(load=True)
        self._buffers = []

    def upload(self, arr: np.ndarray):
        host = np.ascontiguousarray(arr)
        dev = malloc(host.nbytes, runtime=self.runtime)
        self._buffers.append(dev)
        copy_host_to_device(dev, host_array_ptr(host), runtime=self.runtime)
        return dev

    def download(self, dev, like: np.ndarray) -> np.ndarray:
        self.runtime.device_synchronize()
        host = np.empty_like(like)
        copy_device_to_host(host_array_ptr(host), dev, runtime=self.runtime)
        return host

    def close(self) -> None:
        for dev in reversed(self._buffers):
            free(dev, runtime=self.runtime)
        self._buffers.clear()


@pytest.fixture
def device():
    dev = _Device()
    try:
        yield dev
    finally:
        dev.close()


def _tile_map(device: _Device, bounds: np.ndarray, tile_rows: int):
    experts = bounds.size - 1
    capacity = routed_f16_tile_capacity(int(bounds[-1]), experts, tile_rows)
    tiles_host = np.full(capacity + 4, -1, dtype=np.int32)
    count_host = np.full(1, -1, dtype=np.int32)
    bounds_dev = device.upload(bounds)
    tiles_dev = device.upload(tiles_host)
    count_dev = device.upload(count_host)
    qwen4_exp_routed_f16_tile_map(
        bounds_dev.ptr, tiles_dev.ptr, count_dev.ptr, experts,
        tile_rows=tile_rows, tile_capacity=capacity,
        library=device.library, runtime=device.runtime,
    )
    return bounds_dev, tiles_dev, count_dev, capacity, tiles_host, count_host


def _assert_close(actual: np.ndarray, reference: np.ndarray) -> None:
    assert np.isfinite(actual).all()
    error = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    # BF16 output rounding alone is about 2e-3 relative; F16 weight decode
    # adds less. A mis-tiled or mis-decoded block is O(1).
    assert error <= 6e-3
    # Outer safety floor on unit-variance logits: raw expert outputs are not
    # logits, and an unscaled softmax over them is a one-hot tie detector.
    scale = reference.astype(np.float64).std(axis=-1, keepdims=True) + 1e-12
    ref = reference.astype(np.float64) / scale
    cand = actual.astype(np.float64) / scale
    ref_logp = ref - ref.max(axis=-1, keepdims=True)
    cand_logp = cand - cand.max(axis=-1, keepdims=True)
    ref_logp -= np.log(np.exp(ref_logp).sum(axis=-1, keepdims=True))
    cand_logp -= np.log(np.exp(cand_logp).sum(axis=-1, keepdims=True))
    kl = np.sum(np.exp(ref_logp) * (ref_logp - cand_logp), axis=-1)
    assert float(kl.max()) <= 0.05
    assert np.mean(reference.argmax(axis=-1) == actual.argmax(axis=-1)) >= 0.9


@pytest.mark.parametrize("tile_rows", [16, 48, 64, 128])
def test_tile_map_expands_the_grouped_prefix(device, tile_rows) -> None:
    bounds = np.concatenate([[0], np.cumsum(_COUNTS)]).astype(np.int64)
    _, tiles_dev, count_dev, capacity, tiles_host, count_host = _tile_map(
        device, bounds, tile_rows
    )
    expected = [
        expert | (tile << 16)
        for expert, count in enumerate(_COUNTS)
        for tile in range((count + tile_rows - 1) // tile_rows)
    ]
    assert len(expected) <= capacity
    count = device.download(count_dev, count_host)
    tiles = device.download(tiles_dev, tiles_host)
    assert int(count[0]) == len(expected)
    assert tiles[: len(expected)].tolist() == expected
    assert (tiles[capacity:] == -1).all()


@pytest.mark.parametrize(
    "make,gate_up,qtype",
    [
        (make_q4_k_weight, qwen4_exp_q4_k_routed_f16_gate_up_silu_bf16_bf16_out,
         GGMLQuantizationType.Q4_K),
        (make_q5_k_weight, qwen4_exp_q5_k_routed_f16_gate_up_silu_bf16_bf16_out,
         GGMLQuantizationType.Q5_K),
    ],
)
@pytest.mark.parametrize("tile_rows", [64, 128])
@pytest.mark.parametrize("out_features", [64, 160, 150])
def test_gate_up_silu_matches_cpu_reference(
    device, make, gate_up, qtype, tile_rows, out_features
) -> None:
    in_features = 512
    bounds = np.concatenate([[0], np.cumsum(_COUNTS)]).astype(np.int64)
    rows = int(bounds[-1])
    experts = len(_COUNTS)
    x_bits, x = _inputs(rows, in_features, seed=out_features + tile_rows)
    gate = _expert_weights(make, experts, out_features, in_features)
    up = np.ascontiguousarray(gate[::-1])
    reference = _silu(_reference(x, gate, qtype, bounds)) * _reference(
        x, up, qtype, bounds
    )

    bounds_dev, tiles_dev, count_dev, capacity, _, _ = _tile_map(
        device, bounds, tile_rows
    )
    out_host = np.full((rows + 2, out_features), _SENTINEL, dtype=np.uint16)
    out_dev = device.upload(out_host)
    gate_up(
        device.upload(x_bits).ptr, bounds_dev.ptr, tiles_dev.ptr, count_dev.ptr,
        device.upload(gate).ptr, device.upload(up).ptr, out_dev.ptr,
        in_features, out_features,
        tile_rows=tile_rows, tile_capacity=capacity,
        library=device.library, runtime=device.runtime,
    )
    out = device.download(out_dev, out_host)
    assert (out[rows:] == _SENTINEL).all()
    _assert_close(_from_bf16(out[:rows]), reference)


@pytest.mark.parametrize(
    "make,linear,qtype,in_features,tile_rows_set",
    [
        (make_q5_1_weight, qwen4_exp_q5_1_routed_f16_linear_bf16_bf16_out,
         GGMLQuantizationType.Q5_1, 640, (16, 48, 64)),
        (make_q8_0_weight, qwen4_exp_q8_0_routed_f16_linear_bf16_bf16_out,
         GGMLQuantizationType.Q8_0, 640, (16, 48, 64)),
        (make_q4_k_weight, qwen4_exp_q4_k_routed_f16_linear_bf16_bf16_out,
         GGMLQuantizationType.Q4_K, 512, (16, 48)),
        (make_q5_k_weight, qwen4_exp_q5_k_routed_f16_linear_bf16_bf16_out,
         GGMLQuantizationType.Q5_K, 512, (16, 48)),
    ],
)
@pytest.mark.parametrize("out_features", [128, 200, 257])
def test_linear_matches_cpu_reference(
    device, make, linear, qtype, in_features, tile_rows_set, out_features
) -> None:
    bounds = np.concatenate([[0], np.cumsum(_COUNTS)]).astype(np.int64)
    rows = int(bounds[-1])
    experts = len(_COUNTS)
    x_bits, x = _inputs(rows, in_features, seed=out_features)
    weights = _expert_weights(make, experts, out_features, in_features)
    reference = _reference(x, weights, qtype, bounds)
    x_dev = device.upload(x_bits)
    w_dev = device.upload(weights)
    for tile_rows in tile_rows_set:
        bounds_dev, tiles_dev, count_dev, capacity, _, _ = _tile_map(
            device, bounds, tile_rows
        )
        out_host = np.full((rows + 2, out_features), _SENTINEL, dtype=np.uint16)
        out_dev = device.upload(out_host)
        linear(
            x_dev.ptr, bounds_dev.ptr, tiles_dev.ptr, count_dev.ptr,
            w_dev.ptr, out_dev.ptr, in_features, out_features,
            tile_rows=tile_rows, tile_capacity=capacity,
            library=device.library, runtime=device.runtime,
        )
        out = device.download(out_dev, out_host)
        assert (out[rows:] == _SENTINEL).all(), tile_rows
        _assert_close(_from_bf16(out[:rows]), reference)
