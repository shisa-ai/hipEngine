"""Backend-neutral per-output-channel symmetric W8A16 contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hipengine.quant.registry import register_quant

W8A16_LAYOUT = "row_major_int8_per_output_channel_symmetric_f32_scale"
_W8_MAX = np.float32(127.0)
_MIN_ABS_MAX = np.float32(1.0e-8)


@dataclass(frozen=True)
class W8A16Quant:
    """INT8 weights with one FP32 symmetric scale per output row."""

    name: str = "w8a16"
    weight_storage: str = "int8_row_major"
    activation_preprocess: str = "none_fp16"
    compute_dtype: str = "fp32_accum_fp16_output"
    scale_granularity: str = "per_output_channel_symmetric_fp32"
    calibration_artifact: str = "deterministic_absmax_no_calibration"
    kernel_family: str = "w8a16"


@dataclass(frozen=True)
class W8A16HostTensor:
    """Contiguous host representation consumed by W8A16 kernel loaders."""

    qweight: np.ndarray
    scales: np.ndarray
    source_shape: tuple[int, int]
    layout: str = W8A16_LAYOUT

    @property
    def source_fp16_nbytes(self) -> int:
        return int(np.prod(self.source_shape, dtype=np.int64)) * np.dtype(np.float16).itemsize

    @property
    def qweight_nbytes(self) -> int:
        return int(self.qweight.nbytes)

    @property
    def scale_nbytes(self) -> int:
        return int(self.scales.nbytes)

    @property
    def packed_nbytes(self) -> int:
        return self.qweight_nbytes + self.scale_nbytes


def quantize_w8a16_per_output(weight: object) -> W8A16HostTensor:
    """Quantize a rank-2 weight using deterministic row-wise symmetric INT8.

    The row scale is ``max(max(abs(row)), 1e-8) / 127``. Values use NumPy's
    round-to-nearest-even ``rint`` before clipping to ``[-127, 127]``. INT8
    ``-128`` is intentionally unused so the format remains symmetric.
    """

    source = np.asarray(weight)
    if source.ndim != 2:
        raise ValueError("W8A16 weight must have rank 2 [out_features, in_features]")
    if source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError("W8A16 weight dimensions must be positive")
    weight_f32 = np.asarray(source, dtype=np.float32)
    if not bool(np.isfinite(weight_f32).all()):
        raise ValueError("W8A16 weight must contain only finite values")
    absmax = np.max(np.abs(weight_f32), axis=1).astype(np.float32)
    scales = np.maximum(absmax, _MIN_ABS_MAX).astype(np.float32) / _W8_MAX
    quantized = np.rint(weight_f32 / scales[:, None])
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    return W8A16HostTensor(
        qweight=np.ascontiguousarray(quantized),
        scales=np.ascontiguousarray(scales),
        source_shape=(int(source.shape[0]), int(source.shape[1])),
    )


def dequantize_w8a16_per_output(qweight: object, scales: object) -> np.ndarray:
    """Return the FP32 matrix represented by row-major W8A16 storage."""

    quantized = np.asarray(qweight)
    scale = np.asarray(scales)
    if quantized.ndim != 2 or quantized.dtype != np.int8:
        raise ValueError("W8A16 qweight must be rank-2 int8")
    if scale.shape != (quantized.shape[0],) or scale.dtype != np.float32:
        raise ValueError("W8A16 scales must be float32 with one value per output row")
    if not bool(np.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("W8A16 scales must be finite and positive")
    return quantized.astype(np.float32) * scale[:, None]


def w8a16_linear_fp16(
    x: object,
    qweight: object,
    scales: object,
    bias: object | None = None,
) -> np.ndarray:
    """CPU oracle for FP16 activations, FP32 dot/scaling, and FP16 output."""

    hidden = np.asarray(x)
    if hidden.ndim < 1 or hidden.dtype != np.float16:
        raise ValueError("W8A16 input must be float16 with at least one dimension")
    weight = dequantize_w8a16_per_output(qweight, scales)
    if hidden.shape[-1] != weight.shape[1]:
        raise ValueError("W8A16 input width must match weight input width")
    output = hidden.astype(np.float32) @ weight.T
    if bias is not None:
        bias_array = np.asarray(bias)
        if bias_array.dtype != np.float16 or bias_array.shape != (weight.shape[0],):
            raise ValueError("W8A16 bias must be float16 with one value per output row")
        output = output + bias_array.astype(np.float32)
    result = np.asarray(output, dtype=np.float16)
    if not bool(np.isfinite(result).all()):
        raise ValueError("W8A16 output must contain only finite values")
    return result


W8A16 = register_quant(W8A16Quant())


__all__ = [
    "W8A16",
    "W8A16HostTensor",
    "W8A16Quant",
    "W8A16_LAYOUT",
    "dequantize_w8a16_per_output",
    "quantize_w8a16_per_output",
    "w8a16_linear_fp16",
]
