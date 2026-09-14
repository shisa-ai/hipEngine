"""Torch-free VibeVoice host layouts shared by loaders and kernels."""
import numpy as np


def f32_to_bf16_bits(host: np.ndarray) -> np.ndarray:
    """FP32 host array -> BF16 bits (uint16), round-to-nearest-even."""
    array = np.ascontiguousarray(host, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return (rounded >> np.uint32(16)).astype(np.uint16)



def conv_rows_out(prefix_rows: int, length: int, k_len: int, stride: int) -> int:
    """Valid causal outputs for a pass over ``length`` rows plus prefix."""
    padded = prefix_rows + length
    if padded < k_len:
        return 0
    return (padded - k_len) // stride + 1



def transpose_conv_weight_t(w: np.ndarray) -> np.ndarray:
    """nn.Conv1d weight [C_out, C_in, K] -> kernel layout [K, C_in, C_out] bf16 bits."""
    array = np.asarray(w)
    if array.ndim != 3:
        raise ValueError("conv weight must be [C_out, C_in, K]")
    return f32_to_bf16_bits(np.ascontiguousarray(array.transpose(2, 1, 0)))

