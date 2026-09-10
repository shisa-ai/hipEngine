"""Qwen3-VL-compatible image/video encoder for Qwen4Exp."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.vision.qwen4_exp_vision import (
    qwen4_exp_vision_add_bias_residual_f32,
    qwen4_exp_vision_attention_f32,
    qwen4_exp_vision_bias_gelu_tanh_f32,
    qwen4_exp_vision_layernorm_f32,
)
from hipengine.loading.qwen4_exp_vision_materialize import (
    Qwen4ExpVisionResidentWeights,
)
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
)


@dataclass(frozen=True)
class Qwen4ExpVisionFeatures:
    embeddings: np.ndarray
    grid_thw: tuple[int, int, int]
    modality: str


class Qwen4ExpVisionRunner:
    """Encode merge-compatible RGB grids and temporal frame pairs.

    Images use the official duplicated temporal patch. Videos consume adjacent
    frame pairs and duplicate an odd final frame. Attention remains isolated per
    temporal pair, matching Qwen3-VL's frame-wise cumulative sequence lengths.
    """

    _HIDDEN = 1_152
    _INTERMEDIATE = 4_304
    _MERGE = 2

    def __init__(
        self,
        resident: Qwen4ExpVisionResidentWeights,
        *,
        patch_weight0: np.ndarray,
        patch_weight1: np.ndarray,
        patch_bias: np.ndarray,
        position_embedding: np.ndarray,
        max_patch_tokens: int = 256,
    ) -> None:
        self.resident = resident
        self.config = resident.plan.config
        self.runtime = resident.runtime
        self.closed = False
        self.max_patch_tokens = int(max_patch_tokens)
        if self.max_patch_tokens <= 0 or self.max_patch_tokens % 4:
            raise ValueError("Qwen4Exp max_patch_tokens must be a positive multiple of four")
        self.patch_weight0 = np.ascontiguousarray(patch_weight0, dtype=np.float32)
        self.patch_weight1 = np.ascontiguousarray(patch_weight1, dtype=np.float32)
        self.patch_bias = np.ascontiguousarray(patch_bias, dtype=np.float32)
        self.position_embedding = np.ascontiguousarray(
            position_embedding, dtype=np.float32
        )
        expected_patch = (self._HIDDEN, 3, 16, 16)
        if (
            self.patch_weight0.shape != expected_patch
            or self.patch_weight1.shape != expected_patch
            or self.patch_bias.shape != (self._HIDDEN,)
            or self.position_embedding.shape != (2_304, self._HIDDEN)
        ):
            raise ValueError("invalid Qwen4Exp vision patch/position tensors")

        rows = self.max_patch_tokens
        merged = rows // 4
        f32 = np.dtype(np.float32).itemsize
        i32 = np.dtype(np.int32).itemsize
        sizes = {
            "initial": rows * self._HIDDEN * f32,
            "norm": rows * self._HIDDEN * f32,
            "qkv": rows * 3 * self._HIDDEN * f32,
            "attention": rows * self._HIDDEN * f32,
            "projected": rows * self._HIDDEN * f32,
            "ff1": rows * self._INTERMEDIATE * f32,
            "gelu": rows * self._INTERMEDIATE * f32,
            "ff2": rows * self._HIDDEN * f32,
            "pos_h": rows * i32,
            "pos_w": rows * i32,
            "merge_input": rows * self._HIDDEN * f32,
            "merge_hidden": merged * 4 * self._HIDDEN * f32,
            "merge_output": merged * 2_560 * f32,
        }
        self._buffers = {
            name: malloc(size, runtime=self.runtime) for name, size in sizes.items()
        }

    def _w(self, name: str):
        return self.resident.weight(name)

    def _p(self, name: str) -> int:
        return self._w(name).allocation("raw").tensor.ptr

    @staticmethod
    def _normalize_frame(frame: object) -> np.ndarray:
        values = np.asarray(frame)
        if values.ndim != 3 or values.shape[-1] != 3:
            raise ValueError("Qwen4Exp vision expects RGB arrays shaped [height,width,3]")
        if values.dtype == np.uint8:
            result = values.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
        else:
            result = values.astype(np.float32)
            if not np.isfinite(result).all():
                raise ValueError("Qwen4Exp vision input must be finite")
            if result.size and result.min() >= 0.0 and result.max() <= 1.0:
                result = result * np.float32(2.0) - np.float32(1.0)
        return np.ascontiguousarray(result)

    @staticmethod
    def _block_major_indices(grid_h: int, grid_w: int) -> np.ndarray:
        return np.asarray(
            [
                (block_h * 2 + inner_h) * grid_w + block_w * 2 + inner_w
                for block_h in range(grid_h // 2)
                for block_w in range(grid_w // 2)
                for inner_h in range(2)
                for inner_w in range(2)
            ],
            dtype=np.int64,
        )

    def _position_rows(
        self, grid_h: int, grid_w: int, order: np.ndarray
    ) -> np.ndarray:
        side = 48
        raster = np.empty((grid_h * grid_w, self._HIDDEN), dtype=np.float32)
        for row in range(grid_h):
            source_h = np.float32(row * (side - 1) / max(grid_h - 1, 1))
            h0 = int(np.floor(source_h))
            h1 = min(h0 + 1, side - 1)
            wh = np.float32(source_h - h0)
            for col in range(grid_w):
                source_w = np.float32(col * (side - 1) / max(grid_w - 1, 1))
                w0 = int(np.floor(source_w))
                w1 = min(w0 + 1, side - 1)
                ww = np.float32(source_w - w0)
                top = (
                    self.position_embedding[h0 * side + w0] * (np.float32(1.0) - ww)
                    + self.position_embedding[h0 * side + w1] * ww
                )
                bottom = (
                    self.position_embedding[h1 * side + w0] * (np.float32(1.0) - ww)
                    + self.position_embedding[h1 * side + w1] * ww
                )
                raster[row * grid_w + col] = (
                    top * (np.float32(1.0) - wh) + bottom * wh
                )
        return np.ascontiguousarray(raster[order])

    def preprocess_temporal_pair(
        self, first: object, second: object | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
        frame0 = self._normalize_frame(first)
        frame1 = frame0 if second is None else self._normalize_frame(second)
        if frame0.shape != frame1.shape:
            raise ValueError("Qwen4Exp temporal frame pairs must share one RGB shape")
        height, width, _ = frame0.shape
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError(
                "Qwen4Exp image height and width must be positive multiples of 32"
            )
        grid_h, grid_w = height // 16, width // 16
        patch_tokens = grid_h * grid_w
        if patch_tokens > self.max_patch_tokens:
            raise ValueError(
                f"Qwen4Exp vision grid has {patch_tokens} patches; maximum is "
                f"{self.max_patch_tokens}"
            )
        raster0 = (
            frame0.reshape(grid_h, 16, grid_w, 16, 3)
            .transpose(0, 2, 4, 1, 3)
            .reshape(patch_tokens, 3, 16, 16)
        )
        raster1 = (
            frame1.reshape(grid_h, 16, grid_w, 16, 3)
            .transpose(0, 2, 4, 1, 3)
            .reshape(patch_tokens, 3, 16, 16)
        )
        order = self._block_major_indices(grid_h, grid_w)
        patches0 = np.ascontiguousarray(raster0[order])
        patches1 = np.ascontiguousarray(raster1[order])
        embedded = (
            np.einsum("nchw,ochw->no", patches0, self.patch_weight0, optimize=True)
            + np.einsum("nchw,ochw->no", patches1, self.patch_weight1, optimize=True)
            + self.patch_bias
        ).astype(np.float32)
        embedded += self._position_rows(grid_h, grid_w, order)
        positions_h = np.asarray(
            [index // grid_w for index in order], dtype=np.int32
        )
        positions_w = np.asarray(
            [index % grid_w for index in order], dtype=np.int32
        )
        return (
            np.ascontiguousarray(embedded),
            positions_h,
            positions_w,
            (grid_h, grid_w),
        )

    def _encode_pair(
        self, first: object, second: object | None, *, modality: str
    ) -> Qwen4ExpVisionFeatures:
        if self.closed:
            raise RuntimeError("Qwen4Exp vision runner is closed")
        initial, pos_h, pos_w, (grid_h, grid_w) = self.preprocess_temporal_pair(
            first, second
        )
        rows = int(initial.shape[0])
        merged_rows = rows // 4
        copy_host_to_device(
            self._buffers["initial"], host_array_ptr(initial), initial.nbytes,
            runtime=self.runtime,
        )
        copy_host_to_device(
            self._buffers["pos_h"], host_array_ptr(pos_h), pos_h.nbytes,
            runtime=self.runtime,
        )
        copy_host_to_device(
            self._buffers["pos_w"], host_array_ptr(pos_w), pos_w.nbytes,
            runtime=self.runtime,
        )
        current = self._buffers["initial"]
        norm = self._buffers["norm"]
        qkv = self._buffers["qkv"]
        attention = self._buffers["attention"]
        projected = self._buffers["projected"]
        ff1 = self._buffers["ff1"]
        gelu = self._buffers["gelu"]
        ff2 = self._buffers["ff2"]
        for layer in range(27):
            prefix = f"layers.{layer}."
            qwen4_exp_vision_layernorm_f32(
                current.ptr, self._p(prefix + "ln1.weight"),
                self._p(prefix + "ln1.bias"), norm.ptr, rows, self._HIDDEN,
                self.config.norm_epsilon, runtime=self.runtime,
            )
            launch_gguf_linear(
                self._w(prefix + "attn_qkv.weight"), norm.ptr, qkv.ptr,
                rows, self._HIDDEN, 3 * self._HIDDEN,
                activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32, runtime=self.runtime,
            )
            qwen4_exp_vision_attention_f32(
                qkv.ptr, self._p(prefix + "attn_qkv.bias"),
                self._buffers["pos_h"].ptr, self._buffers["pos_w"].ptr,
                attention.ptr, rows, runtime=self.runtime,
            )
            launch_gguf_linear(
                self._w(prefix + "attn_out.weight"), attention.ptr,
                projected.ptr, rows, self._HIDDEN, self._HIDDEN,
                activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32, runtime=self.runtime,
            )
            qwen4_exp_vision_add_bias_residual_f32(
                projected.ptr, self._p(prefix + "attn_out.bias"), current.ptr,
                projected.ptr, rows, self._HIDDEN, runtime=self.runtime,
            )
            qwen4_exp_vision_layernorm_f32(
                projected.ptr, self._p(prefix + "ln2.weight"),
                self._p(prefix + "ln2.bias"), norm.ptr, rows, self._HIDDEN,
                self.config.norm_epsilon, runtime=self.runtime,
            )
            launch_gguf_linear(
                self._w(prefix + "ffn_up.weight"), norm.ptr, ff1.ptr,
                rows, self._HIDDEN, self._INTERMEDIATE,
                activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32, runtime=self.runtime,
            )
            qwen4_exp_vision_bias_gelu_tanh_f32(
                ff1.ptr, self._p(prefix + "ffn_up.bias"), gelu.ptr,
                rows, self._INTERMEDIATE, runtime=self.runtime,
            )
            launch_gguf_linear(
                self._w(prefix + "ffn_down.weight"), gelu.ptr, ff2.ptr,
                rows, self._INTERMEDIATE, self._HIDDEN,
                activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32, runtime=self.runtime,
            )
            qwen4_exp_vision_add_bias_residual_f32(
                ff2.ptr, self._p(prefix + "ffn_down.bias"), projected.ptr,
                current.ptr, rows, self._HIDDEN, runtime=self.runtime,
            )
        qwen4_exp_vision_layernorm_f32(
            current.ptr, self._p("post_norm.weight"), self._p("post_norm.bias"),
            norm.ptr, rows, self._HIDDEN, self.config.norm_epsilon,
            runtime=self.runtime,
        )
        merge_bytes = rows * self._HIDDEN * np.dtype(np.float32).itemsize
        self.runtime.memcpy(
            self._buffers["merge_input"].ptr, norm.ptr, merge_bytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
        )
        launch_gguf_linear(
            self._w("merge.fc1.weight"), self._buffers["merge_input"].ptr,
            self._buffers["merge_hidden"].ptr, merged_rows, 4 * self._HIDDEN,
            4 * self._HIDDEN, activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32, runtime=self.runtime,
        )
        qwen4_exp_vision_bias_gelu_tanh_f32(
            self._buffers["merge_hidden"].ptr, self._p("merge.fc1.bias"),
            self._buffers["merge_input"].ptr, merged_rows, 4 * self._HIDDEN,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self._w("merge.fc2.weight"), self._buffers["merge_input"].ptr,
            self._buffers["merge_output"].ptr, merged_rows, 4 * self._HIDDEN,
            2_560, activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32, runtime=self.runtime,
        )
        qwen4_exp_vision_add_bias_residual_f32(
            self._buffers["merge_output"].ptr, self._p("merge.fc2.bias"), 0,
            self._buffers["merge_output"].ptr, merged_rows, 2_560,
            runtime=self.runtime,
        )
        self.runtime.device_synchronize()
        output = np.empty((merged_rows, 2_560), dtype=np.float32)
        copy_device_to_host(
            host_array_ptr(output), self._buffers["merge_output"], output.nbytes,
            runtime=self.runtime,
        )
        return Qwen4ExpVisionFeatures(
            np.ascontiguousarray(output), (1, grid_h, grid_w), modality
        )

    def encode_image(self, image: object) -> Qwen4ExpVisionFeatures:
        return self._encode_pair(image, None, modality="image")

    def encode(self, image: object) -> np.ndarray:
        """Backward-compatible image-only array result."""
        return self.encode_image(image).embeddings

    def encode_video(self, frames: object) -> Qwen4ExpVisionFeatures:
        values = np.asarray(frames)
        if values.ndim != 4 or values.shape[-1] != 3 or values.shape[0] <= 0:
            raise ValueError("Qwen4Exp video expects [frames,height,width,3]")
        outputs: list[np.ndarray] = []
        grid_h = grid_w = 0
        for start in range(0, values.shape[0], 2):
            second = values[min(start + 1, values.shape[0] - 1)]
            encoded = self._encode_pair(
                values[start], second, modality="video"
            )
            outputs.append(encoded.embeddings)
            _, grid_h, grid_w = encoded.grid_thw
        return Qwen4ExpVisionFeatures(
            np.ascontiguousarray(np.concatenate(outputs, axis=0)),
            (len(outputs), grid_h, grid_w),
            "video",
        )

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(tuple(self._buffers.values())):
            free(buffer, runtime=self.runtime)
        self._buffers.clear()
        self.closed = True


__all__ = ["Qwen4ExpVisionFeatures", "Qwen4ExpVisionRunner"]
