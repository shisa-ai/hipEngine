"""Explicit original-product IU4 FFN device owner for GGUF prefill research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1151.quant.iu4_s4_ffn_product import (
    build_iu4_s4_ffn_product,
    iu4_pfs_linear_bf16_out,
    iu4_pfs_pack_gate_bf16,
    iu4_pfs_pack_swiglu_down_bf16,
)
from hipengine.quant.iu4_ffn_pfs import (
    KAIRIC_QWEN38_FFN_SHA256,
    open_kairic_qwen38_ffn,
    pfs_s4_to_n16_k32_tiles,
)


@dataclass(frozen=True)
class IU4FFNDeviceLayer:
    gate_weight: DeviceBuffer
    gate_scales: DeviceBuffer
    gate_sums: DeviceBuffer
    down_weight: DeviceBuffer
    down_scales: DeviceBuffer
    down_sums: DeviceBuffer


class IU4FFNProductRuntime:
    """Own one fully validated, fully resident PFSIU4F execution product."""

    hidden_size = 5120
    intermediate_size = 17408
    layer_count = 64
    minimum_rows = 96
    maximum_rows = 2048

    def __init__(
        self,
        path: str | Path,
        *,
        runtime: HipRuntime | None = None,
        compiler_version: str | None = None,
        require_cached_build: bool = False,
    ) -> None:
        self.path = Path(path).resolve()
        self.runtime = runtime or get_hip_runtime()
        self.library = build_iu4_s4_ffn_product(
            load=True,
            compiler_version=compiler_version,
            require_cached=bool(require_cached_build),
        )
        self.layers: tuple[IU4FFNDeviceLayer, ...] = ()
        self.buffers: list[DeviceBuffer] = []
        self.launch_count = 0
        self.fallback_count = 0
        try:
            loaded: list[IU4FFNDeviceLayer] = []
            with open_kairic_qwen38_ffn(self.path, verify_sha256=True) as sidecar:
                self.sha256 = sidecar.sha256 or KAIRIC_QWEN38_FFN_SHA256
                for layer_id in range(self.layer_count):
                    layer = sidecar.layer(layer_id)
                    gate_weight = self._upload(
                        pfs_s4_to_n16_k32_tiles(layer.gate_weight)
                    )
                    gate_scales = self._upload(layer.gate_scales)
                    gate_sums = self._upload(layer.gate_sums)
                    down_weight = self._upload(
                        pfs_s4_to_n16_k32_tiles(layer.down_weight)
                    )
                    down_scales = self._upload(layer.down_scales)
                    down_sums = self._upload(layer.down_sums)
                    loaded.append(
                        IU4FFNDeviceLayer(
                            gate_weight=gate_weight,
                            gate_scales=gate_scales,
                            gate_sums=gate_sums,
                            down_weight=down_weight,
                            down_scales=down_scales,
                            down_sums=down_sums,
                        )
                    )
            self.layers = tuple(loaded)
        except BaseException:
            self.close()
            raise

    def _upload(self, values) -> DeviceBuffer:
        import numpy as np

        array = np.ascontiguousarray(values)
        buffer = malloc(array.nbytes, runtime=self.runtime)
        copy_host_to_device(buffer, host_array_ptr(array), runtime=self.runtime)
        self.buffers.append(buffer)
        return buffer

    @classmethod
    def workspace_nbytes(cls, rows: int) -> int:
        if rows <= 0:
            raise ValueError("IU4 FFN workspace rows must be positive")
        return rows * cls.intermediate_size // 2 + rows * 8

    def supports(self, *, layer_id: int, rows: int) -> bool:
        return (
            0 <= layer_id < len(self.layers)
            and self.minimum_rows <= rows <= self.maximum_rows
        )

    def launch(
        self,
        *,
        layer_id: int,
        input_ptr: int,
        gate_up_ptr: int,
        workspace_ptr: int,
        workspace_nbytes: int,
        output_ptr: int,
        rows: int,
        stream: int = 0,
    ) -> bool:
        if not self.supports(layer_id=layer_id, rows=rows):
            self.fallback_count += 1
            return False
        required = self.workspace_nbytes(rows)
        if workspace_nbytes < required:
            raise ValueError(
                f"IU4 FFN workspace requires {required} bytes, got {workspace_nbytes}"
            )
        packed_ptr = int(workspace_ptr)
        scales_ptr = packed_ptr + rows * self.intermediate_size // 2
        zero_points_ptr = scales_ptr + rows * 4
        layer = self.layers[layer_id]

        iu4_pfs_pack_gate_bf16(
            input_ptr,
            packed_ptr,
            scales_ptr,
            zero_points_ptr,
            rows,
            self.hidden_size,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )
        iu4_pfs_linear_bf16_out(
            packed_ptr,
            scales_ptr,
            zero_points_ptr,
            layer.gate_weight.ptr,
            layer.gate_scales.ptr,
            layer.gate_sums.ptr,
            gate_up_ptr,
            rows,
            self.hidden_size,
            2 * self.intermediate_size,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )
        iu4_pfs_pack_swiglu_down_bf16(
            gate_up_ptr,
            packed_ptr,
            scales_ptr,
            zero_points_ptr,
            rows,
            self.intermediate_size,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )
        iu4_pfs_linear_bf16_out(
            packed_ptr,
            scales_ptr,
            zero_points_ptr,
            layer.down_weight.ptr,
            layer.down_scales.ptr,
            layer.down_sums.ptr,
            output_ptr,
            rows,
            self.intermediate_size,
            self.hidden_size,
            stream=stream,
            library=self.library,
            runtime=self.runtime,
        )
        self.launch_count += 1
        return True

    def close(self) -> None:
        for buffer in reversed(getattr(self, "buffers", [])):
            free(buffer, runtime=self.runtime)
        self.buffers = []
        self.layers = ()


__all__ = ["IU4FFNDeviceLayer", "IU4FFNProductRuntime"]
