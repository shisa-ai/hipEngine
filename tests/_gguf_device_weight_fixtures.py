"""Synthetic device-weight metadata; never allocates or loads a GPU runtime."""

from types import MappingProxyType

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.materialize import DeviceTensorAllocation
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q4_K_T16,
    Qwen35GGUFDeviceWeight,
    Qwen35GGUFWeightSpec,
)
from hipengine.quant.gguf import GGMLQuantizationType


def q4_k_t16_weight(
    ptr: int,
    *,
    in_features: int = 256,
    out_features: int = 16,
    qmicro: bool = False,
    backend: str = "hip_gfx1100",
) -> Qwen35GGUFDeviceWeight:
    source = GGUFTensorInfo(
        name=f"weight.{ptr:x}",
        shape=(out_features, in_features),
        ggml_shape=(in_features, out_features),
        ggml_type=int(GGMLQuantizationType.Q4_K),
        ggml_type_name="Q4_K",
        n_elements=out_features * in_features,
        nbytes=out_features * (in_features // 256) * 144,
        offset=0,
        data_offset=0,
        byte_shape=(out_features, (in_features // 256) * 144),
    )
    quant_key = "gguf_q4_k_qmicro_t16_v1" if qmicro else "gguf_q4_k_t16_v1"
    layout = LAYOUT_GGUF_Q4_K_QMICRO_T16 if qmicro else LAYOUT_GGUF_Q4_K_T16
    spec = Qwen35GGUFWeightSpec(
        slot_path=f"layers.0.weight_{ptr:x}",
        source=source,
        quant_key=quant_key,
        layout=layout,
        allocation_names=("tiles",),
    )
    tile_bytes = 2304 if qmicro else 2368
    nbytes = out_features // 16 * (in_features // 256) * tile_bytes
    buffer = DeviceBuffer(ptr=ptr, nbytes=nbytes)
    allocation = DeviceTensorAllocation(
        name=f"{source.name}.t16.tiles",
        source=source,
        buffer=buffer,
        tensor=Tensor.from_handle(ptr, (nbytes,), DType.INT8, Device("hip", 0)),
    )
    return Qwen35GGUFDeviceWeight(
        spec=spec,
        allocations=MappingProxyType({"tiles": allocation}),
        backend=backend,
    )
