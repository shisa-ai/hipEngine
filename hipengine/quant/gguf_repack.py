"""CPU-only source-shape contracts shared by GGUF repackers and admission.

These describe source byte arrays, not allocation sizes or invocation ABIs.
The converter and its resident materialization route use the same immutable
contract. GGUF block sizes come from the canonical quant layout registry.
"""
from dataclasses import dataclass
from typing import Sequence

from hipengine.quant.gguf import GGMLQuantizationType, quant_layout


@dataclass(frozen=True)
class GGUFRepackShape:
    source_type: GGMLQuantizationType
    rank: int
    columns: int

    @property
    def block_bytes(self) -> int:
        return quant_layout(self.source_type).type_size

    def validate(self, byte_shape: Sequence[int]) -> tuple[int, ...]:
        shape = tuple(int(dim) for dim in byte_shape)
        if len(shape) != self.rank:
            axes = "experts, out_features, bytes_per_row" if self.rank == 3 else "out_features, bytes_per_row"
            kind = "expert" if self.rank == 3 else "dense"
            raise ValueError(
                f"raw_qweight must have GGUF {self.source_type.name} "
                f"{kind} byte shape [{axes}] (rank-{self.rank})"
            )
        if self.rank == 3 and shape[0] <= 0:
            raise ValueError("experts must be positive")
        out_features, bytes_per_row = shape[-2:]
        if out_features <= 0 or out_features % self.columns:
            raise ValueError(
                f"out_features must be positive and divisible by {self.columns}"
            )
        if bytes_per_row <= 0 or bytes_per_row % self.block_bytes:
            raise ValueError(
                f"bytes_per_row must be a positive multiple of {self.block_bytes}"
            )
        return shape


Q4_K_PACK8_SHAPE = GGUFRepackShape(GGMLQuantizationType.Q4_K, 2, 8)
Q4_K_T16_SHAPE = GGUFRepackShape(GGMLQuantizationType.Q4_K, 3, 16)
Q5_K_T16_SHAPE = GGUFRepackShape(GGMLQuantizationType.Q5_K, 3, 16)
Q6_K_T16_SHAPE = GGUFRepackShape(GGMLQuantizationType.Q6_K, 3, 16)
Q8_0_T16_SHAPE = GGUFRepackShape(GGMLQuantizationType.Q8_0, 2, 16)
Q4_K_X8_SHAPE = GGUFRepackShape(GGMLQuantizationType.Q4_K, 3, 8)
Q5_K_X8_SHAPE = GGUFRepackShape(GGMLQuantizationType.Q5_K, 3, 8)
Q6_K_X8_SHAPE = GGUFRepackShape(GGMLQuantizationType.Q6_K, 3, 8)
