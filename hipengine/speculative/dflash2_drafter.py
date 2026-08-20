"""Host-side (torch-free, NumPy) DFlash2 reference drafter.

D1 exactness reference: implements the ``DFlash2DraftModel`` forward and
proposal using the CPU-reference primitives (grouped dynamic conv, candidate
selector, DFlash2 attention, rope), verified against golden traces from the
z-lab/dflash torch reference (``~/dflash`` @ 07ebd93). This is the exact-math
slow path that pins the contract for the native kernels in D2; it is not the
production execution path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hipengine.kernels.cpu_reference.dflash2 import (
    candidate_selector_select,
    dflash2_attention_forward,
    grouped_dynamic_conv_finish,
    grouped_dynamic_conv_prepare,
)
from hipengine.kernels.cpu_reference.ops import linear, rmsnorm
from hipengine.loading.dflash import DFlashDraftConfig
from hipengine.loading.safetensors import WeightIndex, read_tensor_storage_bytes

ArrayLike = np.ndarray | list[float] | tuple[float, ...]


def bf16_payload_to_f32(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Bit-cast a BF16 storage payload to a NumPy float32 array.

    BF16 shares the sign/exponent bits of FP32; left-shifting the uint16
    storage by 16 yields the FP32 bit pattern (with zero mantissa low bits).
    """

    bits = np.frombuffer(payload, dtype="<u2").astype(np.uint32) << np.uint32(16)
    out = np.zeros(bits.shape, dtype=np.float32)
    out.view(np.uint32)[...] = bits
    return out.reshape(shape)


def load_dflash2_numpy_weights(index: WeightIndex) -> dict[str, np.ndarray]:
    """Load the DFlash2 BF16 safetensors into a dict of float32 arrays."""

    out: dict[str, np.ndarray] = {}
    for name, info in index.tensors.items():
        if info.dtype != "BF16":
            raise ValueError(f"DFlash2 drafter tensor {name!r} has dtype {info.dtype}, expected BF16")
        out[name] = bf16_payload_to_f32(read_tensor_storage_bytes(info), info.shape)
    return out


def _silu(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    return x_arr / (np.float32(1.0) + np.exp(-x_arr).astype(np.float32))


@dataclass(frozen=True)
class DFlash2Proposal:
    """Greedy selector result over one draft block."""

    path: np.ndarray  # (batch, draft_length) int token ids
    candidates: np.ndarray  # (batch, draft_length, top_k) int token ids
    unary: np.ndarray  # (batch, draft_length, top_k) float32
    logits: np.ndarray  # (batch, draft_length, vocab) float32 draft logits


class DFlash2NumpyDrafter:
    """Exact-math host-side DFlash2 drafter (D1 reference path)."""

    def __init__(self, config: DFlashDraftConfig, weights: dict[str, np.ndarray]) -> None:
        if not config.is_dflash2:
            raise ValueError(f"config is not a DFlash2 drafter: {config.architecture!r}")
        self.config = config
        self.weights = weights
        self.eps = float(config.rms_norm_eps)
        self.group_size = int(config.conv_group_size)

    # -- helpers ------------------------------------------------------------

    def _w(self, name: str) -> np.ndarray:
        try:
            return self.weights[name]
        except KeyError as exc:
            raise KeyError(f"missing DFlash2 drafter weight {name!r}") from exc

    def project_target_hidden(self, taps_concat: ArrayLike) -> np.ndarray:
        """``hidden_norm(fc(taps_concat))``: (batch, ctx, hidden)."""

        h = linear(taps_concat, self._w("fc.weight"))
        return rmsnorm(h, self._w("hidden_norm.weight"), eps=self.eps).astype(np.float32)

    # -- layer --------------------------------------------------------------

    def forward_layer(
        self,
        hidden: ArrayLike,
        target_hidden: ArrayLike,
        positions: ArrayLike,
        layer: int,
    ) -> np.ndarray:
        cfg = self.config
        w = self._w
        eps = self.eps
        gs = self.group_size
        prefix = f"layers.{layer}"
        sliding_window = int(cfg.sliding_windows[layer]) if cfg.sliding_windows else 0

        h = rmsnorm(hidden, w(f"{prefix}.input_layernorm.weight"), eps=eps).astype(np.float32)
        h, attn_kernel = grouped_dynamic_conv_prepare(
            h,
            w(f"{prefix}.attention_conv.kernel_projection.weight"),
            w(f"{prefix}.attention_conv.base_kernel"),
            gs,
        )
        attn = dflash2_attention_forward(
            h,
            target_hidden,
            positions,
            w(f"{prefix}.self_attn.q_proj.weight"),
            w(f"{prefix}.self_attn.k_proj.weight"),
            w(f"{prefix}.self_attn.v_proj.weight"),
            w(f"{prefix}.self_attn.o_proj.weight"),
            w(f"{prefix}.self_attn.q_norm.weight"),
            w(f"{prefix}.self_attn.k_norm.weight"),
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rope_theta=cfg.rope_theta,
            sliding_window=sliding_window,
            is_causal=bool(cfg.causal),
            eps=eps,
        )
        h = grouped_dynamic_conv_finish(
            attn, attn_kernel, w(f"{prefix}.attention_conv.base_kernel"), gs
        )
        h = np.asarray(hidden, dtype=np.float32) + h

        residual = h
        h = rmsnorm(h, w(f"{prefix}.post_attention_layernorm.weight"), eps=eps).astype(np.float32)
        h, mlp_kernel = grouped_dynamic_conv_prepare(
            h,
            w(f"{prefix}.mlp_conv.kernel_projection.weight"),
            w(f"{prefix}.mlp_conv.base_kernel"),
            gs,
        )
        gate = linear(h, w(f"{prefix}.mlp.gate_proj.weight"))
        up = linear(h, w(f"{prefix}.mlp.up_proj.weight"))
        mlp_out = linear(_silu(gate) * up, w(f"{prefix}.mlp.down_proj.weight"))
        h = grouped_dynamic_conv_finish(
            mlp_out, mlp_kernel, w(f"{prefix}.mlp_conv.base_kernel"), gs
        )
        return residual + h

    # -- full forward -------------------------------------------------------

    def forward(
        self,
        target_hidden_taps: ArrayLike,
        noise_embedding: ArrayLike,
        positions: ArrayLike,
    ) -> np.ndarray:
        """Run all draft layers over one block.

        ``target_hidden_taps`` is the concatenated target hidden rows
        (batch, ctx, n_taps*hidden); ``noise_embedding`` is the block input
        (batch, block_len, hidden) starting from the anchor row;
        ``positions`` spans context + block (batch, ctx + block_len). Returns
        the final-normed draft hidden (batch, block_len, hidden).
        """

        projected = self.project_target_hidden(target_hidden_taps)
        hidden = np.asarray(noise_embedding, dtype=np.float32)
        for layer in range(self.config.num_hidden_layers):
            hidden = self.forward_layer(hidden, projected, positions, layer)
        return rmsnorm(hidden, self._w("norm.weight"), eps=self.eps).astype(np.float32)

    # -- logits + proposal --------------------------------------------------

    def compute_logits(
        self,
        hidden: ArrayLike,
        output_head_weight: ArrayLike,
    ) -> np.ndarray:
        """Draft logits from the target output head (``output_multiplier``=1)."""

        cfg = self.config
        logits = linear(hidden, output_head_weight)
        multiplier = float(getattr(cfg, "output_multiplier", 1.0))
        if multiplier != 1.0:
            logits = logits * np.float32(multiplier)
        return logits

    def propose(
        self,
        hidden: ArrayLike,
        output_head_weight: ArrayLike,
        anchor_ids: ArrayLike,
        *,
        top_k: int | None = None,
    ) -> DFlash2Proposal:
        """Greedy DFlash2 proposal over a draft hidden block.

        ``hidden`` is the draft rows to propose over (e.g. the last
        ``block_size - 1`` rows of a forward); ``anchor_ids`` is the last
        verified token per batch row.
        """

        top_k = int(top_k or self.config.selector_top_k)
        logits = self.compute_logits(hidden, output_head_weight)
        result = candidate_selector_select(
            hidden,
            logits,
            anchor_ids,
            self._w("candidate_selector.predecessor_codebook"),
            self._w("candidate_selector.successor_codebook"),
            self._w("candidate_selector.hidden_projection.weight"),
            top_k=top_k,
        )
        return DFlash2Proposal(
            path=result.path,
            candidates=result.candidates,
            unary=result.unary,
            logits=logits,
        )


__all__ = [
    "DFlash2Proposal",
    "DFlash2NumpyDrafter",
    "bf16_payload_to_f32",
    "load_dflash2_numpy_weights",
]
