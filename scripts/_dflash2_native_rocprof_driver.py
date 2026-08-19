#!/usr/bin/env python3
"""Minimal rocprofv3 driver for the native DFlash2 drafter forward + select.

Prebuild the .so cache outside the profiler (run this once warm), then invoke
with ``rocprofv3 --kernel-trace``.  Uses a small-vocab fixture config so the
output-head upload fits the host staging buffer.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("HIPENGINE_DFLASH_DRAFTER_DENSE", "wmma")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from hipengine.speculative.dflash2_drafter import DFlash2NumpyDrafter, DFlashDraftConfig  # noqa: E402
from hipengine.speculative.dflash2_native import DFlash2NativeDrafter, _to_bf16_bits  # noqa: E402

HIDDEN = 5120
BS = 8
CTX = 16
VOCAB = 1280

cfg = DFlashDraftConfig(
    architecture="qwen3mtp-dflash2", block_size=BS, mask_token_id=0,
    target_layer_ids=tuple(range(5)), num_target_layers=5,
    hidden_size=HIDDEN, target_hidden_size=HIDDEN, target_hidden_concat_size=5 * HIDDEN,
    intermediate_size=17408, num_hidden_layers=5, num_attention_heads=32,
    num_key_value_heads=8, head_dim=128, rope_theta=1e7, vocab_size=VOCAB,
    dtype="bfloat16", layer_types=("sliding_attention",) * 5, rms_norm_eps=1e-6,
    sliding_windows=(2048,) * 5, causal=False, conv_kernel_size=2, conv_group_size=16,
    selector_rank=256, selector_top_k=16,
)


def _make_weights(rng: np.random.default_rng) -> dict[str, np.ndarray]:
    w: dict[str, np.ndarray] = {}
    w["fc.weight"] = rng.standard_normal((HIDDEN, 5 * HIDDEN), dtype=np.float32) * 0.02
    w["hidden_norm.weight"] = rng.standard_normal((HIDDEN,), dtype=np.float32)
    w["norm.weight"] = rng.standard_normal((HIDDEN,), dtype=np.float32)
    for l in range(5):
        p = f"layers.{l}"
        w[f"{p}.input_layernorm.weight"] = rng.standard_normal((HIDDEN,), dtype=np.float32)
        w[f"{p}.post_attention_layernorm.weight"] = rng.standard_normal((HIDDEN,), dtype=np.float32)
        w[f"{p}.self_attn.q_proj.weight"] = rng.standard_normal((32 * 128, HIDDEN), dtype=np.float32) * 0.02
        w[f"{p}.self_attn.k_proj.weight"] = rng.standard_normal((8 * 128, HIDDEN), dtype=np.float32) * 0.02
        w[f"{p}.self_attn.v_proj.weight"] = rng.standard_normal((8 * 128, HIDDEN), dtype=np.float32) * 0.02
        w[f"{p}.self_attn.o_proj.weight"] = rng.standard_normal((HIDDEN, 32 * 128), dtype=np.float32) * 0.02
        w[f"{p}.self_attn.q_norm.weight"] = rng.standard_normal((128,), dtype=np.float32)
        w[f"{p}.self_attn.k_norm.weight"] = rng.standard_normal((128,), dtype=np.float32)
        w[f"{p}.attention_conv.base_kernel"] = rng.standard_normal((2, 2, HIDDEN), dtype=np.float32) * 0.1
        w[f"{p}.attention_conv.kernel_projection.weight"] = rng.standard_normal((1280, HIDDEN), dtype=np.float32) * 0.02
        w[f"{p}.mlp.gate_proj.weight"] = rng.standard_normal((17408, HIDDEN), dtype=np.float32) * 0.02
        w[f"{p}.mlp.up_proj.weight"] = rng.standard_normal((17408, HIDDEN), dtype=np.float32) * 0.02
        w[f"{p}.mlp.down_proj.weight"] = rng.standard_normal((HIDDEN, 17408), dtype=np.float32) * 0.02
        w[f"{p}.mlp_conv.base_kernel"] = rng.standard_normal((2, 2, HIDDEN), dtype=np.float32) * 0.1
        w[f"{p}.mlp_conv.kernel_projection.weight"] = rng.standard_normal((1280, HIDDEN), dtype=np.float32) * 0.02
    w["candidate_selector.predecessor_codebook"] = rng.standard_normal((VOCAB, 256), dtype=np.float32) * 0.05
    w["candidate_selector.successor_codebook"] = rng.standard_normal((VOCAB, 256), dtype=np.float32) * 0.05
    w["candidate_selector.hidden_projection.weight"] = rng.standard_normal((256, HIDDEN), dtype=np.float32) * 0.02
    return w


rng = np.random.default_rng(7)
weights = _make_weights(rng)
npd = DFlash2NumpyDrafter(cfg, weights)
taps = rng.standard_normal((CTX, 5 * HIDDEN), dtype=np.float32) * 0.02
noise = rng.standard_normal((BS, HIDDEN), dtype=np.float32) * 0.02
positions = np.arange(CTX + BS, dtype=np.int64)
projected = npd.project_target_hidden(taps[None])[0]
head = rng.standard_normal((VOCAB, HIDDEN), dtype=np.float32) * 0.02

with DFlash2NativeDrafter(cfg, weights, max_context_len=64) as native:
    native.upload_weight("output_head.weight", head)
    native.reset_projected_context(_to_bf16_bits(projected))
    native.runtime.device_synchronize()
    for _ in range(3):
        ptr = native.forward(_to_bf16_bits(noise), positions)
        native.runtime.device_synchronize()
        path, scores = native.select(ptr, native.wdev["output_head.weight"], None, np.asarray([7], dtype=np.int64))
        native.runtime.device_synchronize()
    print("native forward+select cycles OK; final path:", path.tolist())
