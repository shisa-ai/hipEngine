#!/usr/bin/env python3
"""Cross-validate DFlash2NativeDrafter.forward against the NumPy drafter layer
loop on synthetic weights with real shapes (D2b RED)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from hipengine.kernels.cpu_reference.ops import rmsnorm
from hipengine.loading.dflash import DFlashDraftConfig
from hipengine.speculative.dflash2_drafter import DFlash2NumpyDrafter
from hipengine.speculative.dflash2_native import (
    DFlash2NativeDrafter,
    _from_bf16_bits,
    _to_bf16_bits,
)

N_LAYERS = 5
HIDDEN = 5120
INTER = 17408
VOCAB = 248320
BS = 8
CTX = 16


def make_config() -> DFlashDraftConfig:
    return DFlashDraftConfig(
        architecture="DFlash2DraftModel",
        block_size=BS,
        mask_token_id=248070,
        target_layer_ids=(5, 19, 33, 47, 61),
        num_target_layers=64,
        hidden_size=HIDDEN,
        target_hidden_size=HIDDEN,
        target_hidden_concat_size=5 * HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rope_theta=1e7,
        vocab_size=VOCAB,
        dtype="bfloat16",
        layer_types=("sliding_attention",) * N_LAYERS,
        rms_norm_eps=1e-6,
        sliding_windows=(2048,) * N_LAYERS,
        causal=False,
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=256,
        selector_top_k=16,
    )


def make_weights(rng: np.random.default_rng) -> dict[str, np.ndarray]:
    w: dict[str, np.ndarray] = {}
    w["fc.weight"] = rng.standard_normal((HIDDEN, 5 * HIDDEN), dtype=np.float32) * 0.02
    w["hidden_norm.weight"] = rng.standard_normal((HIDDEN,), dtype=np.float32)
    w["norm.weight"] = rng.standard_normal((HIDDEN,), dtype=np.float32)
    for l in range(N_LAYERS):
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
        w[f"{p}.mlp.gate_proj.weight"] = rng.standard_normal((INTER, HIDDEN), dtype=np.float32) * 0.02
        w[f"{p}.mlp.up_proj.weight"] = rng.standard_normal((INTER, HIDDEN), dtype=np.float32) * 0.02
        w[f"{p}.mlp.down_proj.weight"] = rng.standard_normal((HIDDEN, INTER), dtype=np.float32) * 0.02
        w[f"{p}.mlp_conv.base_kernel"] = rng.standard_normal((2, 2, HIDDEN), dtype=np.float32) * 0.1
        w[f"{p}.mlp_conv.kernel_projection.weight"] = rng.standard_normal((1280, HIDDEN), dtype=np.float32) * 0.02
    w["candidate_selector.predecessor_codebook"] = rng.standard_normal((VOCAB, 256), dtype=np.float32) * 0.05
    w["candidate_selector.successor_codebook"] = rng.standard_normal((VOCAB, 256), dtype=np.float32) * 0.05
    w["candidate_selector.hidden_projection.weight"] = rng.standard_normal((256, HIDDEN), dtype=np.float32) * 0.02
    return w


def main() -> int:
    rng = np.random.default_rng(1234)
    config = make_config()
    weights = make_weights(rng)

    # Inputs: taps concat (CTX, 5*HIDDEN), noise (BS, HIDDEN), positions.
    taps = rng.standard_normal((CTX, 5 * HIDDEN), dtype=np.float32) * 0.02
    noise = rng.standard_normal((BS, HIDDEN), dtype=np.float32) * 0.02
    positions = np.arange(CTX + BS, dtype=np.int64)  # context + block positions

    # --- NumPy reference full forward ---
    npd = DFlash2NumpyDrafter(config, weights)
    projected = npd.project_target_hidden(taps[None])[0]
    ref = npd.forward(taps[None], noise[None], positions[None])[0]
    ref_bf16 = _from_bf16_bits(_to_bf16_bits(ref))
    print("numpy ref:", ref_bf16.shape, "finite", np.isfinite(ref_bf16).all())
    # numpy per-layer hidden for comparison
    np_layers = []
    np_hidden = np.asarray(noise, dtype=np.float32)
    for layer in range(N_LAYERS):
        np_hidden = npd.forward_layer(np_hidden[None], projected[None], positions[None], layer)[0]
        np_layers.append(_from_bf16_bits(_to_bf16_bits(np_hidden)))
    from hipengine.kernels.cpu_reference.ops import rmsnorm
    np_loop_final = _from_bf16_bits(_to_bf16_bits(rmsnorm(np_hidden, weights["norm.weight"], eps=config.rms_norm_eps)))
    print("numpy loop final vs forward ref err:", np.abs(np_loop_final - ref_bf16).max())

    # --- Native forward (takes f32 weights; converts to bf16 internally) ---
    with DFlash2NativeDrafter(config, weights, max_context_len=64) as native:
        proj_bf16 = _to_bf16_bits(projected)
        native.reset_projected_context(proj_bf16)
        noise_bf16 = _to_bf16_bits(noise)
        drafts_per_layer = []

        def cb(layer, ptr):
            native.runtime.device_synchronize()
            arr = native._d2h(ptr, (BS, HIDDEN), np.uint16)
            v = _from_bf16_bits(arr)
            drafts_per_layer.append((layer, v))
            npv = np_layers[layer]
            err = np.abs(v.astype(np.float64) - npv.astype(np.float64))
            print(f"layer {layer}: native_max {np.abs(v).max():.2f} np_max {np.abs(npv).max():.2f} max_err {err.max():.4f} mean_err {err.mean():.4f}")
            if layer == 0:
                print("  native[0,:6]", v[0, :6])
                print("  numpy [0,:6]", npv[0, :6])

        draft_ptr = native.forward(noise_bf16, positions, debug_callback=cb)
        got = native._d2h(draft_ptr, (BS - 1, HIDDEN), np.uint16)
        got_bf16 = _from_bf16_bits(got)
        ref7 = ref_bf16[1:]
        denom = np.abs(ref7).astype(np.float64) + 1e-3
        rel = np.abs(got_bf16.astype(np.float64) - ref7.astype(np.float64)) / denom
        rel = rel.ravel()
        print(f"final rel err: p50 {np.percentile(rel,50):.4f} p90 {np.percentile(rel,90):.4f} p99 {np.percentile(rel,99):.4f} max {rel.max():.4f} mean {rel.mean():.4f}")
        print("final native[0,:6]", got_bf16[0, :6])
        print("final numpy [0,:6]", ref7[0, :6])
        # cross-check: what does block_hidden_a hold after forward? (final pre-norm hidden)
        bh_a = _from_bf16_bits(native._d2h(native.block_hidden_a, (BS, HIDDEN), np.uint16))
        print("cb layer4 vs post-forward block_hidden_a err:", np.abs(drafts_per_layer[4][1] - bh_a).max())
        ref_bf16 = ref_bf16[1:]  # draft rows exclude the anchor row 0
        print("native out:", got_bf16.shape, "finite", np.isfinite(got_bf16).all())
        err = np.abs(got_bf16.astype(np.float64) - ref_bf16.astype(np.float64))
        denom = np.abs(ref_bf16).astype(np.float64) + 1e-3
        rel = (err / denom).ravel()
        print("max abs err:", err.max())
        print("rel err p50/p90/p99/max/mean:", round(float(np.percentile(rel, 50)), 4), round(float(np.percentile(rel, 90)), 4),
              round(float(np.percentile(rel, 99)), 4), round(float(rel.max()), 4), round(float(rel.mean()), 4))
        print("mean abs err:", err.mean())
        # bf16-tolerance gate for the draft hidden (post-norm values are small;
        # relative-to-max is the right scale). native is a bf16 path vs the f32
        # numpy oracle, so exact match is not expected.
        ref_max = float(np.abs(ref_bf16).max())
        ok = np.isfinite(got_bf16).all() and (err.mean() <= 0.05 * ref_max) and (err.max() <= 0.20 * ref_max)
        print(f"ref_max {ref_max:.3f} mean<=0.05*rmax {err.mean() <= 0.05 * ref_max} max<=0.20*rmax {err.max() <= 0.20 * ref_max}")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
