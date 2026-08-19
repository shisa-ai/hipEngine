"""D2b wiring tests for the native DFlash2 drafter forward + selector path.

The native module (``hipengine/speculative/dflash2_native.py``) wires the
grouped-dynamic-conv, attention, MLP, and candidate-selector kernels into a
torch-free forward that mirrors ``hipengine/speculative/dflash2_drafter.py``
(DFlash2NumpyDrafter).  These tests validate the *composite* wiring:

- forward is deterministic for a fixed input (guards against races like the
  in-place rotary pair-read bug);
- forward output agrees with the exact numpy oracle to a BF16-tolerance
  (native runs a BF16 path vs the F32 oracle);
- ``select`` produces a valid greedy path that matches the numpy ``propose``
  draft tokens.

The per-kernel strict RED gates live in ``test_dflash2_native_kernels.py``;
this file covers the composite module contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.dflash2 import candidate_selector_select
from hipengine.kernels.cpu_reference.ops import rmsnorm, linear
from hipengine.speculative.dflash2_drafter import DFlash2NumpyDrafter


def _has_gpu() -> bool:
    try:  # noqa: SIM105
        from hipengine.core.hip import get_hip_runtime
    except Exception:
        return False
    try:
        get_hip_runtime()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="HIP runtime not available")


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


def _to_bf16_bits(x_f32: np.ndarray) -> np.ndarray:
    bits = x_f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _from_bf16_bits(x_u16: np.ndarray) -> np.ndarray:
    return (x_u16.astype(np.uint32) << 16).view(np.float32)


HIDDEN = 5120
INTER = 17408
CTX = 16
BS = 8
N_LAYERS = 5
VOCAB = 1280


def _make_config() -> object:
    from hipengine.speculative.dflash2_drafter import DFlashDraftConfig

    return DFlashDraftConfig(
        architecture="qwen3mtp-dflash2",
        block_size=BS,
        mask_token_id=151664,
        target_layer_ids=tuple(range(N_LAYERS)),
        num_target_layers=N_LAYERS,
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


def _make_weights(rng: np.random.default_rng) -> dict[str, np.ndarray]:
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
    # synthetic target output head (vocab, hidden) bf16
    w["output_head.weight"] = rng.standard_normal((VOCAB, HIDDEN), dtype=np.float32) * 0.02
    return w


def _build_native(weights, rng):
    from hipengine.speculative.dflash2_native import DFlash2NativeDrafter

    config = _make_config()
    return DFlash2NativeDrafter(config, weights, max_context_len=64)


def test_dflash2_native_forward_deterministic(_runtime):
    """Same input -> byte-identical forward output (guards the in-place rotary
    pair-read race and other buffer-aliasing bugs)."""
    from hipengine.speculative.dflash2_native import _to_bf16_bits, _from_bf16_bits

    rng = np.random.default_rng(1234)
    weights = _make_weights(rng)
    from hipengine.speculative.dflash2_drafter import DFlash2NumpyDrafter

    npd = DFlash2NumpyDrafter(_make_config(), weights)
    taps = rng.standard_normal((CTX, 5 * HIDDEN), dtype=np.float32) * 0.02
    noise = rng.standard_normal((BS, HIDDEN), dtype=np.float32) * 0.02
    positions = np.arange(CTX + BS, dtype=np.int64)
    projected = npd.project_target_hidden(taps[None])[0]

    with _build_native(weights, rng) as native:
        native.reset_projected_context(_to_bf16_bits(projected))
        native.runtime.device_synchronize()
        out = []
        for _ in range(2):
            ptr = native.forward(_to_bf16_bits(noise), positions)
            native.runtime.device_synchronize()
            out.append(_from_bf16_bits(native._d2h(ptr, (BS - 1, HIDDEN), np.uint16)))
        assert np.isfinite(out[0]).all()
        assert np.array_equal(out[0], out[1]), "native forward is not deterministic"


def test_dflash2_native_forward_bf16_tolerance(_runtime):
    """Composite forward matches the exact numpy oracle to BF16 tolerance."""
    from hipengine.speculative.dflash2_native import _to_bf16_bits, _from_bf16_bits
    from hipengine.speculative.dflash2_drafter import DFlash2NumpyDrafter

    rng = np.random.default_rng(2024)
    weights = _make_weights(rng)
    npd = DFlash2NumpyDrafter(_make_config(), weights)
    taps = rng.standard_normal((CTX, 5 * HIDDEN), dtype=np.float32) * 0.02
    noise = rng.standard_normal((BS, HIDDEN), dtype=np.float32) * 0.02
    positions = np.arange(CTX + BS, dtype=np.int64)
    projected = npd.project_target_hidden(taps[None])[0]
    ref = npd.forward(taps[None], noise[None], positions[None])[0]
    ref_bf16 = _from_bf16_bits(_to_bf16_bits(ref))[1:]  # drop the anchor row

    with _build_native(weights, rng) as native:
        native.reset_projected_context(_to_bf16_bits(projected))
        native.runtime.device_synchronize()
        ptr = native.forward(_to_bf16_bits(noise), positions)
        native.runtime.device_synchronize()
        got = _from_bf16_bits(native._d2h(ptr, (BS - 1, HIDDEN), np.uint16))

    assert np.isfinite(got).all()
    err = np.abs(got.astype(np.float64) - ref_bf16.astype(np.float64))
    ref_max = float(np.abs(ref_bf16).max())
    assert err.mean() <= 0.05 * ref_max, f"mean abs err {err.mean():.3f} vs 0.05*rmax {0.05*ref_max:.3f}"
    assert err.max() <= 0.20 * ref_max, f"max abs err {err.max():.3f} vs 0.20*rmax {0.20*ref_max:.3f}"


def test_dflash2_native_forward_select_matches_numpy(_runtime):
    """Native forward + select produces the same greedy draft tokens as the
    numpy propose path (the D2b composite wiring contract)."""
    from hipengine.speculative.dflash2_native import DFlash2NativeDrafter, _to_bf16_bits, _from_bf16_bits

    rng = np.random.default_rng(3033)
    weights = _make_weights(rng)
    npd = DFlash2NumpyDrafter(_make_config(), weights)
    taps = rng.standard_normal((CTX, 5 * HIDDEN), dtype=np.float32) * 0.02
    noise = rng.standard_normal((BS, HIDDEN), dtype=np.float32) * 0.02
    positions = np.arange(CTX + BS, dtype=np.int64)
    projected = npd.project_target_hidden(taps[None])[0]
    ref = npd.forward(taps[None], noise[None], positions[None])[0]
    ref_bf16 = _from_bf16_bits(_to_bf16_bits(ref))[1:]  # (block_size-1, hidden)
    head = weights["output_head.weight"]
    anchor = np.asarray([7], dtype=np.int64)

    # numpy reference proposal
    logits_ref = linear(ref_bf16, head)
    res = candidate_selector_select(
        ref_bf16[None], logits_ref[None], anchor,
        weights["candidate_selector.predecessor_codebook"],
        weights["candidate_selector.successor_codebook"],
        weights["candidate_selector.hidden_projection.weight"],
        top_k=16,
    )
    np_path = res.path[0]

    with DFlash2NativeDrafter(_make_config(), weights, max_context_len=64) as native:
        native.reset_projected_context(_to_bf16_bits(projected))
        native.runtime.device_synchronize()
        ptr = native.forward(_to_bf16_bits(noise), positions)
        native.runtime.device_synchronize()
        path, scores = native.select(ptr, native.wdev["output_head.weight"], None, anchor)

    assert np.isfinite(scores).all()
    assert len(path) == BS - 1
    assert np.all((path >= 0) & (path < VOCAB))
    assert np.array_equal(path, np_path), (
        f"native select path {path.tolist()} != numpy propose {np_path.tolist()}"
    )
