"""Host-side guards and symbolic numerical proof for the router DPP tree."""
from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.moe import router
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def test_router_dpp_peer_registration_and_strict_fallback():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    clear_registry_for_tests()
    try:
        router.register_qwen35_router_kernels(replace=True)
        register_gfx1151_kernels(replace=True)
        assert resolve(backend="hip_gfx1151", layer="router_logits", quant="f32",
                       variant="f32_hidden_token_tile4_dpp_exact") is (
                           router.qwen35_router_logits_f32_f32w_token_tile4_dpp_exact)
        assert resolve(backend="hip_gfx1151", layer="router_logits", quant="f32",
                       variant="f32_hidden_token_tile4_dense_exact") is (
                           router.qwen35_router_logits_f32_f32w_token_tile4_dense_exact)
    finally:
        clear_registry_for_tests()


def test_router_dpp_register_fold_preserves_parent_tree_bits():
    rng = np.random.default_rng(20261005)
    values = (rng.normal(size=(10000, 256)) *
              np.exp2(rng.integers(-20, 20, (10000, 256)))).astype(np.float32)
    parent = values.copy()
    for stride in (128, 64, 32, 16, 8, 4, 2, 1):
        parent[:, :stride] += parent[:, stride:2*stride]
    # Each of the first32 lanes folds the same inter-wave pairs as the LDS
    # parent, not a sum of pre-reduced waves (which would change association).
    candidate = values[:, :32] + values[:, 128:160]
    candidate += values[:, 64:96] + values[:, 192:224]
    right = values[:, 32:64] + values[:, 160:192]
    right += values[:, 96:128] + values[:, 224:256]
    candidate += right
    for stride in (16, 8, 4, 2, 1):
        candidate[:, :stride] += candidate[:, stride:2*stride]
    np.testing.assert_array_equal(parent[:, 0].view(np.uint32),
                                  candidate[:, 0].view(np.uint32))
    changed = values.reshape(10000, 8, 32).sum(axis=2).sum(axis=1)
    assert np.any(changed.view(np.uint32) != parent[:, 0].view(np.uint32))


@pytest.mark.parametrize("tokens,k,n,threads,match", [
    (0, 2560, 512, 256, "tokens"),
    (512, 0, 512, 256, "hidden_size"),
    (512, 2560, 0, 256, "num_rows"),
    (512, 2560, 512, 128, "256 threads"),
])
def test_router_dpp_shape_guards_precede_library(tokens, k, n, threads, match):
    with pytest.raises(ValueError, match=match):
        router.qwen35_router_logits_f32_f32w_token_tile4_dpp_exact(
            1, 2, 3, tokens, k, n, threads=threads)
