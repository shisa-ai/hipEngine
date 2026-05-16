from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.moe import (
    plan_qwen35_router_build,
    qwen35_router_logits_bf16,
    qwen35_router_logits_fp16,
    qwen35_router_select,
    qwen35_router_topk_shared_coop_out_bf16,
    qwen35_router_topk_shared_coop_out_fp16,
    qwen35_router_topk_shared_out_bf16,
    qwen35_router_topk_shared_out_fp16,
    register_qwen35_router_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_qwen35_router_registers_bf16_and_w4_paro() -> None:
    register_qwen35_router_kernels()

    assert resolve(backend="hip_gfx1100", layer="router_logits", quant="bf16") is qwen35_router_logits_bf16
    assert resolve(backend="hip_gfx1100", layer="router_logits", quant="fp16") is qwen35_router_logits_fp16
    assert (
        resolve(backend="hip_gfx1100", layer="router_logits", quant="w4_paro", variant="fp16_hidden")
        is qwen35_router_logits_fp16
    )
    assert resolve(backend="hip_gfx1100", layer="router_select", quant="fp32") is qwen35_router_select
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="bf16", variant="out")
        is qwen35_router_topk_shared_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="w4_paro", variant="out")
        is qwen35_router_topk_shared_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="w4_paro", variant="out_fp16_hidden")
        is qwen35_router_topk_shared_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="w4_paro", variant="coop_out")
        is qwen35_router_topk_shared_coop_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="w4_paro", variant="coop_out_fp16_hidden")
        is qwen35_router_topk_shared_coop_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="fp16", variant="out")
        is qwen35_router_topk_shared_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="router_topk_shared", quant="fp16", variant="coop_out")
        is qwen35_router_topk_shared_coop_out_fp16
    )


def test_qwen35_router_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_router_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc router test version",
    )

    assert artifact.family == "qwen35_router"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "qwen35_router.so"
    assert artifact.compiler_version == "hipcc router test version"
    assert any(str(path).endswith("router.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_router_wrappers_validate_shape_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_router_logits_bf16(0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="threads must be one of"):
        qwen35_router_logits_bf16(0, 0, 0, 1, 16, 8, threads=32)
    with pytest.raises(ValueError, match="tokens must be positive"):
        qwen35_router_logits_fp16(0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="top_k must be <= 16"):
        qwen35_router_select(0, 0, 0, 1, 8, 8, 17)
    with pytest.raises(ValueError, match="top_k must be <= num_experts"):
        qwen35_router_select(0, 0, 0, 1, 8, 2, 4)
    with pytest.raises(ValueError, match="num_experts must be smaller"):
        qwen35_router_topk_shared_out_bf16(0, 0, 0, 0, 0, 1, 16, 8, 8, 4)
    with pytest.raises(ValueError, match="num_experts must be smaller"):
        qwen35_router_topk_shared_out_fp16(0, 0, 0, 0, 0, 1, 16, 8, 8, 4)
    with pytest.raises(ValueError, match="decode-only"):
        qwen35_router_topk_shared_coop_out_bf16(0, 0, 0, 0, 0, 2, 16, 9, 8, 4)
    with pytest.raises(ValueError, match="num_experts must be smaller"):
        qwen35_router_topk_shared_coop_out_fp16(0, 0, 0, 0, 0, 1, 16, 8, 8, 4)
