"""CPU-only Q8 sidecar planning boundaries, independent of local model files."""

from __future__ import annotations

import pytest

from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.qwen35_gguf_materialize import plan_qwen35_gguf_weight_spec
from hipengine.quant.gguf import GGMLQuantizationType


@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("all_dense", [False, True])
@pytest.mark.parametrize("slot", ["attn_qkv", "attn_gate", "attn_q", "ffn_gate_shexp", "ssm_out"])
def test_dense_q8_raw_sidecar_scope_is_load_time_policy(monkeypatch, sidecar, all_dense, slot):
    monkeypatch.setenv("HIPENGINE_GGUF_Q8_0_RAW_SIDECAR", str(int(sidecar)))
    monkeypatch.setenv("HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL", str(int(all_dense)))
    tensor = GGUFTensorInfo(
        name=f"blk.0.{slot}.weight", shape=(256, 256), ggml_shape=(256, 256),
        ggml_type=int(GGMLQuantizationType.Q8_0), ggml_type_name="Q8_0",
        n_elements=256 * 256, nbytes=256 * (256 // 32) * 34,
        offset=0, data_offset=0, byte_shape=(256, (256 // 32) * 34),
    )

    spec = plan_qwen35_gguf_weight_spec(f"layers.0.{slot}", tensor, decode_repack=True)

    assert spec.quant_key == "gguf_q8_0_t16_v1"
    keep_raw = sidecar and (all_dense or slot in {"attn_qkv", "attn_gate"})
    assert spec.allocation_names == (("tiles", "raw") if keep_raw else ("tiles",))
