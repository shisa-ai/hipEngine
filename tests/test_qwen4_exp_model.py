from __future__ import annotations

import pytest

from hipengine.models import QWEN4_EXP_GGUF, Qwen4ExpGGUFModel, resolve_model


def test_qwen4_exp_model_plugin_resolves_only_its_registered_architecture() -> None:
    plugin = resolve_model("qwen4exp")

    assert plugin is QWEN4_EXP_GGUF
    assert isinstance(plugin, Qwen4ExpGGUFModel)
    assert plugin.name == "qwen4_exp_gguf"
    assert plugin.architectures == ("qwen4exp",)
    assert plugin.default_quant == "gguf_q4_k_m"
    assert plugin.default_backend == "auto"


def test_qwen4_exp_model_plugin_freezes_layer_and_state_geometry() -> None:
    plugin = QWEN4_EXP_GGUF

    kinds = tuple(plugin.attention_kind_for_layer(layer) for layer in range(48))
    assert kinds == ("gdn", "gdn", "gdn", "qsa") * 12
    assert kinds.count("gdn") == 36
    assert kinds.count("qsa") == 12
    assert tuple(layer for layer in range(48) if plugin.has_ple(layer)) == (1,)
    assert plugin.hidden_size == 2560
    assert plugin.residual_branch_count == 4
    assert plugin.residual_width == 10240
    assert plugin.qsa_dense_equivalent_max_tokens == 2051
    assert plugin.native_context_length == 262144
    assert "blk.{layer}.ssm_a" in plugin.weight_name_templates
    assert "blk.{layer}.ssm_dt.bias" in plugin.weight_name_templates
    assert "blk.{layer}.ssm_a.weight" not in plugin.weight_name_templates
    assert "blk.{layer}.ssm_dt.weight" not in plugin.weight_name_templates
    assert plugin.ple_device_resident is False
    assert plugin.vision_supported is False
    assert plugin.mtp_supported is False


def test_qwen4_exp_model_plugin_declares_distinct_gdn_and_qsa_sequences() -> None:
    gdn = QWEN4_EXP_GGUF.decode_layer_sequence(attention_kind="gdn", include_ple=True)
    qsa = QWEN4_EXP_GGUF.decode_layer_sequence(attention_kind="qsa", include_ple=False)

    assert gdn[:3] == ("ple_sparse_gather", "ple_inject", "gr_attn_read")
    assert "gdn_recurrence_fp32" in gdn
    assert "qsa_index_select" not in gdn
    assert "qsa_index_select" in qsa
    assert "kv_live_spans" in qsa
    assert "sparse_paged_attention_decode" in qsa
    assert gdn[-3:] == ("moe_top10_shared", "gr_ffn_write", "residual_state_commit")
    assert qsa[-3:] == ("moe_top10_shared", "gr_ffn_write", "residual_state_commit")


def test_qwen4_exp_model_plugin_rejects_unknown_layers_and_attention_kinds() -> None:
    with pytest.raises(ValueError, match="layer_id"):
        QWEN4_EXP_GGUF.attention_kind_for_layer(-1)
    with pytest.raises(ValueError, match="layer_id"):
        QWEN4_EXP_GGUF.has_ple(48)
    with pytest.raises(ValueError, match="attention_kind"):
        QWEN4_EXP_GGUF.decode_layer_sequence(attention_kind="dense", include_ple=False)
