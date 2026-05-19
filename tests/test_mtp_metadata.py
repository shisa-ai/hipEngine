from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.mtp import (
    qwen35_mtp_config_from_target,
    qwen35_mtp_runtime_tensor_names,
    validate_qwen35_mtp_metadata,
)
from hipengine.loading.safetensors import TensorInfo, WeightIndex
from hipengine.speculative import (
    ChainDraftRequest,
    MtpChainCompiler,
    MtpDraftRequest,
    MtpProposalContext,
    Qwen35MtpDraftProvider,
    TargetVerifyBatch,
    compile_mtp_chain,
)

LOCAL_PACKED_TARGET = Path(
    "/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/"
    "snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e"
)


def test_shared_chain_compiler_materializes_root_only_in_target_batch() -> None:
    draft = compile_mtp_chain(
        [MtpDraftRequest(request_id=7, root_position=100, candidate_tokens=(11, 12, 13), active_count=2)],
        candidate_budget=3,
    )

    assert draft.request_ids == (7,)
    assert draft.candidate_tokens == (11, 12, 13)
    assert draft.parent_positions == (100, 101, 102)
    assert draft.draft_depths == (1, 2, 3)
    assert draft.active_mask == (True, True, False)

    target = TargetVerifyBatch.from_draft(draft, root_tokens=(10,), root_positions=(100,))
    assert target.tokens == (10, 11, 12, 13)
    assert target.positions == (100, 101, 102, 103)
    assert target.parent_rows == (-1, 0, 1, 2)
    assert target.active_mask == (True, True, True, False)
    assert target.mode == "verify_chain"


def test_mtp_chain_compiler_uses_mtp_depth_buckets() -> None:
    assert MtpChainCompiler(candidate_budget=1).candidate_budget == 1
    assert MtpChainCompiler(candidate_budget=5).candidate_budget == 5
    with pytest.raises(ValueError, match="candidate_budget"):
        MtpChainCompiler(candidate_budget=4)


def test_dflash_and_mtp_can_share_chain_request_shape() -> None:
    request = ChainDraftRequest.from_root_prefixed(
        request_id=3,
        root_position=8,
        token_ids=(99, 100, 101),
        expected_root_token=99,
    )
    draft = compile_mtp_chain([request], candidate_budget=2)
    assert draft.candidate_tokens == (100, 101)
    assert draft.parent_positions == (8, 9)


def test_qwen35_mtp_provider_wraps_generated_tokens_as_candidate_only_batch() -> None:
    def generator(context: MtpProposalContext, budget: int):
        assert budget == 2
        assert context.root_tokens == (99,)
        return ((100, 101),)

    provider = Qwen35MtpDraftProvider(generator)
    draft = provider.propose(
        MtpProposalContext(request_ids=(5,), root_tokens=(99,), root_positions=(8,)),
        candidate_budget=2,
    )

    assert draft.request_ids == (5,)
    assert draft.candidate_tokens == (100, 101)
    assert draft.parent_positions == (8, 9)
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(99,), root_positions=(8,))
    assert target.tokens == (99, 100, 101)
    assert target.parent_rows == (-1, 0, 1)


def test_qwen35_mtp_metadata_validation_passes_fake_manifest() -> None:
    index = _mtp_index()

    result = validate_qwen35_mtp_metadata(index)

    assert result.passed is True
    assert result.config.hidden_size == 16
    assert result.config.num_experts == 8
    assert len(result.present) == 19
    assert qwen35_mtp_runtime_tensor_names(result.config)[0] == "mtp.fc.weight"


def test_qwen35_mtp_metadata_validation_reports_missing_shape_dtype() -> None:
    index = _mtp_index()
    tensors = dict(index.tensors)
    tensors.pop("mtp.norm.weight")
    tensors["mtp.fc.weight"] = _tensor("mtp.fc.weight", "F16", (16, 31))
    bad = WeightIndex(index.model_path, index.config, tensors, index.shards)

    result = validate_qwen35_mtp_metadata(bad)

    assert result.passed is False
    assert "mtp.norm.weight" in result.missing
    assert any("mtp.fc.weight" in err and "expected dtype BF16" in err for err in result.dtype_errors)
    assert any("mtp.fc.weight" in err and "expected shape (16, 32)" in err for err in result.shape_errors)


@pytest.mark.skipif(not LOCAL_PACKED_TARGET.exists(), reason="local packed target not cached")
def test_local_packed_target_cleanly_reports_missing_mtp_tensors() -> None:
    from hipengine.loading.safetensors import load_weight_index

    result = validate_qwen35_mtp_metadata(load_weight_index(LOCAL_PACKED_TARGET))

    assert result.passed is False
    assert len(result.present) == 0
    assert "mtp.fc.weight" in result.missing
    assert not result.dtype_errors
    assert not result.shape_errors


def _target_config() -> dict:
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 16,
            "vocab_size": 100,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "partial_rotary_factor": 1.0,
        },
        "quantization_config": {"quant_method": "paroquant", "bits": 4},
    }


def _mtp_index() -> WeightIndex:
    cfg = qwen35_mtp_config_from_target(_target_config())
    tensors = {name: _tensor(name, "BF16", shape) for name, shape in _expected_shapes(cfg).items()}
    return WeightIndex(Path("/fake/mtp-target"), _target_config(), tensors, (Path("fake.safetensors"),))


def _expected_shapes(cfg) -> dict[str, tuple[int, ...]]:
    hidden = cfg.hidden_size
    q = cfg.num_attention_heads * cfg.head_dim
    kv = cfg.num_key_value_heads * cfg.head_dim
    moe = cfg.moe_intermediate_size
    shared = cfg.shared_expert_intermediate_size
    experts = cfg.num_experts
    return {
        "mtp.fc.weight": (hidden, 2 * hidden),
        "mtp.pre_fc_norm_embedding.weight": (hidden,),
        "mtp.pre_fc_norm_hidden.weight": (hidden,),
        "mtp.layers.0.input_layernorm.weight": (hidden,),
        "mtp.layers.0.post_attention_layernorm.weight": (hidden,),
        "mtp.layers.0.self_attn.q_proj.weight": (2 * q, hidden),
        "mtp.layers.0.self_attn.k_proj.weight": (kv, hidden),
        "mtp.layers.0.self_attn.v_proj.weight": (kv, hidden),
        "mtp.layers.0.self_attn.o_proj.weight": (hidden, q),
        "mtp.layers.0.self_attn.q_norm.weight": (cfg.head_dim,),
        "mtp.layers.0.self_attn.k_norm.weight": (cfg.head_dim,),
        "mtp.layers.0.mlp.gate.weight": (experts, hidden),
        "mtp.layers.0.mlp.experts.gate_up_proj": (experts, 2 * moe, hidden),
        "mtp.layers.0.mlp.experts.down_proj": (experts, hidden, moe),
        "mtp.layers.0.mlp.shared_expert_gate.weight": (1, hidden),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (shared, hidden),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": (shared, hidden),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": (hidden, shared),
        "mtp.norm.weight": (hidden,),
    }


def _tensor(name: str, dtype: str, shape: tuple[int, ...]) -> TensorInfo:
    return TensorInfo(name=name, shard_path=Path("fake.safetensors"), dtype=dtype, shape=shape)
