from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hipengine.loading.gguf import scan_gguf


DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")
DEFAULT_STEPFUN_NVFP4_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--stepfun-ai--Step-3.7-Flash-NVFP4"
    / "snapshots/36afbf6e15100cdc2d7a5b79d7e95d276ed33679"
)


def _stepfun_gguf_dir() -> Path:
    return Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))


def _stepfun_nvfp4_snapshot() -> Path:
    return Path(os.environ.get("HIPENGINE_STEPFUN_NVFP4_SNAPSHOT", DEFAULT_STEPFUN_NVFP4_SNAPSHOT))


def _stepfun_gguf_paths() -> tuple[Path, ...]:
    root = _stepfun_gguf_dir()
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return paths


def _stepfun_hf_file(name: str) -> Path:
    path = _stepfun_nvfp4_snapshot() / name
    if not path.is_file():
        pytest.skip(
            f"StepFun NVFP4 metadata file {name} not found; set "
            "HIPENGINE_STEPFUN_NVFP4_SNAPSHOT to the cached HF snapshot"
        )
    return path


def test_stepfun_gguf_headers_match_expected_text_architecture() -> None:
    infos = tuple(scan_gguf(path) for path in _stepfun_gguf_paths())
    metadata = infos[0].metadata

    assert [info.metadata["split.no"] for info in infos] == [0, 1, 2]
    assert {info.metadata["split.count"] for info in infos} == {3}
    assert {info.metadata["split.tensors.count"] for info in infos} == {754}
    assert sum(info.tensor_count for info in infos) == 754
    # llama.cpp split GGUFs carry the full model metadata in shard 0; later
    # shards keep only split bookkeeping and tensor tables.
    assert infos[0].architecture == "step35"
    assert infos[0].file_type_name == "MOSTLY_Q3_K_L"

    assert metadata["general.name"] == "Step-3.7"
    assert metadata["step35.block_count"] == 45
    assert metadata["step35.context_length"] == 262_144
    assert metadata["step35.embedding_length"] == 4096
    assert metadata["step35.feed_forward_length"] == 11_264
    assert metadata["step35.leading_dense_block_count"] == 3
    assert metadata["step35.expert_count"] == 288
    assert metadata["step35.expert_used_count"] == 8
    assert metadata["step35.expert_feed_forward_length"] == 1280
    assert metadata["step35.expert_shared_feed_forward_length"] == 1280
    assert metadata["step35.expert_weights_norm"] is True
    assert metadata["step35.expert_weights_scale"] == pytest.approx(3.0)
    assert metadata["step35.attention.key_length"] == 128
    assert metadata["step35.attention.value_length"] == 128
    assert metadata["step35.attention.sliding_window"] == 512
    assert metadata["step35.attention.layer_norm_rms_epsilon"] == pytest.approx(1.0e-5)
    assert metadata["step35.rope.freq_base"] == pytest.approx(5_000_000.0)
    assert metadata["step35.rope.freq_base_swa"] == pytest.approx(10_000.0)

    sliding_pattern = metadata["step35.attention.sliding_window_pattern"]
    head_counts = metadata["step35.attention.head_count"]
    kv_head_counts = metadata["step35.attention.head_count_kv"]
    assert len(sliding_pattern) == len(head_counts) == len(kv_head_counts) == 45
    assert [idx for idx, is_sliding in enumerate(sliding_pattern) if not is_sliding] == list(
        range(0, 45, 4)
    )
    assert head_counts == [96 if is_sliding else 64 for is_sliding in sliding_pattern]
    assert kv_head_counts == [8] * 45

    assert metadata["tokenizer.ggml.model"] == "gpt2"
    assert metadata["tokenizer.ggml.pre"] == "deepseek-v3"
    assert metadata["tokenizer.ggml.bos_token_id"] == 0
    assert metadata["tokenizer.ggml.eos_token_id"] == 128007
    assert metadata["tokenizer.ggml.padding_token_id"] == 1
    assert len(metadata["tokenizer.ggml.tokens"]) == 128_896
    assert len(metadata["tokenizer.ggml.merges"]) == 127_741


def test_stepfun_gguf_tensor_headers_cover_dense_and_moe_layers() -> None:
    tensors = {
        tensor.name: tensor
        for path in _stepfun_gguf_paths()
        for tensor in scan_gguf(path).tensors
    }

    expected = {
        "token_embd.weight": ((128_896, 4096), "Q8_0"),
        "output.weight": ((128_896, 4096), "Q8_0"),
        "output_norm.weight": ((4096,), "F32"),
        "rope_freqs.weight": ((64,), "F32"),
        "blk.0.attn_q.weight": ((8192, 4096), "Q3_K"),
        "blk.0.attn_gate.weight": ((64, 4096), "Q3_K"),
        "blk.0.ffn_gate.weight": ((11_264, 4096), "Q3_K"),
        "blk.0.ffn_down.weight": ((4096, 11_264), "Q5_K"),
        "blk.3.ffn_gate_inp.weight": ((288, 4096), "F32"),
        "blk.3.exp_probs_b.bias": ((288,), "F32"),
        "blk.3.ffn_gate_exps.weight": ((288, 1280, 4096), "Q3_K"),
        "blk.3.ffn_down_exps.weight": ((288, 4096, 1280), "Q5_K"),
        "blk.3.ffn_gate_shexp.weight": ((1280, 4096), "Q3_K"),
        "blk.3.ffn_down_shexp.weight": ((4096, 1280), "Q5_K"),
        "blk.44.attn_q.weight": ((8192, 4096), "Q3_K"),
        "blk.44.ffn_gate_exps.weight": ((288, 1280, 4096), "Q3_K"),
    }

    assert len(tensors) == 754
    for name, (shape, qtype) in expected.items():
        tensor = tensors[name]
        assert tensor.shape == shape
        assert tensor.ggml_type_name == qtype
        assert tensor.data_offset >= 0
        assert tensor.nbytes > 0


def test_stepfun_hf_metadata_matches_gguf_architecture() -> None:
    with _stepfun_hf_file("config.json").open(encoding="utf-8") as fh:
        config = json.load(fh)
    with _stepfun_hf_file("hf_quant_config.json").open(encoding="utf-8") as fh:
        quant_config = json.load(fh)
    with _stepfun_hf_file("model.safetensors.index.json").open(encoding="utf-8") as fh:
        index = json.load(fh)

    text = config["text_config"]
    assert config["model_type"] == "step3p7"
    assert config["architectures"] == ["Step3p7ForConditionalGeneration"]
    assert text["model_type"] == "step3p5"
    assert text["architectures"] == ["Step3p5ForCausalLM"]
    assert text["num_hidden_layers"] == 45
    assert text["hidden_size"] == 4096
    assert text["vocab_size"] == 128_896
    assert text["max_position_embeddings"] == 262_144
    assert text["eos_token_id"] == [1, 2, 128007]
    assert text["moe_layers_enum"] == ",".join(str(idx) for idx in range(3, 45))
    assert text["moe_num_experts"] == 288
    assert text["moe_top_k"] == 8
    assert text["moe_intermediate_size"] == 1280
    assert text["share_expert_dim"] == 1280
    assert text["moe_router_activation"] == "sigmoid"
    assert text["moe_router_scaling_factor"] == pytest.approx(3.0)
    assert text["need_fp32_gate"] is True
    assert text["norm_expert_weight"] is True
    assert text["use_moe_router_bias"] is True
    assert text["use_head_wise_attn_gate"] is True
    assert text["sliding_window"] == 512
    assert text["rms_norm_eps"] == pytest.approx(1.0e-5)
    assert text["layer_types"] == [
        "sliding_attention" if idx % 4 else "full_attention" for idx in range(45)
    ]
    assert text["partial_rotary_factors"] == [
        1.0 if layer_type == "sliding_attention" else 0.5
        for layer_type in text["layer_types"]
    ]
    assert text["rope_theta"][:45] == [
        10_000.0 if layer_type == "sliding_attention" else 5_000_000.0
        for layer_type in text["layer_types"]
    ]

    quant = quant_config["quantization"]
    assert quant["quant_algo"] == "NVFP4"
    assert quant["kv_cache_quant_algo"] == "FP8"
    assert quant["group_size"] == 16
    assert "model.vision_model*" in quant["exclude_modules"]
    assert "model.vit_large_projector" in quant["exclude_modules"]

    weight_map = index["weight_map"]
    language_layers = {
        int(match.group(1))
        for name in weight_map
        if (match := re.search(r"model\.language_model\.layers\.(\d+)\.", name))
    }
    assert index["metadata"]["total_parameters"] == 103_810_330_432
    assert index["metadata"]["total_size"] == 124_385_012_840
    assert len(weight_map) == 1888
    assert language_layers == set(range(45))


def test_stepfun_metadata_scan_does_not_mmap_payloads() -> None:
    paths = _stepfun_gguf_paths()
    code = "\n".join(
        [
            "import numpy as np",
            "from hipengine.loading.gguf import scan_gguf",
            "def forbidden_memmap(*args, **kwargs):",
            "    raise AssertionError('metadata scan must not mmap tensor payloads')",
            "np.memmap = forbidden_memmap",
            f"paths = {[str(path) for path in paths]!r}",
            "for path in paths:",
            "    scan_gguf(path)",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)
