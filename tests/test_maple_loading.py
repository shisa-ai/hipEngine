"""Checkpoint-layout and raw materialization tests for Maple ternary MLX."""

from __future__ import annotations

import copy
import ctypes
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.loading.maple import (
    MAPLE_EXACT_TENSOR_COUNT,
    MAPLE_EXACT_WEIGHT_BYTES,
    MapleLayoutError,
    load_maple_tensor_to_device,
    maple_tensor_requirements,
    read_maple_tensor,
    validate_maple_weight_index,
)
from hipengine.loading.safetensors import TensorInfo, WeightIndex, load_weight_index
from hipengine.models import MAPLE_LAYER_PATTERN
from hipengine.models.maple import parse_maple_model_spec
from hipengine.quant.maple_ternary import MAPLE_TERNARY2
from hipengine.quant.registry import resolve_quant
from hipengine.tokenization.maple import MapleTokenizer


class FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x1000
        self.buffers: dict[int, bytearray] = {}
        self.freed: list[int] = []

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(nbytes, 1) + 0x100
        self.buffers[ptr] = bytearray(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(ptr)
        self.buffers.pop(ptr, None)

    def memcpy(self, dst: int, src: int, count: int, kind: HipMemcpyKind) -> None:
        assert kind == HipMemcpyKind.HOST_TO_DEVICE
        self.buffers[dst][:count] = ctypes.string_at(src, count)


def maple_config() -> dict:
    quantization = {
        "bits": 2,
        "group_size": 128,
        "mode": "affine",
        "lm_head": {"bits": 4, "group_size": 64},
        "model.word_embeddings": {"bits": 4, "group_size": 64},
    }
    return {
        "architectures": ["MapleForCausalLM"],
        "model_type": "maple",
        "dtype": "bfloat16",
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "first_k_dense_replace": 0,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "layer_types": list(MAPLE_LAYER_PATTERN),
        "max_position_embeddings": 128000,
        "moe_intermediate_size": 512,
        "norm_topk_prob": True,
        "num_attention_heads": 16,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 24,
        "num_key_value_heads": 4,
        "num_shared_experts": 0,
        "partial_rotary_factor": 0.5,
        "quantization": quantization,
        "quantization_config": copy.deepcopy(quantization),
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 10000,
        "router_dtype": "fp32",
        "sliding_window": 512,
        "tie_word_embeddings": False,
        "use_bias": False,
        "use_qk_norm": True,
        "vocab_size": 151936,
    }


def synthetic_index(*, flash_head: bool = True) -> WeightIndex:
    config = maple_config()
    spec = parse_maple_model_spec(config)
    path = Path("/synthetic/maple/model.safetensors")
    tensors = {
        requirement.name: TensorInfo(
            name=requirement.name,
            shard_path=path,
            dtype=requirement.dtype,
            shape=requirement.shape,
        )
        for requirement in maple_tensor_requirements(spec)
    }
    if flash_head:
        tensors["lm_head_flash.token_map"] = TensorInfo(
            name="lm_head_flash.token_map",
            shard_path=path,
            dtype="I32",
            shape=(4748, 32),
        )
    return WeightIndex(
        model_path=path.parent,
        config=config,
        tensors=dict(sorted(tensors.items())),
        shards=(path,),
    )


def test_maple_manifest_pins_exact_checkpoint_footprint_and_optional_flash_head() -> None:
    validation = validate_maple_weight_index(synthetic_index())
    assert len(validation.exact_tensor_names) == MAPLE_EXACT_TENSOR_COUNT == 463
    assert validation.exact_weight_bytes == MAPLE_EXACT_WEIGHT_BYTES == 5_308_186_624
    assert validation.ignored_flash_head_names == ("lm_head_flash.token_map",)
    assert validation.exact_tensor_names[0] == "model.word_embeddings.weight"
    assert validation.exact_tensor_names[-1] == "lm_head.biases"


def test_maple_manifest_rejects_missing_shape_dtype_and_unexpected_tensors() -> None:
    base = synthetic_index(flash_head=False)

    tensors = dict(base.tensors)
    tensors.pop("model.layers.0.self_attn.q_proj.weight")
    with pytest.raises(MapleLayoutError, match="missing Maple tensors"):
        validate_maple_weight_index(
            WeightIndex(base.model_path, base.config, tensors, base.shards)
        )

    tensors = dict(base.tensors)
    old = tensors["model.layers.0.mlp.switch_mlp.down_proj.weight"]
    tensors[old.name] = TensorInfo(old.name, old.shard_path, old.dtype, (256, 2048, 31))
    with pytest.raises(MapleLayoutError, match="shape"):
        validate_maple_weight_index(
            WeightIndex(base.model_path, base.config, tensors, base.shards)
        )

    tensors = dict(base.tensors)
    old = tensors["model.layers.0.mlp.gate.weight"]
    tensors[old.name] = TensorInfo(old.name, old.shard_path, "F16", old.shape)
    with pytest.raises(MapleLayoutError, match="dtype"):
        validate_maple_weight_index(
            WeightIndex(base.model_path, base.config, tensors, base.shards)
        )

    tensors = dict(base.tensors)
    tensors["unexpected.weight"] = TensorInfo(
        "unexpected.weight", old.shard_path, "BF16", (1,)
    )
    with pytest.raises(MapleLayoutError, match="unexpected Maple tensors"):
        validate_maple_weight_index(
            WeightIndex(base.model_path, base.config, tensors, base.shards)
        )


def test_maple_u32_reader_and_device_upload_preserve_packed_bits(tmp_path) -> None:
    packed = np.asarray([[0x01234567, 0x89ABCDEF]], dtype=np.uint32)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    save_file({"packed": packed}, tmp_path / "model.safetensors")
    info = load_weight_index(tmp_path).tensors["packed"]

    got = read_maple_tensor(info)
    assert got.dtype == np.dtype("<u4")
    assert np.array_equal(got, packed)

    runtime = FakeRuntime()
    allocation = load_maple_tensor_to_device(
        info, device=Device("hip", 1), runtime=runtime
    )
    assert allocation.source is info
    assert allocation.tensor.dtype is DType.INT32
    assert allocation.tensor.device == Device("hip", 1)
    assert bytes(runtime.buffers[allocation.buffer.ptr]) == packed.tobytes()
    allocation.free(runtime=runtime)
    assert runtime.freed == [allocation.buffer.ptr]


def test_maple_quant_plugin_describes_official_packing() -> None:
    assert resolve_quant("maple_ternary2") is MAPLE_TERNARY2
    assert MAPLE_TERNARY2.weight_storage == "u32_packed_2bit_row_alpha"
    assert MAPLE_TERNARY2.compute_dtype == "bf16"
    assert MAPLE_TERNARY2.scale_granularity == "per_row"


def test_maple_chat_prompt_matches_checkpoint_generation_suffix() -> None:
    assert MapleTokenizer.format_chat_prompt("Hello") == (
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    assert MapleTokenizer.format_chat_prompt("Hello", system="Be concise.") == (
        "<|im_start|>system\nBe concise.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
