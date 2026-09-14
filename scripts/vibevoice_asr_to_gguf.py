#!/usr/bin/env python3
"""Export microsoft/VibeVoice-ASR-HF to a GGUF v3 container.

The GGUF keeps the checkpoint's own tensor names and BF16 payloads
byte-identical, so a GGUF-loaded model is exactly the HF model until a
quantizer rewrites selected tensors. The backbone is then quantized in a
second step with llama-quantize (which only rewrites 2-D tensors, leaving
the 3-D conv weights and 1-D tensors untouched).

Metadata follows the existing loader contract: the Qwen2 backbone geometry
is stored as ``vibevoice.text.*`` keys mirroring ``config.json``'s
``text_config``, and the encoder geometry as ``vibevoice.acoustic.*`` /
``vibevoice.semantic.*`` so a GGUF-side loader can rebuild
``VibevoiceQwen2Spec`` and the encoder configs without the HF directory.

Usage:
    python3 scripts/vibevoice_asr_to_gguf.py \
        --model microsoft/VibeVoice-ASR-HF --out /tmp/vibevoice-asr-bf16.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import gguf

from hipengine.loading.safetensors import load_weight_index, read_tensor_storage_bytes
from hipengine.loading.hf_cache import resolve_model_path

ENCODER_SECTIONS = (
    ("acoustic", "acoustic_tokenizer_encoder_config"),
    ("semantic", "semantic_tokenizer_encoder_config"),
)
ENCODED_PREFIXES = (
    "acoustic_tokenizer_encoder.",
    "semantic_tokenizer_encoder.",
    "language_model.",
    "multi_modal_projector.",
)

# llama.cpp caps GGUF tensor names at 64 bytes, so long HF names are
# prefix-compressed; the GGUF-side loader expands them back.
NAME_MAP = (
    ("acoustic_tokenizer_encoder.", "ate."),
    ("semantic_tokenizer_encoder.", "ste."),
    ("language_model.", "lm."),
    ("multi_modal_projector.", "mmp."),
)


def gguf_name(hf_name: str) -> str:
    for src, dst in NAME_MAP:
        if hf_name.startswith(src):
            return dst + hf_name[len(src):]
    raise SystemExit(f"unexpected tensor outside known sections: {hf_name}")


# llama.cpp naming for the Qwen2 backbone so llama-quantize accepts the
# file (it only quantizes tensors whose roles it recognizes; encoder and
# projector tensors are left untouched, which is the intended
# mixed-precision layout).
LLAMA_BACKBONE = (
    ("language_model.model.embed_tokens.weight", "token_embd.weight"),
    ("language_model.model.norm.weight", "output_norm.weight"),
    ("language_model.lm_head.weight", "output.weight"),
)


def llama_name(hf_name: str) -> str:
    for src, dst in LLAMA_BACKBONE:
        if hf_name == src:
            return dst
    if hf_name.startswith("language_model.model.layers."):
        rest = hf_name[len("language_model.model.layers."):]
        layer, _, tail = rest.partition(".")
        table = {
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.q_proj.bias": "attn_q.bias",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.k_proj.bias": "attn_k.bias",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.v_proj.bias": "attn_v.bias",
            "self_attn.o_proj.weight": "attn_output.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        llama_tail = table.get(tail)
        if llama_tail is None:
            raise SystemExit(f"unmapped backbone tensor: {hf_name}")
        return f"blk.{layer}.{llama_tail}"
    for src, dst in NAME_MAP:
        if hf_name.startswith(src):
            return dst + hf_name[len(src):]
    raise SystemExit(f"unexpected tensor outside known sections: {hf_name}")
TEXT_SPEC_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
)
TEXT_SPEC_FLOAT_KEYS = ("rms_norm_eps",)


def resolve_snapshot(model: str) -> Path:
    return Path(resolve_model_path(model))


def iter_tensors(snapshot: Path, dtype: str):
    index = load_weight_index(snapshot)
    for name in sorted(index.names_with_prefix("")):
        info = index.require((name,))[0]
        payload = read_tensor_storage_bytes(info)
        if info.dtype == "BF16":
            array = np.frombuffer(payload, dtype=np.uint16).reshape(info.shape)
        elif info.dtype == "F32":
            array = np.frombuffer(payload, dtype=np.float32).reshape(info.shape)
            if dtype == "f16":
                array = array.astype(np.float16)
        else:
            raise SystemExit(f"unsupported checkpoint dtype {info.dtype!r} for {name!r}")
        yield name, array


def add_metadata(writer: gguf.GGUFWriter, config: dict) -> None:
    writer.add_architecture()
    writer.add_string("vibevoice.model_type", str(config.get("model_type", "vibevoice_asr")))
    writer.add_uint32("vibevoice.audio_token_id", int(config["audio_token_id"]))
    writer.add_uint32("vibevoice.audio_bos_token_id", int(config["audio_bos_token_id"]))
    writer.add_uint32("vibevoice.audio_eos_token_id", int(config["audio_eos_token_id"]))
    writer.add_uint32("vibevoice.acoustic_chunk_size", int(config["acoustic_tokenizer_chunk_size"]))

    for short, key in ENCODER_SECTIONS:
        enc = config[f"{key}"]
        prefix = f"vibevoice.{short}."
        writer.add_uint32(prefix + "hidden_size", int(enc["hidden_size"]))
        writer.add_uint32(prefix + "num_filters", int(enc["num_filters"]))
        writer.add_uint32(prefix + "kernel_size", int(enc["kernel_size"]))
        writer.add_array(prefix + "depths", [int(v) for v in enc["depths"]])
        writer.add_array(
            prefix + "downsampling_ratios", [int(v) for v in enc["downsampling_ratios"]]
        )
        writer.add_uint32(prefix + "ffn_expansion", int(enc["ffn_expansion"]))
        writer.add_float32(prefix + "rms_norm_eps", float(enc["rms_norm_eps"]))
        writer.add_float32(prefix + "layer_scale_init_value", float(enc["layer_scale_init_value"]))

    text = config["text_config"]
    for name in TEXT_SPEC_KEYS:
        writer.add_uint32(f"vibevoice.text.{name}", int(text[name]))
    for name in TEXT_SPEC_FLOAT_KEYS:
        writer.add_float32(f"vibevoice.text.{name}", float(text[name]))
    writer.add_float32("vibevoice.text.rope_theta", float(text["rope_parameters"]["rope_theta"]))


def add_llama_metadata(writer: gguf.GGUFWriter, config: dict) -> None:
    """llama.cpp KV keys so llama-quantize accepts the backbone."""
    text = config["text_config"]
    writer.add_uint32("llama.context_length", int(text["max_position_embeddings"]))
    writer.add_uint32("llama.embedding_length", int(text["hidden_size"]))
    writer.add_uint32("llama.block_count", int(text["num_hidden_layers"]))
    writer.add_uint32("llama.feed_forward_length", int(text["intermediate_size"]))
    writer.add_uint32("llama.attention.head_count", int(text["num_attention_heads"]))
    writer.add_uint32("llama.attention.head_count_kv", int(text["num_key_value_heads"]))
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", float(text["rms_norm_eps"]))
    writer.add_float32("llama.rope.freq_base", float(text["rope_parameters"]["rope_theta"]))
    writer.add_uint32("llama.vocab_size", int(text["vocab_size"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--out", required=True, help="output .gguf path")
    parser.add_argument(
        "--llama-names",
        action="store_true",
        help="rename the Qwen2 backbone to llama.cpp names with arch=llama "
        "so llama-quantize can quantize it; encoders keep vibevoice names",
    )
    parser.add_argument(
        "--skip-sections",
        default="",
        help="comma-separated gguf section prefixes to omit (ate,ste,mmp)",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "f16"),
        default="bf16",
        help="storage dtype (bf16 keeps the checkpoint byte-identical)",
    )
    args = parser.parse_args()

    snapshot = resolve_snapshot(args.model)
    config = json.load(open(snapshot / "config.json"))

    arch = "vibevoice-asr"
    name_fn = gguf_name
    if args.llama_names:
        arch = "llama"
        name_fn = llama_name
    writer = gguf.GGUFWriter(path=args.out, arch=arch)
    add_metadata(writer, config)
    if args.llama_names:
        add_llama_metadata(writer, config)

    count = 0
    total = 0
    skip = tuple(f"{s}." for s in args.skip_sections.split(",") if s)
    max_name = 0
    for name, array in iter_tensors(snapshot, args.dtype):
        gguf_name_ = name_fn(name)
        if gguf_name_.startswith(skip):
            continue
        max_name = max(max_name, len(gguf_name_))
        if args.dtype == "bf16":
            writer.add_tensor(gguf_name_, array, raw_dtype=gguf.GGMLQuantizationType.BF16)
            total += array.nbytes
        else:
            payload = array.astype(np.float16)
            writer.add_tensor(gguf_name_, payload, raw_dtype=gguf.GGMLQuantizationType.F16)
            total += payload.nbytes
        count += 1
    print(f"longest gguf tensor name: {max_name} bytes")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {count} tensors ({total / 1e9:.2f} GB payload) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
