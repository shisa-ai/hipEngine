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
TEXT_SPEC_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "rms_norm_eps",
)


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
    writer.add_float32("vibevoice.text.rope_theta", float(text["rope_parameters"]["rope_theta"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR-HF")
    parser.add_argument("--out", required=True, help="output .gguf path")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "f16"),
        default="bf16",
        help="storage dtype (bf16 keeps the checkpoint byte-identical)",
    )
    args = parser.parse_args()

    snapshot = resolve_snapshot(args.model)
    config = json.load(open(snapshot / "config.json"))

    writer = gguf.GGUFWriter(path=args.out, arch="vibevoice-asr")
    add_metadata(writer, config)

    count = 0
    total = 0
    for name, array in iter_tensors(snapshot, args.dtype):
        if not name.startswith(ENCODED_PREFIXES):
            raise SystemExit(f"unexpected tensor outside known sections: {name}")
        if args.dtype == "bf16":
            writer.add_tensor(name, array, raw_dtype=gguf.GGMLQuantizationType.BF16)
            total += array.nbytes
        else:
            payload = array.astype(np.float16)
            writer.add_tensor(name, payload, raw_dtype=gguf.GGMLQuantizationType.F16)
            total += payload.nbytes
        count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {count} tensors ({total / 1e9:.2f} GB payload) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
