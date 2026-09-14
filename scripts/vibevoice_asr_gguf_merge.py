#!/usr/bin/env python3
"""Merge a quantized VibeVoice backbone GGUF with BF16 encoder tensors.

llama-quantize fallback-quantizes unrecognized tensors (the encoder 2-D
FFNs would go Q4_K and the 3-D convs get silently re-rounded BF16->F16),
so the backbone is quantized alone and this script staples the untouched
BF16 encoder/projector tensors from the full BF16 export back on.

Usage:
    python3 scripts/vibevoice_asr_gguf_merge.py \
        --backbone /tmp/vibevoice-backbone-q4km.gguf \
        --full-bf16 /tmp/vibevoice-asr-bf16.gguf \
        --out /tmp/vibevoice-asr-q4km.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import gguf

from hipengine.loading.gguf import GGUFReader, scan_gguf

ENCODER_PREFIXES = ("ate.", "ste.", "mmp.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, help="quantized backbone GGUF")
    parser.add_argument("--full-bf16", required=True, help="full BF16 export GGUF")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    backbone_info = scan_gguf(args.backbone)
    full_info = scan_gguf(args.full_bf16)
    assert full_info.architecture == "vibevoice-asr", full_info.architecture

    writer = gguf.GGUFWriter(path=args.out, arch="vibevoice-asr")

    # KV: prefer the backbone's (llama.* keys are harmless leftovers), but
    # rewrite general.architecture and keep everything else from full.
    meta = dict(full_info.metadata)
    for key, value in backbone_info.metadata.items():
        if key.startswith("general.") or key.startswith("llama."):
            continue
        meta[key] = value
    for key, value in meta.items():
        if key == "general.architecture":
            writer.add_architecture()
        else:
            _add_kv(writer, key, value)

    total = 0
    for tensor in backbone_info.tensors:
        data = np.asarray(GGUFReader(args.backbone).tensor_data(tensor.name))
        writer.add_tensor(tensor.name, data,
                          raw_dtype=gguf.GGMLQuantizationType(tensor.ggml_type))
        total += data.nbytes
        print(f"backbone {tensor.name}: type {tensor.ggml_type}")
    full_reader = GGUFReader(args.full_bf16)
    for tensor in full_info.tensors:
        if not tensor.name.startswith(ENCODER_PREFIXES):
            continue
        data = np.asarray(full_reader.tensor_data(tensor.name))
        assert tensor.ggml_type == 30, (tensor.name, tensor.ggml_type)
        writer.add_tensor(tensor.name, data,
                          raw_dtype=gguf.GGMLQuantizationType(tensor.ggml_type))
        total += data.nbytes
    print(f"merged {backbone_info.tensor_count} backbone + "
          f"{sum(1 for t in full_info.tensors if t.name.startswith(ENCODER_PREFIXES))} "
          f"encoder tensors ({total / 1e9:.2f} GB payload)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return 0


def _add_kv(writer, key, value):
    import gguf as _g

    if isinstance(value, bool):
        writer.add_bool(key, value)
    elif isinstance(value, int):
        if abs(value) < 2**31:
            writer.add_int32(key, value)
        else:
            writer.add_uint64(key, value)
    elif isinstance(value, float):
        writer.add_float32(key, value)
    elif isinstance(value, str):
        writer.add_string(key, value)
    elif isinstance(value, (list, tuple)):
        writer.add_array(key, list(value))
    else:
        raise SystemExit(f"cannot round-trip KV {key} of type {type(value)}")


if __name__ == "__main__":
    sys.exit(main())
