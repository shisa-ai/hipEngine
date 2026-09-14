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
import hashlib
import sys
from pathlib import Path

import numpy as np

from hipengine.loading.gguf import GGUFReader, scan_gguf

ENCODER_PREFIXES = ("ate.", "ste.", "mmp.")

# Backbone tensors the runner reads by name; a merge that silently drops
# one produces a model that loads and then decodes nonsense.
REQUIRED_GLOBAL_BACKBONE = ("token_embd.weight", "output_norm.weight", "output.weight")
REQUIRED_LAYER_PARTS = ("attn_q", "attn_k", "attn_v", "attn_output",
                       "ffn_gate", "ffn_up", "ffn_down")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(backbone: str, full_bf16: str, backbone_info, full_info) -> None:
    """Reject incomplete, mismatched, or non-BF16 merge inputs."""
    # The quantized backbone comes out of llama-quantize, which stamps
    # general.architecture=llama; the BF16 export carries vibevoice-asr.
    # That pair is the intended pipeline input, so the backbone is allowed
    # to be either. The output architecture is taken from the BF16 export.
    if full_info.architecture != "vibevoice-asr":
        raise SystemExit(f"full-bf16 architecture must be 'vibevoice-asr', "
                         f"got {full_info.architecture!r}")
    if backbone_info.architecture not in ("vibevoice-asr", "llama"):
        raise SystemExit(f"unexpected backbone architecture "
                         f"{backbone_info.architecture!r} (expected llama or vibevoice-asr)")
    full_names = {t.name for t in full_info.tensors}
    backbone_names = {t.name for t in backbone_info.tensors}
    overlapping = backbone_names & full_names
    if overlapping:
        raise SystemExit(f"inputs share {len(overlapping)} tensor name(s), e.g. "
                         f"{sorted(overlapping)[:3]}")
    # Every encoder tensor in the output must come from the BF16 export as BF16.
    non_bf16 = [t.name for t in full_info.tensors
                if t.name.startswith(ENCODER_PREFIXES) and t.ggml_type != 30]
    if non_bf16:
        raise SystemExit(f"{len(non_bf16)} encoder tensor(s) are not BF16, e.g. {non_bf16[:3]}")
    encoder_in_full = {t.name for t in full_info.tensors if t.name.startswith(ENCODER_PREFIXES)}
    if not encoder_in_full:
        raise SystemExit("full-bf16 input contains no encoder tensors (wrong file?)")
    # Completeness: the runner requires the full decoder tensor set.
    import re

    missing_global = [n for n in REQUIRED_GLOBAL_BACKBONE if n not in backbone_names]
    layer_ids = sorted({int(m.group(1)) for n in backbone_names
                        if (m := re.match(r"blk\.(\d+)\.", n))})
    if not layer_ids:
        raise SystemExit("backbone contains no blk.<i>. tensors (wrong file?)")
    missing_layers = [
        f"blk.{i}.{part}.weight"
        for i in layer_ids
        for part in REQUIRED_LAYER_PARTS
        if f"blk.{i}.{part}.weight" not in backbone_names
    ]
    if missing_global or missing_layers:
        raise SystemExit(f"backbone is incomplete: missing globals {missing_global[:3]}"
                         f", missing {len(missing_layers)} layer tensor(s) {missing_layers[:3]}")
    print(f"inputs ok: backbone {len(backbone_names)} tensors "
          f"(sha256 {file_sha256(backbone)[:16]}), "
          f"full-bf16 {len(full_names)} tensors (sha256 {file_sha256(full_bf16)[:16]}), "
          f"{len(encoder_in_full)} encoder tensors to staple")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, help="quantized backbone GGUF")
    parser.add_argument("--full-bf16", required=True, help="full BF16 export GGUF")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    backbone_info = scan_gguf(args.backbone)
    full_info = scan_gguf(args.full_bf16)
    assert full_info.architecture == "vibevoice-asr", full_info.architecture
    validate_inputs(args.backbone, args.full_bf16, backbone_info, full_info)

    import gguf

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
