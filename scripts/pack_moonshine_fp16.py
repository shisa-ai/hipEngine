#!/usr/bin/env python3
"""Pack the Moonshine snapshot into a deployable FP16 model artifact.

The source Hugging Face snapshot stores F32 weights (``model.safetensors``,
252,895,696 bytes for shisa-realtime-asr-0.92b).  The resident runtime needs
FP16 weights (126,435,712 bytes = 63,217,856 parameters x 2).  This packer
performs the one allowed load-time F32 -> FP16 conversion once, writes a
single packed ``model.safetensors`` plus the tokenizer/config sidecars, and
records a ``pack_manifest.json`` with the SHA-256 of the packed weights and the
byte counts.  The resulting directory is the deployable model artifact that the
C5 benchmark routes load instead of the F32 source snapshot.

The packed file is byte-stable: the same snapshot always produces the same
packed bytes, so the manifest hash is a reproducible deployment identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hipengine.loading.moonshine import (
    convert_moonshine_weight_to_fp16,
    read_generation_config,
)
from hipengine.loading.safetensors import load_weight_index
from hipengine.models.moonshine import (
    expected_moonshine_weight_shapes,
    parse_moonshine_model_spec,
)

_SIDECAR_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_packed_safetensors(
    snapshot_dir: Path,
    *,
    shard: Path,
    output_path: Path,
) -> None:
    """Convert every F32 tensor to FP16 and write a single packed safetensors.

    Uses ``safe_open`` with ``framework='numpy'`` so the exact
    ``convert_moonshine_weight_to_fp16`` conversion path (finite checks,
    contiguous FP16) is reused, and ``safetensors.numpy.save_file`` so no torch
    import is required.
    """

    from safetensors import safe_open
    from safetensors.numpy import save_file

    index = load_weight_index(snapshot_dir)
    config = json.loads((snapshot_dir / "config.json").read_text())
    generation = read_generation_config(snapshot_dir)
    spec = parse_moonshine_model_spec(config, generation)
    expected = expected_moonshine_weight_shapes(spec)

    packed: dict[str, np.ndarray] = {}
    with safe_open(str(shard), framework="numpy") as handle:
        for name in sorted(expected):
            source = handle.get_tensor(name)
            if source.shape != expected[name]:
                raise ValueError(
                    f"Moonshine weight {name} shape={source.shape}, expected {expected[name]}"
                )
            packed[name] = convert_moonshine_weight_to_fp16(name, source)
    save_file(packed, str(output_path))


def sidecar_bytes(snapshot_dir: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in _SIDECAR_FILES:
        path = snapshot_dir / name
        if path.is_file():
            result[name] = path.stat().st_size
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--shard", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.snapshot_dir.resolve()
    shard = (args.shard or snapshot / "model.safetensors").resolve()
    if not shard.is_file():
        parser.error(f"model shard not found: {shard}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    packed_path = output / "model.safetensors"
    if packed_path.exists():
        packed_path.unlink()
    build_packed_safetensors(snapshot, shard=shard, output_path=packed_path)

    weights_sha256 = sha256_bytes(packed_path.read_bytes())
    weights_bytes = packed_path.stat().st_size

    config = json.loads((snapshot / "config.json").read_text())
    generation = read_generation_config(snapshot)
    spec = parse_moonshine_model_spec(config, generation)
    parameter_count = spec.parameter_count

    sidecars = sidecar_bytes(snapshot)
    for name in _SIDECAR_FILES:
        src = snapshot / name
        if src.is_file():
            dst = output / name
            dst.write_bytes(src.read_bytes())

    manifest = {
        "artifact": "moonshine_packed_fp16",
        "model": "shisa-ai/shisa-realtime-asr-0.92b",
        "snapshot": str(snapshot),
        "shard": str(shard),
        "packed": {
            "model.safetensors_sha256": weights_sha256,
            "model.safetensors_bytes": weights_bytes,
            "fp16_payload_bytes": parameter_count * 2,
            "parameter_count": parameter_count,
            "sidecar_bytes": sidecars,
            "sidecar_total_bytes": sum(sidecars.values()),
        },
        "packed_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "pack_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
