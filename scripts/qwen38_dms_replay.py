#!/usr/bin/env python3
"""Replay an external Qwen3.8 DMS sidecar and screen no-evict/CR2/CR4/CR8."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.kvcache import load_dms_retrofit_config
from hipengine.kvcache.dms_sidecar import (
    load_external_dms_sidecar,
    screen_external_sidecar,
)

_EXPECTED_PHYSICAL_LAYERS = tuple(range(3, 64, 4))


def run(args: argparse.Namespace) -> dict:
    model = args.model.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    labels = args.labels.expanduser().resolve()
    if labels.is_dir():
        labels = labels / "label_manifest.json"
    config = load_dms_retrofit_config(
        model,
        metadata_path=metadata,
        expected_artifact_fingerprint=args.expected_artifact,
        expected_physical_layer_ids=_EXPECTED_PHYSICAL_LAYERS,
    )
    source = load_external_dms_sidecar(config)
    ratios = tuple(int(value) for value in args.compression_ratios.split(",") if value)
    result = screen_external_sidecar(
        labels,
        source,
        compression_ratios=ratios,
        output_path=args.output,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-artifact", required=True)
    parser.add_argument("--compression-ratios", default="2,4,8")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
