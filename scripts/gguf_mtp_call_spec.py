#!/usr/bin/env python3
"""Dump validated Qwen35 GGUF MTP CPU-reference call specs as JSON.

This script reads GGUF metadata/tensor headers only. It does not materialize
weights, run kernels, or touch runtime KV state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_mtp_draft_tensor_plans


def summarize(path: str | Path, *, strict: bool = True) -> dict[str, Any]:
    reader = GGUFReader(path)
    plans = build_qwen35_gguf_mtp_draft_tensor_plans(reader.info, strict=strict)
    return {
        "path": str(reader.info.path),
        "architecture": reader.info.architecture,
        "tensor_count": reader.info.tensor_count,
        "mtp_draft_call_specs": [plan.cpu_reference_call_spec.as_dict() for plan in plans],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="GGUF file(s) or directories containing .gguf files",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="allow unknown extra MTP tensors while still requiring known slots",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level")
    args = parser.parse_args(argv)

    summaries: list[dict[str, Any]] = []
    for raw_path in args.paths:
        for path in discover_gguf_files(raw_path):
            summaries.append(summarize(path, strict=not args.non_strict))
    json.dump(summaries, sys.stdout, indent=args.indent, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
