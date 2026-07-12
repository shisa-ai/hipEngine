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
from typing import Any, Collection

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_mtp_draft_tensor_plans


class MissingMTPDraftLayerError(ValueError):
    """Raised when an explicit layer filter does not match a draft layer."""


class MissingMTPDraftSpecError(ValueError):
    """Raised when MTP draft call specs are required but absent."""


def summarize(
    path: str | Path,
    *,
    strict: bool = True,
    layers: Collection[int] | None = None,
    require_mtp: bool = False,
) -> dict[str, Any]:
    reader = GGUFReader(path)
    plans = build_qwen35_gguf_mtp_draft_tensor_plans(reader.info, strict=strict)
    layer_filter = None if layers is None else set(layers)
    if layer_filter is not None:
        available_layers = {plan.layer_id for plan in plans}
        missing_layers = sorted(layer_filter - available_layers)
        if missing_layers:
            requested = ", ".join(str(layer) for layer in missing_layers)
            available = ", ".join(str(layer) for layer in sorted(available_layers))
            if not available:
                available = "none"
            raise MissingMTPDraftLayerError(
                f"{reader.info.path}: requested MTP draft layer(s) {requested} "
                f"not found; available layer(s): {available}"
            )
        plans = tuple(plan for plan in plans if plan.layer_id in layer_filter)
    if require_mtp and not plans:
        raise MissingMTPDraftSpecError(
            f"{reader.info.path}: no MTP draft call specs found"
        )
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
    parser.add_argument(
        "--layer",
        dest="layers",
        action="append",
        type=int,
        help="only include the selected MTP draft layer id; repeat for multiple layers",
    )
    parser.add_argument(
        "--require-mtp",
        action="store_true",
        help="return an error if a GGUF has no MTP draft call specs after filtering",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level")
    args = parser.parse_args(argv)

    summaries: list[dict[str, Any]] = []
    try:
        for raw_path in args.paths:
            for path in discover_gguf_files(raw_path):
                summaries.append(
                    summarize(
                        path,
                        strict=not args.non_strict,
                        layers=args.layers,
                        require_mtp=args.require_mtp,
                    )
                )
    except (MissingMTPDraftLayerError, MissingMTPDraftSpecError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(summaries, sys.stdout, indent=args.indent, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
