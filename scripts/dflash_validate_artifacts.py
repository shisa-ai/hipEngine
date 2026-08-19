#!/usr/bin/env python3
"""Validate DFlash target/drafter metadata without materializing tensors.

Supports two target kinds:
  * ``paro`` (default): PARO-packed HF safetensors target artifact.
  * ``gguf``: a GGUF target file (e.g. the Qwen3.8-27B resident model); the
    drafter is pair-checked against the GGUF's dense layer count, hidden size,
    and vocab size.
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

from hipengine.loading import (  # noqa: E402
    DFLASH_DRAFTER_MODEL,
    DFLASH_PACKED_TARGET_MODEL,
    load_weight_index,
    validate_dflash_drafter_against_gguf_target,
    validate_dflash_drafter_metadata,
    validate_dflash_target_metadata,
)
from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.loading.qwen35_gguf import qwen35_gguf_config_from_metadata  # noqa: E402


def _pair_errors(drafter, target) -> list[str]:
    errors: list[str] = []
    if target.hidden_size != drafter.config.target_hidden_size:
        errors.append(
            f"target hidden_size {target.hidden_size} != drafter target_hidden_size {drafter.config.target_hidden_size}"
        )
    if target.vocab_size != drafter.config.vocab_size:
        errors.append(f"target vocab_size {target.vocab_size} != drafter vocab_size {drafter.config.vocab_size}")
    if target.num_hidden_layers != drafter.config.num_target_layers:
        errors.append(
            f"target layers {target.num_hidden_layers} != drafter num_target_layers {drafter.config.num_target_layers}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DFLASH_PACKED_TARGET_MODEL)
    parser.add_argument("--drafter-model", default=DFLASH_DRAFTER_MODEL)
    parser.add_argument("--target-kind", choices=("paro", "gguf"), default="paro")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--raise-on-error", action="store_true")
    args = parser.parse_args()

    drafter_index = load_weight_index(args.drafter_model)
    if args.target_kind == "gguf":
        target = scan_gguf(args.target_model)
        gguf_cfg = qwen35_gguf_config_from_metadata(target)
        drafter = validate_dflash_drafter_metadata(drafter_index, raise_on_error=args.raise_on_error)
        gguf_errors = validate_dflash_drafter_against_gguf_target(
            drafter.config,
            num_target_layers=gguf_cfg.block_count,
            hidden_size=gguf_cfg.hidden_size,
            vocab_size=gguf_cfg.vocab_size,
        )
        if args.raise_on_error and gguf_errors:
            raise ValueError("; ".join(gguf_errors))
        target_passed = True
        target_dict: dict[str, Any] = {
            "artifact_kind": "gguf_target",
            "model_path": str(args.target_model),
            "architecture": gguf_cfg.architecture,
            "block_count": gguf_cfg.block_count,
            "hidden_size": gguf_cfg.hidden_size,
            "vocab_size": gguf_cfg.vocab_size,
        }
        pair_errors = list(gguf_errors)
    else:
        target_index = load_weight_index(args.target_model)
        target = validate_dflash_target_metadata(target_index, raise_on_error=args.raise_on_error)
        drafter = validate_dflash_drafter_metadata(
            drafter_index,
            target_config=target.config,
            raise_on_error=args.raise_on_error,
        )
        pair_errors = _pair_errors(drafter, target)
        if args.raise_on_error and pair_errors:
            raise ValueError("; ".join(pair_errors))
        target_passed = target.passed
        target_dict = target.to_json_dict()

    output: dict[str, Any] = {
        "schema": 2,
        "target_kind": args.target_kind,
        "target_model": str(args.target_model),
        "drafter_model": str(args.drafter_model),
        "passed": target_passed and drafter.passed and not pair_errors,
        "target": target_dict,
        "drafter": drafter.to_json_dict(),
        "pair_errors": pair_errors,
        "materialized_tensors": False,
    }
    text = json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
