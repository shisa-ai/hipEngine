#!/usr/bin/env python3
"""Validate DFlash target/drafter metadata without materializing tensors."""

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
    validate_dflash_drafter_metadata,
    validate_dflash_target_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DFLASH_PACKED_TARGET_MODEL)
    parser.add_argument("--drafter-model", default=DFLASH_DRAFTER_MODEL)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--raise-on-error", action="store_true")
    args = parser.parse_args()

    target_index = load_weight_index(args.target_model)
    target = validate_dflash_target_metadata(target_index, raise_on_error=args.raise_on_error)
    drafter_index = load_weight_index(args.drafter_model)
    drafter = validate_dflash_drafter_metadata(
        drafter_index,
        target_config=target.config,
        raise_on_error=args.raise_on_error,
    )
    pair_errors: list[str] = []
    if target.config.hidden_size != drafter.config.target_hidden_size:
        pair_errors.append(
            f"target hidden_size {target.config.hidden_size} != drafter target_hidden_size {drafter.config.target_hidden_size}"
        )
    if target.config.vocab_size != drafter.config.vocab_size:
        pair_errors.append(f"target vocab_size {target.config.vocab_size} != drafter vocab_size {drafter.config.vocab_size}")
    if target.config.num_hidden_layers != drafter.config.num_target_layers:
        pair_errors.append(
            f"target layers {target.config.num_hidden_layers} != drafter num_target_layers {drafter.config.num_target_layers}"
        )
    output: dict[str, Any] = {
        "schema": 1,
        "target_model": str(args.target_model),
        "drafter_model": str(args.drafter_model),
        "passed": target.passed and drafter.passed and not pair_errors,
        "target": target.to_json_dict(),
        "drafter": drafter.to_json_dict(),
        "pair_errors": pair_errors,
        "materialized_tensors": False,
    }
    if args.raise_on_error and pair_errors:
        raise ValueError("; ".join(pair_errors))
    text = json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
