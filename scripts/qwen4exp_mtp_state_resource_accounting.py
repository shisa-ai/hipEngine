#!/usr/bin/env python3
"""Account Qwen4Exp MTP state-row verifier workspace by candidate budget."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.loading.qwen4_exp_gguf import qwen4_exp_gguf_config_from_metadata


def account(model_root: Path, budgets: list[int]) -> dict:
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    config = qwen4_exp_gguf_config_from_metadata(index)
    gdn_layers = config.layer_types.count("gdn")
    matrix_per_row = (
        config.gdn_time_step_rank * config.gdn_state_size * config.gdn_state_size * 4
    )
    conv_per_row = (
        (2 * config.gdn_group_count * config.gdn_state_size + config.gdn_inner_size)
        * config.gdn_conv_kernel * 4
    )
    rows = []
    for budget in budgets:
        if budget < 1:
            raise ValueError("budget must be positive")
        verifier_rows = budget + 1
        components = {
            "gdn_matrix_rows": gdn_layers * verifier_rows * matrix_per_row,
            "gdn_conv_rows": gdn_layers * verifier_rows * conv_per_row,
            "ple_out": verifier_rows * config.residual_width * 4,
            "argmax_ids": verifier_rows * 8,
        }
        rows.append({
            "candidate_budget": budget,
            "verifier_rows": verifier_rows,
            "components_bytes": components,
            "total_bytes": sum(components.values()),
            "total_mib": sum(components.values()) / (1024 ** 2),
        })
    return {
        "schema": 1,
        "kind": "qwen4exp_mtp_state_row_resource_accounting",
        "status": "diagnostic_complete",
        "performance_claim": False,
        "model_root": str(model_root),
        "model_architecture": config.architecture,
        "geometry": {
            "block_count": config.block_count,
            "gdn_layers": gdn_layers,
            "gdn_time_step_rank": config.gdn_time_step_rank,
            "gdn_state_size": config.gdn_state_size,
            "gdn_group_count": config.gdn_group_count,
            "gdn_inner_size": config.gdn_inner_size,
            "gdn_conv_kernel": config.gdn_conv_kernel,
            "residual_width": config.residual_width,
            "bytes_per_fp32": 4,
            "bytes_per_int64": 8,
        },
        "accounting_contract": "Qwen4ExpGGUFResidentModelRunner._verify_rows_workspace",
        "rows": rows,
        "conclusion": (
            "State-row verification has a linear per-row workspace charge. This is "
            "the real verifier-state cost; the current provider still uses cursor "
            "trim for draft rollback and does not snapshot this target state."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--budgets", default="1,2,3,5,7")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budgets = [int(item) for item in args.budgets.split(",") if item]
    payload = account(args.model_root, budgets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
