#!/usr/bin/env python3
"""Gate MTP-GGUF cross-engine parity preconditions before metrics compare.

This helper is intentionally mechanical: it compares exact prompt token IDs and,
when supplied/required, exact sampling settings.  It does not run generation,
load model weights, or compute accepted/output metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_prompt_token_inventories import compare_prompt_token_inventories, load_json  # noqa: E402


JsonObject = dict[str, Any]


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_sampling_settings(path: Path) -> JsonObject:
    payload = load_json(path)
    settings = payload.get("sampling") if isinstance(payload.get("sampling"), dict) else payload
    if not isinstance(settings, dict) or not settings:
        raise ValueError(f"{path} did not contain non-empty sampling settings")
    return settings


def compare_sampling_settings(
    hipengine: JsonObject | None,
    llamacpp: JsonObject | None,
    *,
    require_sampling: bool = False,
) -> JsonObject:
    if hipengine is None and llamacpp is None:
        return {
            "checked": False,
            "passed": not require_sampling,
            "reason": "sampling settings were not provided",
            "mismatches": [],
        }
    if hipengine is None or llamacpp is None:
        return {
            "checked": True,
            "passed": False,
            "reason": "sampling settings must be provided for both engines",
            "mismatches": [
                {
                    "path": "<root>",
                    "hipengine": "missing" if hipengine is None else "present",
                    "llamacpp": "missing" if llamacpp is None else "present",
                }
            ],
        }

    mismatches = _json_mismatches(hipengine, llamacpp)
    return {
        "checked": True,
        "passed": not mismatches,
        "hipengine_sampling_sha256": stable_json_sha256(hipengine),
        "llamacpp_sampling_sha256": stable_json_sha256(llamacpp),
        "mismatches": mismatches,
    }


def build_parity_precheck(
    *,
    hipengine_token_inventory: JsonObject,
    llamacpp_token_inventory: JsonObject,
    hipengine_sampling: JsonObject | None = None,
    llamacpp_sampling: JsonObject | None = None,
    context_tokens: int = 8,
    require_sampling: bool = False,
) -> JsonObject:
    token_comparison = compare_prompt_token_inventories(
        hipengine_token_inventory,
        llamacpp_token_inventory,
        left_label="hipengine",
        right_label="llamacpp",
        context_tokens=context_tokens,
    )
    sampling_comparison = compare_sampling_settings(
        hipengine_sampling,
        llamacpp_sampling,
        require_sampling=require_sampling,
    )
    all_pass = bool(token_comparison["all_match"]) and bool(sampling_comparison["passed"])
    return {
        "schema": 1,
        "kind": "gguf_mtp_parity_precheck",
        "all_pass": all_pass,
        "token_ids": token_comparison,
        "sampling": sampling_comparison,
        "warning": (
            "This gate only checks parity preconditions. Do not compare MTP accepted/output "
            "metrics unless all_pass is true and the numeric KL/top-1 gate also passes."
        ),
    }


def _json_mismatches(left: Any, right: Any, *, path: str = "<root>") -> list[JsonObject]:
    if isinstance(left, dict) and isinstance(right, dict):
        mismatches: list[JsonObject] = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}" if path != "<root>" else str(key)
            if key not in left:
                mismatches.append({"path": child_path, "hipengine": None, "llamacpp": right[key]})
            elif key not in right:
                mismatches.append({"path": child_path, "hipengine": left[key], "llamacpp": None})
            else:
                mismatches.extend(_json_mismatches(left[key], right[key], path=child_path))
        return mismatches
    if isinstance(left, list) and isinstance(right, list):
        mismatches = []
        for index in range(max(len(left), len(right))):
            child_path = f"{path}[{index}]"
            if index >= len(left):
                mismatches.append({"path": child_path, "hipengine": None, "llamacpp": right[index]})
            elif index >= len(right):
                mismatches.append({"path": child_path, "hipengine": left[index], "llamacpp": None})
            else:
                mismatches.extend(_json_mismatches(left[index], right[index], path=child_path))
        return mismatches
    if left != right:
        return [{"path": path, "hipengine": left, "llamacpp": right}]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipengine-token-inventory", required=True, type=Path)
    parser.add_argument("--llamacpp-token-inventory", required=True, type=Path)
    parser.add_argument("--hipengine-sampling", type=Path, help="JSON object or artifact containing a 'sampling' object")
    parser.add_argument("--llamacpp-sampling", type=Path, help="JSON object or artifact containing a 'sampling' object")
    parser.add_argument("--context-tokens", type=int, default=8)
    parser.add_argument("--require-sampling", action="store_true")
    parser.add_argument("--out", type=Path, help="write precheck JSON to this path")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    result = build_parity_precheck(
        hipengine_token_inventory=load_json(args.hipengine_token_inventory),
        llamacpp_token_inventory=load_json(args.llamacpp_token_inventory),
        hipengine_sampling=load_sampling_settings(args.hipengine_sampling) if args.hipengine_sampling else None,
        llamacpp_sampling=load_sampling_settings(args.llamacpp_sampling) if args.llamacpp_sampling else None,
        context_tokens=args.context_tokens,
        require_sampling=bool(args.require_sampling),
    )
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.fail_on_mismatch and not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
