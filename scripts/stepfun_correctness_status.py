#!/usr/bin/env python3
"""Summarize StepFun Q3_K_L correctness artifacts and remaining blockers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

DEFAULT_PROMPT_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json"
)
DEFAULT_ORACLE_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-llamacpp-oracle-exec-attempt.json"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-artifact", type=Path, default=DEFAULT_PROMPT_ARTIFACT)
    parser.add_argument("--oracle-artifact", type=Path, default=DEFAULT_ORACLE_ARTIFACT)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON output to this path instead of stdout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def build_status(prompt_artifact: Path, oracle_artifact: Path) -> dict[str, object]:
    prompt = _load(prompt_artifact)
    oracle = _load(oracle_artifact)
    all_layer_prompt_smoke = (
        prompt.get("status") == "partial_prompt_smoke"
        and prompt.get("execution_mode") == "chunked"
        and prompt.get("layer_count") == 45
        and prompt.get("skipped_layers") == []
        and prompt.get("no_vision_projector_mtp_slots") is True
        and prompt.get("memory_stats_after_free", {}).get("active_allocations") == 0
        and prompt.get("memory_stats_after_free", {}).get("current_allocated_bytes") == 0
    )
    oracle_parity = (
        oracle.get("status") == "executed"
        and oracle.get("returncode") == 0
        and oracle.get("text_matches_expected_exact") is True
    )
    blockers: list[dict[str, object]] = []
    if not oracle_parity:
        blockers.append(
            {
                "kind": "oracle_parity_blocked",
                "detail": oracle.get("oracle_blocker_detail")
                or "llama.cpp/CPU oracle result has not matched the StepFun artifact yet",
                "artifact": str(oracle_artifact),
                "oracle_blocker_kind": oracle.get("oracle_blocker_kind"),
            }
        )
    blockers.append(
        {
            "kind": "kv_backed_decode_not_wired",
            "detail": (
                "Current all-layer prompt smoke is host-composed prefill/logits; "
                "final KV-backed one-token decode runner remains open."
            ),
            "artifact": str(prompt_artifact),
        }
    )
    return {
        "status": "blocked" if blockers else "ready",
        "model": "Step-3.7-flash-Q3_K_L",
        "backend": prompt.get("backend", "hip_gfx1151"),
        "prompt_artifact": str(prompt_artifact),
        "oracle_artifact": str(oracle_artifact),
        "all_layer_prompt_smoke": all_layer_prompt_smoke,
        "all_layer_prompt_next_token_id": prompt.get("next_token_id"),
        "all_layer_prompt_next_token_text": prompt.get("next_token_text"),
        "all_layer_prompt_peak_resident_weight_nbytes": prompt.get("peak_resident_weight_nbytes"),
        "oracle_parity": oracle_parity,
        "oracle_blocker_kind": oracle.get("oracle_blocker_kind"),
        "step35_supported_by_local_llama_cpp": oracle.get("step35_supported"),
        "kv_backed_decode_ready": False,
        "e2e_inference_ready": False,
        "blockers": blockers,
        "note": (
            "Host-composed all-layer prompt smoke is present; true e2e inference still needs "
            "oracle parity and KV-backed decode."
        ),
    }


def _emit_json(result: dict[str, object], *, pretty: bool, output: Path | None) -> None:
    text = json.dumps(result, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _emit_json(
        build_status(args.prompt_artifact, args.oracle_artifact),
        pretty=args.pretty,
        output=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
