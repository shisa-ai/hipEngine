#!/usr/bin/env python3
"""hipEngine GGUF MTP B1 prompt-suite child harness.

The native GGUF MTP execution path is still under construction.  This harness is
therefore a correctness-first preflight child: it verifies the model exposes a
validated MTP block, verifies prompt-token and sampling parity inputs, and emits a
compact blocked artifact instead of silently comparing accepted/output metrics
before the runtime exists.
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

from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.loading.qwen35_gguf import validate_qwen35_gguf_mtp_blocks  # noqa: E402
from scripts.gguf_mtp_parity_precheck import (  # noqa: E402
    build_parity_precheck,
    load_json,
    load_sampling_settings,
)
from scripts.gguf_prompt_token_inventory import load_prompt_suite  # noqa: E402


DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PROMPTS = Path("benchmarks/fixtures/llamacpp_mtp_bench_prompts.json")
DEFAULT_HIPENGINE_TOKENS = Path(
    "benchmarks/fixtures/hipengine_gguf_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json"
)
DEFAULT_LLAMACPP_TOKENS = Path(
    "benchmarks/fixtures/llamacpp_hip_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json"
)
DEFAULT_SAMPLING = Path("benchmarks/fixtures/gguf_mtp_b1_sampling_greedy_seed12345.json")


class B1PromptSuitePreflightError(RuntimeError):
    """Raised when the B1 harness cannot build a preflight artifact."""


def build_b1_prompt_suite_artifact(
    *,
    model: Path,
    prompts_file: Path,
    hipengine_token_inventory: Path,
    llamacpp_token_inventory: Path,
    hipengine_sampling: Path,
    llamacpp_sampling: Path,
    prompt_limit: int | None = None,
) -> dict[str, Any]:
    prompts_payload = load_prompt_suite(prompts_file)
    prompts = list(prompts_payload.get("prompts", []))
    if prompt_limit is not None:
        prompts = prompts[: int(prompt_limit)]
    if not prompts:
        raise B1PromptSuitePreflightError("prompt suite is empty after filtering")

    model_info = scan_gguf(model)
    mtp_blocks = validate_qwen35_gguf_mtp_blocks(model_info)
    if not mtp_blocks:
        raise B1PromptSuitePreflightError(f"{model}: no validated MTP blocks found")

    parity = build_parity_precheck(
        hipengine_token_inventory=load_json(hipengine_token_inventory),
        llamacpp_token_inventory=load_json(llamacpp_token_inventory),
        hipengine_sampling=load_sampling_settings(hipengine_sampling),
        llamacpp_sampling=load_sampling_settings(llamacpp_sampling),
        require_sampling=True,
    )
    blockers: list[dict[str, Any]] = []
    if not parity["all_pass"]:
        blockers.append(
            {
                "code": "parity_precheck_failed",
                "detail": "token-id and sampling parity must pass before B1 accepted/output metrics are comparable",
                "token_match": bool(parity["token_ids"]["all_match"]),
                "sampling_match": bool(parity["sampling"]["passed"]),
            }
        )
    if parity["all_pass"]:
        blockers.append(
            {
                "code": "native_gguf_mtp_runtime_missing",
                "detail": (
                    "Native GGUF MTP draft execution is not implemented yet; this harness "
                    "stops after metadata/token/sampling preflight instead of reporting metrics."
                ),
            }
        )

    return {
        "schema": 1,
        "kind": "hipengine_gguf_mtp_b1_prompt_suite",
        "mode": "preflight",
        "status": "blocked" if blockers else "ready",
        "model": str(model),
        "model_metadata": {
            "architecture": model_info.architecture,
            "file_type_name": model_info.file_type_name,
            "tensor_count": model_info.tensor_count,
        },
        "prompts_file": str(prompts_file),
        "prompt_count": len(prompts),
        "prompt_names": [str(prompt.get("name", index)) for index, prompt in enumerate(prompts)],
        "budget": "B1",
        "draft_max": 1,
        "validated_mtp_blocks": [
            {
                "layer_id": int(block.layer_id),
                "tensor_count": len(block.tensor_names),
                "nextn_tensor_count": len(block.nextn_tensor_names),
                "optional_fallback_tensor_names": dict(block.optional_fallback_tensor_names),
            }
            for block in mtp_blocks
        ],
        "parity_precheck": parity,
        "execution": {
            "implemented": False,
            "exactness_gate": "not_run",
            "accepted_output_metrics": "not_run",
            "next_action": "implement native GGUF MTP draft execution and re-run this harness",
        },
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--hipengine-token-inventory", type=Path, default=DEFAULT_HIPENGINE_TOKENS)
    parser.add_argument("--llamacpp-token-inventory", type=Path, default=DEFAULT_LLAMACPP_TOKENS)
    parser.add_argument("--hipengine-sampling", type=Path, default=DEFAULT_SAMPLING)
    parser.add_argument("--llamacpp-sampling", type=Path, default=DEFAULT_SAMPLING)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--out", type=Path, help="write JSON artifact to this path")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="return exit code 2 when the artifact status is blocked",
    )
    args = parser.parse_args(argv)

    artifact = build_b1_prompt_suite_artifact(
        model=args.model,
        prompts_file=args.prompts_file,
        hipengine_token_inventory=args.hipengine_token_inventory,
        llamacpp_token_inventory=args.llamacpp_token_inventory,
        hipengine_sampling=args.hipengine_sampling,
        llamacpp_sampling=args.llamacpp_sampling,
        prompt_limit=args.prompt_limit,
    )
    payload = json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.fail_on_blocked and artifact["status"] == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
