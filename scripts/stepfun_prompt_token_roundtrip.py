#!/usr/bin/env python3
"""Emit a StepFun host prompt/tokenizer round-trip artifact.

The round-trip check tokenizes the retained host-composed prompt with the same
llama.cpp ``llama-tokenize --no-bos`` mode used by the oracle diagnostics and
compares it with the prompt-smoke ``input_ids``. This narrows oracle blocker
evidence only: it can rule out prompt-token drift, but it does not prove oracle
parity, KV readiness, e2e readiness, or performance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod
from scripts import stepfun_oracle_artifact_check as oracle_check
from scripts import stepfun_oracle_rank_check as rank_check
from scripts import stepfun_oracle_token_mismatch as token_mismatch

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-prompt-token-roundtrip.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        default=status_mod.DEFAULT_PROMPT_ARTIFACT,
        help="Canonical StepFun prompt/logit artifact with host prompt and input_ids.",
    )
    parser.add_argument(
        "--llama-tokenize",
        type=Path,
        default=token_mismatch.DEFAULT_LLAMA_TOKENIZE,
        help="llama.cpp llama-tokenize binary used for prompt diagnostics.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=token_mismatch.DEFAULT_TOKENIZER_MODEL,
        help="GGUF model to pass to llama-tokenize.",
    )
    parser.add_argument(
        "--tokenizer-timeout-s",
        type=float,
        default=60.0,
        help="Prompt tokenization timeout in seconds.",
    )
    parser.add_argument(
        "--artifact-date",
        default=DEFAULT_ARTIFACT_DATE,
        help="Date string to record in the artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write JSON output atomically to this path instead of stdout. Use "
            "--default-output for the canonical StepFun artifact path."
        ),
    )
    parser.add_argument(
        "--default-output",
        action="store_true",
        help=f"Write to the canonical artifact path: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--status-only", action="store_true", help="Emit only status.")
    parser.add_argument(
        "--roundtrip-only",
        action="store_true",
        help="Emit only whether llama.cpp prompt tokenization matches host input_ids.",
    )
    parser.add_argument(
        "--first-mismatch-only",
        action="store_true",
        help="Emit only the first token mismatch record, or null.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the artifact payload.",
    )
    return parser.parse_args(argv)


def _as_int_list(value: object) -> list[int] | None:
    if not isinstance(value, list):
        return None
    output: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        output.append(item)
    return output


def _first_mismatch(host_input_ids: list[int], llama_token_ids: list[int]) -> dict[str, object] | None:
    shared_len = min(len(host_input_ids), len(llama_token_ids))
    for index in range(shared_len):
        host_id = host_input_ids[index]
        llama_id = llama_token_ids[index]
        if host_id != llama_id:
            return {
                "index": index,
                "host_token_id": host_id,
                "llama_token_id": llama_id,
                "kind": "token_id_mismatch",
            }
    if len(host_input_ids) != len(llama_token_ids):
        return {
            "index": shared_len,
            "host_token_id": host_input_ids[shared_len] if shared_len < len(host_input_ids) else None,
            "llama_token_id": llama_token_ids[shared_len] if shared_len < len(llama_token_ids) else None,
            "kind": "token_count_mismatch",
        }
    return None


def build_prompt_token_roundtrip(
    *,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    llama_tokenize: Path = token_mismatch.DEFAULT_LLAMA_TOKENIZE,
    tokenizer_model: Path = token_mismatch.DEFAULT_TOKENIZER_MODEL,
    tokenizer_timeout_s: float = 60.0,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the host prompt text/input-id round-trip report."""

    prompt_payload = rank_check._load_json_object(prompt_artifact)
    prompt_text = prompt_payload.get("prompt")
    host_input_ids = _as_int_list(prompt_payload.get("input_ids"))
    missing_evidence: list[str] = []
    tokenization: dict[str, object] | None = None
    llama_token_ids: list[int] | None = None
    if not isinstance(prompt_text, str):
        missing_evidence.append("host_prompt_text_present")
    if host_input_ids is None:
        missing_evidence.append("host_input_ids_present")
    if isinstance(prompt_text, str):
        tokenization = oracle_check._tokenize_text_with_llamacpp(
            text=prompt_text,
            label="host-prompt",
            llama_tokenize=llama_tokenize,
            model=tokenizer_model,
            timeout_s=tokenizer_timeout_s,
        )
        token_ids = tokenization.get("token_ids")
        if isinstance(token_ids, list) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in token_ids
        ):
            llama_token_ids = list(token_ids)
        else:
            missing_evidence.append("llama_prompt_token_ids_present")
        if tokenization.get("status") != "passed":
            missing_evidence.append("llama_prompt_tokenization_passed")
    first_mismatch = None
    roundtrip_matches = False
    if host_input_ids is not None and llama_token_ids is not None:
        first_mismatch = _first_mismatch(host_input_ids, llama_token_ids)
        roundtrip_matches = first_mismatch is None
        if not roundtrip_matches:
            missing_evidence.append("host_prompt_input_ids_match_llama_tokenize")
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_prompt_token_roundtrip",
        "date": artifact_date,
        "status": "passed" if roundtrip_matches else "failed",
        "ready": roundtrip_matches,
        "prompt_artifact": str(prompt_artifact),
        "prompt_artifact_sha256": status_mod._stable_json_sha256(prompt_payload),
        "llama_tokenize": str(llama_tokenize),
        "tokenizer_model": str(tokenizer_model),
        "mode": "llama-tokenize --no-bos",
        "prompt_text_len": len(prompt_text) if isinstance(prompt_text, str) else None,
        "host_prompt_length": prompt_payload.get("prompt_length"),
        "host_input_id_count": len(host_input_ids) if host_input_ids is not None else None,
        "host_input_ids": host_input_ids,
        "llama_token_count": len(llama_token_ids) if llama_token_ids is not None else None,
        "llama_token_ids": llama_token_ids,
        "tokenization_record": tokenization,
        "prompt_tokenization_matches_host_input_ids": roundtrip_matches,
        "first_mismatch": first_mismatch,
        "missing_evidence": missing_evidence,
        "conclusion": (
            "host prompt input_ids match llama.cpp no-BOS prompt tokenization"
            if roundtrip_matches
            else "host prompt input_ids do not match llama.cpp no-BOS prompt tokenization"
        ),
        "oracle_blocker_interpretation": (
            "prompt tokenization is tokenizer-coherent; the retained llama.cpp "
            "oracle mismatch should be investigated as logits/backend parity, not "
            "prompt-token drift"
            if roundtrip_matches
            else "prompt tokenization coherence is not fully established"
        ),
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This artifact only checks host prompt input_ids against llama-tokenize; "
                "it does not compare generated text to the target."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_prompt_token_roundtrip(
        prompt_artifact=args.prompt_artifact,
        llama_tokenize=args.llama_tokenize,
        tokenizer_model=args.tokenizer_model,
        tokenizer_timeout_s=args.tokenizer_timeout_s,
        artifact_date=args.artifact_date,
    )
    if args.status_only:
        payload: object = report["status"]
    elif args.roundtrip_only:
        payload = report["prompt_tokenization_matches_host_input_ids"]
    elif args.first_mismatch_only:
        payload = report["first_mismatch"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
