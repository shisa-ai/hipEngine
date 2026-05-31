#!/usr/bin/env python3
"""Build or run a llama.cpp one-token oracle command for StepFun artifacts."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Sequence

DEFAULT_LLAMA_CLI = Path("/home/lhl/ai/llama.cpp-cpu/llama-cli")
DEFAULT_MODEL = Path("/data/models/gguf/Step-3.7-flash-Q3_K_L-00001-of-00003.gguf")
DEFAULT_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--llama-cli", type=Path, default=DEFAULT_LLAMA_CLI)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--n-predict", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--llama-arg",
        action="append",
        default=None,
        help="Additional argument appended to llama-cli; repeat for each token (use --llama-arg=--flag for flags).",
    )
    parser.add_argument("--execute", action="store_true", help="Run llama-cli instead of only emitting the plan.")
    parser.add_argument(
        "--diagnostic-logs",
        action="store_true",
        help="Do not pass --log-disable, so llama.cpp load errors are captured in stderr.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON output to this path instead of stdout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args(argv)


def _llama_version(llama_cli: Path) -> str | None:
    if not llama_cli.exists():
        return None
    try:
        completed = subprocess.run(
            [str(llama_cli), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unavailable: {type(exc).__name__}: {exc}"
    return (completed.stdout + completed.stderr).strip() or None


def _emit_json(result: dict[str, object], *, pretty: bool, output: Path | None) -> None:
    text = json.dumps(result, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _comparison_fields(generated_text: str, expected_text: object) -> dict[str, object]:
    if not isinstance(expected_text, str):
        return {
            "generated_text": generated_text,
            "text_matches_expected_exact": None,
            "text_matches_expected_stripped": None,
        }
    return {
        "generated_text": generated_text,
        "text_matches_expected_exact": generated_text == expected_text,
        "text_matches_expected_stripped": generated_text.strip() == expected_text.strip(),
    }


def _blocker_fields(stderr: str) -> dict[str, object]:
    if "unknown model architecture: 'step35'" in stderr:
        return {
            "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
            "oracle_blocker_detail": "local llama.cpp build reports unknown model architecture: 'step35'",
            "step35_supported": False,
        }
    return {
        "oracle_blocker_kind": None,
        "oracle_blocker_detail": None,
        "step35_supported": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact = json.loads(args.artifact.read_text())
    prompt = artifact["prompt"]
    command = [
        str(args.llama_cli),
        "--model",
        str(args.model),
        "--prompt",
        prompt,
        "--predict",
        str(args.n_predict),
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--min-p",
        "0",
        "--repeat-penalty",
        "1",
        "--seed",
        "0",
        "--no-display-prompt",
        "--simple-io",
    ]
    if args.llama_arg:
        command.extend(args.llama_arg)
    if not args.diagnostic_logs:
        command.append("--log-disable")
    result: dict[str, object] = {
        "status": "planned",
        "artifact": str(args.artifact),
        "llama_cli": str(args.llama_cli),
        "llama_cpp_version": _llama_version(args.llama_cli),
        "model": str(args.model),
        "n_predict": args.n_predict,
        "diagnostic_logs": bool(args.diagnostic_logs),
        "extra_llama_args": list(args.llama_arg or ()),
        "command": command,
        "command_shell": shlex.join(command),
        "prompt": prompt,
        "prompt_length": artifact.get("prompt_length"),
        "expected_next_token_id": artifact.get("next_token_id"),
        "expected_next_token_text": artifact.get("next_token_text"),
        "expected_next_token_logit": artifact.get("next_token_logit"),
        "expected_top_tokens": artifact.get("top_tokens"),
        "comparison_policy": {
            "generated_text_source": "llama-cli stdout with --no-display-prompt --simple-io",
            "exact_text_match_field": "text_matches_expected_exact",
            "stripped_text_match_field": "text_matches_expected_stripped",
            "expected_text_field": "expected_next_token_text",
        },
        "note": (
            "Dry-run oracle plan by default. Use --execute only when the machine can afford a llama.cpp "
            "one-token run over the StepFun Q3_K_L GGUF shards; compare output/tokenization manually or with "
            "a follow-up parser before claiming parity."
        ),
    }
    if args.execute:
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=args.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _as_text(exc.stdout)
            stderr = _as_text(exc.stderr)
            result.update(
                {
                    "status": "timeout",
                    "timeout_s": args.timeout_s,
                    "elapsed_s": time.perf_counter() - started,
                    "stdout": stdout,
                    "stderr": stderr,
                    **_comparison_fields(stdout, result.get("expected_next_token_text")),
                    **(
                        _blocker_fields(stderr)
                        if _blocker_fields(stderr)["oracle_blocker_kind"] is not None
                        else {
                            "oracle_blocker_kind": "llama_cpp_oracle_timeout",
                            "oracle_blocker_detail": "llama.cpp oracle timed out before producing a comparable token",
                            "step35_supported": None,
                        }
                    ),
                }
            )
        else:
            result.update(
                {
                    "status": "executed",
                    "returncode": completed.returncode,
                    "elapsed_s": time.perf_counter() - started,
                    "stdout": _as_text(completed.stdout),
                    "stderr": _as_text(completed.stderr),
                    **_comparison_fields(_as_text(completed.stdout), result.get("expected_next_token_text")),
                    **_blocker_fields(_as_text(completed.stderr)),
                }
            )
    _emit_json(result, pretty=args.pretty, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
