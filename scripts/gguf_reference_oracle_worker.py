#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate exact GGUF serving oracles in an isolated GPU process."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hipengine import LLM, SamplingParams
from scripts.gguf_live_server_bench import _read_compiler_version, _run_reference
from scripts.gguf_production_load_gate import WorkloadRequest, _prompt_manifest


def run(args: argparse.Namespace) -> None:
    specs_payload = json.loads(args.specs_json.read_text(encoding="utf-8"))
    if not isinstance(specs_payload, list) or not specs_payload:
        raise ValueError("oracle specs must be a non-empty list")
    specs = tuple(WorkloadRequest(**row) for row in specs_payload)
    compiler_version = _read_compiler_version(args.compiler_version_file)
    if args.require_cached_build and compiler_version is None:
        raise ValueError("require-cached-build requires compiler-version-file")

    llm = LLM(
        args.model,
        backend=str(args.backend),
        max_active_requests=int(args.max_active_requests),
    )
    adapter = llm._get_text_generator()
    llm.prepare(
        max_sequence_length=int(args.max_sequence_length),
        sampling_params=SamplingParams(max_tokens=int(args.max_output_tokens)),
    )
    runner = adapter._runner
    prompt_rows = _prompt_manifest(runner.generator.tokenizer, specs)

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    serializable_rows = {
        key: {
            **row,
            "token_ids": list(row["token_ids"]),
        }
        for key, row in prompt_rows.items()
    }
    reference_tokens: dict[str, list[int]] = {}
    ordered_rows = tuple(sorted(prompt_rows.items()))
    for index, (key, row) in enumerate(ordered_rows, start=1):
        session = Qwen35GGUFResidentSession(
            args.model,
            backend=str(args.backend),
            runtime=runner._shared_runner.runtime,
            shared_runner=runner._shared_runner,
            max_sequence_length=int(args.max_sequence_length),
            use_wmma_prefill=True,
            use_gemv_decode=True,
            compiler_version=compiler_version,
            require_cached_build=bool(args.require_cached_build),
        )
        try:
            result = _run_reference(
                session,
                row["token_ids"],
                max(spec.max_tokens for spec in specs if spec.oracle_key == key),
            )
        except BaseException:
            session.close()
            raise
        reference_tokens[key] = [int(token) for token in result.generated_tokens]
        is_last = index == len(ordered_rows)
        if is_last:
            args.output_json.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "prompt_rows": serializable_rows,
                        "reference_tokens": reference_tokens,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            session.close()
        print(
            f"reference {index}/{len(ordered_rows)}: {key}",
            file=sys.stderr,
            flush=True,
        )
        if is_last:
            # The worker owns this complete HIP generation. Exit before final
            # diagnostic session teardown can race queued device completion;
            # process teardown releases all oracle allocations atomically.
            os._exit(0)
    raise AssertionError("oracle worker did not process any rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--max-active-requests", type=int, required=True)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--specs-json", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    raise AssertionError("oracle worker returned without exiting")


if __name__ == "__main__":
    raise SystemExit(main())
