#!/usr/bin/env python3
"""Correctness smoke for the Qwen3.5/PARO serial c>N resident slot bridge.

This is not a benchmark.  It compares a shared ``max_batch_size=2`` resident
session using ``step_batch_serial`` against independent c=1 resident sessions
for deterministic prompt slices.
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

from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession

DEFAULT_MODEL = (
    "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/"
    "snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"
)
DEFAULT_FIXTURE = "fixtures/qwen35_paro/parent_512_32_seed1234.json"


def _load_prompt_slices(path: Path, *, prompt_length: int, batch_size: int) -> list[list[int]]:
    fixture = json.loads(path.read_text())
    tokens = [int(token) for token in fixture["prompt_ids"]]
    needed = prompt_length * batch_size
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if len(tokens) < needed:
        raise ValueError(f"fixture contains {len(tokens)} tokens, need at least {needed}")
    return [tokens[row * prompt_length : (row + 1) * prompt_length] for row in range(batch_size)]


def _compiler_version(path: str | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text()


def _run_c1(
    runner: Qwen35ParoNextTokenRunner,
    prompt: list[int],
    *,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> dict[str, Any]:
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=len(prompt) + 4,
        max_layers=max_layers,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
    ) as session:
        seed = None
        for pos, token in enumerate(prompt):
            seed = session.step(token, position=pos, sample=(pos == len(prompt) - 1))
        if seed is None:
            raise RuntimeError("prefill did not produce a seed token")
        decode = session.step(seed.token_id, position=len(prompt), sample=True)
        if decode is None:
            raise RuntimeError("decode did not produce a token")
    return {
        "seed": seed.token_id,
        "decode": decode.token_id,
        "seed_logit": seed.logit,
        "decode_logit": decode.logit,
    }


def _run_batch_serial(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> list[dict[str, Any]]:
    prompt_lengths = {len(prompt) for prompt in prompts}
    if len(prompt_lengths) != 1:
        raise ValueError("current smoke expects equal prompt lengths")
    prompt_length = prompt_lengths.pop()
    slots = list(range(len(prompts)))
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=prompt_length + 4,
        max_layers=max_layers,
        max_batch_size=len(prompts),
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
    ) as session:
        seed_results = None
        for pos in range(prompt_length):
            seed_results = session.step_batch_serial(
                [prompt[pos] for prompt in prompts],
                positions=[pos] * len(prompts),
                slots=slots,
                sample=(pos == prompt_length - 1),
            )
        if seed_results is None or any(result is None for result in seed_results):
            raise RuntimeError("batch prefill did not produce seed tokens")
        decode_results = session.step_batch_serial(
            [result.token_id for result in seed_results if result is not None],
            positions=[prompt_length] * len(prompts),
            slots=slots,
            sample=True,
        )
        if any(result is None for result in decode_results):
            raise RuntimeError("batch decode did not produce tokens")
    return [
        {
            "seed": seed.token_id,
            "decode": decode.token_id,
            "seed_logit": seed.logit,
            "decode_logit": decode.logit,
        }
        for seed, decode in zip(seed_results, decode_results, strict=True)
        if seed is not None and decode is not None
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--compiler-version-file")
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument("--json", type=Path, help="Optional path to write JSON output")
    args = parser.parse_args(argv)

    if args.batch_size != 2:
        raise ValueError("this smoke currently supports --batch-size 2")
    prompts = _load_prompt_slices(Path(args.fixture), prompt_length=args.prompt_length, batch_size=args.batch_size)
    compiler_version = _compiler_version(args.compiler_version_file)
    runner = Qwen35ParoNextTokenRunner(Path(args.model))
    expected = [
        _run_c1(
            runner,
            prompt,
            max_layers=args.max_layers,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached,
        )
        for prompt in prompts
    ]
    actual = _run_batch_serial(
        runner,
        prompts,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached,
    )
    command = (
        "python3 scripts/qwen35_batch_serial_correctness.py "
        f"--prompt-length {args.prompt_length} --max-layers {args.max_layers} --batch-size {args.batch_size}"
    )
    if args.json is not None:
        command += f" --json {args.json}"
    payload = {
        "schema": 1,
        "status": "accepted_correctness_smoke",
        "model": str(Path(args.model)),
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "mode": "resident_c2_serial_slot_runner_correctness",
        "command": command,
        "batch_size": args.batch_size,
        "prompt_lengths": [len(prompt) for prompt in prompts],
        "max_layers": args.max_layers,
        "expected_c1": expected,
        "batch_serial": actual,
        "passed": actual == expected,
        "notes": [
            "Correctness-first serial c>N bridge over batch-shaped resident slot buffers; not a throughput claim.",
            "Compares a shared resident session against independent c=1 resident sessions.",
        ],
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json is not None:
        args.json.write_text(text + "\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
