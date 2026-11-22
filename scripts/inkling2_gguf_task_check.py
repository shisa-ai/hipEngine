#!/usr/bin/env python3
"""Compare Inkling GGUF strict and fast on fixed known-answer prompts."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys

from hipengine import LLM, SamplingParams


# Freeze prompts and answers before either profile runs. Cases are scored
# independently: an improvement cannot offset a regression on another case.
_CASES = (
    ("arithmetic_product", "arithmetic", "What is 17 multiplied by 23?", "391"),
    ("arithmetic_remainder", "arithmetic", "What is the remainder when 198 is divided by 17?", "11"),
    ("arithmetic_distance", "arithmetic", "A cyclist travels 8 km every 20 minutes at a constant speed. How many kilometers does she travel in 50 minutes?", "20"),
    ("record_revision", "extraction", "Old entry: Dock 4, code MINT-941. Corrected entry: Dock 4, code MINT-419. What is the corrected code?", "MINT-419"),
    ("record_status", "extraction", "Inventory: ember = red; shale = gray; pearl = white. What color is shale?", "gray"),
    ("record_filter", "extraction", "Tickets: F11 closed, F12 waiting, F13 closed. Which ticket is waiting?", "F12"),
    ("python_slice", "code_reading", "In Python, what integer is len([10, 20, 30, 40, 50][1:4])?", "3"),
    ("python_dict", "code_reading", "In Python, what integer is {'one': 1, 'two': 2}.get('three', 7)?", "7"),
    ("python_range", "code_reading", "In Python, what integer is sum(range(2, 7, 2))?", "12"),
    ("japanese_days", "japanese", "火曜日の翌日は何曜日ですか。", "水曜日"),
    ("japanese_record", "japanese", "出発駅は東京、到着駅は新大阪です。到着駅の駅名を答えてください。", "新大阪"),
    ("japanese_addition", "japanese", "リンゴが9個あり、さらに6個を買いました。全部で何個ですか。数字で答えてください。", "15"),
)
_PREFIX = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>"
_SUFFIX = "<|end_message|><|message_model|><|content_text|>"


def _render(category: str, prompt: str) -> str:
    instruction = (
        "答えだけを書いてください。句読点や説明は不要です。" if category == "japanese" else
        "Give only the requested result, with no explanation, suffix, or punctuation."
    )
    return _PREFIX + f"{prompt}\n{instruction}" + _SUFFIX


def _digest(ids) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def evaluate(model: Path, backend: str, profile: str):
    sampling = SamplingParams(temperature=0, max_tokens=24)
    with closing(LLM(model=str(model), backend=backend, max_sequence_length=256, prefix_cache="off",
                     speculative_mtp_serving="off", execution_profile=profile)) as llm:
        if llm.resolved_execution_profile != profile or llm.execution_profile_fell_back_to_strict:
            raise RuntimeError(f"Inkling task check did not resolve profile {profile!r}")
        print(json.dumps({"profile": profile, "manifest_sha256": llm.execution_profile_manifest_sha256}), flush=True)
        captured = {}
        for key, category, prompt, expected in _CASES:
            ids = llm.tokenize(_render(category, prompt))
            output, = llm.generate_detailed(ids, sampling)
            text = output.text.strip()
            row = dict(case=key, profile=profile, category=category, prompt_digest=_digest(ids),
                       prompt_tokens=len(ids), ids=output.generated_token_ids, answer=text, expected=expected,
                       correct=text == expected, finish=output.finish_details.reason)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            captured[key] = row
        return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--reverse", action="store_true", help="run fast before strict")
    args = parser.parse_args()
    order = ("fast", "strict") if args.reverse else ("strict", "fast")
    results = {profile: evaluate(args.model, args.backend, profile) for profile in order}
    strict, fast = results["strict"], results["fast"]
    if any(strict[key]["prompt_digest"] != fast[key]["prompt_digest"] for key in strict):
        raise RuntimeError("strict and fast task inputs differed")
    regressed = [key for key in strict if strict[key]["correct"] and not fast[key]["correct"]]
    print(json.dumps(dict(order=order, strict=sum(r["correct"] for r in strict.values()),
                          fast=sum(r["correct"] for r in fast.values()), cases=len(strict),
                          regressed=regressed)), flush=True)
    return bool(regressed)


if __name__ == "__main__":
    sys.exit(main())
