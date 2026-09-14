"""Run tests from manually reviewed code responses in the Q8 task capture.

This is not a security sandbox. Inspect generated source before invocation.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def extract_code(text):
    text = text.strip().removesuffix("<|im_end|>").strip()
    if text.startswith("```python\n") and text.endswith("```"):
        text = text[len("```python\n"):-3].strip()
    ast.parse(text)
    return text + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--arm", choices=("strict", "q8_fallback"), required=True)
    parser.add_argument("--reviewed-source", action="store_true", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.capture.read_bytes()
    capture = json.loads(raw)
    cases = [row for row in capture["arms"][args.arm]["cases"]
             if row["category"] == "code"]
    if len(cases) != 6:
        raise ValueError("all six code prompts required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        if not case["deterministic"] or case["repeats"][0]["finish_reason"] != "eos":
            raise ValueError("incomplete/nondeterministic code response")
        source = extract_code(case["repeats"][0]["text"])
        if not case["id"].replace("_", "").isalnum():
            raise ValueError("invalid case id")
        path = args.output_dir / ("test_" + case["id"] + ".py")
        path.write_text(source)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-c", "/dev/null", "-p", "no:cacheprovider",
             "-q", str(path.resolve())],
            cwd=args.output_dir, capture_output=True, text=True, timeout=60,
        )
        main_result = subprocess.run(
            [sys.executable, str(path.resolve())], cwd=args.output_dir,
            capture_output=True, text=True, timeout=60,
        )
        records.append(dict(id=case["id"], source_sha256=hashlib.sha256(
            source.encode()).hexdigest(), returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr,
            main_returncode=main_result.returncode, main_stdout=main_result.stdout,
            main_stderr=main_result.stderr,
            tests_present=any(isinstance(node, ast.Assert)
                              for node in ast.walk(ast.parse(source)))))
    report = dict(capture_sha256=hashlib.sha256(raw).hexdigest(), arm=args.arm,
                  command=sys.argv, records=records,
                  passed=all(r["tests_present"] and r["returncode"] in (0, 5)
                             and r["main_returncode"] == 0
                             for r in records),
                  scope="generated self-tests, not an independent semantic proof")
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
