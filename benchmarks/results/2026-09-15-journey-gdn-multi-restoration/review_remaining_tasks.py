"""Preserve complete task comparisons and execute the inspected changed code."""

import argparse
import ast
import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path
import random

CANDIDATE = "production_gdn_multi_restore"
REVIEWED = {
    ("strict", "code_markdown_table"): "c74023f8283f773e7281a7358d990650f51588ff37b462b789f4f05cea474365",
    ("strict", "heldout_code_interval_schedule"): "ed8b4e8ac9a3bbc438c0d826597b6a9ace18152b838450c10ea02d18c7756df4",
    (CANDIDATE, "code_markdown_table"): "3ac92eea92a175f56bc885fa1b7c33473794ed20ebb7f2cc570a643407318a84",
    (CANDIDATE, "heldout_code_interval_schedule"): "4621fcc8e42d8386cf74aae40635dc44979e9ee6dfa1019bb44313d2b1ee4359",
}


def execute_reviewed(arm, prompt, text):
    lines = text.removesuffix("<|im_end|>").strip().splitlines()
    if lines[0] != "```python" or lines[-1] != "```":
        raise ValueError("expected one complete Python block")
    source = "\n".join(lines[1:-1]) + "\n"
    sha = hashlib.sha256(source.encode()).hexdigest()
    if sha != REVIEWED[(arm, prompt)]:
        raise ValueError("changed generated code needs inspection before execution")
    namespace = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.parse(source), f"<reviewed-{arm}-{prompt}>", "exec"), namespace)
    return namespace, sha


def check_markdown(fn):
    rng = random.Random(824)
    for _ in range(1000):
        rows = []
        for _ in range(rng.randrange(6)):
            keys = rng.sample(("a", "b", "c"), rng.randrange(4))
            rows.append({key: rng.choice(("", "x|y", 1, 2.5, True)) for key in keys})
        columns = list(dict.fromkeys(key for row in rows for key in row))
        expected = ""
        if columns:
            lines = ["| " + " | ".join(columns) + " |",
                     "| " + " | ".join("---" for _ in columns) + " |"]
            lines.extend("| " + " | ".join(
                str(row.get(key, "")).replace("|", "\\|") for key in columns) + " |"
                for row in rows)
            expected = "\n".join(lines)
        assert fn(rows) == expected
    # None rendering is intentionally not an oracle requirement.
    return {"oracle_cases": 1000, "none_rendering": fn([{"a": None}])}


def check_intervals(fn):
    rng = random.Random(842)
    for _ in range(500):
        intervals = []
        for _ in range(rng.randrange(8)):
            start = rng.randrange(8)
            intervals.append((start, start + rng.randrange(1, 5)))
        maximum = 0
        for count in range(len(intervals) + 1):
            for chosen in itertools.combinations(intervals, count):
                ordered = sorted(chosen)
                if all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])):
                    maximum = max(maximum, count)
        result = fn(intervals)
        assert len(result) == maximum
        assert all(a[1] <= b[0] for a, b in zip(result, result[1:]))
        assert all(value in intervals for value in result)
    assert fn([(2, 3), (1, 3), (3, 4)]) == [(1, 3), (3, 4)]
    return {"oracle_cases": 500, "domain": "nonempty half-open intervals",
            "tie_case_passed": True}


def assemble(path):
    raw = path.read_bytes()
    packet = json.loads(raw)
    if (packet["status"] != "captured_requires_manual_review"
            or not packet["source"]["tracked_clean"] or packet["complete_suite"]
            or packet["candidate"] != CANDIDATE or len(packet["prompts"]) != 17
            or packet["arms"][CANDIDATE]["dispatch"]["calls"] != 510
            or any(row["current_allocated_bytes"] for row in packet["lifecycle"].values())):
        raise ValueError("invalid remaining-task capture")
    strict = packet["arms"]["strict"]["cases"]
    candidate = packet["arms"][CANDIDATE]["cases"]
    prompt_ids = [row["id"] for row in packet["prompts"]]
    if (len(set(prompt_ids)) != 17 or "heldout_code_rate_limiter" in prompt_ids
            or [row["id"] for row in strict] != prompt_ids
            or [row["id"] for row in candidate] != prompt_ids):
        raise ValueError("task cases do not cover the declared distinct prompt subset")
    comparisons, code_checks = [], {}
    for before, after in zip(strict, candidate, strict=True):
        if before["id"] != after["id"]:
            raise ValueError("unpaired prompt")
        record = {"id": before["id"], "category": before["category"], "arms": {}}
        for arm, case in (("strict", before), (CANDIDATE, after)):
            runs = case["repeats"]
            if len(runs) != 2 or not case["deterministic"] or runs[0] != runs[1]:
                raise ValueError("nonrepeatable task output")
            result = runs[0]
            record["arms"][arm] = {
                "tokens": len(result["ids"]), "text": result["text"],
                "finish": result["finish_reason"],
                "ids_sha256": hashlib.sha256(
                    json.dumps(result["ids"], separators=(",", ":")).encode()).hexdigest(),
            }
            if (arm, case["id"]) in REVIEWED:
                namespace, sha = execute_reviewed(arm, case["id"], result["text"])
                checks = (check_markdown(namespace["markdown_table"])
                          if case["id"] == "code_markdown_table"
                          else check_intervals(namespace["max_non_overlapping"]))
                code_checks[f"{arm}:{case['id']}"] = {
                    "source_sha256": sha, "generated_tests_passed": True, **checks}
        record["output_ids_exact"] = before["repeats"][0]["ids"] == after["repeats"][0]["ids"]
        record["both_eos"] = all(value["finish"] == "eos" for value in record["arms"].values())
        comparisons.append(record)
    if len(comparisons) != 17:
        raise ValueError("missing task cases")
    return {
        "schema": 1, "status": "captured_for_paired_review",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "metadata": {key: value for key, value in packet.items() if key != "arms"},
        "arm_metadata": {arm: {key: value for key, value in data.items() if key != "cases"}
                         for arm, data in packet["arms"].items()},
        "comparisons": comparisons, "code_checks": code_checks,
        "summary": {
            "prompts": len(comparisons),
            "complete_pairs": sum(row["both_eos"] for row in comparisons),
            "exact_pairs": sum(row["output_ids_exact"] for row in comparisons),
            "truncated_prompts": [row["id"] for row in comparisons if not row["both_eos"]],
        },
        "limits": [
            "Capture completion and executable code checks do not decide prose task quality.",
            "Japanese plan is truncated in both arms and must not be counted as a completed task.",
            "The separate rate-limiter capture supplies the eighteenth prompt.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("remaining-task-capture.json").write_text(
        json.dumps(assemble(args.capture), indent=2, ensure_ascii=False) + "\n")
