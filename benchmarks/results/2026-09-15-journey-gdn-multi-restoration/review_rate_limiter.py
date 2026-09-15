"""Execute the inspected rate-limiter responses against the same allow oracle.

Only run on the reviewed local capture. Generated code is not a sandbox.
The optional remaining() cost probe is separate from the requested allow API.
"""

import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import random

REVIEWED_SOURCES = {
    "strict": "5535e32bedf212357f91a749cefb6f6edd72d60332ddc277b0b183e15a4ddb99",
    "production_gdn_multi_restore": "72ba555bed820487224ffbb7c0a50b2bab7e3b2c9554eea6105f8a0525d658c1",
}


def source_from_response(text):
    lines = text.removesuffix("<|im_end|>").strip().splitlines()
    if lines[0] != "```python" or lines[-1] != "```":
        raise ValueError("expected one complete Python response")
    source = "\n".join(lines[1:-1]) + "\n"
    tree = ast.parse(source)
    if any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
           for node in tree.body):
        raise ValueError("response differs from the reviewed class-only structure")
    return source, tree


def check_allow(cls):
    checks = 0
    for seed in range(4):
        for capacity in (1, 2, 5):
            for window in (0.5, 1.0, 10.0):
                limiter = cls(capacity, window)
                reference = collections.defaultdict(list)
                rng = random.Random(seed)
                timestamp = 0.0
                for _ in range(1000):
                    timestamp += rng.choice((0.0, 0.125, 0.5, 1.0, 10.0))
                    key = rng.choice(("alpha", "beta", "gamma"))
                    reference[key] = [
                        value for value in reference[key] if value > timestamp - window]
                    expected = len(reference[key]) < capacity
                    actual = limiter.allow(key, timestamp)
                    if actual is not expected:
                        raise AssertionError((seed, capacity, window, key, timestamp))
                    if expected:
                        reference[key].append(timestamp)
                    checks += 1
    return checks


def remaining_cost(cls):
    class CountedDeque(collections.deque):
        visits = 0

        def __iter__(self):
            for value in super().__iter__():
                self.visits += 1
                yield value

    limiter = cls(128, 10000.0)
    for timestamp in range(128):
        assert limiter.allow("alpha", timestamp)
    mapping = getattr(limiter, "_requests", None)
    if mapping is None:
        mapping = limiter._windows
    queue = CountedDeque(mapping["alpha"])
    mapping["alpha"] = queue
    for _ in range(100):
        assert limiter.remaining("alpha", 128.0) == 0
    return {"stored_timestamps": 128, "queries": 100, "iteration_visits": queue.visits}


def review(path):
    raw = path.read_bytes()
    packet = json.loads(raw)
    candidate = "production_gdn_multi_restore"
    if (packet["status"] != "captured_requires_manual_review"
            or not packet["source"]["tracked_clean"] or packet["complete_suite"]
            or packet["candidate"] != candidate
            or [row["id"] for row in packet["prompts"]] != ["heldout_code_rate_limiter"]
            or packet["arms"][candidate]["dispatch"]["calls"] != 30
            or any(row["current_allocated_bytes"] for row in packet["lifecycle"].values())):
        raise ValueError("invalid targeted task capture")
    results = {}
    for arm in ("strict", candidate):
        cases = packet["arms"][arm]["cases"]
        if len(cases) != 1 or not cases[0]["deterministic"]:
            raise ValueError("incomplete task repeats")
        runs = cases[0]["repeats"]
        if len(runs) != 2 or runs[0] != runs[1] or runs[0]["finish_reason"] != "eos":
            raise ValueError("task output is truncated or not repeatable")
        source, tree = source_from_response(runs[0]["text"])
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        if source_hash != REVIEWED_SOURCES[arm]:
            raise ValueError("changed generated code requires inspection before execution")
        namespace = {}
        exec(compile(tree, f"<reviewed-{arm}-response>", "exec"), namespace)
        cls = namespace["SlidingWindowRateLimiter"]
        results[arm] = {
            "source_sha256": source_hash,
            "tokens": len(runs[0]["ids"]),
            "allow_oracle_checks": check_allow(cls),
            "generated_test_functions": sum(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_") for node in ast.walk(tree)),
            "optional_remaining_cost": remaining_cost(cls),
        }
    return {
        "schema": 1, "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "capture": packet, "execution_checks": results,
        "verdict": "requested_allow_api_noninferior_shared_test_omission",
        "full_task_suite_passed": False, "promotion_claim": False,
        "review": [
            "Both allow implementations use the same deque expiry/append algorithm on monotone timestamps.",
            "Each passes 36000 independent allow checks across capacities, windows, expiry, repeated times and keys.",
            "Both omit requested pytest tests: a shared instruction-following defect, not a new candidate defect.",
            "Both wrap code in a fence despite asking for only code; this is also shared.",
            "Candidate remaining() scans all stored timestamps per query; strict expires deque heads.",
            "The optional remaining() cost difference is real, but remaining() was not requested; it is not used as an allow-API veto.",
            "Different constructor keyword names and unused imports are not failures of the unspecified constructor API.",
            "This targeted comparison does not certify other prompts, dynamic isolation, arbitrary timestamps or performance.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    Path(__file__).with_name("rate-limiter-review.json").write_text(
        json.dumps(review(args.capture), indent=2) + "\n")
