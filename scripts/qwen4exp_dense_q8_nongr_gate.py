"""Run the canonical Q8 gate with generic IU8 excluded from GR reads.

This is a single-threaded diagnostic, not a production dispatch policy or a
performance harness. The underlying linear cache keys the IU8 environment bit.
"""

import argparse
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLAG = "HIPENGINE_QWEN4_EXP_Q8_IU8_WMM"


@contextmanager
def exclude_gr_reads(module):
    original = module.run_qwen4_exp_gr_read
    counter = {"calls": 0, "rows": {}}

    @wraps(original)
    def excluded(*args, **kwargs):
        previous = os.environ.get(FLAG)
        if previous in (None, "", "0", "false", "False"):
            return original(*args, **kwargs)
        rows = int(kwargs["rows"])
        if rows > 256:
            counter["calls"] += 1
            counter["rows"][rows] = counter["rows"].get(rows, 0) + 1
        os.environ[FLAG] = "0"
        try:
            return original(*args, **kwargs)
        finally:
            os.environ[FLAG] = previous

    module.run_qwen4_exp_gr_read = excluded
    try:
        yield counter
    finally:
        module.run_qwen4_exp_gr_read = original


def validate_exclusion(packet, counter):
    if (packet["status"] != "completed"
            or packet["candidate"] != "production_dense_q8_restore"
            or counter["calls"] <= 0 or packet["candidate_dispatch_calls"] <= 0):
        raise ValueError("missing completed non-GR intervention")
    shapes = packet["candidate_dispatch_shapes"]
    if any(row["arguments"][1:] == [10240, 320] for row in shapes):
        raise ValueError("GR-down IU8 escaped the role exclusion")
    if sum(row["calls"] for row in shapes) != packet["candidate_dispatch_calls"]:
        raise ValueError("inconsistent candidate dispatch counts")
    if packet["protocol"]["complete_fixture"]:
        if (packet["protocol"]["chunk"] != 1024
                or packet["protocol"]["repeats"] != 3
                or counter != {"calls": 6912, "rows": {512: 1152, 1024: 5760}}
                or packet["candidate_dispatch_calls"] != 21744):
            raise ValueError("unexpected full-matrix intervention counts")


def main():
    from hipengine.runtime import qwen4_exp_runner
    from scripts.qwen4exp_q8_repair_depth_gate import main as depth_gate

    # Delegate argument validation and GPU serialization to the canonical gate.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate")
    args, _ = parser.parse_known_args()
    if args.candidate is not None:
        raise ValueError("this diagnostic fixes the candidate; omit --candidate")
    sys.argv.extend(["--candidate", "production_dense_q8_restore"])
    try:
        with exclude_gr_reads(qwen4_exp_runner) as counter:
            depth_gate()
        output = args.output
        packet = json.loads(output.read_bytes())
        packet["role_exclusion"] = {
            "scope": "run_qwen4_exp_gr_read",
            "flag": FLAG,
            "eligible_calls": counter["calls"],
            "rows": [{"rows": rows, "calls": calls}
                     for rows, calls in sorted(counter["rows"].items())],
            "production_policy": False,
        }
        try:
            validate_exclusion(packet, counter)
        except ValueError as error:
            packet["status"] = "invalid_capture"
            packet["role_exclusion"]["error"] = str(error)
            raise
        finally:
            output.write_text(json.dumps(packet, indent=2) + "\n")
    finally:
        del sys.argv[-2:]


if __name__ == "__main__":
    main()
