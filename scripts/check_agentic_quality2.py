#!/usr/bin/env python3
"""Validate frozen AGENTIC-QUALITY2 fixtures, oracles, and host sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.agentic_quality2 import (  # noqa: E402
    evaluate_quality2_fail_safe_control,
    execute_reference_case,
    load_agentic_quality2_suite,
)
from hipengine.benchmark.agentic_quality2_sandbox import (  # noqa: E402
    AgenticQuality2Sandbox,
    SandboxLimits,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("benchmarks/prompts/agentic-quality2-v2.json"),
    )
    parser.add_argument("--json", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "status",
            "reason",
            "tests_attempted",
            "tests_passed",
            "failure",
            "returncode",
            "stdout_sha256",
            "stderr_sha256",
            "output_truncated",
            "network_isolated",
            "filesystem_isolated",
            "device_isolated",
            "environment_cleared",
            "hidden_expected_exposed",
            "process_group_killed",
            "scratch_cleaned",
            "limits",
        )
        if key in result
    }


def _sandbox_probes() -> tuple[dict[str, Any], bool]:
    limits = SandboxLimits(
        wall_seconds=0.4,
        cpu_seconds=1,
        memory_bytes=96 << 20,
        file_bytes=4096,
        output_bytes=1024,
        processes=1,
        open_files=32,
    )
    sandbox = AgenticQuality2Sandbox(limits=limits)
    with tempfile.TemporaryDirectory(prefix="aq2-validation-") as temporary:
        root = Path(temporary)
        probes: dict[str, dict[str, Any]] = {}
        probes["valid"] = sandbox.run_code_case(
            source=(
                "def clamp_readings(values, low, high):\n"
                "    return [max(low, min(high, value)) for value in values]\n"
            ),
            entry_point="clamp_readings",
            hidden_tests=[{"args": [[-2, 4, 11], 0, 8], "kwargs": {}, "expected": [0, 4, 8]}],
            scratch_root=root / "valid",
        )
        escape_specs = {
            "network": (
                "def probe():\n import socket\n socket.create_connection(('127.0.0.1',9),0.1)\n",
                ("socket",),
            ),
            "filesystem": ("def probe():\n return open('/etc/passwd').read()\n", ()),
            "home_repository_model": (
                "def probe():\n return open('/home/lhl/hipEngine/README.md').read()\n",
                (),
            ),
            "device": ("def probe():\n return open('/dev/kfd','rb').read(1)\n", ()),
            "environment": (
                "def probe():\n import os\n return os.getenv('AQ2_SANDBOX_SECRET')\n",
                ("os",),
            ),
            "process": (
                (
                    "def probe():\n import subprocess\n"
                    " subprocess.Popen(['/usr/bin/sleep','30'])\n return 1\n"
                ),
                ("subprocess",),
            ),
            "hidden_expected": (
                (
                    "def probe():\n import os\n"
                    " return '|'.join(open('/input/' + name).read() "
                    "for name in sorted(os.listdir('/input')) "
                    "if os.path.isfile('/input/' + name))\n"
                ),
                ("os",),
            ),
        }
        previous = os.environ.get("AQ2_SANDBOX_SECRET")
        os.environ["AQ2_SANDBOX_SECRET"] = "must-not-leak"
        try:
            for name, (source, allowed_imports) in escape_specs.items():
                expected = (
                    "hidden-expected-must-never-enter-sandbox"
                    if name == "hidden_expected"
                    else "never"
                )
                probes[name] = sandbox.run_code_case(
                    source=source,
                    entry_point="probe",
                    hidden_tests=[{"args": [], "kwargs": {}, "expected": expected}],
                    scratch_root=root / name,
                    allowed_imports=allowed_imports,
                )
        finally:
            if previous is None:
                os.environ.pop("AQ2_SANDBOX_SECRET", None)
            else:
                os.environ["AQ2_SANDBOX_SECRET"] = previous
        probes["wall_timeout"] = sandbox.run_code_case(
            source="def probe():\n    while True: pass\n",
            entry_point="probe",
            hidden_tests=[{"args": [], "kwargs": {}, "expected": 1}],
            scratch_root=root / "timeout",
        )
        probes["memory"] = sandbox.run_code_case(
            source="def probe():\n    return bytearray(200_000_000)\n",
            entry_point="probe",
            hidden_tests=[{"args": [], "kwargs": {}, "expected": None}],
            scratch_root=root / "memory",
        )
        probes["file_size"] = sandbox.run_code_case(
            source=(
                "def probe():\n"
                "    with open('/work/payload','wb') as handle:\n"
                "        handle.write(b'x' * 20000)\n"
                "    return 1\n"
            ),
            entry_point="probe",
            hidden_tests=[{"args": [], "kwargs": {}, "expected": 1}],
            scratch_root=root / "file",
        )
        probes["output"] = sandbox.run_code_case(
            source="def probe():\n    print('x' * 20000)\n    return 1\n",
            entry_point="probe",
            hidden_tests=[{"args": [], "kwargs": {}, "expected": 1}],
            scratch_root=root / "output",
        )
    valid = probes["valid"]["status"] == "passed"
    escape_pass = all(probes[name]["status"] == "failed" for name in escape_specs)
    timeout_pass = (
        probes["wall_timeout"]["status"] == "timeout"
        and probes["wall_timeout"]["process_group_killed"] is True
    )
    resource_pass = (
        all(probes[name]["status"] == "failed" for name in ("memory", "file_size", "output"))
        and probes["output"]["output_truncated"] is True
    )
    isolation_pass = all(
        result["network_isolated"]
        and result["filesystem_isolated"]
        and result["device_isolated"]
        and result["environment_cleared"]
        and result["scratch_cleaned"]
        for result in probes.values()
    )
    sensitive_clean = all(
        "must-not-leak" not in result.get("stdout", "")
        and "must-not-leak" not in result.get("stderr", "")
        and "hidden-expected-must-never-enter-sandbox" not in result.get("stdout", "")
        and "hidden-expected-must-never-enter-sandbox" not in result.get("stderr", "")
        for result in probes.values()
    )
    qualified = bool(
        valid
        and escape_pass
        and timeout_pass
        and resource_pass
        and isolation_pass
        and sensitive_clean
    )
    return {name: _probe_summary(result) for name, result in probes.items()}, qualified


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    suite = load_agentic_quality2_suite(args.suite)
    reference = [execute_reference_case(suite, case_id) for case_id in suite.workloads]
    controls = [
        evaluate_quality2_fail_safe_control(suite, row["id"])
        for row in suite.oracle["fail_safe_controls"]
    ]
    probes, sandbox_qualified = _sandbox_probes()
    references_pass = all(row["passed"] is True for row in reference)
    controls_pass = all(row["passed"] is True for row in controls)
    qualified = bool(references_pass and controls_pass and sandbox_qualified)
    artifact = {
        "kind": "hipengine_agentic_quality2_validation",
        "schema_version": 1,
        "status": "qualified" if qualified else "blocked",
        "date": "2026-08-26",
        "performance_claim": False,
        "source": {
            "git_base_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"),
                cwd=REPO_ROOT,
                text=True,
            ).strip(),
            "checker_sha256": _sha256(Path(__file__)),
            "loader_sha256": _sha256(REPO_ROOT / "hipengine/benchmark/agentic_quality2.py"),
            "sandbox_sha256": _sha256(
                REPO_ROOT / "hipengine/benchmark/agentic_quality2_sandbox.py"
            ),
        },
        "suite": suite.identity(),
        "coverage": {
            "workloads": len(suite.workloads),
            "development": len(suite.development_ids),
            "heldout": len(suite.heldout_ids),
            "reference_cases_passed": sum(row["passed"] is True for row in reference),
            "reference_cases_total": len(reference),
            "fail_safe_controls_passed": sum(row["passed"] is True for row in controls),
            "fail_safe_controls_total": len(controls),
            "sandbox_probes_passed": sum(
                (
                    row["status"] == "passed"
                    if name == "valid"
                    else row["status"] in {"failed", "timeout"}
                )
                for name, row in probes.items()
            ),
            "sandbox_probes_total": len(probes),
        },
        "reference_rollup": {
            "passed": references_pass,
            "by_kind": {
                kind: sum(row["kind"] == kind and row["passed"] is True for row in reference)
                for kind in sorted({row["kind"] for row in reference})
            },
            "case_result_sha256": {row["case_id"]: row["result_sha256"] for row in reference},
        },
        "fail_safe_rollup": {
            "passed": controls_pass,
            "controls": [
                {
                    "control_id": row["control_id"],
                    "class": row["class"],
                    "split": row["split"],
                    "passed": row["passed"],
                }
                for row in controls
            ],
        },
        "sandbox": {
            "qualified": sandbox_qualified,
            "backend": "bubblewrap+prlimit+python-isolated",
            "bwrap_version": subprocess.check_output(
                ("/usr/bin/bwrap", "--version"),
                text=True,
            ).strip(),
            "python_version": subprocess.check_output(
                ("/usr/bin/python3", "--version"),
                text=True,
            ).strip(),
            "network": "new empty network namespace",
            "filesystem": "only read-only /usr,/lib,/lib64 plus private /proc,/dev,/tmp,/work",
            "environment": "clearenv with fixed PATH and LANG",
            "hidden_tests": "one input per fresh namespace; expected values remain host-only",
            "process": "new PID/session namespace, nproc limit, kill process group on timeout",
            "resources": "wall, CPU, AS, file size, process, FD, core, and output caps",
            "probes": probes,
        },
        "aggregation": {
            "deterministic_order_independent": True,
            "heldout_details_sealed_by_default": True,
            "blocked_unscorable_excluded_with_explicit_denominator": True,
            "large_raw_compact_separation": True,
        },
        "decision": {
            "qualified_for_live_baseline": qualified,
            "generated_code_policy": (
                "execute only through qualified sandbox" if sandbox_qualified else "blocked_sandbox"
            ),
            "candidate_implementation_admitted": False,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"AGENTIC-QUALITY2 validation: references {artifact['coverage']['reference_cases_passed']}/"
        f"{artifact['coverage']['reference_cases_total']}, controls "
        f"{artifact['coverage']['fail_safe_controls_passed']}/"
        f"{artifact['coverage']['fail_safe_controls_total']}, sandbox={sandbox_qualified} "
        f"-> {args.json}"
    )
    return 0 if qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
