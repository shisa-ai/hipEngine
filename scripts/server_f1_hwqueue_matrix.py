#!/usr/bin/env python3
"""Run a resume-safe, counterbalanced gfx1151 F1 hardware-queue matrix.

The driver launches one server benchmark child at a time.  It never treats an
SLO/product failure as a process-health failure, but it stops immediately on a
missing artifact, control/ownership failure, nondeterminism, route failure,
child exception, timeout, or a surviving KFD process.  ``unset`` suppresses both
the harness value and hipEngine's gfx1151 package queue default so ROCm receives
no ``GPU_MAX_HW_QUEUES`` limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
_ALLOWED_QUEUE_POLICIES = ("1", "2", "4", "8", "unset")
_RUN_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_KFD_PID_RE = re.compile(r"^\s*(\d+)\s+\S+\s+", re.MULTILINE)


class MatrixError(RuntimeError):
    """Raised when matrix provenance, health, or resume contracts fail."""


def parse_queue_policies(raw: str) -> tuple[str, ...]:
    policies = tuple(part.strip().lower() for part in str(raw).split(",") if part.strip())
    if not policies:
        raise ValueError("queue policies must include at least one of 1,2,4,8,unset")
    if len(set(policies)) != len(policies):
        raise ValueError("queue policies must not contain duplicate values")
    if any(policy not in _ALLOWED_QUEUE_POLICIES for policy in policies):
        raise ValueError("queue policies must be selected from 1,2,4,8,unset")
    return policies


def _parse_widths(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values or len(set(values)) != len(values) or any(value < 1 or value > 32 for value in values):
        raise argparse.ArgumentTypeError("concurrencies must be unique values in c1-c32")
    return values


def counterbalanced_queue_blocks(
    policies: Sequence[str],
    *,
    blocks: int,
) -> tuple[tuple[str, ...], ...]:
    values = tuple(str(policy) for policy in policies)
    if not values or int(blocks) < 1:
        raise ValueError("counterbalanced schedule requires policies and at least one block")
    result: list[tuple[str, ...]] = []
    for block_index in range(int(blocks)):
        pair_index = block_index // 2
        rotated = values[pair_index % len(values) :] + values[: pair_index % len(values)]
        result.append(rotated if block_index % 2 == 0 else tuple(reversed(rotated)))
    return tuple(result)


def queue_environment(base: Mapping[str, str], policy: str) -> dict[str, str]:
    normalized = str(policy)
    if normalized not in _ALLOWED_QUEUE_POLICIES:
        raise ValueError("queue policy must be one of 1,2,4,8,unset")
    environment = {str(key): str(value) for key, value in base.items()}
    if normalized == "unset":
        environment.pop("GPU_MAX_HW_QUEUES", None)
        environment["HIPENGINE_GPU_MAX_HW_QUEUES_POLICY"] = "runtime_default"
    else:
        environment["GPU_MAX_HW_QUEUES"] = normalized
        environment["HIPENGINE_GPU_MAX_HW_QUEUES_POLICY"] = "explicit"
    return environment


def run_leaf(
    lane_root: Path,
    *,
    commit: str,
    profile: str,
    queue_policy: str,
    suite: str,
    run_tag: str,
    block_index: int,
) -> Path:
    return (
        Path(lane_root)
        / "runs"
        / str(commit)
        / str(profile)
        / f"hwq-{queue_policy}"
        / str(suite)
        / f"{run_tag}-block{int(block_index):02d}"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resume_result(
    leaf: Path,
    *,
    spec_sha256: str,
    expected_widths: Sequence[int],
) -> dict[str, Any] | None:
    leaf = Path(leaf)
    spec_path = leaf / "spec.json"
    result_path = leaf / "result.json"
    if not spec_path.exists() and not result_path.exists():
        return None
    if not spec_path.is_file():
        raise ValueError(f"resume spec is missing: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping) or spec.get("spec_sha256") != str(spec_sha256):
        raise ValueError(f"resume spec hash does not match: {spec_path}")
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"resume result root must be an object: {result_path}")
    expected = {str(int(width)) for width in expected_widths}
    if (
        result.get("schema") != 2
        or not result.get("completed_at")
        or set(result.get("rows", {})) != expected
    ):
        raise ValueError(f"resume result is incomplete or mismatched: {result_path}")
    return result


def classify_child_result(
    payload: Mapping[str, Any],
    *,
    expected_widths: Sequence[int],
) -> dict[str, Any]:
    failures: list[str] = []
    if payload.get("schema") != 2:
        failures.append("artifact_schema_not_2")
    if payload.get("status") == "failed_exception":
        failures.append("child_failed_exception")
    rows = payload.get("rows")
    rows = rows if isinstance(rows, Mapping) else {}
    for width in expected_widths:
        row = rows.get(str(int(width)))
        if not isinstance(row, Mapping):
            failures.append(f"c{width}_missing")
            continue
        correctness = row.get("correctness")
        correctness = correctness if isinstance(correctness, Mapping) else {}
        for gate_name in ("warmups", "measured", "repeat_determinism"):
            gate = correctness.get(gate_name)
            if not isinstance(gate, Mapping) or gate.get("passed") is not True:
                failures.append(f"c{width}_{gate_name}_failed")
        live = correctness.get("live_admission")
        if live is not None and (
            not isinstance(live, Mapping) or live.get("passed") is not True
        ):
            failures.append(f"c{width}_live_control_failed")
        execution = row.get("execution")
        if not isinstance(execution, Mapping) or execution.get("route_ok") is not True:
            failures.append(f"c{width}_route_failed")
        streaming = row.get("streaming")
        if streaming is not None:
            if not isinstance(streaming, Mapping) or streaming.get("passed") is not True:
                failures.append(f"c{width}_stream_correctness_failed")
            route = streaming.get("route") if isinstance(streaming, Mapping) else None
            if not isinstance(route, Mapping) or route.get("passed") is not True:
                failures.append(f"c{width}_stream_route_failed")
    mechanical_passed = not failures
    return {
        "mechanical_passed": mechanical_passed,
        "product_gate_passed": payload.get("passed") is True,
        "safe_to_continue": mechanical_passed,
        "failures": failures,
    }


def _capture(command: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _kfd_pids() -> tuple[int, ...]:
    capture = _capture(["rocm-smi", "--showpids"])
    if capture.returncode != 0:
        raise MatrixError(f"rocm-smi --showpids failed:\n{capture.stdout}")
    return tuple(int(value) for value in _KFD_PID_RE.findall(capture.stdout))


def _gpu_state() -> dict[str, float]:
    capture = _capture(["rocm-smi", "--showtemp", "--showuse", "--csv"])
    if capture.returncode != 0:
        raise MatrixError(f"rocm-smi health capture failed:\n{capture.stdout}")
    lines = [line for line in capture.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise MatrixError(f"rocm-smi health capture is incomplete:\n{capture.stdout}")
    fields = [field.strip() for field in lines[1].split(",")]
    if len(fields) < 3:
        raise MatrixError(f"rocm-smi health row is incomplete: {lines[1]}")
    return {"edge_temperature_c": float(fields[1]), "gpu_use_percent": float(fields[2])}


def _wait_for_idle(
    *,
    max_temperature_c: float,
    stable_samples: int,
    sample_interval_seconds: float,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    stable = 0
    while stable < int(stable_samples):
        pids = _kfd_pids()
        state = _gpu_state()
        observation = {
            "observed_at": datetime.now(UTC).isoformat(),
            "kfd_pids": list(pids),
            **state,
        }
        observations.append(observation)
        if (
            not pids
            and state["gpu_use_percent"] == 0.0
            and state["edge_temperature_c"] <= float(max_temperature_c)
        ):
            stable += 1
        else:
            stable = 0
        if stable < int(stable_samples):
            time.sleep(float(sample_interval_seconds))
    return observations


def _terminate_owned_process(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30.0)


def _run_child(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    timeout_seconds: float,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return int(process.wait(timeout=float(timeout_seconds)))
        except subprocess.TimeoutExpired as exc:
            _terminate_owned_process(process)
            raise MatrixError(
                f"child timed out after {timeout_seconds}s; log={log_path}"
            ) from exc


def _git_head(source_root: Path) -> str:
    capture = _capture(["git", "rev-parse", "HEAD"], cwd=source_root)
    if capture.returncode != 0:
        raise MatrixError(f"cannot resolve source HEAD:\n{capture.stdout}")
    return capture.stdout.strip()


def _git_clean(source_root: Path) -> bool:
    capture = _capture(["git", "status", "--porcelain=v1"], cwd=source_root)
    return capture.returncode == 0 and not capture.stdout.strip()


def _build_child_command(
    args: argparse.Namespace,
    *,
    queue_policy: str,
    leaf: Path,
    port_base: int,
) -> list[str]:
    command = [
        str(args.python),
        "scripts/server_f1_concurrency_bench.py",
        "--engine",
        "hipengine",
        "--model",
        str(args.model),
        "--backend",
        str(args.backend),
        "--quant",
        str(args.quant),
        "--served-model-name",
        f"qwen38-gfx1151-hwq-{queue_policy}",
        "--correctness-profile",
        "production",
        "--production-correctness-artifact",
        str(args.production_correctness_artifact),
        "--hipengine-route-expectation",
        "native",
        "--hipengine-prefill-decode-policy",
        "fair",
        "--hipengine-prefill-chunk-tokens",
        "256",
        "--hipengine-kv-storage",
        "bf16",
        "--gpu",
        str(args.gpu),
        "--gpu-max-hw-queues",
        str(queue_policy),
        "--compiler-version-file",
        str(args.compiler_version_file),
        "--concurrencies",
        ",".join(str(value) for value in args.concurrencies),
        "--live-concurrency",
        str(args.live_concurrency),
        "--same-server-oracle",
        "--streaming-primary",
        "--stream-warmup-runs",
        str(args.stream_warmup_runs),
        "--stream-measured-runs",
        str(args.stream_measured_runs),
        "--prompt-token-id",
        str(args.prompt_token_id),
        "--prompt-length",
        str(args.prompt_length),
        "--decode-tokens",
        str(args.decode_tokens),
        "--oracle-rows",
        str(args.oracle_rows),
        "--warmup-runs",
        str(args.warmup_runs),
        "--measured-runs",
        str(args.measured_runs),
        "--ctx-per-seq",
        str(args.ctx_per_seq),
        "--memory-domain",
        "gtt",
        "--memory-sample-through-shutdown",
        "--port-base",
        str(port_base),
        "--work-dir",
        str(leaf / "work"),
        "--json",
        str(leaf / "result.json"),
    ]
    if 1 not in args.concurrencies:
        command.append("--focused-width-repair")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--production-correctness-artifact", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--lane-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--suite", default="core")
    parser.add_argument("--queue-policies", type=parse_queue_policies, default=_ALLOWED_QUEUE_POLICIES)
    parser.add_argument("--order-blocks", type=int, default=2)
    parser.add_argument("--concurrencies", type=_parse_widths, default=(17, 32))
    parser.add_argument("--live-concurrency", type=int, default=32)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port-base", type=int, default=22300)
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--oracle-rows", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--stream-warmup-runs", type=int, default=1)
    parser.add_argument("--stream-measured-runs", type=int, default=3)
    parser.add_argument("--ctx-per-seq", type=int, default=256)
    parser.add_argument("--max-idle-temperature-c", type=float, default=45.0)
    parser.add_argument("--idle-stable-samples", type=int, default=4)
    parser.add_argument("--idle-sample-interval-seconds", type=float, default=15.0)
    parser.add_argument("--child-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve()
    lane_root = args.lane_root.expanduser().resolve()
    if not _RUN_TAG_RE.fullmatch(str(args.run_tag)):
        raise ValueError("run-tag must contain only letters, digits, dot, underscore, or dash")
    if int(args.order_blocks) < 1:
        raise ValueError("order-blocks must be at least one")
    if int(args.live_concurrency) not in args.concurrencies:
        raise ValueError("live-concurrency must appear in concurrencies")
    for path, label in (
        (args.model, "model"),
        (args.production_correctness_artifact, "production correctness artifact"),
        (args.compiler_version_file, "compiler version file"),
    ):
        if not path.expanduser().is_file():
            raise ValueError(f"{label} is unavailable: {path}")
    if not _git_clean(source_root):
        raise MatrixError(f"source root must be tracked/untracked clean: {source_root}")
    commit = _git_head(source_root)
    policies = tuple(args.queue_policies)
    schedule = counterbalanced_queue_blocks(policies, blocks=int(args.order_blocks))
    matrix_path = lane_root / "matrices" / f"{args.run_tag}.json"
    if matrix_path.exists() and not args.resume:
        raise MatrixError(f"matrix artifact already exists; use --resume: {matrix_path}")
    matrix: dict[str, Any]
    if matrix_path.exists():
        loaded = json.loads(matrix_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise MatrixError(f"matrix artifact root must be an object: {matrix_path}")
        matrix = loaded
    else:
        matrix = {
            "schema_version": 1,
            "kind": "gfx1151_f1_hardware_queue_matrix",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "planned" if args.dry_run else "running",
            "source_commit": commit,
            "source_root": str(source_root),
            "queue_policies": list(policies),
            "counterbalanced_blocks": [list(block) for block in schedule],
            "concurrencies": list(args.concurrencies),
            "suite": str(args.suite),
            "run_tag": str(args.run_tag),
            "runs": [],
        }
    if matrix.get("source_commit") != commit:
        raise MatrixError("resume matrix source commit does not match")
    existing_runs_by_spec = {
        str(row.get("spec_sha256")): row
        for row in matrix.get("runs", ())
        if isinstance(row, Mapping) and row.get("spec_sha256")
    }

    ordinal = 0
    for block_index, block in enumerate(schedule):
        for queue_policy in block:
            leaf = run_leaf(
                lane_root,
                commit=commit,
                profile="fp16-production",
                queue_policy=queue_policy,
                suite=str(args.suite),
                run_tag=str(args.run_tag),
                block_index=block_index,
            )
            port_base = int(args.port_base) + block_index * 1000 + ordinal * 50
            command = _build_child_command(
                args,
                queue_policy=queue_policy,
                leaf=leaf,
                port_base=port_base,
            )
            selected_environment = queue_environment(os.environ, queue_policy)
            selected_environment["HIPENGINE_HIP_ARCH"] = "gfx1151"
            selected_environment["HIPENGINE_GGUF_FP16_RECURRENT_STATE"] = "1"
            selected_environment.pop("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", None)
            spec_without_sha = {
                "schema_version": 1,
                "source_commit": commit,
                "queue_policy": queue_policy,
                "block_index": block_index,
                "ordinal": ordinal,
                "command": command,
                "command_shell": shlex.join(command),
                "selected_environment": {
                    key: selected_environment.get(key)
                    for key in (
                        "HIPENGINE_HIP_ARCH",
                        "HIPENGINE_GGUF_FP16_RECURRENT_STATE",
                        "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS",
                        "HIPENGINE_GPU_MAX_HW_QUEUES_POLICY",
                        "GPU_MAX_HW_QUEUES",
                    )
                },
                "model_sha256_sampled_by_child": True,
                "production_correctness_artifact": str(
                    args.production_correctness_artifact.expanduser().resolve()
                ),
                "production_correctness_artifact_sha256": _file_sha256(
                    args.production_correctness_artifact.expanduser().resolve()
                ),
                "compiler_version_file": str(args.compiler_version_file.expanduser().resolve()),
                "compiler_version_file_sha256": _file_sha256(
                    args.compiler_version_file.expanduser().resolve()
                ),
                "timeout_seconds": float(args.child_timeout_seconds),
            }
            spec_sha256 = _payload_sha256(spec_without_sha)
            spec = {**spec_without_sha, "spec_sha256": spec_sha256}
            resumed = resume_result(
                leaf,
                spec_sha256=spec_sha256,
                expected_widths=args.concurrencies,
            ) if args.resume else None
            if resumed is not None:
                result = resumed
                existing_run = existing_runs_by_spec.get(spec_sha256)
                preflight = (
                    list(existing_run.get("preflight", ()))
                    if isinstance(existing_run, Mapping)
                    else []
                )
                postflight = (
                    dict(existing_run.get("postflight", {}))
                    if isinstance(existing_run, Mapping)
                    else {}
                )
                returncode = None
                resumed_existing = True
            elif args.dry_run:
                leaf.mkdir(parents=True, exist_ok=True)
                _write_json(leaf / "spec.json", spec)
                ordinal += 1
                continue
            else:
                if leaf.exists() and any(leaf.iterdir()):
                    existing_spec = leaf / "spec.json"
                    if not existing_spec.is_file():
                        raise MatrixError(f"non-empty run leaf has no spec: {leaf}")
                    observed = json.loads(existing_spec.read_text(encoding="utf-8"))
                    if observed.get("spec_sha256") != spec_sha256:
                        raise MatrixError(f"run leaf spec mismatch: {leaf}")
                leaf.mkdir(parents=True, exist_ok=True)
                _write_json(leaf / "spec.json", spec)
                preflight = _wait_for_idle(
                    max_temperature_c=float(args.max_idle_temperature_c),
                    stable_samples=int(args.idle_stable_samples),
                    sample_interval_seconds=float(args.idle_sample_interval_seconds),
                )
                _write_json(leaf / "preflight.json", {"samples": preflight})
                returncode = _run_child(
                    command,
                    cwd=source_root,
                    environment=selected_environment,
                    log_path=leaf / "child.log",
                    timeout_seconds=float(args.child_timeout_seconds),
                )
                surviving_pids = _kfd_pids()
                if surviving_pids:
                    raise MatrixError(
                        f"KFD process survived child exit: {leaf}: {surviving_pids}"
                    )
                postflight = {
                    "observed_at": datetime.now(UTC).isoformat(),
                    "kfd_pids": [],
                    **_gpu_state(),
                }
                result_path = leaf / "result.json"
                if not result_path.is_file():
                    raise MatrixError(f"child produced no result artifact (rc={returncode}): {leaf}")
                loaded_result = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(loaded_result, dict):
                    raise MatrixError(f"child result root must be an object: {result_path}")
                result = loaded_result
                resumed_existing = False
            classification = classify_child_result(
                result,
                expected_widths=args.concurrencies,
            )
            result_sha256 = _file_sha256(leaf / "result.json")
            existing_run = existing_runs_by_spec.get(spec_sha256)
            if isinstance(existing_run, Mapping):
                if existing_run.get("result_sha256") != result_sha256:
                    raise MatrixError(f"resume result hash changed: {leaf}")
            else:
                run_record = {
                    "block_index": block_index,
                    "ordinal": ordinal,
                    "queue_policy": queue_policy,
                    "leaf": str(leaf),
                    "spec_sha256": spec_sha256,
                    "result_sha256": result_sha256,
                    "returncode": returncode,
                    "resumed": resumed_existing,
                    "preflight": preflight,
                    "postflight": postflight,
                    "classification": classification,
                }
                matrix["runs"].append(run_record)
                existing_runs_by_spec[spec_sha256] = run_record
            matrix["status"] = "running"
            _write_json(matrix_path, matrix)
            if not classification["safe_to_continue"]:
                matrix["status"] = "stopped_mechanical_failure"
                matrix["completed_at"] = datetime.now(UTC).isoformat()
                _write_json(matrix_path, matrix)
                raise MatrixError(
                    f"mechanical gate failed for hwq={queue_policy}: "
                    f"{classification['failures']}"
                )
            ordinal += 1
    if args.dry_run:
        matrix["planned_run_count"] = sum(len(block) for block in schedule)
        matrix["status"] = "planned"
    else:
        matrix["status"] = "complete"
        matrix["mechanical_passed"] = all(
            bool(run["classification"]["mechanical_passed"])
            for run in matrix["runs"]
        )
        matrix["all_product_gates_passed"] = all(
            bool(run["classification"]["product_gate_passed"])
            for run in matrix["runs"]
        )
        matrix["completed_at"] = datetime.now(UTC).isoformat()
    _write_json(matrix_path, matrix)
    return matrix


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_tag": result["run_tag"],
                "runs": len(result.get("runs", ())),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
