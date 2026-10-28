#!/usr/bin/env python3
"""Prepare an isolated JIT cache and profile one direct GGUF owner child.

The workflow has three explicit process boundaries:

1. an unprofiled build child populates a new scoped cache;
2. an unprofiled cache-only warm child proves the same workload can start with
   ``HIPENGINE_REQUIRE_CACHED_BUILD=1`` and a compiler guard; and
3. optionally, rocprofv3 wraps only the final direct owner child.

Cache trees are content/mode/mtime hashed before and after every cache-only
stage.  The compiler guard and descendant-process monitor make compiler activity
a mechanical failure rather than a log-review convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_model_identity  # noqa: E402
from hipengine.benchmark.rocprof_cache import (  # noqa: E402
    RocprofCacheError,
    _cache_only_environment,
    _default_roctx_sdk,
    _file_sha256,
    _prepare_roctx_override,
    _run_monitored,
    _write_json,
    cache_build_manifest_summary,
    prepare_compiler_guard,
    snapshot_cache_tree,
    validate_cache_only_stage,
)

_RUN_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def profile_command(
    *,
    rocprofv3: str,
    trace_dir: Path,
    child_command: Sequence[str],
) -> list[str]:
    return [
        str(rocprofv3),
        "--kernel-trace",
        "--marker-trace",
        "--hip-runtime-trace",
        "--memory-copy-trace",
        "--output-format",
        "csv",
        "-d",
        str(trace_dir),
        "--",
        *[str(value) for value in child_command],
    ]


def _capture(command: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _git_value(source_root: Path, *arguments: str) -> str:
    result = _capture(["git", *arguments], cwd=source_root)
    if result.returncode != 0:
        raise RocprofCacheError(f"git {' '.join(arguments)} failed:\n{result.stdout}")
    return result.stdout.strip()


def _git_clean(source_root: Path) -> bool:
    return not _git_value(source_root, "status", "--porcelain=v1")


def _child_command(
    args: argparse.Namespace,
    *,
    output: Path,
    profile: bool,
    require_cached: bool,
) -> list[str]:
    command = [
        str(args.python),
        "scripts/gguf_continuous_owner_profile_child.py",
        "--model",
        str(args.model),
        "--backend",
        str(args.backend),
        "--quant",
        str(args.quant),
        "--concurrency",
        str(args.concurrency),
        "--prompt-length",
        str(args.prompt_length),
        "--decode-tokens",
        str(args.decode_tokens),
        "--prompt-token-id",
        str(args.prompt_token_id),
        "--marker-index",
        str(args.marker_index),
        "--compiler-version-file",
        str(args.compiler_version_file),
        "--cache-root",
        str(args.cache_root),
        "--out",
        str(output),
    ]
    if profile:
        command.append("--profile")
    if require_cached:
        command.append("--require-cached-build")
    return command


def _queue_environment(environment: Mapping[str, str], queue_policy: str) -> dict[str, str]:
    selected = {str(key): str(value) for key, value in environment.items()}
    selected["GPU_MAX_HW_QUEUES"] = str(int(queue_policy))
    selected.pop("HIPENGINE_GPU_MAX_HW_QUEUES_POLICY", None)
    return selected


def _trace_queue_observation(trace_dir: Path) -> dict[str, Any]:
    queue_ids: set[int] = set()
    files: list[str] = []
    for path in sorted(trace_dir.rglob("*_kernel_trace.csv")):
        files.append(str(path))
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw = next(
                    (
                        row.get(name)
                        for name in ("Queue_Id", "QueueId", "Queue_ID", "queue_id")
                        if row.get(name) not in (None, "")
                    ),
                    None,
                )
                if raw is not None:
                    queue_ids.add(int(raw))
    return {
        "kernel_trace_files": files,
        "runtime_queue_ids": sorted(queue_ids) if queue_ids else None,
        "runtime_queue_count": len(queue_ids) if queue_ids else None,
        "observed": bool(queue_ids),
        "reason": (
            None
            if queue_ids
            else "kernel trace did not expose a recognized queue-ID column"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--marker-index", type=int, default=3)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--gpu-max-hw-queues", choices=("1", "2", "4", "8"), default="2")
    parser.add_argument(
        "--fp16-recurrent-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--rocprofv3", default="rocprofv3")
    parser.add_argument("--roctx-sdk", type=Path, default=_default_roctx_sdk())
    parser.add_argument("--child-timeout-seconds", type=float, default=2700.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve()
    model = args.model.expanduser().resolve()
    compiler_file = args.compiler_version_file.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve() / str(args.run_tag)
    output = args.out.expanduser().resolve()
    args.model = model
    args.compiler_version_file = compiler_file
    args.cache_root = cache_root
    args.python = args.python.expanduser().resolve()
    if not _RUN_TAG_RE.fullmatch(str(args.run_tag)):
        raise ValueError("run-tag must contain only letters, digits, dot, underscore, or dash")
    if not model.is_file() or not compiler_file.is_file():
        raise ValueError("model and compiler-version file must exist")
    if not _git_clean(source_root):
        raise RocprofCacheError(f"source root must be clean: {source_root}")
    source_commit = _git_value(source_root, "rev-parse", "HEAD")
    source_tree = _git_value(source_root, "rev-parse", "HEAD^{tree}")
    if run_root.exists() and any(run_root.iterdir()):
        raise RocprofCacheError(f"run root must be new and empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    if args.rebuild:
        if cache_root.exists() and any(cache_root.iterdir()):
            raise RocprofCacheError(
                "--rebuild requires a new or empty isolated cache root; refusing deletion"
            )
        cache_root.mkdir(parents=True, exist_ok=True)
    elif not cache_root.is_dir() or not any(cache_root.iterdir()):
        raise RocprofCacheError("cache root must be populated unless --rebuild is set")

    environment = _queue_environment(os.environ, str(args.gpu_max_hw_queues))
    environment["HIPENGINE_HIP_ARCH"] = "gfx1151"
    environment["HIPENGINE_COMPILER_VERSION_FILE"] = str(compiler_file)
    environment["HIPENGINE_BUILD_CACHE_ROOT"] = str(cache_root)
    environment["HIPENGINE_GGUF_FP16_RECURRENT_STATE"] = (
        "1" if bool(args.fp16_recurrent_state) else "0"
    )
    environment.pop("HIPENGINE_REQUIRE_CACHED_BUILD", None)

    stages: dict[str, Any] = {}
    if args.rebuild:
        build_command = _child_command(
            args,
            output=run_root / "build-child.json",
            profile=False,
            require_cached=False,
        )
        build_stage = _run_monitored(
            build_command,
            cwd=source_root,
            environment=environment,
            stdout_path=run_root / "build.stdout.log",
            stderr_path=run_root / "build.stderr.log",
            timeout_seconds=float(args.child_timeout_seconds),
        )
        if build_stage["returncode"] != 0:
            raise RocprofCacheError(f"unprofiled build child failed: {build_stage}")
        stages["build"] = {"command": build_command, **build_stage}

    cache_after_build = snapshot_cache_tree(cache_root)
    if cache_after_build["file_count"] < 1:
        raise RocprofCacheError("isolated cache contains no files after build")
    build_manifests = cache_build_manifest_summary(cache_root)
    if not build_manifests:
        raise RocprofCacheError("isolated cache contains no build manifests")
    guard = prepare_compiler_guard(run_root / "compiler-guard")
    cache_environment = _cache_only_environment(environment, guard=guard)

    warm_command = _child_command(
        args,
        output=run_root / "cache-warm-child.json",
        profile=False,
        require_cached=True,
    )
    warm_stage = _run_monitored(
        warm_command,
        cwd=source_root,
        environment=cache_environment,
        stdout_path=run_root / "cache-warm.stdout.log",
        stderr_path=run_root / "cache-warm.stderr.log",
        timeout_seconds=float(args.child_timeout_seconds),
    )
    cache_after_warm = snapshot_cache_tree(cache_root)
    validate_cache_only_stage(
        before=cache_after_build,
        after=cache_after_warm,
        compiler_guard_marker=guard["marker"],
        observed_compiler_processes=warm_stage["observed_compiler_processes"],
    )
    if warm_stage["returncode"] != 0:
        raise RocprofCacheError(f"cache-only warm child failed: {warm_stage}")
    stages["cache_warm"] = {"command": warm_command, **warm_stage}

    queue_observation = {
        "runtime_queue_ids": None,
        "runtime_queue_count": None,
        "observed": False,
        "reason": "prepare-only; task #18 profile supplies queue trace",
    }
    if args.profile:
        roctx_override, dependencies = _prepare_roctx_override(args.roctx_sdk, run_root)
        profile_environment = dict(cache_environment)
        ld_prefix = os.pathsep.join(
            [str(roctx_override), *(str(path) for path in dependencies)]
        )
        profile_environment["LD_LIBRARY_PATH"] = (
            ld_prefix + os.pathsep + profile_environment.get("LD_LIBRARY_PATH", "")
        )
        trace_dir = run_root / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        final_child = _child_command(
            args,
            output=run_root / "profile-child.json",
            profile=True,
            require_cached=True,
        )
        command = profile_command(
            rocprofv3=str(args.rocprofv3),
            trace_dir=trace_dir,
            child_command=final_child,
        )
        profile_stage = _run_monitored(
            command,
            cwd=source_root,
            environment=profile_environment,
            stdout_path=run_root / "profile.stdout.log",
            stderr_path=run_root / "profile.stderr.log",
            timeout_seconds=float(args.child_timeout_seconds),
        )
        cache_after_profile = snapshot_cache_tree(cache_root)
        validate_cache_only_stage(
            before=cache_after_warm,
            after=cache_after_profile,
            compiler_guard_marker=guard["marker"],
            observed_compiler_processes=profile_stage["observed_compiler_processes"],
        )
        if profile_stage["returncode"] != 0:
            raise RocprofCacheError(f"profile child failed: {profile_stage}")
        child_payload = json.loads((run_root / "profile-child.json").read_text(encoding="utf-8"))
        if child_payload.get("profile") is not True or child_payload.get(
            "require_cached_build"
        ) is not True:
            raise RocprofCacheError("profile child did not report profile+cache-only mode")
        stages["profile"] = {"command": command, **profile_stage}
        queue_observation = _trace_queue_observation(trace_dir)
    else:
        cache_after_profile = cache_after_warm

    artifact = {
        "schema_version": 1,
        "kind": "gguf_continuous_owner_rocprof_cache_workflow",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "profiled" if args.profile else "prepared_cache_only",
        "source": {
            "root": str(source_root),
            "commit": source_commit,
            "tree": source_tree,
            "clean": True,
        },
        "model": collect_model_identity(model),
        "backend": str(args.backend),
        "quant": str(args.quant),
        "concurrency": int(args.concurrency),
        "queue_policy": str(args.gpu_max_hw_queues),
        "fp16_recurrent_state": bool(args.fp16_recurrent_state),
        "compiler_version_file": str(compiler_file),
        "compiler_version_file_sha256": _file_sha256(compiler_file),
        "cache_root": str(cache_root),
        "cache": {
            "build_manifests": build_manifests,
            "after_build": cache_after_build,
            "after_cache_warm": cache_after_warm,
            "after_profile": cache_after_profile,
            "immutable_cache_only": (
                cache_after_build["tree_sha256"]
                == cache_after_warm["tree_sha256"]
                == cache_after_profile["tree_sha256"]
            ),
        },
        "compiler_guard": {
            "marker": str(guard["marker"]),
            "invoked": guard["marker"].exists(),
        },
        "stages": stages,
        "runtime_queue_observation": queue_observation,
        "profile_wraps_direct_child_only": True,
    }
    _write_json(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = run(args)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "cache_tree_sha256": artifact["cache"]["after_build"]["tree_sha256"],
                "runtime_queue_count": artifact["runtime_queue_observation"][
                    "runtime_queue_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
