#!/usr/bin/env python3
"""Early scaffold smokes.

Default modes are CPU-only and safe before GPU clearance. ``smoke-add-hip`` is the explicit
GPU/JIT path for the first raw-pointer HIP smoke kernel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.dispatch.fusion import FusionPlanner, resolve_plan
from hipengine.kernels.registry import MissingKernelError
from hipengine.models import resolve_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("registry", "cpu-fixtures", "smoke-add-plan", "smoke-add-hip"),
        default="registry",
    )
    parser.add_argument("--n", type=int, default=1024, help="Element count for smoke-add-hip.")
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=None,
        help=(
            "Read the precomputed hipcc --version text from this file before building/loading "
            "smoke_add. Use under rocprofv3 to avoid spawning hipcc inside the profiler."
        ),
    )
    parser.add_argument(
        "--require-cached-build",
        action="store_true",
        help="Fail instead of invoking hipcc if the expected smoke_add cache artifact is absent.",
    )
    args = parser.parse_args()
    if args.mode == "registry":
        return registry_smoke()
    if args.mode == "cpu-fixtures":
        return cpu_fixture_smoke()
    if args.mode == "smoke-add-plan":
        return smoke_add_plan_smoke()
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = _read_compiler_version(args.compiler_version_file)
    return smoke_add_hip_smoke(
        args.n,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )


def registry_smoke() -> int:
    model = resolve_model("HipEngineToyForCausalLM")
    planner = FusionPlanner(backend="hip_gfx1100", quant="fp16")
    plan = planner.plan(model.layer_sequence())
    print("plan:", " -> ".join(step.layer for step in plan))
    try:
        resolve_plan(plan)
    except MissingKernelError as exc:
        print("expected missing kernel:", exc)
        return 0
    print("unexpected: plan resolved even though no kernels are registered")
    return 1


def cpu_fixture_smoke() -> int:
    from hipengine.kernels.cpu_reference import register_cpu_reference_kernels
    from hipengine.kernels.cpu_reference.fixtures import load_fixture, run_fixture

    register_cpu_reference_kernels()
    fixture_dir = Path("tests/fixtures/cpu_reference")
    failed = 0
    for path in sorted(fixture_dir.glob("*.json")):
        result = run_fixture(load_fixture(path))
        print(f"{'PASS' if result.passed else 'FAIL'} {path} max_abs={result.max_abs:.6g}")
        failed += 0 if result.passed else 1
    return 1 if failed else 0


def smoke_add_plan_smoke() -> int:
    from hipengine.kernels.hip_gfx1100.smoke import plan_smoke_add_build

    artifact = plan_smoke_add_build()
    print("family:", artifact.family)
    print("profile:", artifact.profile.name)
    print("output:", artifact.output_path)
    print("command:", " ".join(artifact.command))
    print("dry-run only: no hipcc invocation, no GPU access")
    return 0


def _read_compiler_version(path: Path) -> str:
    text = path.expanduser().read_text().strip()
    if not text:
        raise ValueError(f"compiler version file is empty: {path}")
    return text


def smoke_add_hip_smoke(
    n: int,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> int:
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add, smoke_add_f32

    if n < 1:
        raise ValueError("--n must be >= 1")

    a_host = np.arange(n, dtype=np.float32)
    b_host = np.arange(n, dtype=np.float32) * 2.0 + 1.0
    out_host = np.empty_like(a_host)

    runtime = get_hip_runtime()
    library = build_smoke_add(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    a_dev = b_dev = out_dev = None
    try:
        a_dev = malloc(a_host.nbytes, runtime=runtime)
        b_dev = malloc(b_host.nbytes, runtime=runtime)
        out_dev = malloc(out_host.nbytes, runtime=runtime)
        copy_host_to_device(a_dev, host_array_ptr(a_host), runtime=runtime)
        copy_host_to_device(b_dev, host_array_ptr(b_host), runtime=runtime)
        smoke_add_f32(a_dev.ptr, b_dev.ptr, out_dev.ptr, n, library=library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_host), out_dev, runtime=runtime)
    finally:
        for buffer in (out_dev, b_dev, a_dev):
            if buffer is not None:
                free(buffer, runtime=runtime)

    expected = a_host + b_host
    max_abs = float(np.max(np.abs(out_host - expected)))
    print(f"n={n} max_abs={max_abs}")
    print("first5=", out_host[:5].tolist())
    return 0 if np.allclose(out_host, expected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
