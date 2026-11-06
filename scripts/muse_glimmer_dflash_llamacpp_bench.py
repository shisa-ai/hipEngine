#!/usr/bin/env python3
"""llama.cpp Muse Glimmer draft-dflash mtpbench baseline (Gate B target).

External diagnostic, outside hipEngine's runtime path. Starts a local
llama-server with the Glimmer target + mmproj + dflash draft
(``--spec-type draft-dflash``), runs the natural-prompt mtpbench suite at
temperature=0, and writes one compact JSON artifact. The decode rate uses the
canonical cross-engine boundary (docs/campaigns/MTP-LLAMACPP-PARITY.md):
request N+1 outputs, report ``sum(predicted_n - 1) * 1000 / sum(predicted_ms)``
timed transitions. A ``base`` (no-spec) mode gives the same-suite true AR
control. This is the target hipEngine DFlash must beat (campaign MUSE-GLIMMER.md
Gate B); it is not a hipEngine performance claim.

Reuses the proven natural-protocol + summary helpers from llamacpp_mtp_bench so
the timing boundary is identical to the retained Qwen MTP diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llamacpp_mtp_bench as base  # noqa: E402
from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402

DEFAULT_DIR = Path("/models/gguf/Muse-Glimmer-30B-GGUF")
DEFAULT_SERVER_BIN = Path.home() / "llama.cpp/llama.cpp-hip/build/bin/llama-server"
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"


def _server_command(args: argparse.Namespace, mode: str) -> list[str]:
    cmd = [
        str(args.server_bin),
        "--jinja",
        "-m", str(args.model),
        "--mmproj", str(args.mmproj),
        "-ngl", str(args.gpu_layers),
        "-fa", args.flash_attn,
        "-ctk", args.cache_type_k,
        "-ctv", args.cache_type_v,
        "-c", str(args.ctx_size),
        "-np", "1",
        "--host", args.host,
        "--port", str(args.port),
        "--alias", args.alias,
        "--no-cache-prompt",
    ]
    if mode == "dflash":
        cmd += ["--spec-type", "draft-dflash", "--spec-draft-model", str(args.draft)]
    cmd += args.server_extra_arg
    return cmd


def _run_mode(args: argparse.Namespace, mode: str, logs_dir: Path) -> dict[str, Any]:
    if args.no_spawn:
        # Client-only: benchmark an already-running server (e.g. the production
        # Glimmer draft-dflash instance) without starting or touching a process.
        return {
            "server_command": f"<existing server at {args.host}:{args.port}>",
            "server_log": None,
            "no_spawn": True,
            "natural": base._run_natural(args),
        }
    log_path = logs_dir / f"server-{mode}.log"
    command = _server_command(args, mode)
    payload: dict[str, Any] = {"server_command": command, "server_log": str(log_path)}
    proc: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("wb") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        base._wait_for_health(args.host, args.port, args.server_start_timeout)
        payload["natural"] = base._run_natural(args)
    finally:
        if proc is not None:
            base._terminate(proc)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN)
    parser.add_argument("--model", type=Path, default=DEFAULT_DIR / "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
    parser.add_argument("--mmproj", type=Path, default=DEFAULT_DIR / "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf")
    parser.add_argument("--draft", type=Path, default=DEFAULT_DIR / "dflash-Muse-Glimmer-30B-Q4_K_M.gguf")
    parser.add_argument("--alias", default="muse-glimmer-30b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8016)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--flash-attn", default="1")
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--mode", choices=("base", "dflash", "both"), default="both")
    parser.add_argument("--no-spawn", action="store_true",
                        help="Benchmark an existing server at --host/--port instead of starting one. "
                             "Forces --mode dflash (the caller asserts the server's spec config).")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--server-start-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--server-extra-arg", action="append", default=[])
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.no_spawn:
        args.mode = "dflash"
    else:
        for path in (args.model, args.mmproj, args.draft, args.server_bin):
            if not Path(path).exists():
                parser.error(f"required artifact not found: {path}")

    modes = ["base", "dflash"] if args.mode == "both" else [args.mode]
    logs_dir = args.log_dir or (args.output.with_suffix("").parent / (args.output.with_suffix("").name + "-logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, Any] = {}
    for mode in modes:
        print(f"=== llama-server mode={mode} ===", flush=True)
        t0 = time.perf_counter()
        runs[mode] = _run_mode(args, mode, logs_dir)
        summary = runs[mode]["natural"]["summary"]
        rate = summary.get("aggregate_decode_transition_per_second")
        acc = summary.get("draft_acceptance")
        print(f"  {mode}: decode {rate:.3f} tok/s (transition-normalized), draft_acceptance={acc}, {time.perf_counter()-t0:.1f}s", flush=True)

    prov = collect_artifact_provenance(
        repo_root=REPO_ROOT, configured_backend="llamacpp_hip", resolved_backend="hip_gfx1151",
        target_arch="gfx1151", model_path=str(args.model), quant="gguf_q4_k_m", kv_dtype="f16/f16",
        command=[str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]],
        timing_protocol="llama.cpp server predicted_n/predicted_ms; N+1 outputs, N timed transitions",
        repetitions=1,
    )
    artifact: dict[str, Any] = {
        "schema": 1,
        "kind": "muse_glimmer_llamacpp_dflash_mtpbench_baseline",
        "status": "diagnostic_retained",
        "performance_claim": False,
        "date": time.strftime("%Y-%m-%d"),
        "hardware": "Radeon 8060S / gfx1151 (Ryzen AI MAX+ PRO 395, 128 GiB unified)",
        "model_path": str(args.model),
        "mmproj_path": str(args.mmproj),
        "draft_path": str(args.draft),
        "prompt_suite": str(args.prompts),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "provenance": prov,
        "llama_cpp_commit": base._git_rev_parse(Path(args.server_bin).resolve().parents[2]),
        "runs": runs,
        "notes": [
            "Gate B target for hipEngine Muse Glimmer DFlash (campaign MUSE-GLIMMER.md).",
            "External llama.cpp diagnostic; no hipEngine correctness gate is implied.",
            "Decode rate is the cross-engine transition-normalized boundary: "
            "sum(predicted_n - 1) * 1000 / sum(predicted_ms).",
            "base mode is the same-suite true no-spec AR control.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())