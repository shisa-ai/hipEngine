#!/usr/bin/env python3
"""Run a shared-token Qwen3.5-0.8B hipEngine/llama HIP/Vulkan comparison.

Each quant runs six fresh-process blocks.  Engine order rotates cyclically, so
all three engines occupy every order position twice.  Every child performs one
internal warmup and one measured exact-core and public-path p512/tg128 row.
Raw child JSON/logs remain outside the repository; the requested output is a
compact aggregate with hashes and complete sample arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = (
    ROOT / "benchmarks" / "fixtures" / "qwen35_08b_vulkan_parity_p512_t128.json"
)
DEFAULT_COMPILER_VERSION = Path("/tmp/d08-c0/hipcc-version.txt")
DEFAULT_MODELS = {
    "q4": Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"),
    "q8": Path("/models/gguf/Qwen3.5-0.8B-Q8_0.gguf"),
}
ENGINES = ("hipengine", "llamacpp_hip", "llamacpp_vulkan")
METRICS = (
    "prefill_tok_s",
    "decode_tok_s",
    "public_prefill_tok_s",
    "public_decode_tok_s",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def rotated_order(block: int) -> list[str]:
    shift = block % len(ENGINES)
    return list(ENGINES[shift:] + ENGINES[:shift])


def _digest_ids(values: list[int]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_hipengine(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        metric: float(payload[metric]["median"])
        for metric in METRICS
    }
    row.update(
        {
            "finite": bool(
                payload["timed_all_finite"]
                and payload["public_all_finite"]
                and payload["top1_all_finite"]
                and payload["public_top1_all_finite"]
            ),
            "deterministic": bool(
                payload["top1_repeat_exact"]
                and payload["public_top1_repeat_exact"]
            ),
            "core_top1_sha256": _digest_ids(payload["top1_ids"]),
            "public_top1_sha256": _digest_ids(payload["public_top1_ids"]),
            "core_graph_nodes": payload["core_graph_nodes"],
            "public_graph_nodes": payload["public_graph_nodes"],
            "memory": payload["memory"],
        }
    )
    return row


def normalize_llamacpp(payload: dict[str, Any]) -> dict[str, Any]:
    repetitions = int(payload["repetitions"])
    if repetitions != 1:
        raise ValueError("three-way children must use one measured repetition")
    prompt_tokens = int(payload["prompt_tokens"])
    forced_tokens = int(payload["forced_tokens"])
    timings = {
        "prefill_tok_s": prompt_tokens * 1000.0 / float(payload["prefill_ms"][0]),
        "decode_tok_s": forced_tokens * 1000.0 / float(payload["decode_ms"][0]),
        "public_prefill_tok_s": (
            prompt_tokens * 1000.0 / float(payload["public_prefill_ms"][0])
        ),
        "public_decode_tok_s": (
            forced_tokens * 1000.0 / float(payload["public_decode_ms"][0])
        ),
    }
    return {
        **timings,
        "finite": True,
        "deterministic": bool(
            payload["top1_deterministic"] and payload["public_top1_deterministic"]
        ),
        "core_top1_sha256": _digest_ids(payload["top1_ids"]),
        "public_top1_sha256": _digest_ids(payload["public_top1_ids"]),
        "reported_engine": payload["engine"],
    }


def summarize(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    by_engine = {
        engine: [
            next(row for row in block["execution"] if row["engine"] == engine)
            for block in blocks
        ]
        for engine in ENGINES
    }
    engines: dict[str, Any] = {}
    for engine, rows in by_engine.items():
        engines[engine] = {
            metric: {
                "samples": [float(row[metric]) for row in rows],
                "median": float(statistics.median(float(row[metric]) for row in rows)),
                "min": float(min(float(row[metric]) for row in rows)),
                "max": float(max(float(row[metric]) for row in rows)),
            }
            for metric in METRICS
        }
        engines[engine].update(
            {
                "all_finite": all(bool(row["finite"]) for row in rows),
                "all_deterministic": all(bool(row["deterministic"]) for row in rows),
                "core_top1_sha256": sorted(
                    {str(row["core_top1_sha256"]) for row in rows}
                ),
                "public_top1_sha256": sorted(
                    {str(row["public_top1_sha256"]) for row in rows}
                ),
            }
        )

    comparisons: dict[str, Any] = {}
    for peer in ENGINES[1:]:
        comparisons[peer] = {}
        for metric in METRICS:
            current = [float(row[metric]) for row in by_engine["hipengine"]]
            control = [float(row[metric]) for row in by_engine[peer]]
            current_median = float(statistics.median(current))
            control_median = float(statistics.median(control))
            comparisons[peer][metric] = {
                "hipengine_median": current_median,
                "peer_median": control_median,
                "hipengine_over_peer": current_median / control_median,
                "peer_over_hipengine": control_median / current_median,
                "hipengine_paired_wins": sum(
                    left > right
                    for left, right in zip(current, control, strict=True)
                ),
                "blocks": len(current),
            }

    core_digests = {
        digest
        for engine in ENGINES
        for digest in engines[engine]["core_top1_sha256"]
    }
    public_digests = {
        digest
        for engine in ENGINES
        for digest in engines[engine]["public_top1_sha256"]
    }
    return {
        "engines": engines,
        "comparisons": comparisons,
        "correctness": {
            "all_finite": all(engines[name]["all_finite"] for name in ENGINES),
            "all_deterministic": all(
                engines[name]["all_deterministic"] for name in ENGINES
            ),
            "cross_engine_core_top1_exact": len(core_digests) == 1,
            "cross_engine_public_top1_exact": len(public_digests) == 1,
            "core_top1_sha256": sorted(core_digests),
            "public_top1_sha256": sorted(public_digests),
        },
    }


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        self.prompt_id = int(self.fixture["prompt"]["token_ids_rle"][0][0])
        self.prompt_tokens = int(self.fixture["prompt"]["count"])
        self.teacher_id = int(
            self.fixture["teacher_forced_continuation"]["token_ids_rle"][0][0]
        )
        self.forced_tokens = int(self.fixture["teacher_forced_continuation"]["count"])

    def _run_process(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        stem: Path,
    ) -> tuple[subprocess.CompletedProcess[str], float]:
        started = time.perf_counter()
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
        stem.with_suffix(".stdout.log").write_text(process.stdout, encoding="utf-8")
        stem.with_suffix(".stderr.log").write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0:
            print(process.stdout[-2000:])
            print(process.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f"child failed after {elapsed:.1f}s: {command[0]}")
        return process, elapsed

    def hipengine_child(
        self,
        quant: str,
        model: Path,
        block: int,
        order_index: int,
    ) -> dict[str, Any]:
        artifact = self.args.raw_dir / f"{quant}-b{block:02d}-{order_index}-hipengine.json"
        env = os.environ.copy()
        env["HIPENGINE_HIP_ARCH"] = "gfx1151"
        env["HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING"] = "0"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "qwen35_08b_exact_core.py"),
            "--model",
            str(model),
            "--fixture",
            str(self.args.fixture),
            "--compiler-version-file",
            str(self.args.compiler_version_file),
            "--repetitions",
            "1",
            "--output",
            str(artifact),
        ]
        _process, elapsed = self._run_process(
            command, env=env, stem=artifact.with_suffix("")
        )
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        return {
            "engine": "hipengine",
            "order_index": order_index,
            "elapsed_seconds": elapsed,
            "raw_json": str(artifact),
            "raw_sha256": sha256(artifact),
            **normalize_hipengine(payload),
        }

    def llamacpp_child(
        self,
        engine: str,
        helper: Path,
        model: Path,
        quant: str,
        block: int,
        order_index: int,
    ) -> dict[str, Any]:
        stem = self.args.raw_dir / f"{quant}-b{block:02d}-{order_index}-{engine}"
        artifact = stem.with_suffix(".json")
        env = os.environ.copy()
        env["QWEN35_08B_ENGINE_LABEL"] = engine
        if engine == "llamacpp_vulkan":
            env["VK_ICD_FILENAMES"] = self.args.vulkan_icd
        command = [
            str(helper),
            str(model),
            str(self.prompt_id),
            str(self.prompt_tokens),
            str(self.teacher_id),
            str(self.forced_tokens),
            "1",
        ]
        process, elapsed = self._run_process(command, env=env, stem=stem)
        artifact.write_text(process.stdout, encoding="utf-8")
        payload = json.loads(process.stdout)
        return {
            "engine": engine,
            "order_index": order_index,
            "elapsed_seconds": elapsed,
            "raw_json": str(artifact),
            "raw_sha256": sha256(artifact),
            **normalize_llamacpp(payload),
        }

    def child(
        self,
        engine: str,
        quant: str,
        model: Path,
        block: int,
        order_index: int,
    ) -> dict[str, Any]:
        if engine == "hipengine":
            return self.hipengine_child(quant, model, block, order_index)
        helper = (
            self.args.hip_helper
            if engine == "llamacpp_hip"
            else self.args.vulkan_helper
        )
        return self.llamacpp_child(
            engine, helper, model, quant, block, order_index
        )


def _source_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "commit": git("rev-parse", "HEAD", cwd=path),
        "describe": git("describe", "--always", cwd=path),
        "tracked_clean": not bool(
            git("status", "--porcelain=v1", "--untracked-files=no", cwd=path)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hip-helper", type=Path, required=True)
    parser.add_argument("--vulkan-helper", type=Path, required=True)
    parser.add_argument("--hip-source", type=Path, required=True)
    parser.add_argument("--vulkan-source", type=Path, required=True)
    parser.add_argument("--q4-model", type=Path, default=DEFAULT_MODELS["q4"])
    parser.add_argument("--q8-model", type=Path, default=DEFAULT_MODELS["q8"])
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--compiler-version-file", type=Path, default=DEFAULT_COMPILER_VERSION
    )
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vulkan-icd",
        default="/usr/share/vulkan/icd.d/radeon_icd.json",
    )
    args = parser.parse_args()
    if args.blocks <= 0 or args.blocks % len(ENGINES):
        parser.error("--blocks must be a positive multiple of three")
    required = (
        args.hip_helper,
        args.vulkan_helper,
        args.q4_model,
        args.q8_model,
        args.fixture,
        args.compiler_version_file,
    )
    for path in required:
        if not path.exists():
            parser.error(f"required path does not exist: {path}")
    status = git("status", "--porcelain=v1")
    if status:
        parser.error("hipEngine tracked source must be clean before measurement")

    hip_source = _source_metadata(args.hip_source)
    vulkan_source = _source_metadata(args.vulkan_source)
    if hip_source["commit"] != vulkan_source["commit"]:
        parser.error("llama.cpp HIP and Vulkan helpers must use one source commit")
    if not hip_source["tracked_clean"] or not vulkan_source["tracked_clean"]:
        parser.error("llama.cpp HIP and Vulkan tracked source must be clean")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args)
    models = {"q4": args.q4_model, "q8": args.q8_model}
    quant_results: dict[str, Any] = {}
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for quant, model in models.items():
        blocks: list[dict[str, Any]] = []
        for block in range(args.blocks):
            order = rotated_order(block)
            execution: list[dict[str, Any]] = []
            for order_index, engine in enumerate(order):
                row = runner.child(engine, quant, model, block, order_index)
                execution.append(row)
                print(
                    quant,
                    block,
                    engine,
                    f"pp={row['prefill_tok_s']:.1f}",
                    f"tg={row['decode_tok_s']:.2f}",
                    f"public_tg={row['public_decode_tok_s']:.2f}",
                    flush=True,
                )
            blocks.append({"block": block, "order": order, "execution": execution})
        quant_results[quant] = {
            "model": str(model.resolve()),
            "model_sha256": sha256(model),
            "blocks": blocks,
            "summary": summarize(blocks),
        }

    payload = {
        "schema": 1,
        "task": "Qwen3.5-0.8B current-HEAD exact three-way HIP/Vulkan comparison",
        "status": "diagnostic",
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "host": platform.node(),
            "platform": platform.platform(),
            "arch": "gfx1151",
            "gpu": "AMD Radeon 8060S Graphics",
        },
        "hipengine": {
            "commit": git("rev-parse", "HEAD"),
            "describe": git("describe", "--always"),
            "tracked_clean": True,
        },
        "llamacpp": {
            "hip_source": hip_source,
            "vulkan_source": vulkan_source,
            "hip_helper": str(args.hip_helper.resolve()),
            "hip_helper_sha256": sha256(args.hip_helper),
            "vulkan_helper": str(args.vulkan_helper.resolve()),
            "vulkan_helper_sha256": sha256(args.vulkan_helper),
        },
        "protocol": {
            "fixture": str(args.fixture.resolve()),
            "fixture_sha256": sha256(args.fixture),
            "hipengine_driver": "scripts/qwen35_08b_exact_core.py",
            "hipengine_driver_sha256": sha256(
                ROOT / "scripts" / "qwen35_08b_exact_core.py"
            ),
            "parent_harness": str(Path(__file__).resolve()),
            "parent_harness_sha256": sha256(Path(__file__).resolve()),
            "llamacpp_helper_source": "benchmarks/llama.cpp/qwen35_08b_exact_core.cpp",
            "llamacpp_helper_source_sha256": sha256(
                ROOT / "benchmarks" / "llama.cpp" / "qwen35_08b_exact_core.cpp"
            ),
            "compiler_version_file": str(args.compiler_version_file.resolve()),
            "compiler_version_sha256": sha256(args.compiler_version_file),
            "blocks": args.blocks,
            "child_repetitions": 1,
            "order": (
                "cyclic counter-rotation; every engine occupies every order "
                "position equally"
            ),
            "scope": (
                "shared repeated-token p512/tg128 fixture; exact forced-token "
                "core plus native greedy public path"
            ),
            "vulkan_icd": args.vulkan_icd,
        },
        "quants": quant_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for quant, result in quant_results.items():
        print("SUMMARY", quant, json.dumps(result["summary"], indent=2), flush=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
