#!/usr/bin/env python3
"""Localize the per-request first-token cost of the GGUF server route.

The C1-C8 matrix times a whole wave with one boundary, so a fixed cost inside
``admission + prompt processing + first visible token`` never shows up in the
published rate. The wave decomposition
(``scripts/gguf_engine_submodule_decomposition.py``) showed this engine's
first-token cost stays near 305-335 ms per lane across C1-C8 while llama.cpp's
falls from 223 ms to 106 ms, i.e. this engine pays it per request instead of per
wave, and swapping in the cheapest measured admission would move it from losing
C2/C6-C8 to winning every cell.

This probe measures that submodule alone, on one engine, through the same
in-process server route the matrix uses, one sequential request at a time so the
request wall equals the wave's per-lane admission cost. ``--length-scan``
separates a fixed cost from prompt-processing throughput (flat across lengths
means padding or host work, proportional means real prefill cost), and
``--profile`` attributes the wall to host functions.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import io
import json
import pstats
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_matrix_path = REPO_ROOT / "scripts" / "llamacpp_c1c8_engine_matrix.py"
_spec = importlib.util.spec_from_file_location("matrix", _matrix_path)
matrix = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(matrix)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")


def synthetic(target_tokens: int) -> str:
    """ChatML prompt whose token count is close to ``target_tokens``."""

    words = max(1, int(target_tokens * 0.75))
    body = " ".join(f"alpha{index}" for index in range(words))
    return "<|im_start|>user\n" + body + "\n<|im_start|>assistant\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=matrix.DEFAULT_PROMPTS)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--execution-profile", default="production")
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--batch-window-ms", type=float, default=20.0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--length-scan",
        default="",
        help="comma-separated target prompt token counts, measured one sequential "
        "request each, to separate fixed cost from prefill throughput",
    )
    parser.add_argument(
        "--wave-widths",
        default="",
        help="comma-separated widths; times one barrier-synchronized wave per width of "
        "equal-length prompts, which is what tests packed multi-request prefill",
    )
    parser.add_argument("--resident-capacity", type=int, default=8)
    parser.add_argument("--wave-prompt-tokens", type=int, default=64)
    parser.add_argument("--wave-repeats", type=int, default=2)
    parser.add_argument(
        "--packed-prefill-max-rows",
        type=int,
        default=0,
        help="diagnostic override of the backend package GGUF_C2_PACKED_PREFILL_MAX_ROWS "
        "(0 leaves the package value); read before the model runner is built",
    )
    parser.add_argument("--profile", action="store_true", help="cProfile the measured requests")
    parser.add_argument(
        "--prof-out",
        type=Path,
        default=None,
        help="dump the raw cProfile stats here for offline pstats analysis",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from fastapi.testclient import TestClient

    from hipengine import LLM
    from hipengine.server.api import ServerConfig, create_app

    prompts = matrix.load_prompts(Path(args.prompts).resolve())
    if int(args.packed_prefill_max_rows) > 0:
        from importlib import import_module

        module = import_module(f"hipengine.kernels.{args.backend}")
        setattr(module, "GGUF_C2_PACKED_PREFILL_MAX_ROWS", int(args.packed_prefill_max_rows))
    llm = LLM(
        str(args.model),
        backend=args.backend,
        execution_profile=args.execution_profile,
        max_active_requests=int(args.resident_capacity),
        max_sequence_length=int(args.max_sequence_length),
    )
    llm.prepare(max_sequence_length=int(args.max_sequence_length))
    app = create_app(
        ServerConfig(
            model=str(args.model),
            backend=args.backend,
            served_model_name="probe",
            eager_load=False,
            max_context_tokens=int(args.max_sequence_length),
            max_active_requests=int(args.resident_capacity),
            generation_batch_window_ms=float(args.batch_window_ms),
            shutdown_grace_seconds=5.0,
        ),
        llm=llm,
    )

    def one(client: TestClient, prompt: str) -> dict:
        started = time.perf_counter()
        response = client.post(
            "/v1/completions",
            json={
                "model": "probe",
                "prompt": prompt,
                "max_tokens": int(args.max_tokens),
                "temperature": 0.0,
                "top_p": 1.0,
            },
        )
        ended = time.perf_counter()
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        body = response.json()
        usage = body.get("usage") or {}
        return {
            "wall_seconds": ended - started,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }
    with TestClient(app) as client:
        for target in sorted(
            {int(v) for v in str(args.length_scan).split(",") if v.strip()}
        ):
            row = one(client, synthetic(target))
            print(
                f"length {target}: wall={row['wall_seconds']:.4f}s "
                f"prompt={row['prompt_tokens']}",
                flush=True,
            )

        def wave(width: int, prompt: str) -> dict:
            """Time one barrier-synchronized wave, the same boundary the matrix uses."""

            barrier = threading.Barrier(width)

            def lane(_: int) -> tuple[float, float]:
                barrier.wait(timeout=180.0)
                started = time.perf_counter()
                row = one(client, prompt)
                return started, started + row["wall_seconds"]

            with ThreadPoolExecutor(max_workers=width) as pool:
                spans = list(pool.map(lane, range(width)))
            wall = max(end for _, end in spans) - min(start for start, _ in spans)
            return {"width": width, "wave_wall_seconds": wall}

        for index in range(int(args.warmup)):
            row = one(client, prompts[index % len(prompts)]["rendered"])
            print(f"warmup {index}: wall={row['wall_seconds']:.4f}s", flush=True)

        waves: list[dict] = []
        for width in sorted({int(v) for v in str(args.wave_widths).split(",") if v.strip()}):
            prompt = synthetic(int(args.wave_prompt_tokens))
            walls = [wave(width, prompt)["wave_wall_seconds"] for _ in range(int(args.wave_repeats))]
            waves.append(
                {
                    "width": width,
                    "wave_prompt_tokens": int(args.wave_prompt_tokens),
                    "walls": walls,
                    "wall_min_s": min(walls),
                    "wall_mean_s": sum(walls) / len(walls),
                }
            )
            print(
                f"wave C{width}: wall_min={min(walls):.4f}s mean={sum(walls) / len(walls):.4f}s",
                flush=True,
            )

        profiler = cProfile.Profile() if args.profile else None
        if profiler is not None:
            profiler.enable()
        rows = []
        for index in range(int(args.requests)):
            row = one(client, prompts[index % len(prompts)]["rendered"])
            rows.append(row)
            print(
                f"request {index}: wall={row['wall_seconds']:.4f}s "
                f"prompt={row['prompt_tokens']} completion={row['completion_tokens']}",
                flush=True,
            )
        if profiler is not None:
            profiler.disable()
            if args.prof_out is not None:
                args.prof_out.parent.mkdir(parents=True, exist_ok=True)
                profiler.dump_stats(str(args.prof_out))
        walls = [row["wall_seconds"] for row in rows]
        summary = {
            "model": str(args.model),
            "packed_prefill_max_rows_requested": int(args.packed_prefill_max_rows),
            "waves": waves,
            "backend": args.backend,
            "execution_profile": args.execution_profile,
            "max_tokens": int(args.max_tokens),
            "requests": len(rows),
            "prompt_tokens": rows[0]["prompt_tokens"],
            "wall_mean_s": sum(walls) / len(walls),
            "wall_min_s": min(walls),
            "wall_max_s": max(walls),
            "rows": rows,
        }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    if args.profile and profiler is not None:
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("tottime").print_stats(45)
        stats.sort_stats("cumulative").print_callers(18, "prefill_batch_native")
        summary["profile"] = stream.getvalue()
        print(stream.getvalue()[:8000])
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
