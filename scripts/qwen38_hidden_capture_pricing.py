#!/usr/bin/env python3
"""Price layer-output hidden capture against prefill wall time (XTX).

Runs matched prefill workloads with and without
`capture_layer_output_hidden` on a resident GGUF session and reports the
capture overhead in wall seconds and copied bytes. Diagnostic pricing only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession  # noqa: E402
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402

_CAPTURE_PREFILL_GDN_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"
_GDN_PREFILL_MODE_ENV = "HIPENGINE_GGUF_GDN_PREFILL_MODE"


def _timed_prefills(
    session: Qwen35GGUFResidentSession,
    prompt_ids: list[int],
    *,
    capture: bool,
    layer_ids: tuple[int, ...],
    repeats: int,
) -> dict[str, Any]:
    walls: list[float] = []
    captured_layer_count = 0
    for _ in range(repeats):
        session.reset()
        started = time.perf_counter()
        result = session.prefill(
            prompt_ids,
            use_bulk=True,
            bulk_attention_mode="bulk",
            return_logits=True,
            capture_layer_output_hidden=(list(layer_ids) if capture else None),
        )
        walls.append(time.perf_counter() - started)
        if capture:
            captured = getattr(session, "last_layer_output_hidden", None) or {}
            captured_layer_count = len(captured)
            assert captured_layer_count == len(layer_ids), (
                f"captured {captured_layer_count} layers, expected {len(layer_ids)}"
            )
        token = int(result.token_id)
    return {
        "wall_seconds": walls,
        "median_wall_seconds": statistics.median(walls),
        "final_token": token,
        "captured_layers": captured_layer_count if capture else 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    model_path = Path(args.model).resolve()
    compiler_version = Path(args.compiler_version_file).read_text(encoding="utf-8").strip()
    info = scan_gguf(model_path)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    prompt_ids = [int(t) for t in np.random.default_rng(20260907).integers(100, 150_000, size=args.prompt_tokens)]

    import os

    os.environ[_CAPTURE_PREFILL_GDN_ENV] = "1"
    os.environ[_GDN_PREFILL_MODE_ENV] = "exact"

    results: dict[str, Any] = {}
    with Qwen35GGUFResidentSession(
        model_path,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        backend=args.backend,
        max_sequence_length=args.max_sequence_length,
        use_wmma_prefill=False,
        use_gemv_decode=False,
    ) as session:
        layer_ids = tuple(range(len(session.runner.weights.config.layer_types)))
        hidden_bytes = int(session.runner.weights.config.hidden_size) * 4 * len(layer_ids)
        results["layer_count"] = len(layer_ids)
        results["capture_bytes_per_prefill_fp32"] = hidden_bytes
        results["plain"] = _timed_prefills(
            session, prompt_ids, capture=False, layer_ids=layer_ids, repeats=args.repeats
        )
        results["captured"] = _timed_prefills(
            session, prompt_ids, capture=True, layer_ids=layer_ids, repeats=args.repeats
        )

    plain_med = results["plain"]["median_wall_seconds"]
    cap_med = results["captured"]["median_wall_seconds"]
    results["overhead"] = {
        "median_overhead_seconds": cap_med - plain_med,
        "median_overhead_pct": (cap_med - plain_med) / plain_med * 100.0,
    }
    return {
        "schema": 1,
        "kind": "qwen38_hidden_capture_prefill_pricing",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "diagnostic",
        "performance_claim": False,
        "model": {"path": str(model_path), "size_bytes": model_path.stat().st_size},
        "workload": {
            "prompt_tokens": args.prompt_tokens,
            "repeats": args.repeats,
            "max_sequence_length": args.max_sequence_length,
        },
        "results": results,
        "note": "capture copies every layer's output hidden (fp32) for "
        "MTP/DFlash draft consumption; pricing captures the prefill-phase "
        "cost only, decode-step capture is priced separately by the MTP "
        "campaign",
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=4224)
    parser.add_argument("--compiler-version-file", type=Path, default=Path("/tmp/he-hipcc.txt"))
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
