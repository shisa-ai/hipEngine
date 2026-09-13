#!/usr/bin/env python3
"""Assemble the same-GGUF gfx1151 comparison tables from raw/*.json.

Reads the artifacts produced by run_comparison.sh, emits summary.json next to
this script, and prints the Markdown tables used in README.md.

Usage: python3 assemble.py [--check]
       --check  exit 1 if summary.json on disk differs from the recomputed one
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
SHAPES = ("512/128", "1K/128", "4K/128")
MODEL_SHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"

# (key, artifact, display name, protocol tier, source pin)
# DIAGNOSTIC_KEYS arms are reported but excluded from the "fastest" marking.
DIAGNOSTIC_KEYS = {"halobox-hip-blocking"}
ENGINES = (
    ("hipengine", "hipengine-production-ar.json", "hipEngine production AR",
     "resident sweep (explicit token ids)", "a2f62c881"),
    ("upstream-hip", "upstream-hip.json", "llama.cpp HIP",
     "llama-bench split timing", "002a12ad2 (build 10939)"),
    ("upstream-vulkan", "upstream-vulkan.json", "llama.cpp Vulkan",
     "llama-bench split timing", "37b3a9e0c (build 10940)"),
    ("halobox-hip", "halobox-hip.json", "strix-llama.cpp HIP",
     "llama-bench split timing", "654803517 (build 372)"),
    ("halobox-vulkan", "halobox-vulkan.json", "strix-llama.cpp Vulkan",
     "llama-bench split timing", "654803517 (build 372)"),
    ("halobox-hip-blocking", "halobox-hip-launch-blocking.json",
     "strix-llama.cpp HIP, HIP_LAUNCH_BLOCKING=1",
     "llama-bench split timing, diagnostic", "654803517 (build 372)"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hipengine(path: Path) -> dict:
    data = json.loads(path.read_text())
    rows = {}
    for shape, summary in data["summaries"].items():
        key = {"512": "512/128", "1024": "1K/128", "4096": "4K/128"}[shape]
        rows[key] = {
            "prefill_tok_s": summary["prefill_tok_s"]["median"],
            "decode_tok_s": summary["decode_tok_s"]["median"],
            "prefill_stdev_pct": summary["prefill_tok_s"]["stdev_pct_of_median"],
            "decode_stdev_pct": summary["decode_tok_s"]["stdev_pct_of_median"],
            "final_token_ids": summary.get("final_token_ids"),
            "tracked_peak_gib": summary.get("tracked_peak_allocated_gib", {}).get("median"),
        }
    return {
        "rows": rows,
        "pin": data["provenance"].get("hipengine_commit", "")[:9],
        "protocol": data["provenance"].get("timing_protocol"),
        "graph_gate_passed": all(g.get("passed") for g in data.get("graph_eager_gate", [])),
        "graph_gate_cases": len(data.get("graph_eager_gate", [])),
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def load_llamacpp(path: Path) -> dict:
    data = json.loads(path.read_text())
    rows = {}
    for row in data["rows"]:
        shape = "1K/128" if row["workload"] == "1K/128" else row["workload"]
        phases = {p["phase"]: p["llamacpp_record"] for p in data["phase_records"]
                  if p["workload"] == row["workload"]}
        prefill, decode = phases["prefill"], phases["decode"]
        rows[shape] = {
            "prefill_tok_s": _median(prefill["samples_ts"]),
            "decode_tok_s": _median(decode["samples_ts"]),
            "prefill_tok_s_mean": row["prefill_tok_s"],
            "decode_tok_s_mean": row["decode_tok_s"],
            "prefill_cv_percent": 100.0 * prefill["stddev_ts"] / prefill["avg_ts"],
            "decode_cv_percent": 100.0 * decode["stddev_ts"] / decode["avg_ts"],
            "peak_delta_gib": row.get("peak_delta_gib"),
        }
    return {
        "rows": rows,
        "pin": f"{data.get('build_commit')} (build {data.get('build_number')})",
        "binary": data.get("llama_bench_binary"),
        "status": data.get("status"),
    }


def build() -> dict:
    engines = {}
    for key, filename, label, tier, pin in ENGINES:
        path = RAW / filename
        if not path.exists():
            continue
        loaded = load_hipengine(path) if key == "hipengine" else load_llamacpp(path)
        engines[key] = {
            "label": label,
            "tier": tier,
            "pin": pin,
            "artifact": f"raw/{filename}",
            "artifact_sha256": sha256(path),
            **loaded,
        }
    reference = engines["hipengine"]["rows"]
    for key, engine in engines.items():
        if key == "hipengine":
            continue
        for shape, row in engine["rows"].items():
            base = reference.get(shape)
            if not base:
                continue
            row["prefill_vs_hipengine_percent"] = 100.0 * (row["prefill_tok_s"] / base["prefill_tok_s"] - 1.0)
            row["decode_vs_hipengine_percent"] = 100.0 * (row["decode_tok_s"] / base["decode_tok_s"] - 1.0)
    return {
        "schema": 1,
        "kind": "qwen38_27b_gfx1151_same_gguf_engine_comparison",
        "date": "2026-09-13",
        "status": "diagnostic_comparison",
        "performance_claim": False,
        "model": {
            "path": "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
            "sha256": MODEL_SHA256,
            "quant": "Q4_K_M",
            "kv_storage": "bf16",
        },
        "shapes": list(SHAPES),
        "engines": engines,
        "protocol_note": (
            "hipEngine rows come from its resident sweep with explicit token ids; "
            "llama.cpp-family rows come from llama-bench split prefill/decode timing. "
            "The two tiers are separate and are not interchangeable."
        ),
    }


def markdown(summary: dict) -> str:
    engines = summary["engines"]
    order = [k for k, *_ in ENGINES if k in engines]
    out = []
    for metric, unit in (("prefill_tok_s", "tok/s"), ("decode_tok_s", "tok/s")):
        title = "Prompt processing" if metric == "prefill_tok_s" else "Text generation"
        leaders = {
            shape: max(engines[k]["rows"][shape][metric] for k in order
                       if k not in DIAGNOSTIC_KEYS and shape in engines[k]["rows"])
            for shape in SHAPES
            if any(k not in DIAGNOSTIC_KEYS and shape in engines[k]["rows"] for k in order)
        }
        out.append(f"### {title} ({unit})\n")
        out.append("| Engine | Pin | 512/128 | 1K/128 | 4K/128 |")
        out.append("| --- | --- | ---: | ---: | ---: |")
        for key in order:
            engine = engines[key]
            cells = []
            for shape in SHAPES:
                row = engine["rows"].get(shape)
                if not row:
                    cells.append("—")
                    continue
                value = row[metric]
                text = f"{value:.3f}"
                if value == leaders.get(shape):
                    text = f"**{text}**"
                cells.append(text)
            out.append(f"| {engine['label']} | `{engine['pin']}` | " + " | ".join(cells) + " |")
        out.append("")
    out.append("### Versus hipEngine\n")
    out.append("| Engine | Prefill 512/128 | Prefill 1K/128 | Prefill 4K/128 | Decode 512/128 | Decode 1K/128 | Decode 4K/128 |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in order:
        engine = engines[key]
        if key == "hipengine":
            continue
        cells = []
        for shape in SHAPES:
            row = engine["rows"].get(shape)
            if not row or "prefill_vs_hipengine_percent" not in row:
                cells.append("—")
                continue
            cells.append(f"{row['prefill_vs_hipengine_percent']:+.2f}%")
        for shape in SHAPES:
            row = engine["rows"].get(shape)
            if not row or "decode_vs_hipengine_percent" not in row:
                cells.append("—")
                continue
            cells.append(f"{row['decode_vs_hipengine_percent']:+.2f}%")
        out.append(f"| {engine['label']} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify summary.json matches the recomputed summary")
    args = parser.parse_args()

    summary = build()
    rendered = json.dumps(summary, indent=1, sort_keys=False) + "\n"
    target = HERE / "summary.json"
    if args.check:
        if not target.exists() or target.read_text() != rendered:
            print("summary.json is out of date", file=sys.stderr)
            return 1
        print("summary.json is current")
        return 0
    target.write_text(rendered)
    print(markdown(summary))
    print(f"[wrote] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
