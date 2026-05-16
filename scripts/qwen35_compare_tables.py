#!/usr/bin/env python3
"""Print hardcoded Qwen3.5/PARO comparison tables.

The values here are intentionally static: they summarize the current retained
resident-runner hipENGINE diagnostics and the external comparison rows we use
for quick status checks.  They are not a benchmark runner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Row:
    workload: str
    prefill_tok_s: float | None
    decode_tok_s: float | None
    peak_gib: float | None


@dataclass(frozen=True)
class Baseline:
    key: str
    display: str
    source: str
    notes: str
    rows: tuple[Row, ...]


HIPENGINE_SOURCE = "benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json"
HIPENGINE_NOTES = (
    "hipENGINE rows are Qwen3.5-35B-A3B-PARO w4_paro resident-runner diagnostics "
    "with --attn-aotriton-min-tokens 512 --graph-replay-decode and parent-style chunk flags "
    "(linear/MoE/post/RoPE 1024, full-attn query 4096)."
)

HIPENGINE_ROWS: tuple[Row, ...] = (
    Row("512/128", 2216.487, 109.105, 18.581),
    Row("4K/128", 2504.959, 110.117, 19.875),
    Row("32K/128", 1886.344, 93.923, 20.688),
    Row("128K/128", 1002.409, 61.051, 23.656),
)

BASELINES: dict[str, Baseline] = {
    "nano-vllm-amd": Baseline(
        key="nano-vllm-amd",
        display="nano-vllm-amd parent",
        source="~/amd-gpu-tuning/docs/OPTIMAL.md Latest Results (2026-05-13)",
        notes="Qwen3.5-35B-A3B-PARO parent compact-WMMA + graph-replay rows, graph/step true.",
        rows=(
            Row("512/128", 2557.0, 115.7, 18.86),
            Row("4K/128", 2703.0, 112.0, 21.64),
            Row("32K/128", 1880.0, 98.8, 21.37),
            Row("128K/128", 914.0, 62.6, 27.42),
        ),
    ),
    "llama.cpp-hip": Baseline(
        key="llama.cpp-hip",
        display="llama.cpp HIP",
        source="~/amd-gpu-tuning/PLAN-LONGCONTEXT.md split rows",
        notes=(
            "Qwen3.6-35B-A3B UD-Q4_K_M GGUF, f16 KV, split pp/tg rows with decode depth. "
            "Memory was not retained for these split rows."
        ),
        rows=(
            Row("512/128", 2436.049, 85.487, None),
            Row("4K/128", 2176.905, 87.375, None),
            Row("32K/128", 1496.409, 76.994, None),
            Row("128K/128", 710.213, 57.341, None),
        ),
    ),
    "llama.cpp-vulkan": Baseline(
        key="llama.cpp-vulkan",
        display="llama.cpp Vulkan",
        source="~/amd-gpu-tuning/PLAN-LONGCONTEXT.md split rows",
        notes=(
            "Qwen3.6-35B-A3B UD-Q4_K_M GGUF, f16 KV, split pp/tg rows with decode depth. "
            "Memory was not retained for these split rows."
        ),
        rows=(
            Row("512/128", 1816.927, 127.515, None),
            Row("4K/128", 1705.093, 120.163, None),
            Row("32K/128", 1128.554, 98.073, None),
            Row("128K/128", 480.539, 64.478, None),
        ),
    ),
}

ALIASES = {
    "nano": "nano-vllm-amd",
    "nano-vllm": "nano-vllm-amd",
    "nano-vllm-amd": "nano-vllm-amd",
    "parent": "nano-vllm-amd",
    "llama.cpp hip": "llama.cpp-hip",
    "llama.cpp-hip": "llama.cpp-hip",
    "llamacpp-hip": "llama.cpp-hip",
    "hip": "llama.cpp-hip",
    "llama.cpp vulkan": "llama.cpp-vulkan",
    "llama.cpp-vulkan": "llama.cpp-vulkan",
    "llamacpp-vulkan": "llama.cpp-vulkan",
    "vulkan": "llama.cpp-vulkan",
}


def _normalize_baseline(value: str) -> str:
    key = " ".join(value.strip().lower().replace("_", "-").split())
    key = ALIASES.get(key, key)
    if key not in BASELINES and key != "all":
        valid = ", ".join(["nano-vllm-amd", "llama.cpp-hip", "llama.cpp-vulkan", "all"])
        raise SystemExit(f"unknown baseline {value!r}; choose one of: {valid}")
    return key


def _row_map(rows: Iterable[Row]) -> dict[str, Row]:
    return {row.workload: row for row in rows}


def _fmt_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _fmt_gib(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _fmt_pct(current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None or baseline == 0:
        return "—"
    return f"{(current / baseline - 1.0) * 100.0:+.1f}%"


def _fmt_gib_delta(current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None:
        return "—"
    return f"{current - baseline:+.2f} GiB"


def _print_table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    print(f"### {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" if idx == 0 else "---:" for idx, _ in enumerate(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def print_comparison(baseline: Baseline) -> None:
    hip = _row_map(HIPENGINE_ROWS)
    base = _row_map(baseline.rows)
    workloads = [row.workload for row in HIPENGINE_ROWS]

    print(f"## hipENGINE vs {baseline.display}\n")
    print(f"hipENGINE source: {HIPENGINE_SOURCE}")
    print(f"baseline source: {baseline.source}")
    print(f"notes: {HIPENGINE_NOTES} {baseline.notes}\n")

    _print_table(
        "Prefill",
        ("Workload", "hipENGINE tok/s", f"{baseline.display} tok/s", "Delta"),
        [
            (
                workload,
                _fmt_rate(hip[workload].prefill_tok_s),
                _fmt_rate(base[workload].prefill_tok_s),
                _fmt_pct(hip[workload].prefill_tok_s, base[workload].prefill_tok_s),
            )
            for workload in workloads
        ],
    )

    _print_table(
        "Decode",
        ("Workload", "hipENGINE tok/s", f"{baseline.display} tok/s", "Delta"),
        [
            (
                workload,
                _fmt_rate(hip[workload].decode_tok_s),
                _fmt_rate(base[workload].decode_tok_s),
                _fmt_pct(hip[workload].decode_tok_s, base[workload].decode_tok_s),
            )
            for workload in workloads
        ],
    )

    _print_table(
        "Memory / peak GiB",
        ("Workload", "hipENGINE tracked peak GiB", f"{baseline.display} peak GiB", "Delta"),
        [
            (
                workload,
                _fmt_gib(hip[workload].peak_gib),
                _fmt_gib(base[workload].peak_gib),
                _fmt_gib_delta(hip[workload].peak_gib, base[workload].peak_gib),
            )
            for workload in workloads
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baseline",
        nargs="?",
        default="nano-vllm-amd",
        help=(
            "Comparison baseline: nano-vllm-amd, llama.cpp-hip, llama.cpp-vulkan, or all. "
            "Aliases with spaces such as 'llama.cpp HIP' are accepted."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key = _normalize_baseline(args.baseline)
    if key == "all":
        for index, baseline in enumerate(BASELINES.values()):
            if index:
                print("---\n")
            print_comparison(baseline)
    else:
        print_comparison(BASELINES[key])


if __name__ == "__main__":
    main()
