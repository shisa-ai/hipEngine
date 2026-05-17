#!/usr/bin/env python3
"""Print hardcoded Qwen/PARO comparison tables.

The values here are intentionally static: they summarize the current retained
resident-runner hipEngine diagnostics and the external comparison rows we use
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
class Series:
    key: str
    display: str
    source: str
    notes: str
    rows: tuple[Row, ...]


QWEN35_SOURCE = "benchmarks/results/2026-05-17-hipengine-qwen35-d31-d33-grouped-gqa-long-context-diagnostic.json"
SHISA_PACKED_SOURCE = "benchmarks/results/2026-05-18-hipengine-gfx1100-shisa-qwen36-packed-gt1k-default-diagnostic.json"
SHISA_LEGACY_SOURCE = "benchmarks/results/2026-05-17-hipengine-qwen36-shisa-packed-vs-legacy-refresh-diagnostic.json"
SHISA_GFX1151_SOURCE = "benchmarks/results/2026-05-17-hipengine-gfx1151-shisa-qwen36-packed-chunk256-sweep-diagnostic.json"
LLAMACPP_GFX1151_SOURCE = "benchmarks/results/2026-05-17-llamacpp-upstream-gfx1151-qwen36-gguf-rerun-diagnostic.json"

TARGETS: dict[str, Series] = {
    "qwen35-current": Series(
        key="qwen35-current",
        display="hipEngine Qwen3.5 current",
        source=QWEN35_SOURCE,
        notes=(
            "Qwen3.5-35B-A3B-PARO w4_paro resident-runner diagnostic with current defaults: "
            "AOTriton prefill threshold 512, graph-replay decode, Marlin-K decode, and D3.1-D3.3 "
            "grouped-GQA long-context decode. Long rows use parent-style chunk flags."
        ),
        rows=(
            Row("512/128", 2177.649, 115.627, 18.176),
            Row("4K/128", 2449.055, 116.263, 20.047),
            Row("32K/128", 1964.345, 99.560, 20.320),
            Row("128K/128", 1015.761, 63.368, 23.288),
        ),
    ),
    "shisa-packed": Series(
        key="shisa-packed",
        display="hipEngine shisa Qwen3.6 packed PARO",
        source=SHISA_PACKED_SOURCE,
        notes=(
            "shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed checkpoint forced to "
            "shared_expert_format=packed_paro_w4 on W7900/gfx1100; packed is the default A-side "
            "for shisa comparisons. The current default chunk policy keeps 512 unchunked and "
            "uses 1024/1024/4096/1024/1024 chunks for prompts above 1K."
        ),
        rows=(
            Row("512/128", 2500.565, 111.516, 18.123),
            Row("4K/128", 2899.685, 113.094, 19.455),
            Row("32K/128", 2115.050, 97.594, 20.267),
            Row("128K/128", 1054.291, 62.027, 23.235),
        ),
    ),
    "shisa-packed-gfx1151": Series(
        key="shisa-packed-gfx1151",
        display="hipEngine shisa Qwen3.6 packed PARO (gfx1151)",
        source=SHISA_GFX1151_SOURCE,
        notes=(
            "shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed on Strix Halo/Radeon 8060S, "
            "hip_gfx1151, graph-replay decode, AOTriton threshold 512, and 256-row prefill "
            "chunks for linear/MoE/full-attn surfaces. Diagnostic single-run; no shisa KL/top-1 gate yet."
        ),
        rows=(
            Row("512/128", 983.206, 62.060, 17.997),
            Row("4K/128", 1029.402, 63.605, 18.097),
            Row("32K/128", 792.296, 50.629, 18.909),
            Row("128K/128", 413.489, 30.245, 21.877),
            Row("4K/4K", 1001.266, 62.438, 18.210),
        ),
    ),
    "shisa-legacy": Series(
        key="shisa-legacy",
        display="hipEngine shisa Qwen3.6 legacy shared expert",
        source=SHISA_LEGACY_SOURCE,
        notes=(
            "same shisa unstripped checkpoint forced to shared_expert_format=legacy_fp16. "
            "Use --target shisa-legacy to make legacy the A-side, or --against-target with no value "
            "to compare packed A against legacy B."
        ),
        rows=(
            Row("512/128", 2272.088, 115.324, 18.176),
            Row("4K/128", 2487.298, 116.688, 20.047),
            Row("32K/128", 1974.833, 99.746, 20.320),
            Row("128K/128", 1002.841, 63.190, 23.288),
        ),
    ),
}

BASELINES: dict[str, Series] = {
    "nano-vllm-amd": Series(
        key="nano-vllm-amd",
        display="nano-vllm-amd parent",
        source="~/amd-gpu-tuning/docs/OPTIMAL.md Latest Results plus local 2026-05-13 reruns",
        notes="Qwen3.5-35B-A3B-PARO parent compact-WMMA + graph-replay rows, graph/step true.",
        rows=(
            Row("512/128", 2696.4, 116.05, 18.80),
            Row("4K/128", 2741.5, 113.05, 21.64),
            Row("32K/128", 1880.0, 98.8, 21.37),
            Row("128K/128", 914.0, 62.6, 27.42),
        ),
    ),
    "llama.cpp-hip": Series(
        key="llama.cpp-hip",
        display="llama.cpp HIP",
        source="~/amd-gpu-tuning/PLAN-LONGCONTEXT.md split rows",
        notes=(
            "Qwen3.6-35B-A3B UD-Q4_K_M GGUF, f16 KV, split pp/tg rows with decode depth. "
            "Peak GiB from benchmarks/results/2026-05-17-llamacpp-hip-qwen36-peak.json."
        ),
        rows=(
            Row("512/128", 2436.049, 85.487, 21.125),
            Row("4K/128", 2176.905, 87.375, 21.197),
            Row("32K/128", 1496.409, 76.994, 21.738),
            Row("128K/128", 710.213, 57.341, 23.605),
        ),
    ),
    "llama.cpp-vulkan": Series(
        key="llama.cpp-vulkan",
        display="llama.cpp Vulkan",
        source="~/amd-gpu-tuning/PLAN-LONGCONTEXT.md split rows",
        notes=(
            "Qwen3.6-35B-A3B UD-Q4_K_M GGUF, f16 KV, split pp/tg rows with decode depth. "
            "Peak GiB from benchmarks/results/2026-05-17-llamacpp-vulkan-qwen36-peak.json."
        ),
        rows=(
            Row("512/128", 1816.927, 127.515, 20.844),
            Row("4K/128", 1705.093, 120.163, 20.969),
            Row("32K/128", 1128.554, 98.073, 21.533),
            Row("128K/128", 480.539, 64.478, 23.596),
        ),
    ),
    "llama.cpp-hip-gfx1151": Series(
        key="llama.cpp-hip-gfx1151",
        display="llama.cpp HIP upstream (gfx1151)",
        source=LLAMACPP_GFX1151_SOURCE,
        notes=(
            "Qwen3.6-35B-A3B UD-Q4_K_M GGUF with upstream llama.cpp-hip "
            "build-gfx1151-unroll600, f16 KV, flash attention, split pp/tg rows. "
            "Memory table is intentionally omitted for gfx1151 comparisons because sysfs/rocm-smi "
            "report only the Strix Halo 512 MiB aperture."
        ),
        rows=(
            Row("512/128", 1058.738, 50.537, None),
            Row("4K/128", 1004.220, 49.379, None),
            Row("32K/128", 735.534, 43.435, None),
            Row("128K/128", 376.070, 31.286, None),
            Row("4K/4K", 990.726, 49.071, None),
        ),
    ),
    "llama.cpp-vulkan-gfx1151": Series(
        key="llama.cpp-vulkan-gfx1151",
        display="llama.cpp Vulkan upstream (gfx1151)",
        source=LLAMACPP_GFX1151_SOURCE,
        notes=(
            "Qwen3.6-35B-A3B UD-Q4_K_M GGUF with upstream llama.cpp-vulkan build, "
            "f16 KV, flash attention, split pp/tg rows. Memory table is intentionally omitted "
            "for gfx1151 comparisons because sysfs/rocm-smi report only the Strix Halo 512 MiB aperture."
        ),
        rows=(
            Row("512/128", 638.008, 57.615, None),
            Row("4K/128", 595.400, 55.027, None),
            Row("32K/128", 407.984, 44.576, None),
            Row("128K/128", 181.453, 26.935, None),
            Row("4K/4K", 590.391, 54.241, None),
        ),
    ),
}

BASELINE_ALIASES = {
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
    "llama.cpp hip gfx1151": "llama.cpp-hip-gfx1151",
    "llama.cpp-hip-gfx1151": "llama.cpp-hip-gfx1151",
    "llamacpp-hip-gfx1151": "llama.cpp-hip-gfx1151",
    "hip-gfx1151": "llama.cpp-hip-gfx1151",
    "upstream-hip-gfx1151": "llama.cpp-hip-gfx1151",
    "llama.cpp vulkan gfx1151": "llama.cpp-vulkan-gfx1151",
    "llama.cpp-vulkan-gfx1151": "llama.cpp-vulkan-gfx1151",
    "llamacpp-vulkan-gfx1151": "llama.cpp-vulkan-gfx1151",
    "vulkan-gfx1151": "llama.cpp-vulkan-gfx1151",
    "upstream-vulkan-gfx1151": "llama.cpp-vulkan-gfx1151",
}

TARGET_ALIASES = {
    "qwen35": "qwen35-current",
    "qwen3.5": "qwen35-current",
    "qwen35-current": "qwen35-current",
    "current": "qwen35-current",
    "hipengine": "qwen35-current",
    "shisa": "shisa-packed",
    "qwen36": "shisa-packed",
    "qwen3.6": "shisa-packed",
    "packed": "shisa-packed",
    "packed-paro": "shisa-packed",
    "packed-paro-w4": "shisa-packed",
    "packed_paro_w4": "shisa-packed",
    "shisa-packed": "shisa-packed",
    "gfx1151": "shisa-packed-gfx1151",
    "shisa-gfx1151": "shisa-packed-gfx1151",
    "qwen36-gfx1151": "shisa-packed-gfx1151",
    "qwen3.6-gfx1151": "shisa-packed-gfx1151",
    "packed-gfx1151": "shisa-packed-gfx1151",
    "shisa-packed-gfx1151": "shisa-packed-gfx1151",
    "legacy": "shisa-legacy",
    "legacy-fp16": "shisa-legacy",
    "legacy_fp16": "shisa-legacy",
    "unpacked": "shisa-legacy",
    "shisa-legacy": "shisa-legacy",
}


def _normalized_key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", "-").split())


def _normalize_baseline(value: str) -> str:
    key = _normalized_key(value)
    key = BASELINE_ALIASES.get(key, key)
    if key not in BASELINES and key != "all":
        valid = ", ".join([*BASELINES, "all"])
        raise SystemExit(f"unknown baseline {value!r}; choose one of: {valid}")
    return key


def _normalize_target(value: str) -> str:
    key = _normalized_key(value)
    key = TARGET_ALIASES.get(key, key)
    if key not in TARGETS:
        valid = ", ".join(TARGETS)
        raise SystemExit(f"unknown target {value!r}; choose one of: {valid}")
    return key


def _auto_compare_target(target_key: str) -> str:
    return "shisa-packed" if target_key == "shisa-legacy" else "shisa-legacy"


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


def _shared_workloads(left: Series, right: Series) -> list[str]:
    right_workloads = set(_row_map(right.rows))
    return [row.workload for row in left.rows if row.workload in right_workloads]


def print_comparison(target: Series, baseline: Series, *, include_memory: bool = True) -> None:
    left = _row_map(target.rows)
    right = _row_map(baseline.rows)
    workloads = _shared_workloads(target, baseline)

    print(f"## {target.display} vs {baseline.display}\n")
    print(f"A target source: {target.source}")
    print(f"B baseline source: {baseline.source}")
    print(f"notes: {target.notes} {baseline.notes}\n")

    _print_table(
        "Prefill",
        ("Workload", f"{target.display} tok/s", f"{baseline.display} tok/s", "Delta A vs B"),
        [
            (
                workload,
                _fmt_rate(left[workload].prefill_tok_s),
                _fmt_rate(right[workload].prefill_tok_s),
                _fmt_pct(left[workload].prefill_tok_s, right[workload].prefill_tok_s),
            )
            for workload in workloads
        ],
    )

    _print_table(
        "Decode",
        ("Workload", f"{target.display} tok/s", f"{baseline.display} tok/s", "Delta A vs B"),
        [
            (
                workload,
                _fmt_rate(left[workload].decode_tok_s),
                _fmt_rate(right[workload].decode_tok_s),
                _fmt_pct(left[workload].decode_tok_s, right[workload].decode_tok_s),
            )
            for workload in workloads
        ],
    )

    if include_memory:
        _print_table(
            "Memory / peak GiB",
            ("Workload", f"{target.display} peak GiB", f"{baseline.display} peak GiB", "Delta A vs B"),
            [
                (
                    workload,
                    _fmt_gib(left[workload].peak_gib),
                    _fmt_gib(right[workload].peak_gib),
                    _fmt_gib_delta(left[workload].peak_gib, right[workload].peak_gib),
                )
                for workload in workloads
            ],
        )


def print_target_comparison(target: Series, compare_target: Series, *, include_memory: bool = True) -> None:
    left = _row_map(target.rows)
    right = _row_map(compare_target.rows)
    workloads = _shared_workloads(target, compare_target)

    print(f"## {target.display} (A) vs {compare_target.display} (B)\n")
    print(f"A source: {target.source}")
    print(f"B source: {compare_target.source}")
    print(f"A notes: {target.notes}")
    print(f"B notes: {compare_target.notes}\n")

    _print_table(
        "Prefill",
        ("Workload", "A tok/s", "B tok/s", "Delta A vs B"),
        [
            (
                workload,
                _fmt_rate(left[workload].prefill_tok_s),
                _fmt_rate(right[workload].prefill_tok_s),
                _fmt_pct(left[workload].prefill_tok_s, right[workload].prefill_tok_s),
            )
            for workload in workloads
        ],
    )

    _print_table(
        "Decode",
        ("Workload", "A tok/s", "B tok/s", "Delta A vs B"),
        [
            (
                workload,
                _fmt_rate(left[workload].decode_tok_s),
                _fmt_rate(right[workload].decode_tok_s),
                _fmt_pct(left[workload].decode_tok_s, right[workload].decode_tok_s),
            )
            for workload in workloads
        ],
    )

    if include_memory:
        _print_table(
            "Memory / peak GiB",
            ("Workload", "A peak GiB", "B peak GiB", "Delta A vs B"),
            [
                (
                    workload,
                    _fmt_gib(left[workload].peak_gib),
                    _fmt_gib(right[workload].peak_gib),
                    _fmt_gib_delta(left[workload].peak_gib, right[workload].peak_gib),
                )
                for workload in workloads
            ],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baseline",
        nargs="?",
        default="all",
        help=(
            "Comparison baseline: nano-vllm-amd, llama.cpp-hip, llama.cpp-vulkan, or all. "
            "Ignored when --against-target is set. Default: all."
        ),
    )
    parser.add_argument(
        "--target",
        default="qwen35-current",
        help=(
            "A-side hipEngine target: qwen35-current, shisa-packed, or shisa-legacy. "
            "Aliases: qwen35, shisa/packed, legacy/unpacked. Default: qwen35-current."
        ),
    )
    parser.add_argument(
        "--against-target",
        "--compare-target",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Compare the A-side target against another hipEngine target instead of external baselines. "
            "With no value, uses legacy when A is packed/shisa and packed when A is legacy."
        ),
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Only print prefill/decode throughput tables; skip peak-memory tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_key = _normalize_target(args.target)
    target = TARGETS[target_key]

    if args.against_target is not None:
        compare_key = _auto_compare_target(target_key) if args.against_target == "auto" else _normalize_target(args.against_target)
        print_target_comparison(target, TARGETS[compare_key], include_memory=not args.no_memory)
        return

    key = _normalize_baseline(args.baseline)
    if key == "all":
        for index, baseline in enumerate(BASELINES.values()):
            if index:
                print("---\n")
            print_comparison(target, baseline, include_memory=not args.no_memory)
    else:
        print_comparison(target, BASELINES[key], include_memory=not args.no_memory)


if __name__ == "__main__":
    main()
