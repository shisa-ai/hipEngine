#!/usr/bin/env python3
"""Render a compact markdown report from a Qwen4Exp fresh-profile artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _tok(value: float | int | None) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def _ms(value: float | int | None) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def _gap(value: float | int | None) -> str:
    return "—" if value is None else f"{float(value):.2f}x"


def _row(name: str, ours: float | int | None, llama: float | int | None, gap: float | int | None) -> str:
    return f"| {name} | {_ms(ours)} | {_ms(llama)} | {_gap(gap)} |"


def render_report(artifact: dict[str, Any]) -> str:
    end_to_end = artifact["end_to_end"]
    ours_e2e = end_to_end["hipengine_production"]
    llama_hip = end_to_end["llama_head_hip"]
    llama_vulkan = end_to_end["llama_head_vulkan"]
    ratios = end_to_end["ratios"]
    prefill_window = artifact["profile_windows"]["prefill"]
    decode_window = artifact["profile_windows"]["decode"]
    prefill = artifact["prefill_modules"]
    decode = artifact["decode_modules_per_token"]
    census = artifact["launch_and_host_census"]
    software = artifact["software"]

    lines = [
        "# Qwen4Exp gfx1151 performance gap report",
        "",
        f"- Date: `{artifact['date']}`",
        f"- hipEngine commit: `{software['hipengine_commit']}`",
        f"- llama.cpp remote HEAD: `{software['llama_cpp_remote_head']}`",
        f"- Production manifest: `{software['production_manifest_sha256']}`",
        f"- Strict manifest: `{software['strict_manifest_sha256']}`",
        "",
        "## End-to-end comparison",
        "",
        "| Workload | hipEngine production | llama.cpp HIP | llama.cpp Vulkan | HIP gap | Vulkan gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| p508 prefill tok/s | {_tok(ours_e2e['pp508_tok_s'])} | {_tok(llama_hip['pp508_tok_s'])} | "
            f"{_tok(llama_vulkan['pp508_tok_s'])} | {_gap(ratios['prefill_vs_llama_hip'])} | "
            f"{_gap(ratios['prefill_vs_llama_vulkan'])} |"
        ),
        (
            f"| tg32 decode tok/s | {_tok(ours_e2e['tg32_steady_tok_s'])} | {_tok(llama_hip['tg32_tok_s'])} | "
            f"{_tok(llama_vulkan['tg32_tok_s'])} | {_gap(ratios['decode_vs_llama_hip'])} | "
            f"{_gap(ratios['decode_vs_llama_vulkan'])} |"
        ),
        "",
        "## Device-kernel windows",
        "",
        "| Window | hipEngine kernel sum | llama HIP kernel sum | llama advantage | hipEngine rows | llama rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| p508 | {_ms(prefill_window['hipengine']['kernel_sum_ms'])} | "
            f"{_ms(prefill_window['llama_hip_selected_measured_graph']['kernel_sum_ms'])} | "
            f"{_gap(prefill_window['kernel_sum_gap'])} | {prefill_window['hipengine']['kernel_rows']} | "
            f"{prefill_window['llama_hip_selected_measured_graph']['kernel_rows']} |"
        ),
        (
            f"| tg decode per token | {_ms(decode_window['hipengine_per_token']['kernel_sum_ms'])} | "
            f"{_ms(decode_window['llama_hip_per_output']['kernel_sum_ms'])} | "
            f"{_gap(decode_window['kernel_sum_gap'])} | {decode_window['hipengine_per_token']['kernel_rows']:.0f} | "
            f"{decode_window['llama_hip_per_output']['kernel_rows']:.0f} |"
        ),
        "",
        "## Prefill module ranking",
        "",
        "| Module | hipEngine ms | llama HIP ms | llama advantage |",
        "| --- | ---: | ---: | ---: |",
        _row("Total device", prefill["total_device"]["hipengine_ms"], prefill["total_device"]["llama_hip_ms"], prefill["total_device"]["gap"]),
        _row("Selected Q4 gate/up", prefill["selected_q4_gate_up"]["hipengine_ms"], prefill["selected_q4_gate_up"]["llama_hip_ms"], prefill["selected_q4_gate_up"]["gap"]),
        _row("Selected Q5_1 down", prefill["selected_q5_1_down"]["hipengine_ms"], prefill["selected_q5_1_down"]["llama_hip_ms"], prefill["selected_q5_1_down"]["gap"]),
        _row("Layer-2 Q5_K gate/up", prefill["layer2_q5_k_gate_up"]["hipengine_ms"], prefill["layer2_q5_k_gate_up"]["llama_hip_ms"], prefill["layer2_q5_k_gate_up"]["gap"]),
        _row("GDN prefill", prefill["gdn_conv_recurrence"]["hipengine_ms"], prefill["gdn_conv_recurrence"]["llama_hip_ms"], prefill["gdn_conv_recurrence"]["gap"]),
        _row("QSA prefill", prefill["qsa_mixer"]["hipengine_ms"], prefill["qsa_mixer"]["llama_hip_ms"], prefill["qsa_mixer"]["gap"]),
        "",
        "## Decode module ranking per token",
        "",
        "| Module | hipEngine ms | llama HIP ms | llama advantage |",
        "| --- | ---: | ---: | ---: |",
        _row("Total device", decode["total_device"]["hipengine_ms"], decode["total_device"]["llama_hip_ms"], decode["total_device"]["gap"]),
        _row("Dense Q8", decode["dense_q8"]["hipengine_ms"], decode["dense_q8"]["llama_hip_ms"], decode["dense_q8"]["gap"]),
        _row("Selected Q4 gate/up", decode["selected_q4_gate_up"]["hipengine_ms"], decode["selected_q4_gate_up"]["llama_hip_ms"], decode["selected_q4_gate_up"]["gap"]),
        _row("Selected Q5_1 down", decode["selected_q5_1_down"]["hipengine_ms"], decode["selected_q5_1_down"]["llama_hip_ms"], decode["selected_q5_1_down"]["gap"]),
        _row("GDN decode", decode["gdn_conv_recurrence"]["hipengine_ms"], decode["gdn_conv_recurrence"]["llama_hip_ms"], decode["gdn_conv_recurrence"]["gap"]),
        _row("QSA decode", decode["qsa_attention"]["hipengine_ms"], decode["qsa_attention"]["llama_hip_ms"], decode["qsa_attention"]["gap"]),
        "",
        "## Launch and host census",
        "",
        (
            f"- Prefill: {census['prefill_hipengine']['direct_kernel_launch_api_calls']} direct launches, "
            f"{census['prefill_hipengine']['kernel_rows']} kernel rows, "
            f"{census['prefill_hipengine']['copy_or_fill_kernel_rows']} copy/fill rows, span-minus-sum "
            f"{census['prefill_hipengine']['kernel_span_minus_sum_ms']:.2f} ms."
        ),
        (
            f"- Decode: {census['decode_hipengine_per_token']['direct_kernel_launch_api_calls']:.0f} direct launches, "
            f"{census['decode_hipengine_per_token']['graph_launches']} graph launches, "
            f"{census['decode_hipengine_per_token']['graph_expanded_kernel_rows']} graph-expanded kernels, and "
            f"{census['decode_hipengine_per_token']['kernel_rows']:.0f} kernel rows per token."
        ),
        f"- llama.cpp HIP: {census['llama_hip']['prefill']}; {census['llama_hip']['decode']}",
        "",
        "## Ordered next units",
        "",
    ]
    for unit in artifact["ordered_next_units"]:
        lines.extend(
            [
                f"{unit['rank']}. **{unit['unit']}**",
                f"   - Evidence: {unit['evidence']}",
                f"   - Action: {unit['action']}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact = json.loads(args.artifact.read_text())
    report = render_report(artifact)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
