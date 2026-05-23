#!/usr/bin/env python3
"""Run ``llama-bench`` for split prefill/decode workloads while sampling
amdgpu VRAM in the background, and emit a benchmark-results JSON capturing
the peak VRAM each run touched.

llama-bench itself does not log any peak GPU-memory number (its JSON only
contains tok/s). We need an external observer because:

* Prefill (``-p X -n 0 -d 0``) and decode-at-offset (``-p 0 -n Y -d X``) are
  separate llama-bench invocations and the second can allocate a wider KV
  buffer than the first.
* Vulkan and HIP share the amdgpu kernel-driver allocator, so the cleanest
  cross-backend signal is ``/sys/class/drm/card*/device/mem_info_vram_used``.

Polling interval is configurable via ``--poll`` (milliseconds). Sysfs reads
take a few microseconds, so polling at 1-10 ms is safe even though we do not
care about a small perturbation here — the user explicitly asked for "max
numbers, we do not care about tok/s perturbation".

Usage::

    python3 scripts/llamacpp_bench_with_peak.py \
        --llama-bench /home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench \
        --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
        --backend hip \
        --workloads 512/128 4K/128 32K/128 128K/128 \
        --poll 10 \
        --output benchmarks/results/2026-05-17-llamacpp-hip-qwen35-peak.json

The script always runs ``-r 1`` by default; pass ``--repetitions`` to
override. It prints a Markdown summary table to stdout suitable for pasting
into a worklog entry, and writes a JSON artifact to ``--output``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow `python3 scripts/llamacpp_bench_with_peak.py` from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hipengine.util.amdgpu_vram import (  # noqa: E402
    AmdgpuCard,
    VramSampler,
    VramSamples,
    list_amdgpu_cards,
    select_card,
)

_GIB = 1 << 30
_MIB = 1 << 20

# Match the llama-bench stderr lines that report planned allocations:
#   load_tensors:        ROCm0 model buffer size =  3577.56 MiB
#   load_tensors:      Vulkan0 model buffer size =  3577.55 MiB
#   llama_kv_cache:      ROCm0 KV buffer size =   256.00 MiB
#   llama_kv_cache:    Vulkan0 KV buffer size =   128.00 MiB
#   sched_reserve:      ROCm0 compute buffer size =    97.51 MiB
#   sched_reserve:    Vulkan0 compute buffer size =     6.06 MiB
#   llama_context:  ROCm_Host  output buffer size =     0.12 MiB
#   llama_context: Vulkan_Host  output buffer size =     0.12 MiB
_BUFFER_LINE_RE = re.compile(
    r"""^
    (?P<phase>load_tensors|llama_kv_cache|sched_reserve|llama_context)
    :\s*
    (?P<device>\S+)\s+
    (?P<kind>model|KV|compute|output)
    \s+buffer\s+size\s+=\s+
    (?P<mib>[0-9.]+)\s+MiB
    """,
    re.VERBOSE,
)


def _parse_workload_token(token: str) -> tuple[int, int, str]:
    """Parse a workload string like ``32K/128`` into ``(prompt, gen, label)``."""

    token = token.strip()
    if "/" not in token:
        raise ValueError(f"workload {token!r} must look like '<prompt>/<gen>'")
    left, right = token.split("/", 1)
    return _parse_count(left), _parse_count(right), token


def _parse_count(value: str) -> int:
    """Parse '512', '4K', '128K', '1M' into an int."""

    s = value.strip().upper()
    multiplier = 1
    if s.endswith("K"):
        multiplier = 1024
        s = s[:-1]
    elif s.endswith("M"):
        multiplier = 1024 * 1024
        s = s[:-1]
    if not s:
        raise ValueError(f"empty count in {value!r}")
    return int(s) * multiplier


@dataclass
class LlamaBenchResult:
    """Parsed output of one llama-bench process."""

    command: list[str]
    returncode: int
    duration_s: float
    json_records: list[dict[str, Any]]
    stderr_text: str
    buffer_lines: list[dict[str, Any]]
    avg_tok_s: float | None
    avg_ns: int | None
    build_commit: str | None
    build_number: int | None
    gpu_info: str | None
    model_type: str | None
    backends: str | None

    def buffers_by_device_mib(self) -> dict[str, float]:
        """Sum buffer-size lines per device prefix (ROCm0, Vulkan0, ...)."""

        totals: dict[str, float] = {}
        for entry in self.buffer_lines:
            dev = entry["device"]
            totals[dev] = totals.get(dev, 0.0) + entry["mib"]
        return totals


def _detect_backend_from_stderr(text: str) -> str | None:
    if "ggml_vulkan:" in text or "Vulkan device" in text or "Vulkan0" in text:
        return "vulkan"
    if "ggml_cuda_init:" in text or "ROCm devices" in text or "ROCm0" in text:
        return "hip"
    return None


def _parse_llama_bench_stderr(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        m = _BUFFER_LINE_RE.match(line)
        if not m:
            continue
        rows.append(
            {
                "phase": m["phase"],
                "device": m["device"],
                "kind": m["kind"].lower(),
                "mib": float(m["mib"]),
            }
        )
    return rows


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Pull the JSON array llama-bench prints when ``-o json`` is set.

    llama-bench prepends its CUDA/ROCm/Vulkan banner to stdout, then writes
    a single ``[...]`` array. We find the first ``[`` and parse from there.
    """

    start = text.find("[")
    if start < 0:
        return []
    chunk = text[start:].strip()
    try:
        payload = json.loads(chunk)
    except json.JSONDecodeError:
        # Some builds print extra trailing text; trim to the matching ].
        depth = 0
        end = -1
        for idx, ch in enumerate(chunk):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end < 0:
            return []
        try:
            payload = json.loads(chunk[:end])
        except json.JSONDecodeError:
            return []
    if isinstance(payload, list):
        return payload
    return []


def _summarize_records(records: list[dict[str, Any]]) -> tuple[float | None, int | None]:
    if not records:
        return None, None
    last = records[-1]
    return last.get("avg_ts"), last.get("avg_ns")


def run_llama_bench(
    *,
    binary: Path,
    args: list[str],
    sampler: VramSampler,
    env: dict[str, str] | None = None,
) -> LlamaBenchResult:
    """Launch ``binary`` with ``args`` under an active VRAM sampler."""

    cmd = [str(binary), *args]
    sampler.start()
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    finally:
        sampler.stop()
    t1 = time.perf_counter()

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    records = _extract_json_array(stdout)
    buffer_lines = _parse_llama_bench_stderr(stderr)
    avg_tok_s, avg_ns = _summarize_records(records)
    first = records[0] if records else {}
    return LlamaBenchResult(
        command=cmd,
        returncode=proc.returncode,
        duration_s=t1 - t0,
        json_records=records,
        stderr_text=stderr,
        buffer_lines=buffer_lines,
        avg_tok_s=avg_tok_s,
        avg_ns=avg_ns,
        build_commit=first.get("build_commit"),
        build_number=first.get("build_number"),
        gpu_info=first.get("gpu_info"),
        model_type=first.get("model_type"),
        backends=first.get("backends"),
    )


def _phase_artifact(
    *,
    phase: str,
    workload_label: str,
    prompt_len: int,
    decode_tokens: int,
    bench: LlamaBenchResult,
    samples: VramSamples,
) -> dict[str, Any]:
    buffers = bench.buffers_by_device_mib()
    payload: dict[str, Any] = {
        "phase": phase,
        "workload": workload_label,
        "prompt_length": prompt_len,
        "decode_tokens": decode_tokens,
        "command": bench.command,
        "returncode": bench.returncode,
        "duration_s": bench.duration_s,
        "tok_s": bench.avg_tok_s,
        "avg_ns": bench.avg_ns,
        "vram": samples.to_dict(),
        "llamacpp_buffers_mib": [
            {
                "phase": entry["phase"],
                "device": entry["device"],
                "kind": entry["kind"],
                "mib": entry["mib"],
            }
            for entry in bench.buffer_lines
        ],
        "llamacpp_buffer_totals_mib": buffers,
        "llamacpp_buffer_totals_gib": {dev: mib / 1024.0 for dev, mib in buffers.items()},
    }
    if bench.json_records:
        # Strip the very large arrays llama-bench echoes back; keep the
        # measurement-relevant fields. The full record is still useful for
        # provenance, so retain the last entry verbatim.
        payload["llamacpp_record"] = bench.json_records[-1]
    if bench.returncode != 0:
        payload["stderr_tail"] = bench.stderr_text[-2000:]
    return payload


def _row_summary(prefill: dict[str, Any], decode: dict[str, Any]) -> dict[str, Any]:
    p_peak = prefill["vram"]["peak_gib"]
    d_peak = decode["vram"]["peak_gib"]
    row_peak = max(p_peak, d_peak)
    return {
        "workload": prefill["workload"],
        "prompt_length": prefill["prompt_length"],
        "decode_tokens": decode["decode_tokens"],
        "prefill_tok_s": prefill["tok_s"],
        "decode_tok_s": decode["tok_s"],
        "prefill_peak_vram_gib": p_peak,
        "decode_peak_vram_gib": d_peak,
        "peak_vram_gib": row_peak,
        "prefill_peak_delta_gib": prefill["vram"]["peak_delta_gib"],
        "decode_peak_delta_gib": decode["vram"]["peak_delta_gib"],
        "peak_delta_gib": max(
            prefill["vram"]["peak_delta_gib"], decode["vram"]["peak_delta_gib"]
        ),
    }


def _format_markdown(
    *,
    backend_label: str,
    rows: list[dict[str, Any]],
    card: AmdgpuCard,
    poll_ms: float,
) -> str:
    buf = io.StringIO()
    buf.write(f"## llama.cpp {backend_label} peak VRAM (card {card.card_name}, "
              f"pci {card.pci_id}, total {card.vram_total_gib:.3f} GiB, "
              f"poll {poll_ms:.1f} ms)\n\n")
    buf.write("| Workload | prefill tok/s | decode tok/s | prefill peak GiB | "
              "decode peak GiB | row peak GiB | row peak Δ GiB |\n")
    buf.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:

        def _fmt(v: float | None, prec: int = 3) -> str:
            return "—" if v is None else f"{v:.{prec}f}"

        buf.write(
            "| "
            + " | ".join(
                [
                    row["workload"],
                    _fmt(row["prefill_tok_s"]),
                    _fmt(row["decode_tok_s"]),
                    _fmt(row["prefill_peak_vram_gib"]),
                    _fmt(row["decode_peak_vram_gib"]),
                    _fmt(row["peak_vram_gib"]),
                    _fmt(row["peak_delta_gib"]),
                ]
            )
            + " |\n"
        )
    return buf.getvalue()


def _format_count_label(n: int) -> str:
    if n >= 1024 * 1024 and n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llama-bench",
        required=True,
        type=Path,
        help="Path to the llama-bench binary to invoke",
    )
    parser.add_argument(
        "--model", "-m", required=True, type=Path, help="GGUF model path",
    )
    parser.add_argument(
        "--backend",
        choices=["hip", "vulkan", "auto"],
        default="auto",
        help="Label only; detected from stderr if 'auto'",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["512/128", "4K/128", "32K/128", "128K/128"],
        help="Workload tokens, e.g. 512/128 4K/128 32K/128 128K/128",
    )
    parser.add_argument(
        "--repetitions",
        "-r",
        type=int,
        default=1,
        help="llama-bench -r value (default 1 for single-shot capture)",
    )
    parser.add_argument(
        "--ngl",
        type=int,
        default=99,
        help="-ngl passed to llama-bench (default 99 to offload all layers)",
    )
    parser.add_argument(
        "--flash-attn",
        type=int,
        choices=[0, 1],
        default=1,
        help="-fa value passed to llama-bench (default 1)",
    )
    parser.add_argument(
        "--cache-type-k",
        default="f16",
        help="-ctk value passed to llama-bench (default f16)",
    )
    parser.add_argument(
        "--cache-type-v",
        default="f16",
        help="-ctv value passed to llama-bench (default f16)",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra arguments appended to every llama-bench invocation (shell-split)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Pass --no-warmup through to llama-bench (peak still captures load+warmup if absent)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=50.0,
        help="VRAM sampling interval in milliseconds (default 50; e.g. 10 or 5 for tighter peak capture)",
    )
    parser.add_argument(
        "--keep-samples",
        action="store_true",
        help="Retain the full VRAM sample trace in the output JSON",
    )
    parser.add_argument("--card-name", help="amdgpu card name override (e.g. card1)")
    parser.add_argument("--pci-id", help="amdgpu PCI id override (e.g. 0000:c3:00.0)")
    parser.add_argument("--card-index", type=int, help="amdgpu enumeration-index override")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write the JSON artifact to this path (default: stdout JSON only on --json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the full JSON artifact to stdout",
    )
    parser.add_argument(
        "--phases",
        default="prefill,decode",
        help="Comma-separated subset of phases to run for each workload (default: prefill,decode)",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Free-text note recorded under 'notes' in the JSON artifact",
    )
    parser.add_argument(
        "--status",
        default="diagnostic_retained",
        help="Status field for the JSON artifact (default diagnostic_retained)",
    )
    return parser.parse_args(argv)


def _build_phase_args(
    *,
    args: argparse.Namespace,
    phase: str,
    prompt_len: int,
    decode_tokens: int,
) -> list[str]:
    common = [
        "-m", str(args.model),
        "-ngl", str(args.ngl),
        "-fa", str(args.flash_attn),
        "-ctk", args.cache_type_k,
        "-ctv", args.cache_type_v,
        "-r", str(args.repetitions),
        "-o", "json",
    ]
    if args.no_warmup:
        common.append("--no-warmup")
    if phase == "prefill":
        phase_args = ["-p", str(prompt_len), "-n", "0", "-d", "0"]
    elif phase == "decode":
        phase_args = ["-p", "0", "-n", str(decode_tokens), "-d", str(prompt_len)]
    else:
        raise ValueError(f"unknown phase {phase!r}; choose prefill or decode")
    extra = shlex.split(args.extra_args) if args.extra_args else []
    return [*common, *phase_args, *extra]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.llama_bench.exists():
        print(f"ERROR: --llama-bench {args.llama_bench} not found", file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"ERROR: --model {args.model} not found", file=sys.stderr)
        return 2

    try:
        card = select_card(
            card_name=args.card_name, pci_id=args.pci_id, index=args.card_index
        )
    except (RuntimeError, KeyError, IndexError) as exc:
        cards = list_amdgpu_cards()
        print(f"ERROR: card selection failed: {exc}", file=sys.stderr)
        if cards:
            print("Known cards:", file=sys.stderr)
            for c in cards:
                print(f"  {c.card_name} pci={c.pci_id} total={c.vram_total_gib:.3f} GiB",
                      file=sys.stderr)
        return 2

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    workloads = [_parse_workload_token(tok) for tok in args.workloads]

    rows: list[dict[str, Any]] = []
    phase_records: list[dict[str, Any]] = []
    detected_backend: str | None = None

    for prompt_len, decode_tokens, workload_label in workloads:
        per_phase: dict[str, dict[str, Any]] = {}
        for phase in phases:
            phase_args = _build_phase_args(
                args=args,
                phase=phase,
                prompt_len=prompt_len,
                decode_tokens=decode_tokens,
            )
            sampler = VramSampler(
                card=card,
                interval_ms=args.poll,
                keep_samples=args.keep_samples,
            )
            print(
                f"[run] workload={workload_label} phase={phase} "
                f"cmd: {args.llama_bench} {' '.join(phase_args)}",
                file=sys.stderr,
                flush=True,
            )
            bench = run_llama_bench(
                binary=args.llama_bench,
                args=phase_args,
                sampler=sampler,
            )
            samples = sampler.result()
            if detected_backend is None:
                detected_backend = _detect_backend_from_stderr(bench.stderr_text)
            phase_payload = _phase_artifact(
                phase=phase,
                workload_label=workload_label,
                prompt_len=prompt_len,
                decode_tokens=decode_tokens,
                bench=bench,
                samples=samples,
            )
            phase_records.append(phase_payload)
            per_phase[phase] = phase_payload

            tok_s_disp = (
                f"{bench.avg_tok_s:.3f}" if bench.avg_tok_s is not None else "—"
            )
            print(
                f"[done] workload={workload_label} phase={phase} "
                f"returncode={bench.returncode} tok/s={tok_s_disp} "
                f"baseline={samples.baseline_gib:.3f} GiB "
                f"peak={samples.peak_gib:.3f} GiB "
                f"delta={samples.peak_delta_gib:.3f} GiB "
                f"samples={samples.samples_count}",
                file=sys.stderr,
                flush=True,
            )
            if bench.returncode != 0:
                print(
                    f"[error] llama-bench exited {bench.returncode}; tail:\n"
                    f"{bench.stderr_text[-1500:]}",
                    file=sys.stderr,
                    flush=True,
                )

        if "prefill" in per_phase and "decode" in per_phase:
            rows.append(_row_summary(per_phase["prefill"], per_phase["decode"]))
        else:
            # Single-phase workload (e.g. --phases prefill); still emit a row
            # that reuses whichever phase ran.
            only = next(iter(per_phase.values()))
            rows.append(
                {
                    "workload": only["workload"],
                    "prompt_length": only["prompt_length"],
                    "decode_tokens": only["decode_tokens"],
                    "prefill_tok_s": only["tok_s"] if only["phase"] == "prefill" else None,
                    "decode_tok_s": only["tok_s"] if only["phase"] == "decode" else None,
                    "prefill_peak_vram_gib": only["vram"]["peak_gib"]
                    if only["phase"] == "prefill" else None,
                    "decode_peak_vram_gib": only["vram"]["peak_gib"]
                    if only["phase"] == "decode" else None,
                    "peak_vram_gib": only["vram"]["peak_gib"],
                    "peak_delta_gib": only["vram"]["peak_delta_gib"],
                }
            )

    backend_label = args.backend
    if backend_label == "auto":
        backend_label = detected_backend or "unknown"

    artifact: dict[str, Any] = {
        "schema": 1,
        "status": args.status,
        "performance_claim": False,
        "date": _dt.date.today().isoformat(),
        "hardware": f"amdgpu {card.card_name} pci {card.pci_id} total {card.vram_total_gib:.3f} GiB",
        "host": socket.gethostname(),
        "tool": "scripts/llamacpp_bench_with_peak.py",
        "tool_version": 1,
        "model_path": str(args.model),
        "model_filename": args.model.name,
        "backend": f"llamacpp_{backend_label}",
        "llama_bench_binary": str(args.llama_bench),
        "build_commit": next(
            (r["llamacpp_record"]["build_commit"]
             for r in phase_records if r.get("llamacpp_record", {}).get("build_commit")),
            None,
        ),
        "build_number": next(
            (r["llamacpp_record"]["build_number"]
             for r in phase_records if r.get("llamacpp_record", {}).get("build_number")),
            None,
        ),
        "gpu_info": next(
            (r["llamacpp_record"]["gpu_info"]
             for r in phase_records if r.get("llamacpp_record", {}).get("gpu_info")),
            None,
        ),
        "model_type": next(
            (r["llamacpp_record"]["model_type"]
             for r in phase_records if r.get("llamacpp_record", {}).get("model_type")),
            None,
        ),
        "common_args": {
            "ngl": args.ngl,
            "flash_attn": args.flash_attn,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "repetitions": args.repetitions,
            "no_warmup": bool(args.no_warmup),
            "extra_args": args.extra_args,
        },
        "phases_run": phases,
        "workloads_requested": args.workloads,
        "poll_ms": args.poll,
        "card": card.to_dict(),
        "rows": rows,
        "phase_records": phase_records,
        "notes": [
            "VRAM peak captured by /sys/class/drm/<card>/device/mem_info_vram_used "
            "polled by hipengine/util/amdgpu_vram.py. Whole-card VRAM, includes "
            "everything committed through the amdgpu kernel driver.",
            "Per-row peak_vram_gib = max(prefill_peak, decode_peak). Delta is "
            "peak minus pre-run baseline so other processes' VRAM is excluded.",
            f"tok/s comes from llama-bench's --output json avg_ts with --repetitions {args.repetitions}.",
        ],
    }
    if args.note:
        artifact["notes"].append(args.note)

    md = _format_markdown(
        backend_label=backend_label,
        rows=rows,
        card=card,
        poll_ms=args.poll,
    )
    print(md)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
        print(f"[wrote] {args.output}", file=sys.stderr)

    if args.json:
        json.dump(artifact, sys.stdout, indent=2)
        sys.stdout.write("\n")

    return 0 if all(rec["returncode"] == 0 for rec in phase_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
