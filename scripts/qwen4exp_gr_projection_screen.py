#!/usr/bin/env python3
"""Operation-complete screen for the Qwen3.8-Flash-Next hyper-connection projections.

The fused GR up+sigmoid+mean kernel is the largest single kernel in a
``code-p4096`` prefill: 2589.5 ms of a 17114 ms window across 384 launches
(96 read sites x 4 chunks), i.e. **6.5 ms per launch** at rows=1024, K=320,
N=10240. That is 1.03 TFLOP/s, or 3.9% of the gfx1151 FP32 peak.

Two equal-accumulator retiles of that kernel were already measured and rejected
on 2026-09-06 (``gr-row-reuse-rejected``, ``gr-col-reuse-rejected``), and that
work is not repeated here. This screen asks a different question: **what does
the same projection cost when it is expressed as a tiled GEMM with the weights
dequantized once into shared memory, which is the mechanism the comparator uses
and the mechanism this tree already ports as ``dense_wide256``?**

Arms, all on identical operands taken from real layers of the model file:

    fused_exact     the production fused kernel (up + sigmoid + branch mean)
    unfused_exact   the registered strict chain: exact coltile GEMM, then the
                    standalone sigmoid and gated-mean kernels
    wide_f32in      ``dense_wide256`` with the f32 activation read in the K loop
    wide_f16in      one f32->f16 conversion pass, then the same tile reading a
                    pre-converted activation

``fused_exact`` and ``unfused_exact`` are both exact-F32 arithmetic, so they must
agree bit for bit; the screen asserts that rather than assuming it. The two
``wide`` arms change arithmetic (F16 operands, F32 accumulate) and are measured
as an **upper bound on achievable speed, not as a route**: the GR family's drift
is cliff-shaped, and the much more accurate iu8 candidate was already rejected at
mean KL 1.4e-3. Any promotion from the wide arms needs the prefill-boundary KL
screen and then the production gate.

The GR **down** projections (K=10240, N=320, 896.3 ms of the same window) are
measured in the same run with the same two mechanisms.

    python3 scripts/qwen4exp_gr_projection_screen.py \\
        --model-root <gguf> --output benchmarks/results/<dir>/artifact.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (  # noqa: E402
    qwen4_exp_gated_mean_f32,
    qwen4_exp_sigmoid_f32,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (  # noqa: E402
    gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out,
    gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out,
    gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide import (  # noqa: E402
    f32_to_f16,
    gguf_q8_0_dense_wide256_f16in_f32_f32_out,
    gguf_q8_0_dense_wide256_f32_f32_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_gr_staged_f32 import (  # noqa: E402
    gr_staged_f32_up,
)
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata  # noqa: E402
from scripts.qwen4exp_dense_projection_ab import load_layer_weights  # noqa: E402

FP32_FMA_PEAK_GFLOPS = 61_300.0
WMMA_F16_PEAK_GFLOPS = 122_600.0
LAYERS = (0, 9, 22, 34, 46)
BRANCHES = 4
HIDDEN = 2560
LOW_RANK = 320


def _bits(tensor: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest()


def _error(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, np.float64)
    actual = np.asarray(actual, np.float64)
    error = np.abs(actual - reference)
    scale = max(float(np.max(np.abs(reference))), 1e-30)
    return dict(
        elements=int(error.size),
        bit_identical=bool(np.array_equal(reference, actual)),
        differing=int(np.count_nonzero(error)),
        max_abs=float(error.max()),
        max_relative_to_peak=float(error.max() / scale),
        relative_l2=float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-30)),
    )


class Device:
    """Owns the buffers one shape case needs and the arms that write into them."""

    def __init__(self, runtime, rows: int, in_features: int, out_features: int) -> None:
        self.runtime = runtime
        self.rows, self.in_features, self.out_features = rows, in_features, out_features
        self.x = malloc(rows * in_features * 4, runtime=runtime)
        self.w = None  # set per layer
        self.out = malloc(rows * out_features * 4, runtime=runtime)
        self.extra = malloc(rows * out_features * 4, runtime=runtime)
        self.half = malloc(rows * in_features * 2, runtime=runtime)
        # ``normalized`` is the grouped-RMSNorm output the epilogue consumes: it
        # is always the full residual width, never the low-rank activation. An
        # arm that read the wrong tensor here would understate its own traffic.
        self.normalized = malloc(rows * out_features * 4, runtime=runtime)
        self.mixed = malloc(rows * (out_features // BRANCHES) * 4, runtime=runtime)

    def close(self) -> None:
        for buffer in (
            self.mixed, self.normalized, self.half, self.extra, self.out, self.x,
        ):
            free(buffer, runtime=self.runtime)

    def upload(self, x: np.ndarray, normalized: np.ndarray, weight_bytes: np.ndarray) -> None:
        copy_host_to_device(self.x, host_array_ptr(x), runtime=self.runtime)
        copy_host_to_device(
            self.normalized, host_array_ptr(normalized), runtime=self.runtime
        )
        if self.w is None:
            self.w = malloc(weight_bytes.nbytes, runtime=self.runtime)
        elif self.w.nbytes != weight_bytes.nbytes:
            raise ValueError("weight shape changed inside one case")
        copy_host_to_device(self.w, host_array_ptr(weight_bytes), runtime=self.runtime)

    def read(self, buffer, rows: int, width: int) -> np.ndarray:
        value = np.empty((rows, width), dtype=np.float32)
        copy_device_to_host(host_array_ptr(value), buffer, runtime=self.runtime)
        return value


def _up_arms(device: Device, weight) -> dict[str, Any]:
    """The four up-projection arms. Each returns the mixed tensor it produced."""
    rows, k, n = device.rows, device.in_features, device.out_features
    hidden = n // BRANCHES

    def fused() -> None:
        gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32(
            device.x.ptr, weight.ptr, device.normalized.ptr, device.out.ptr,
            device.mixed.ptr, rows, k, BRANCHES, hidden, runtime=device.runtime,
        )

    def unfused() -> None:
        gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out(
            device.x.ptr, weight.ptr, device.out.ptr, rows, k, n, runtime=device.runtime,
        )
        qwen4_exp_sigmoid_f32(
            device.out.ptr, device.out.ptr, rows * n, runtime=device.runtime,
        )
        qwen4_exp_gated_mean_f32(
            device.normalized.ptr, device.out.ptr, device.mixed.ptr, rows, BRANCHES,
            hidden, runtime=device.runtime,
        )

    def wide_f32in() -> None:
        gguf_q8_0_dense_wide256_f32_f32_out(
            device.x.ptr, weight.ptr, device.extra.ptr, rows, k, n,
            threads=256, runtime=device.runtime,
        )
        qwen4_exp_sigmoid_f32(
            device.extra.ptr, device.extra.ptr, rows * n, runtime=device.runtime,
        )
        qwen4_exp_gated_mean_f32(
            device.normalized.ptr, device.extra.ptr, device.mixed.ptr, rows, BRANCHES,
            hidden, runtime=device.runtime,
        )

    def wide_f16in() -> None:
        f32_to_f16(device.x.ptr, device.half.ptr, rows * k, runtime=device.runtime)
        gguf_q8_0_dense_wide256_f16in_f32_f32_out(
            device.half.ptr, weight.ptr, device.extra.ptr, rows, k, n,
            threads=256, runtime=device.runtime,
        )
        qwen4_exp_sigmoid_f32(
            device.extra.ptr, device.extra.ptr, rows * n, runtime=device.runtime,
        )
        qwen4_exp_gated_mean_f32(
            device.normalized.ptr, device.extra.ptr, device.mixed.ptr, rows, BRANCHES,
            hidden, runtime=device.runtime,
        )

    def staged_f32() -> None:
        gr_staged_f32_up(
            device.x.ptr, weight.ptr, device.extra.ptr, rows, k, n,
            runtime=device.runtime,
        )
        qwen4_exp_sigmoid_f32(
            device.extra.ptr, device.extra.ptr, rows * n, runtime=device.runtime,
        )
        qwen4_exp_gated_mean_f32(
            device.normalized.ptr, device.extra.ptr, device.mixed.ptr, rows, BRANCHES,
            hidden, runtime=device.runtime,
        )

    return dict(
        fused_exact=dict(run=fused, gate=True, mixed=device.mixed),
        unfused_exact=dict(run=unfused, gate=True, mixed=device.mixed),
        wide_f32in=dict(run=wide_f32in, gate=False, mixed=device.mixed),
        wide_f16in=dict(run=wide_f16in, gate=False, mixed=device.mixed),
        staged_f32=dict(run=staged_f32, gate=False, mixed=device.mixed),
    )


def _down_arms(device: Device, weight) -> dict[str, Any]:
    """The down projection has no epilogue, so both arms write ``out`` directly."""
    rows, k, n = device.rows, device.in_features, device.out_features

    def exact() -> None:
        gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out(
            device.x.ptr, weight.ptr, device.out.ptr, rows, k, n, runtime=device.runtime,
        )

    def wide_f32in() -> None:
        gguf_q8_0_dense_wide256_f32_f32_out(
            device.x.ptr, weight.ptr, device.out.ptr, rows, k, n,
            threads=256, runtime=device.runtime,
        )

    def wide_f16in() -> None:
        f32_to_f16(device.x.ptr, device.half.ptr, rows * k, runtime=device.runtime)
        gguf_q8_0_dense_wide256_f16in_f32_f32_out(
            device.half.ptr, weight.ptr, device.out.ptr, rows, k, n,
            threads=256, runtime=device.runtime,
        )

    def staged_f32() -> None:
        gr_staged_f32_up(
            device.x.ptr, weight.ptr, device.out.ptr, rows, k, n,
            runtime=device.runtime,
        )

    return dict(
        exact=dict(run=exact, gate=False, mixed=device.out),
        wide_f32in=dict(run=wide_f32in, gate=False, mixed=device.out),
        wide_f16in=dict(run=wide_f16in, gate=False, mixed=device.out),
        staged_f32=dict(run=staged_f32, gate=False, mixed=device.out),
    )


def _time(runtime, run, repetitions: int) -> float:
    start, stop = runtime.event_create(), runtime.event_create()
    try:
        run()
        runtime.device_synchronize()
        runtime.event_record(start)
        for _ in range(repetitions):
            run()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return runtime.event_elapsed_time_ms(start, stop) / repetitions
    finally:
        runtime.event_destroy(start)
        runtime.event_destroy(stop)


def _case(
    label: str,
    suffix: str,
    arm_builder,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    model_root: str,
    repetitions: int,
    layers: tuple[int, ...],
) -> dict[str, Any]:
    runtime = get_hip_runtime()
    weights = load_layer_weights(model_root, suffix, list(layers))
    if any(ggml_type != "Q8_0" for _, _, ggml_type in weights):
        raise SystemExit(f"{suffix}: expected Q8_0 weights")
    device = Device(runtime, rows, in_features, out_features)
    rng = np.random.default_rng(20260926)
    x = rng.normal(0.0, 0.2, (rows, in_features)).astype(np.float32)
    normalized = rng.normal(0.0, 0.2, (rows, out_features)).astype(np.float32)
    per_arm: dict[str, list[float]] = {}
    digests: dict[str, list[str]] = {}
    reference: dict[str, np.ndarray] = {}
    try:
        for index in range(repetitions):
            layer, raw, _ = weights[index % len(weights)]
            device.upload(x, normalized, raw)
            arms = arm_builder(device, device.w)
            # Order rotates so a warm MALL or a clock ramp cannot favour one arm.
            order = list(arms)
            shift = index % len(order)
            order = order[shift:] + order[:shift]
            for name in order:
                spec = arms[name]
                run = spec["run"]
                for _ in range(2):
                    run()
                runtime.device_synchronize()
                if name not in reference:
                    reference[name] = device.read(spec["mixed"], rows, spec["mixed"].nbytes // (rows * 4))
                elapsed = _time(runtime, run, 3)
                per_arm.setdefault(name, []).append(elapsed)
                runtime.device_synchronize()
                digest = _bits(device.read(spec["mixed"], rows, spec["mixed"].nbytes // (rows * 4)))
                digests.setdefault(name, []).append(digest)
                print(f"  {label} {name:14s} layer{layer:2d} {elapsed:8.4f} ms", flush=True)
    finally:
        device.close()

    flops = 2.0 * rows * in_features * out_features
    production = "fused_exact" if label == "up" else "exact"
    baseline = statistics.median(per_arm[production])
    rows_out = []
    for name, samples in per_arm.items():
        median = statistics.median(samples)
        rows_out.append(dict(
            arm=name,
            median_ms=median,
            samples_ms=samples,
            output_sha256=sorted(set(digests[name])),
            deterministic=len(set(digests[name])) == 1,
            speedup_vs_production=baseline / median,
            achieved_gflops=flops / (median * 1e6),
            fraction_of_fp32_peak=flops / (median * 1e6) / FP32_FMA_PEAK_GFLOPS,
            error_vs_production=_error(reference[production], reference[name]),
        ))
    rows_out.sort(key=lambda row: row["median_ms"])
    return dict(
        label=label,
        tensor_suffix=suffix,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        layers=[layer for layer, _, _ in weights],
        flops=flops,
        production_arm=production,
        arms=rows_out,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=len(LAYERS))
    parser.add_argument("--up-suffix", default="hc_ffn_up.weight")
    parser.add_argument("--down-suffix", default="hc_ffn_down.weight")
    parser.add_argument("--skip-down", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("positive repetitions required")

    packet: dict[str, Any] = dict(
        schema=1,
        kind="qwen4exp_gr_projection_screen",
        command=sys.argv,
        source=_git_metadata(REPO_ROOT),
        host=_host_metadata(),
        model="Qwen3.8-Flash-Next UD-Q4_K_XL",
        boundary="GR projection, operation-complete host call, warm builds",
        timing="hip events around 3 launches after 2 warmups, layer rotated, arm order rotated",
        rows=args.rows,
        branches=BRANCHES,
        hidden=HIDDEN,
        low_rank=LOW_RANK,
        fp32_peak_gflops=FP32_FMA_PEAK_GFLOPS,
        wmma_f16_peak_gflops=WMMA_F16_PEAK_GFLOPS,
        performance_claim=False,
        cases=[],
    )
    try:
        packet["cases"].append(_case(
            "up", args.up_suffix, _up_arms,
            rows=args.rows, in_features=LOW_RANK, out_features=BRANCHES * HIDDEN,
            model_root=args.model_root, repetitions=args.repetitions, layers=LAYERS,
        ))
        if not args.skip_down:
            packet["cases"].append(_case(
                "down", args.down_suffix, _down_arms,
                rows=args.rows, in_features=BRANCHES * HIDDEN, out_features=LOW_RANK,
                model_root=args.model_root, repetitions=args.repetitions, layers=LAYERS,
            ))
        up = packet["cases"][0]
        exact = next(row for row in up["arms"] if row["arm"] == "unfused_exact")
        if not exact["error_vs_production"]["bit_identical"]:
            packet["status"] = "failed"
            raise SystemExit(
                "strict chain is not bit-identical to the fused production kernel; "
                "the screen's exact-parity precondition does not hold"
            )
        packet["exact_parity_precondition"] = True
        packet["status"] = "completed_diagnostic"
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(packet, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
