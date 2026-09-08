#!/usr/bin/env python3
"""QSA ordered-v2 decode drift probe: repetition-order timing with telemetry.

Reproduces the canonical A/B's per-case structure (fresh prefill, then a
128-step decode window per repetition, off/on arms counterbalanced and
interleaved) while recording every per-step wall, GPU sclk/temperature/power
samples and per-step CPU-frequency snapshots. Preserves every repetition,
including slow ones; this is a diagnostic, not a promotion gate.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import qwen4exp_row4_state_gate as gate
from scripts.qwen4exp_cpu_frequency import CpuFrequency
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.qwen4_exp_profiles import (
    register_qwen4_exp_gfx1151_profiles, QWEN4_EXP_MODEL,
    QWEN4_EXP_BACKEND, QWEN4_EXP_QUANTS,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture

FLAG = "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2"


def telemetry():
    values = {}
    for path in list(Path("/sys/class/drm").glob("card*/device/pp_dpm_sclk")):
        try:
            values[f"sclk:{path.parts[-3]}"] = path.read_text().strip()
        except OSError:
            pass
    for path in list(Path("/sys/class/hwmon").glob("hwmon*/temp1_input")):
        try:
            values[f"temp:{path.parent.name}"] = path.read_text().strip()
        except OSError:
            pass
    for path in list(Path("/sys/class/hwmon").glob("hwmon*/power1_average")):
        try:
            values[f"power:{path.parent.name}"] = path.read_text().strip()
        except OSError:
            pass
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=8)
    args = parser.parse_args()
    if args.repetitions < 1 or args.steps < 1 or args.warmup_steps < 0:
        parser.error("repetitions/steps must be positive; warmup-steps >= 0")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    check_host()
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
    fixture, digest = load_fixture(DEFAULT_FIXTURE)
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend="hip_gfx1151", max_sequence_length=4352,
        prefill_chunk_size=1024))
    runner = generator.runner
    key = KernelKey("hip_gfx1151", "qsa_sparse_attention", "bf16_kv",
                    "strict_ordered_three_pass_v2_spans")
    original = resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                       variant=key.variant)
    calls = [0]

    def counted(*a, **kw):
        calls[0] += 1
        return original(*a, **kw)

    frequency = CpuFrequency()
    report = {
        "schema": 1,
        "kind": "qwen4exp_qsa_v2_drift_probe",
        "command": sys.argv,
        "source": gate._git_metadata(ROOT) if hasattr(gate, "_git_metadata") else None,
        "host": None,
        "fixture_sha256": digest,
        "manifest_sha256": resolved.manifest_sha256,
        "flag": FLAG,
        "protocol": (
            "Fresh prefill per repetition; off/on decode arms counterbalanced and "
            "interleaved per repetition; every per-step wall preserved; GPU "
            "sclk/temperature/power and CPU-frequency snapshots around each step; "
            "state/output equality asserted between arms"),
        "cases": [],
    }
    from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
    report["source"] = _git_metadata(ROOT)
    report["host"] = _host_metadata()

    try:
        register(key, counted, replace=True)
        for name in args.case_id:
            case = next(c for c in fixture["cases"] if c["id"] == name)
            rows = []
            expected = None
            for repetition in range(args.repetitions):
                order = ("off", "on") if repetition % 2 == 0 else ("on", "off")
                for arm in order:
                    telemetry_before = telemetry()
                    os.environ[FLAG] = "0"
                    prefill_start = time.perf_counter()
                    root = runner.prefill(case["prompt_token_ids"], capture_logits=False,
                                          capture_target_hidden=False)
                    runner.runtime.device_synchronize()
                    prefill_seconds = time.perf_counter() - prefill_start
                    os.environ[FLAG] = "1" if arm == "on" else "0"
                    # warmup steps (not recorded)
                    token = root.token_id
                    for _ in range(args.warmup_steps):
                        token = runner.step(token, capture_logits=False,
                                            capture_target_hidden=False).token_id
                    runner.runtime.device_synchronize()
                    initial_calls = calls[0]
                    step_seconds = []
                    frequency_steps = []
                    decode_start = time.perf_counter()
                    for _ in range(args.steps):
                        frequency_before = frequency.sample()
                        step_start = time.perf_counter()
                        token = runner.step(token, capture_logits=False,
                                            capture_target_hidden=False).token_id
                        step_seconds.append(time.perf_counter() - step_start)
                        frequency_steps.append((frequency_before, frequency.sample()))
                    runner.runtime.device_synchronize()
                    decode_seconds = time.perf_counter() - decode_start
                    telemetry_after = telemetry()
                    state = gate._state_summary(runner)
                    result = (token, state["state_sha256"], state["layout_sha256"])
                    if expected is None:
                        expected = result
                    if result != expected or not state["finite"]:
                        raise AssertionError(f"arm {arm} altered state/output")
                    engaged = calls[0] - initial_calls
                    if arm == "on" and case["prompt_tokens"] == 4096:
                        if engaged != args.steps * 12 + args.warmup_steps * 12:
                            raise AssertionError(f"v2 engagement mismatch: {engaged}")
                    if arm == "off" and engaged != 0:
                        raise AssertionError("v2 engaged in off arm")
                    rows.append({
                        "repetition": repetition, "arm": arm,
                        "prefill_seconds": prefill_seconds,
                        "decode_seconds": decode_seconds,
                        "step_seconds": step_seconds,
                        "frequency_steps": frequency_steps,
                        "telemetry_before": telemetry_before,
                        "telemetry_after": telemetry_after,
                        "v2_calls": engaged,
                        "state_sha256": state["state_sha256"],
                    })
                    print(f"{name} rep{repetition} {arm} decode "
                          f"{decode_seconds:.3f}s prefill {prefill_seconds:.2f}s "
                          f"v2calls {engaged}", flush=True)
            summary = {}
            for arm in ("off", "on"):
                walls = [r["decode_seconds"] for r in rows if r["arm"] == arm]
                steps = [s for r in rows if r["arm"] == arm for s in r["step_seconds"]]
                summary[arm] = {
                    "decode_seconds": walls,
                    "decode_mean": statistics.fmean(walls),
                    "step_p50_ms": statistics.median(steps) * 1e3,
                    "step_max_ms": max(steps) * 1e3,
                    "slow_step_fraction_over_2x_p50": (
                        sum(1 for s in steps if s > 2 * statistics.median(steps))
                        / len(steps)),
                }
            report["cases"].append({"id": name, "rows": rows, "summary": summary})
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        register(key, original, replace=True)
        try:
            generator.close()
        except Exception:
            pass
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "cases": {c["id"]: {
            arm: {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in c["summary"][arm].items()}
            for arm in ("off", "on")} for c in report["cases"]},
    }, indent=1))


if __name__ == "__main__":
    main()
