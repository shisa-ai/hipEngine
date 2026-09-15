"""Replay GR projections on identical production inputs without substituting outputs."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sample_indices(size, count):
    if size <= 0 or count <= 0:
        raise ValueError("positive sample geometry required")
    return np.unique(np.linspace(0, size - 1, min(size, count), dtype=np.int64))


def reconstruct_planes(values, planes=3):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 32 or planes < 1:
        raise ValueError("expected rows of 32-value activation blocks")
    work = values.reshape(values.shape[0], -1, 32).copy()
    reconstructed = np.zeros(work.shape, dtype=np.float64)
    for _ in range(planes):
        scale = np.max(np.abs(work), axis=-1, keepdims=True) / np.float32(127)
        inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale > 0)
        codes = np.clip(np.rint(work * inverse), -127, 127).astype(np.float32)
        product = codes.astype(np.float64) * scale.astype(np.float64)
        reconstructed += product
        # Simulate the kernel's fused residual update, with one FP32 rounding.
        work = (work.astype(np.float64) - product).astype(np.float32)
    return reconstructed.reshape(values.shape)


def error_metrics(reference, actual):
    reference, actual = np.asarray(reference, np.float64), np.asarray(actual, np.float64)
    if reference.shape != actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("finite aligned arrays required")
    error = np.abs(actual - reference)
    return dict(elements=int(error.size), changed=int(np.count_nonzero(error)),
                max_abs=float(error.max()), mse=float(np.mean(error * error)),
                relative_l2=float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-30)))


def main():
    from hipengine.core.memory import malloc, free, memory_stats
    from hipengine.core.runtime import MemcpyKind
    from hipengine.runtime import qwen4_exp_runner as runtime_module
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import gguf_q8_0_iu8_wmma_prefill_f32_f32
    from hipengine.kernels.registry import resolve
    from scripts.qwen4exp_layer2_profile_gate import _make_generator, _state_summary
    from scripts.qwen4exp_q8_repair_depth_gate import resolve_allocation_profile
    from scripts.qwen4exp_q8_boundary_replay import dequant
    from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata
    from scripts.qwen4exp_framework_family_refresh import check_host, model_identity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", default="general_ja-p512")
    parser.add_argument("--projection-variant", choices=("original", "p4", "compensated"),
                        default="original")
    args = parser.parse_args()
    check_host()
    source = _git_metadata(ROOT)
    if not source["tracked_clean"]:
        parser.error("committed tracked source required")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    args.prefill_chunk_size, args.max_sequence_length = 1024, 4352
    fixture, fixture_hash = load_fixture(DEFAULT_FIXTURE)
    case = next(row for row in fixture["cases"] if row["id"] == args.case_id)
    if case["prompt_tokens"] != 512:
        parser.error("this bounded localization probe requires a 512-token case")
    packet = dict(status="running", source=source, host=_host_metadata(),
                  model=model_identity(args.model_root), command=sys.argv,
                  fixture_sha256=fixture_hash, case_id=args.case_id, records=[],
                  performance_claim=False, promotion_claim=False,
                  projection_variant=args.projection_variant,
                  sampling="Layers0/23/47, both GR roles, 3 evenly spaced rows and64 output columns",
                  limits="CPU plane simulation is diagnostic, not captured GPU quantizer output; "
                         "FP64 sample is a projection oracle, not a full-model teacher.")
    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resolve_allocation_profile()
        generator, profile, _ = _make_generator(args, "production")
        runtime = generator.runner.runtime
        projection = (
            gguf_q8_0_iu8_wmma_prefill_f32_f32 if args.projection_variant == "original"
            else resolve(backend="hip_gfx1151", layer="linear", quant="gguf_q8_0",
                         variant=f"iu8_{args.projection_variant}_prefill_f32_f32_out"))
        original = runtime_module.run_qwen4_exp_gr_read
        seen = set()

        def download(ptr, shape, dtype):
            value = np.empty(shape, dtype=dtype)
            runtime.memcpy(value.ctypes.data, ptr, value.nbytes, MemcpyKind.DEVICE_TO_HOST)
            return value

        def replay(weight, x_ptr, rows, k, n, scratch, leg):
            if weight.spec.quant_key != "gguf_q8_0":
                raise ValueError("GR replay requires raw Q8 weights")
            buffers = []
            try:
                for _ in range(2):
                    buffers.append(malloc(rows * n * 4, runtime=runtime))
                runtime_module.launch_gguf_linear(
                    weight, x_ptr, buffers[0].ptr, rows, k, n,
                    activation_dtype="f32", output_dtype="f32", runtime=runtime)
                projection(
                    x_ptr, weight.allocation("raw").tensor.ptr, buffers[1].ptr,
                    rows, k, n, runtime=runtime)
                runtime.device_synchronize()
                parent = download(buffers[0].ptr, (rows, n), np.float32)
                candidate = download(buffers[1].ptr, (rows, n), np.float32)
                ri, ci = sample_indices(rows, 3), sample_indices(n, 64)
                x = np.stack([download(x_ptr + int(row) * k * 4, (k,), np.float32) for row in ri])
                row_bytes = k // 32 * 34
                raw = np.stack([download(weight.allocation("raw").tensor.ptr + int(col) * row_bytes,
                                         (row_bytes,), np.uint8) for col in ci])
                weights = dequant(raw, k)
                oracle = x.astype(np.float64) @ weights.T
                reconstructed = reconstruct_planes(
                    x, planes=4 if args.projection_variant == "p4" else 3)
                quantized_oracle = reconstructed @ weights.T
                record = dict(
                    weight=weight.spec.slot_path, leg=leg, shape=[rows, k, n],
                    sampled_rows=ri.tolist(), sampled_columns=ci.tolist(),
                    input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                    sampled_weight_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                    input_max_abs=float(np.max(np.abs(x))),
                    parent_vs_candidate=error_metrics(parent, candidate),
                    fp64_vs_parent=error_metrics(oracle, parent[np.ix_(ri, ci)]),
                    fp64_vs_candidate=error_metrics(oracle, candidate[np.ix_(ri, ci)]),
                    fp64_vs_reconstructed_activation=error_metrics(oracle, quantized_oracle),
                    activation_reconstruction=error_metrics(x, reconstructed),
                    sampled_fp64=oracle.tolist(),
                    sampled_parent=parent[np.ix_(ri, ci)].tolist(),
                    sampled_candidate=candidate[np.ix_(ri, ci)].tolist())
                if leg == "up":
                    for name, buffer in zip(("parent", "candidate"), buffers):
                        mixed = malloc(rows * (n // 4) * 4, runtime=runtime)
                        try:
                            runtime_module.qwen4_exp_sigmoid_f32(
                                buffer.ptr, buffer.ptr, rows * n, runtime=runtime)
                            runtime_module.qwen4_exp_gated_mean_f32(
                                scratch.normalized.ptr, buffer.ptr, mixed.ptr,
                                rows, 4, n // 4, runtime=runtime)
                            runtime.device_synchronize()
                            actual = download(mixed.ptr, (rows, n // 4), np.float32)
                            baseline = download(scratch.mixed.ptr, (rows, n // 4), np.float32)
                            record[name + "_epilogue_vs_fused"] = error_metrics(baseline, actual)
                        finally:
                            free(mixed, runtime=runtime)
                packet["records"].append(record)
            finally:
                for buffer in reversed(buffers):
                    free(buffer, runtime=runtime)

        def hooked(residual, norm, down, up, inject, scratch, **kwargs):
            result = original(residual, norm, down, up, inject, scratch, **kwargs)
            parts = up.spec.slot_path.split(".")
            if (parts[0] != "layers" or int(parts[1]) not in {0, 23, 47}
                    or up.spec.slot_path in seen or kwargs["rows"] <= 256):
                return result
            if kwargs.get("stream", 0):
                raise ValueError("diagnostic requires default stream")
            if kwargs["branches"] != 4:
                raise ValueError("diagnostic requires four residual branches")
            seen.add(up.spec.slot_path)
            runtime.device_synchronize()
            rows, hidden, low_rank = kwargs["rows"], kwargs["hidden"], kwargs["low_rank"]
            replay(down, scratch.normalized.ptr, rows, 4 * hidden, low_rank, scratch, "down")
            replay(up, scratch.low_rank.ptr, rows, low_rank, 4 * hidden, scratch, "up")
            print(up.spec.slot_path, "replayed", flush=True)
            return result

        try:
            for flag in ("GR_IU8", "GR_IU8_DOWN", "Q8_IU8_WMM", "Q8_MMQ_PREFILL"):
                if os.environ.get("HIPENGINE_QWEN4_EXP_" + flag) != "0":
                    raise ValueError("production reference must keep failed GR/dense arms off")
            baseline = generator.runner.prefill(case["prompt_token_ids"])
            baseline_logits = np.array(baseline.logits, copy=True)
            baseline_state = _state_summary(generator.runner)
            runtime_module.run_qwen4_exp_gr_read = hooked
            result = generator.runner.prefill(case["prompt_token_ids"])
            if len(packet["records"]) != 12:
                raise ValueError("expected both projections of both GR roles at three layers")
            packet["first_token"] = int(result.token_id)
            np.testing.assert_array_equal(result.logits, baseline_logits)
            if _state_summary(generator.runner) != baseline_state:
                raise ValueError("diagnostic changed model state")
            packet["control_logits_state_exact"] = True
            packet["manifest"] = profile.manifest_sha256
            packet["status"] = "completed_diagnostic"
        except Exception as error:
            packet["status"] = "failed"
            packet["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            runtime_module.run_qwen4_exp_gr_read = original
            generator.close()
            packet["after_close"] = memory_stats()
            if packet["after_close"]["current_allocated_bytes"] or _git_metadata(ROOT) != source:
                packet["status"] = "invalid_capture"
            args.output.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
