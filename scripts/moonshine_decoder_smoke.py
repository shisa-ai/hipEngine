#!/usr/bin/env python3
"""Run the complete resident Moonshine FP16 decoder against a golden fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    host_array_ptr,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.attention.moonshine_attention import (
    build_moonshine_attention,
)
from hipengine.kernels.hip_gfx1100.fused.moonshine_glue import build_moonshine_glue
from hipengine.kernels.hip_gfx1100.fused.moonshine_mlp import build_moonshine_mlp
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import build_dense_gemv
from hipengine.kernels.hip_gfx1100.linear.moonshine_projection import (
    build_moonshine_projection,
)
from hipengine.kernels.hip_gfx1100.linear.moonshine_w8a16 import (
    build_moonshine_w8a16,
)
from hipengine.kernels.hip_gfx1100.norm.moonshine_layernorm import (
    build_moonshine_layernorm,
)
from hipengine.runtime.moonshine import (
    MOONSHINE_DEFAULT_LM_HEAD_ROUTE,
    MOONSHINE_LM_HEAD_ROUTES,
    MoonshineResidentRuntime,
)

BOUNDARY_MAX_ABS = 1.0
BOUNDARY_MAX_RELATIVE_L2 = 0.01
CERTIFIED_ENCODER_BUCKETS = (40, 207, 1248)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--pad-to-certified-bucket",
        action="store_true",
        help="zero-pad encoder hidden/mask and cross-cache references to 40/207/1248",
    )
    parser.add_argument("--prebuild-only", action="store_true")
    parser.add_argument(
        "--w8a16-families",
        default="",
        help="comma-separated selective families: lm_head,mlp,attention",
    )
    parser.add_argument(
        "--lm-head-route",
        choices=MOONSHINE_LM_HEAD_ROUTES,
        default=MOONSHINE_DEFAULT_LM_HEAD_ROUTE,
        help="use exact wave8 partial top-1 (default) or the unfused full-logit fallback",
    )
    parser.add_argument(
        "--token-route",
        choices=("eager", "graph"),
        default="eager",
        help="dispatch each resident token through Python wrappers or a captured HIP graph",
    )
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def _build_all(
    compiler_version: str,
    *,
    load: bool,
    require_cached: bool,
    include_w8a16: bool = False,
):
    arguments = {
        "compiler_version": compiler_version,
        "load": load,
        "require_cached": require_cached,
    }
    builders = [
        build_moonshine_projection,
        build_dense_gemv,
        build_moonshine_layernorm,
        build_moonshine_glue,
        build_moonshine_mlp,
        build_moonshine_attention,
    ]
    if include_w8a16:
        builders.append(build_moonshine_w8a16)
    return tuple(builder(**arguments) for builder in builders)


def _fixture_first_eos_position(manifest: dict[str, object]) -> int:
    producer = manifest.get("producer")
    if not isinstance(producer, dict) or producer.get("first_eos_position") is None:
        raise ValueError("fixture producer must record first_eos_position")
    position = int(producer["first_eos_position"])
    if position < 0:
        raise ValueError("fixture first_eos_position must be nonnegative")
    return position


def _token_mismatch_required(
    position: int,
    first_eos_position: int,
    selected_positions: set[int],
) -> bool:
    return position <= first_eos_position or position in selected_positions


def _certified_encoder_bucket(source_frames: int) -> int:
    for bucket in CERTIFIED_ENCODER_BUCKETS:
        if source_frames <= bucket:
            return bucket
    raise ValueError(
        f"encoder frame count {source_frames} exceeds certified buckets "
        f"{CERTIFIED_ENCODER_BUCKETS}"
    )


def _pad_encoder_inputs(
    hidden: np.ndarray,
    mask: np.ndarray,
    bucket_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_frames = int(hidden.shape[1])
    if mask.shape != (1, source_frames):
        raise ValueError(f"encoder mask shape {mask.shape} does not match {source_frames}")
    if bucket_frames < source_frames:
        raise ValueError(f"bucket {bucket_frames} is smaller than source {source_frames}")
    padded_hidden = np.zeros((1, bucket_frames, hidden.shape[2]), dtype=np.float16)
    padded_mask = np.zeros((1, bucket_frames), dtype=np.int32)
    padded_hidden[:, :source_frames] = hidden
    padded_mask[:, :source_frames] = mask
    return padded_hidden, padded_mask


def _pad_cross_reference(expected: np.ndarray, bucket_frames: int) -> np.ndarray:
    source_frames = int(expected.shape[2])
    if bucket_frames < source_frames:
        raise ValueError(f"bucket {bucket_frames} is smaller than source {source_frames}")
    padded = np.zeros(
        (expected.shape[0], expected.shape[1], bucket_frames, expected.shape[3]),
        dtype=expected.dtype,
    )
    padded[:, :, :source_frames] = expected
    return padded


def _download_fp16(resident: MoonshineResidentRuntime, tensor) -> np.ndarray:
    resident.runtime.stream_synchronize(resident.stream)
    host = np.empty(tensor.shape, dtype=np.float16)
    copy_device_to_host(
        host_array_ptr(host),
        _buffer_view(tensor),
        runtime=resident.runtime,
    )
    return host


def _buffer_view(tensor):
    from hipengine.core.memory import DeviceBuffer

    return DeviceBuffer(tensor.ptr, tensor.numel * tensor.dtype.itemsize)


def _compare(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    report: dict[str, object],
) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        report["failures"].append(
            f"{name}: shape/dtype {actual.shape}/{actual.dtype} != "
            f"{expected.shape}/{expected.dtype}"
        )
        return
    if np.issubdtype(actual.dtype, np.floating) and not bool(np.isfinite(actual).all()):
        report["failures"].append(f"{name}: non-finite output")
        return
    if not np.issubdtype(actual.dtype, np.floating):
        if not np.array_equal(actual, expected):
            report["failures"].append(f"{name}: integer values differ")
        return
    difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    max_abs = float(np.max(difference)) if difference.size else 0.0
    relative_l2 = float(
        np.linalg.norm(difference.ravel())
        / max(np.linalg.norm(expected.astype(np.float32).ravel()), 1.0e-12)
    )
    report["max_abs"] = max(float(report["max_abs"]), max_abs)
    report["max_relative_l2"] = max(float(report["max_relative_l2"]), relative_l2)
    if (
        bool(report.get("boundary_limit_enforced", True))
        and (max_abs > BOUNDARY_MAX_ABS or relative_l2 > BOUNDARY_MAX_RELATIVE_L2)
    ):
        report["failures"].append(
            f"{name}: max_abs={max_abs:.6g}, relative_l2={relative_l2:.6g}"
        )


def main() -> int:
    args = parse_args()
    compiler_version = args.compiler_version_file.read_text()
    if args.prebuild_only:
        for artifact in _build_all(
            compiler_version,
            load=False,
            require_cached=False,
            include_w8a16=bool(args.w8a16_families),
        ):
            print(artifact.output_path)
        return 0
    if args.model_path is None or args.fixture is None:
        raise SystemExit("--model-path and --fixture are required unless --prebuild-only")
    manifest_path = args.fixture.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    token_ids = tuple(int(value) for value in manifest["decoder"]["token_ids"])
    positions = tuple(int(value) for value in manifest["decoder"]["positions"])
    first_eos_position = _fixture_first_eos_position(manifest)
    source_frames = int(manifest["input"]["encoder_frames"])
    encoder_frames = (
        _certified_encoder_bucket(source_frames)
        if args.pad_to_certified_bucket
        else source_frames
    )
    report: dict[str, object] = {
        "all_passed": False,
        "boundary_max_abs_limit": BOUNDARY_MAX_ABS,
        "boundary_max_relative_l2_limit": BOUNDARY_MAX_RELATIVE_L2,
        "boundary_limit_enforced": not bool(args.w8a16_families),
        "boundary_policy": (
            "FP16 boundary limits are recorded but the explicit KL/top-1 and token gates "
            "control W8A16 qualification"
            if args.w8a16_families
            else "strict FP16 boundary limits"
        ),
        "encoder_frames": encoder_frames,
        "source_frames": source_frames,
        "failures": [],
        "max_abs": 0.0,
        "max_relative_l2": 0.0,
        "positions": list(positions),
        "first_eos_position": first_eos_position,
        "generation_tokens_exact": True,
        "selected_tokens_exact": True,
        "post_eos_unselected_token_mismatches": [],
        "timed_step_allocations": 0,
        "token_route": args.token_route,
        "lm_head": None,
        "w8a16_families": [
            value.strip() for value in args.w8a16_families.split(",") if value.strip()
        ],
        "boundary_capture": "all_layer_boundaries" if args.token_route == "eager" else "final_hidden_only",
        "token_graph": None,
    }
    reference_logits: list[np.ndarray] = []
    actual_logits: list[np.ndarray] = []
    with np.load(args.fixture, allow_pickle=False) as fixture:
        with MoonshineResidentRuntime(
            model_path=args.model_path,
            encoder_frames=encoder_frames,
            w8a16_families=args.w8a16_families,
            lm_head_route=args.lm_head_route,
        ) as resident:
            resident.prepare_decoder_kernels(
                compiler_version=compiler_version,
                require_cached=args.require_cached_build,
            )
            report["lm_head"] = resident.lm_head_contract()
            encoder_hidden = fixture["encoder.output"]
            encoder_mask = fixture["encoder.attention_mask"]
            if encoder_frames != source_frames:
                encoder_hidden, encoder_mask = _pad_encoder_inputs(
                    encoder_hidden,
                    encoder_mask,
                    encoder_frames,
                )
            resident.set_encoder_state(encoder_hidden, encoder_mask)
            resident.precompute_cross_kv()
            if args.token_route == "graph":
                resident.capture_token_graphs()
            for layer in range(resident.spec.decoder_layers):
                cache = resident.cross_cache(layer)
                for kind, tensor in (("key", cache.key), ("value", cache.value)):
                    _compare(
                        f"cross.layer_{layer}.{kind}",
                        _download_fp16(resident, tensor),
                        (
                            _pad_cross_reference(
                                fixture[f"cross.layer_{layer}.{kind}"],
                                encoder_frames,
                            )
                            if encoder_frames != source_frames
                            else fixture[f"cross.layer_{layer}.{kind}"]
                        ),
                        report,
                    )

            selected_positions = set(positions)
            for position in range(resident.spec.self_cache_capacity):
                captures: dict[str, np.ndarray] = {}

                def capture(name: str, tensor) -> None:
                    captures[name] = _download_fp16(resident, tensor)

                resident.set_decode_state(token_id=token_ids[position], position=position)
                before = memory_stats()["total_allocated_bytes"]
                with resident.no_allocation_region(f"decoder-position-{position}"):
                    if args.token_route == "graph":
                        resident.graph_token_step()
                    else:
                        resident.token_step(
                            boundary_callback=capture if position in selected_positions else None
                        )
                after = memory_stats()["total_allocated_bytes"]
                report["timed_step_allocations"] = int(report["timed_step_allocations"]) + (
                    after - before
                )
                selected = resident.read_token()
                expected_selected = token_ids[position + 1]
                if selected != expected_selected:
                    mismatch = {
                        "position": position,
                        "selected": selected,
                        "expected": expected_selected,
                    }
                    if _token_mismatch_required(
                        position,
                        first_eos_position,
                        selected_positions,
                    ):
                        report["selected_tokens_exact"] = False
                        if position <= first_eos_position:
                            report["generation_tokens_exact"] = False
                        report["failures"].append(
                            f"position {position}: selected {selected} != {expected_selected}"
                        )
                    else:
                        report["post_eos_unselected_token_mismatches"].append(mismatch)
                if position not in selected_positions:
                    continue
                if args.token_route == "graph":
                    captures["final_hidden"] = _download_fp16(
                        resident,
                        resident.tensor("normalized"),
                    )
                prefix = f"decoder.position_{position}"
                for suffix, actual in captures.items():
                    _compare(
                        f"{prefix}.{suffix}",
                        actual,
                        fixture[f"{prefix}.{suffix}"],
                        report,
                    )
                logits = _download_fp16(resident, resident.tensor("logits"))
                expected_logits = fixture[f"{prefix}.logits"]
                _compare(f"{prefix}.logits", logits, expected_logits, report)
                actual_logits.append(logits.astype(np.float32))
                reference_logits.append(expected_logits.astype(np.float32))
                for layer in range(resident.spec.decoder_layers):
                    cache = resident.self_cache(layer)
                    visible = position + 1
                    for kind, tensor in (("key", cache.key), ("value", cache.value)):
                        actual = _download_fp16(resident, tensor)[:, :, :visible]
                        _compare(
                            f"{prefix}.layer_{layer}.self_{kind}",
                            actual,
                            fixture[f"{prefix}.layer_{layer}.self_{kind}"],
                            report,
                        )
            logits_gate = evaluate_logits(
                np.concatenate(reference_logits, axis=0),
                np.concatenate(actual_logits, axis=0),
            )
            report["logit_kl_mean"] = logits_gate.kl_mean
            report["logit_kl_max"] = logits_gate.kl_max
            report["logit_top1_agreement"] = logits_gate.top1_agreement
            report["logit_gate_passed"] = logits_gate.passed
            if args.token_route == "graph":
                report["token_graph"] = resident.token_graph_contract()
            report["quantization"] = resident.quantization_contract()
            before_close = resident.allocation_contract()
        after_close = memory_stats()
    report["resident_nbytes"] = before_close["resident_nbytes"]
    report["teardown_current_bytes"] = after_close["current_allocated_bytes"]
    report["teardown_active_allocations"] = after_close["active_allocations"]
    if not logits_gate.passed:
        report["failures"].append("logit KL/top-1 gate failed")
    if after_close["current_allocated_bytes"] != before_close["baseline_allocated_bytes"]:
        report["failures"].append("teardown bytes did not return to baseline")
    if after_close["active_allocations"] != before_close["baseline_active_allocations"]:
        report["failures"].append("teardown allocations did not return to baseline")
    report["all_passed"] = not report["failures"]
    text = json.dumps(report, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
