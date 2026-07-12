#!/usr/bin/env python3
"""Plan the next layer-0 BF16 bisection step at conv/GDN state effects."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader  # noqa: E402

DEFAULT_PROJECTION_ARTIFACT = Path(
    "benchmarks/results/mtp-gguf-iter323-layer0-bf16-projection-oracle.json"
)
DEFAULT_RUNNER = Path("hipengine/runtime/qwen35_gguf_runner.py")
DEFAULT_CONV_KERNEL = Path("hipengine/kernels/hip_gfx1100/linear_attn/conv.hip")
DEFAULT_GDN_KERNEL = Path("hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip")
DEFAULT_LLAMACPP_QWEN35MOE = Path(
    "/home/lhl/llama.cpp/llama.cpp-hip/src/models/qwen35moe.cpp"
)
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter324-layer0-conv-gdn-plan.json")
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")

ModelMetadataLoader = Callable[[Path], dict[str, Any]]

REQUIRED_PROJECTION_CLASSIFICATIONS = {
    "layer0_projections_match_bf16_oracle_exactly",
    "layer0_projections_match_bf16_oracle_within_rounding",
}

LINEAR_ATTENTION_TENSORS = {
    "ssm_conv1d": "ssm_conv1d.weight",
    "ssm_dt_bias": "ssm_dt.bias",
    "ssm_a": "ssm_a",
    "ssm_norm": "ssm_norm.weight",
    "ssm_out": "ssm_out.weight",
    "ssm_alpha": "ssm_alpha.weight",
    "ssm_beta": "ssm_beta.weight",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-artifact", type=Path, default=DEFAULT_PROJECTION_ARTIFACT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--conv-kernel", type=Path, default=DEFAULT_CONV_KERNEL)
    parser.add_argument("--gdn-kernel", type=Path, default=DEFAULT_GDN_KERNEL)
    parser.add_argument("--llamacpp-qwen35moe", type=Path, default=DEFAULT_LLAMACPP_QWEN35MOE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=324)
    args = parser.parse_args()

    artifact = build_layer0_conv_gdn_plan(
        projection_artifact_path=args.projection_artifact,
        runner_path=args.runner,
        conv_kernel_path=args.conv_kernel,
        gdn_kernel_path=args.gdn_kernel,
        llamacpp_qwen35moe_path=args.llamacpp_qwen35moe,
        model_path=args.model,
        iteration=args.iteration,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "conclusion": artifact["conclusion"],
                "selected_strategy": artifact["decision"]["selected_strategy"],
                "first_probe": artifact["next_probe_plan"]["first_probe"],
                "next_action": artifact["next_action"],
            },
            indent=2,
        )
    )


def build_layer0_conv_gdn_plan(
    *,
    projection_artifact_path: Path,
    runner_path: Path,
    conv_kernel_path: Path,
    gdn_kernel_path: Path,
    llamacpp_qwen35moe_path: Path,
    model_path: Path,
    iteration: int = 324,
    metadata_loader: ModelMetadataLoader = None,
) -> dict[str, Any]:
    projection = json.loads(projection_artifact_path.read_text())
    runner_text = runner_path.read_text()
    conv_text = conv_kernel_path.read_text()
    gdn_text = gdn_kernel_path.read_text()
    llama_text = llamacpp_qwen35moe_path.read_text()
    selected_loader = metadata_loader or load_model_metadata
    model = selected_loader(model_path)
    prereq = audit_projection_prerequisite(projection)
    runtime = audit_runtime_conv_gdn_sequence(runner_text)
    kernels = audit_kernel_state_dependencies(conv_text, gdn_text)
    llama = audit_llamacpp_linear_attention_anchors(llama_text)
    decision = decide_conv_gdn_strategy(
        projection_ready=prereq["ready"],
        runtime_ready=runtime["ready"],
        kernels_ready=kernels["ready"],
        llama_ready=llama["ready"],
        model_ready=model["ready"],
    )
    status = "ready" if decision["ready"] else "blocked"
    return {
        "schema": 1,
        "kind": "layer0_conv_gdn_bisection_plan",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": status,
        "projection_artifact_path": str(projection_artifact_path),
        "runner_path": str(runner_path),
        "conv_kernel_path": str(conv_kernel_path),
        "gdn_kernel_path": str(gdn_kernel_path),
        "llamacpp_qwen35moe_path": str(llamacpp_qwen35moe_path),
        "model": str(model_path),
        "layer_id": projection.get("layer_id"),
        "position": projection.get("position"),
        "token_id": projection.get("token_id"),
        "projection_prerequisite": prereq,
        "runtime_sequence": runtime,
        "kernel_state_dependencies": kernels,
        "llamacpp_anchors": llama,
        "model_metadata": model,
        "decision": decision,
        "next_probe_plan": build_next_probe_plan(model),
        "constraints": build_constraints(),
        "conclusion": decision["conclusion"],
        "external_checkout_modified": False,
        "next_action": decision["next_action"],
    }


def audit_projection_prerequisite(projection: Mapping[str, Any]) -> dict[str, Any]:
    qkv = projection.get("projection_results", {}).get("linear_qkv_f32", {})
    z = projection.get("projection_results", {}).get("linear_z_f32", {})
    facts = {
        "status_ready": projection.get("status") == "ready",
        "classification_ready": projection.get("classification")
        in REQUIRED_PROJECTION_CLASSIFICATIONS,
        "attn_norm_exact": projection.get("attn_norm_oracle", {})
        .get("delta_vs_hip", {})
        .get("exact_match")
        is True,
        "linear_qkv_usable": str(qkv.get("classification", "")).startswith(
            "projection_matches_bf16_oracle"
        ),
        "linear_z_usable": str(z.get("classification", "")).startswith(
            "projection_matches_bf16_oracle"
        ),
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "evidence": {
            "classification": projection.get("classification"),
            "qkv_classification": qkv.get("classification"),
            "qkv_bf16_max_abs_diff": (qkv.get("delta_bf16_oracle_vs_hip") or {}).get(
                "max_abs_diff"
            ),
            "linear_z_classification": z.get("classification"),
            "linear_z_bf16_max_abs_diff": (z.get("delta_bf16_oracle_vs_hip") or {}).get(
                "max_abs_diff"
            ),
            "next_action": projection.get("next_action"),
        },
    }


def audit_runtime_conv_gdn_sequence(runner_text: str) -> dict[str, Any]:
    body = extract_function_body(runner_text, "_run_linear_attention_attn_only")
    capture_body = extract_function_body(runner_text, "capture_linear_attention_boundary")
    facts = {
        "attn_norm_before_projection": ordered(body, "gguf_rmsnorm", "launch_gguf_linear_pair"),
        "projection_before_alpha_beta": ordered(body, "attn_qkv", "ssm_alpha"),
        "alpha_beta_before_conv": ordered(body, "ssm_beta", "qwen35_linear_attn_conv_decode_bf16"),
        "conv_before_gdn": ordered(
            body,
            "qwen35_linear_attn_conv_decode_bf16",
            "qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16",
        ),
        "gdn_before_recurrent_cast": ordered(
            body,
            "qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16",
            "f32_to_bf16",
        ),
        "recurrent_cast_before_ssm_out": ordered(body, "f32_to_bf16", "ssm_out"),
        "captures_conv_out": "conv_out_f32=" in capture_body,
        "captures_recurrent_out": "recurrent_out_f32=" in capture_body,
        "captures_recurrent_bf16": "recurrent_bf16_f32=" in capture_body,
        "captures_attn_out": "attn_out_f32=" in capture_body,
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "fields_to_compare": [
            "conv_out_f32",
            "recurrent_out_f32",
            "recurrent_bf16_f32",
            "attn_out_f32",
        ],
        "stateful_inputs": [
            "scratch.layer_conv_states[layer_id]",
            "scratch.layer_recurrent_states[layer_id]",
        ],
    }


def audit_kernel_state_dependencies(conv_text: str, gdn_text: str) -> dict[str, Any]:
    facts = {
        "conv_decode_lowp_reads_bf16_input": "const scalar_t* hidden_states" in conv_text,
        "conv_decode_updates_state_shift": "conv_state[offset + idx] = value" in conv_text,
        "conv_decode_appends_newest": "conv_state[offset + kernel_size - 1] = newest" in conv_text,
        "conv_decode_outputs_silu": "out[channel] = silu_f32(acc)" in conv_text,
        "gdn_reads_recurrent_state": "recurrent_state[state_col" in gdn_text,
        "gdn_updates_recurrent_state": "recurrent_state[state_offset] = new_state" in gdn_text,
        "gdn_uses_sigmoid_beta": "sigmoid_f32(scalar_to_float_qwen35(b[v_head]))" in gdn_text,
        "gdn_uses_softplus_decay": "softplus_f32(scalar_to_float_qwen35(a[v_head])" in gdn_text,
        "gdn_rmsnorm_gates_output": "norm_weight[value_idx] * silu_f32" in gdn_text,
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "position0_zero_state_feasible": facts["conv_decode_updates_state_shift"]
        and facts["gdn_reads_recurrent_state"],
        "warm_position_requires_state_replay_or_capture": True,
        "state_mutations": [
            "conv_state shifts left and appends the current BF16 linear_qkv row",
            "recurrent_state decays and updates per v_head/value/key element",
        ],
    }


def audit_llamacpp_linear_attention_anchors(llama_text: str) -> dict[str, Any]:
    facts = {
        "build_conv_state_present": "build_conv_state" in llama_text,
        "ggml_ssm_conv_present": "ggml_ssm_conv" in llama_text,
        "conv_output_raw_callback": 'cb(conv_output_proper, "conv_output_raw", il)' in llama_text,
        "conv_output_silu_callback": 'cb(conv_output_silu, "conv_output_silu", il)' in llama_text,
        "qkv_views_present": "q_conv = ggml_view_4d" in llama_text
        and "v_conv = ggml_view_4d" in llama_text,
        "recurrent_attn_present": "build_recurrent_attn" in llama_text,
        "final_output_callback": 'cb(final_output, "final_output", il)' in llama_text,
        "linear_attn_out_callback": 'cb(cur, "linear_attn_out", il)' in llama_text,
    }
    return {
        "ready": all(facts.values()),
        "facts": facts,
        "candidate_taps": [
            "conv_output_silu",
            "q_conv_predelta",
            "k_conv_predelta",
            "v_conv_predelta",
            "final_output",
            "linear_attn_out",
        ],
    }


def load_model_metadata(model_path: Path) -> dict[str, Any]:
    reader = GGUFReader(model_path)
    metadata = reader.info.metadata
    dims = {
        "ssm_conv_kernel": int(metadata["qwen35moe.ssm.conv_kernel"]),
        "ssm_group_count": int(metadata["qwen35moe.ssm.group_count"]),
        "ssm_inner_size": int(metadata["qwen35moe.ssm.inner_size"]),
        "ssm_state_size": int(metadata["qwen35moe.ssm.state_size"]),
        "ssm_time_step_rank": int(metadata["qwen35moe.ssm.time_step_rank"]),
    }
    dims["linear_qkv_width"] = (
        2 * dims["ssm_group_count"] * dims["ssm_state_size"]
        + dims["ssm_inner_size"]
    )
    dims["ssm_value_dim"] = dims["ssm_inner_size"] // dims["ssm_time_step_rank"]
    dims["conv_state_floats"] = dims["linear_qkv_width"] * dims["ssm_conv_kernel"]
    dims["recurrent_state_floats"] = (
        dims["ssm_time_step_rank"] * dims["ssm_state_size"] * dims["ssm_value_dim"]
    )
    tensors = {}
    for slot, suffix in LINEAR_ATTENTION_TENSORS.items():
        name = f"blk.0.{suffix}"
        info = reader.tensor_info(name)
        tensors[slot] = {
            "tensor_name": name,
            "ggml_type": info.ggml_type_name,
            "shape": list(info.shape),
            "nbytes": int(info.nbytes),
        }
    facts = {
        "conv_kernel_is_4": dims["ssm_conv_kernel"] == 4,
        "qkv_width_matches_conv_channels": tensors["ssm_conv1d"]["shape"]
        == [dims["linear_qkv_width"], dims["ssm_conv_kernel"]],
        "alpha_beta_f32_to_bf16_materialization_expected": tensors["ssm_alpha"]["ggml_type"]
        == "F32"
        and tensors["ssm_beta"]["ggml_type"] == "F32",
        "ssm_out_quantized": tensors["ssm_out"]["ggml_type"] == "Q8_0",
    }
    return {
        "ready": all(facts.values()),
        "dimensions": dims,
        "tensors": tensors,
        "facts": facts,
    }


def decide_conv_gdn_strategy(
    *,
    projection_ready: bool,
    runtime_ready: bool,
    kernels_ready: bool,
    llama_ready: bool,
    model_ready: bool,
) -> dict[str, Any]:
    ready = all((projection_ready, runtime_ready, kernels_ready, llama_ready, model_ready))
    if ready:
        return {
            "ready": True,
            "selected_strategy": "position0_stateless_conv_gdn_oracle_first",
            "conclusion": "layer0_conv_gdn_plan_ready",
            "reason": (
                "Current position 16 depends on warmed conv/recurrent state. "
                "A position-0 oracle has zero initial state and can validate "
                "conv/GDN math and weight contracts before adding replay/capture "
                "for the warm-state position-16 comparison."
            ),
            "next_action": "build_position0_layer0_conv_gdn_oracle",
        }
    return {
        "ready": False,
        "selected_strategy": "blocked",
        "conclusion": "layer0_conv_gdn_plan_missing_required_fact",
        "reason": "One or more projection/runtime/kernel/llama/model facts are missing.",
        "next_action": "inspect_layer0_conv_gdn_plan_blockers",
    }


def build_next_probe_plan(model: Mapping[str, Any]) -> dict[str, Any]:
    dims = model.get("dimensions", {})
    return {
        "first_probe": "position0_conv_out_recurrent_out_attn_out",
        "why_position0_first": (
            "position 0 starts from zero conv/recurrent state, while position 16 "
            "requires replaying the prompt or capturing pre-token recurrent state"
        ),
        "position0_expected_oracles": [
            {
                "field": "conv_out_f32",
                "formula": (
                    "silu(BF16(linear_qkv)[channel] * ssm_conv1d[channel, "
                    "kernel_size - 1]) with zero prior conv state"
                ),
                "dtype_contract": "BF16 linear_qkv input, F32 conv state/output",
            },
            {
                "field": "recurrent_out_f32",
                "formula": (
                    "GDN recurrent update from zero recurrent state using conv_out, "
                    "BF16 linear_z/alpha/beta, ssm_dt_bias, log(-ssm_a), and ssm_norm"
                ),
                "dtype_contract": "F32 recurrent output before BF16 cast",
            },
            {
                "field": "recurrent_bf16_f32",
                "formula": "BF16(recurrent_out_f32) copied back to F32",
                "dtype_contract": "BF16 output copied to F32",
            },
            {
                "field": "attn_out_f32",
                "formula": "ssm_out(Q8_0) projection from recurrent_bf16",
                "dtype_contract": "BF16 input/output copied to F32",
            },
        ],
        "warm_position16_followup": {
            "requires": [
                "replay tokens 0..15 through the same contracted CPU conv/GDN path",
                "or add a diagnostic pre-token conv/recurrent state capture",
            ],
            "state_sizes": {
                "conv_state_floats": dims.get("conv_state_floats"),
                "recurrent_state_floats": dims.get("recurrent_state_floats"),
            },
        },
    }


def build_constraints() -> dict[str, Any]:
    return {
        "must_not_change_hot_path": True,
        "no_torch_hot_path": True,
        "no_backend_or_quant_dispatch_branch": True,
        "no_performance_claim": True,
        "oracle_exactness_before_deeper_bisection": True,
        "selected_oracle_policy": "bf16_contracted_resident_activation",
        "mtp_seed_contract_unchanged": "post_output_norm fp32 hidden seed",
    }


def extract_function_body(text: str, name: str) -> str:
    pattern = rf"^    def {re.escape(name)}\(.*?(?=^    def |^class |\Z)"
    match = re.search(pattern, text, re.S | re.M)
    return match.group(0) if match else ""


def ordered(text: str, first: str, second: str) -> bool:
    first_idx = text.find(first)
    second_idx = text.find(second)
    return first_idx >= 0 and second_idx >= 0 and first_idx < second_idx


if __name__ == "__main__":
    main()
