#!/usr/bin/env python3
"""Summarize StepFun Q3_K_L correctness artifacts and remaining blockers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

DEFAULT_PROMPT_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json"
)
DEFAULT_ORACLE_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-llamacpp-step35-timeout.json"
)
DEFAULT_RESOURCE_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-text-resource-dry-run.json"
)
DEFAULT_DOCS_PATH = Path("docs/STEPFUN.md")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-artifact", type=Path, default=DEFAULT_PROMPT_ARTIFACT)
    parser.add_argument("--oracle-artifact", type=Path, default=DEFAULT_ORACLE_ARTIFACT)
    parser.add_argument("--resource-artifact", type=Path, default=DEFAULT_RESOURCE_ARTIFACT)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_PATH)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON output to this path instead of stdout.")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return exit code 2 when the summarized status is not ready.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _artifact_record(path: Path) -> dict[str, object]:
    """Return stable provenance metadata for an input artifact."""

    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _source_artifacts(
    *,
    prompt_artifact: Path,
    oracle_artifact: Path,
    resource_artifact: Path,
    docs_path: Path,
) -> dict[str, object]:
    return {
        "prompt": _artifact_record(prompt_artifact),
        "oracle": _artifact_record(oracle_artifact),
        "text_resource": _artifact_record(resource_artifact),
        "docs": _artifact_record(docs_path),
    }


def _docs_checklist_status(docs_path: Path) -> dict[str, object]:
    text = docs_path.read_text()
    block = text.split("### P0", 1)[1].split("### P13", 1)[0]
    start_line = text[: text.index("### P0")].count("\n") + 1
    items: list[dict[str, object]] = []
    for offset, line in enumerate(block.splitlines(), start=0):
        match = re.match(r"^- \[( |~)\] (.*)", line)
        if match:
            items.append(
                {
                    "line": start_line + offset,
                    "state": "partial" if match.group(1) == "~" else "open",
                    "text": match.group(2),
                }
            )
    return {
        "docs_path": str(docs_path),
        "open_or_partial_count_p0_p12": len(items),
        "open_or_partial_items_p0_p12": items,
    }


_ATTENTION_LINEAR_SUFFIXES = ("attn_q", "attn_k", "attn_v", "attn_gate", "attn_output")
_DENSE_MLP_LINEAR_SUFFIXES = ("ffn_gate", "ffn_up", "ffn_down")
_MOE_ROUTER_SUFFIXES = ("ffn_gate_inp",)
_MOE_EXPERT_LINEAR_SUFFIXES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
_MOE_SHARED_LINEAR_SUFFIXES = ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")
_RESIDENT_LINEAR_SUFFIXES = (
    *_ATTENTION_LINEAR_SUFFIXES,
    *_DENSE_MLP_LINEAR_SUFFIXES,
    *_MOE_EXPERT_LINEAR_SUFFIXES,
    *_MOE_SHARED_LINEAR_SUFFIXES,
)


def _linear_projection_progress(prompt: dict[str, object]) -> dict[str, object]:
    """Summarize resident linear projection coverage from a prompt artifact."""

    slots = [slot for slot in prompt.get("selected_slots", []) if isinstance(slot, str)]
    layer_suffixes: dict[int, set[str]] = {}
    for slot in slots:
        if not slot.startswith("layers."):
            continue
        parts = slot.split(".")
        if len(parts) != 3:
            continue
        try:
            layer_id = int(parts[1])
        except ValueError:
            continue
        layer_suffixes.setdefault(layer_id, set()).add(parts[2])

    suffix_counts: dict[str, int] = {}
    for suffixes in layer_suffixes.values():
        for suffix in suffixes:
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

    def suffix_count(suffixes: Sequence[str]) -> int:
        return sum(suffix_counts.get(suffix, 0) for suffix in suffixes)

    def complete_layers(suffixes: Sequence[str]) -> int:
        required = set(suffixes)
        return sum(1 for layer_slots in layer_suffixes.values() if required <= layer_slots)

    try:
        layer_count = int(prompt.get("layer_count", 0))
    except (TypeError, ValueError):
        layer_count = 0
    selected_layer_count = len(layer_suffixes)
    expected_layer_count = layer_count or selected_layer_count
    root_lm_head_present = "root.lm_head" in slots
    attention_slot_count = suffix_count(_ATTENTION_LINEAR_SUFFIXES)
    dense_slot_count = suffix_count(_DENSE_MLP_LINEAR_SUFFIXES)
    moe_router_slot_count = suffix_count(_MOE_ROUTER_SUFFIXES)
    moe_expert_slot_count = suffix_count(_MOE_EXPERT_LINEAR_SUFFIXES)
    moe_shared_slot_count = suffix_count(_MOE_SHARED_LINEAR_SUFFIXES)
    resident_linear_projection_slot_count = int(root_lm_head_present) + suffix_count(_RESIDENT_LINEAR_SUFFIXES)
    attention_expected = expected_layer_count * len(_ATTENTION_LINEAR_SUFFIXES)
    return {
        "source": "prompt_artifact.selected_slots",
        "execution_mode": prompt.get("execution_mode"),
        "prompt_status": prompt.get("status"),
        "layer_count": layer_count,
        "selected_layer_count": selected_layer_count,
        "selected_slot_count": prompt.get("selected_slot_count", len(slots)),
        "root_lm_head_present": root_lm_head_present,
        "resident_linear_projection_slot_count": resident_linear_projection_slot_count,
        "host_reference_router_projection_slot_count": moe_router_slot_count,
        "attention": {
            "slot_count": attention_slot_count,
            "expected_slot_count": attention_expected,
            "complete_layer_count": complete_layers(_ATTENTION_LINEAR_SUFFIXES),
            "all_selected_layers_complete": (
                selected_layer_count == expected_layer_count
                and attention_slot_count == attention_expected
                and complete_layers(_ATTENTION_LINEAR_SUFFIXES) == expected_layer_count
            ),
        },
        "dense_mlp": {
            "slot_count": dense_slot_count,
            "complete_layer_count": complete_layers(_DENSE_MLP_LINEAR_SUFFIXES),
        },
        "moe_router": {
            "slot_count": moe_router_slot_count,
            "complete_layer_count": complete_layers(_MOE_ROUTER_SUFFIXES),
            "execution_note": "router weights are copied to the host CPU-reference router in current probes",
        },
        "moe_expert": {
            "slot_count": moe_expert_slot_count,
            "complete_layer_count": complete_layers(_MOE_EXPERT_LINEAR_SUFFIXES),
        },
        "moe_shared_expert": {
            "slot_count": moe_shared_slot_count,
            "complete_layer_count": complete_layers(_MOE_SHARED_LINEAR_SUFFIXES),
        },
        "note": (
            "Derived from selected slots in the all-layer host-composed prompt smoke. "
            "This records GGUF linear projection coverage only; it is not KV-backed decode, "
            "oracle parity, or performance evidence."
        ),
    }


def _oracle_progress(oracle: dict[str, object]) -> dict[str, object]:
    """Summarize the current deterministic oracle target and blocker."""

    stdout = str(oracle.get("stdout", ""))
    stderr = str(oracle.get("stderr", ""))
    generated = str(oracle.get("generated_text", ""))
    expected_top_tokens = oracle.get("expected_top_tokens", [])
    return {
        "source": "oracle_artifact",
        "status": oracle.get("status"),
        "oracle_blocker_kind": oracle.get("oracle_blocker_kind"),
        "oracle_blocker_detail": oracle.get("oracle_blocker_detail"),
        "llama_cli": oracle.get("llama_cli"),
        "llama_cpp_version": oracle.get("llama_cpp_version"),
        "model": oracle.get("model"),
        "prompt_length": oracle.get("prompt_length"),
        "n_predict": oracle.get("n_predict"),
        "timeout_s": oracle.get("timeout_s"),
        "elapsed_s": oracle.get("elapsed_s"),
        "extra_llama_args": list(oracle.get("extra_llama_args", [])),
        "command_shell": oracle.get("command_shell"),
        "expected_next_token_id": oracle.get("expected_next_token_id"),
        "expected_next_token_text": oracle.get("expected_next_token_text"),
        "expected_next_token_logit": oracle.get("expected_next_token_logit"),
        "expected_top_tokens": expected_top_tokens if isinstance(expected_top_tokens, list) else [],
        "generated_text_len": len(generated),
        "stdout_len": len(stdout),
        "stderr_len": len(stderr),
        "text_matches_expected_exact": oracle.get("text_matches_expected_exact") is True,
        "text_matches_expected_stripped": oracle.get("text_matches_expected_stripped") is True,
        "comparison_policy": dict(oracle.get("comparison_policy", {})),
        "step35_supported_by_oracle": oracle.get("step35_supported"),
        "note": (
            "This records the deterministic oracle target and current blocker only. "
            "It is not oracle parity unless a comparable generated token matches the expected text/logit policy."
        ),
    }


def _kv_decode_dispatch_progress(resource: dict[str, object]) -> dict[str, object]:
    """Summarize KV dispatch coverage from the text resource artifact."""

    plan = dict(resource.get("text_decode_resource_plan", {}))
    kv_plan = dict(plan.get("kv_decode_kernel_plan", {}))
    registered = dict(kv_plan.get("registered", {}))
    return {
        "source": "resource_artifact.text_decode_resource_plan.kv_decode_kernel_plan",
        "resource_status": resource.get("status"),
        "backend": kv_plan.get("backend") or plan.get("backend"),
        "model_quant": kv_plan.get("model_quant"),
        "kv_storage_dtype": kv_plan.get("kv_storage_dtype"),
        "decode_attention_kind": kv_plan.get("decode_attention_kind"),
        "max_context": kv_plan.get("max_context"),
        "max_new_tokens": kv_plan.get("max_new_tokens"),
        "max_prompt_rows": kv_plan.get("max_prompt_rows"),
        "attention_block_size": kv_plan.get("attention_block_size"),
        "attention_block_table_len": kv_plan.get("attention_block_table_len"),
        "attention_capacity_tokens": kv_plan.get("attention_capacity_tokens"),
        "decode_span": dict(kv_plan.get("decode_span", {})),
        "prompt_span": dict(kv_plan.get("prompt_span", {})),
        "decode_span_shape_compatible": kv_plan.get("decode_span_shape_compatible") is True,
        "prompt_span_shape_compatible": kv_plan.get("prompt_span_shape_compatible") is True,
        "span_shape_compatible": kv_plan.get("span_shape_compatible") is True,
        "dispatch_keys": dict(kv_plan.get("dispatch_keys", {})),
        "registered": registered,
        "all_registered": kv_plan.get("all_registered") is True and all(bool(v) for v in registered.values()),
        "note": (
            "Derived from the text resource dry-run artifact. Registry dispatch readiness "
            "does not mean the streaming KV-backed runner or oracle parity is complete."
        ),
    }


def _status_refresh_command(
    *,
    prompt_artifact: Path,
    oracle_artifact: Path,
    resource_artifact: Path,
    docs_path: Path,
    output_artifact: Path = Path("benchmarks/results/2026-05-31-stepfun-q3kl-correctness-status.json"),
) -> str:
    return (
        "python3 scripts/stepfun_correctness_status.py "
        f"--prompt-artifact {prompt_artifact} "
        f"--oracle-artifact {oracle_artifact} "
        f"--resource-artifact {resource_artifact} "
        f"--docs {docs_path} "
        f"--output {output_artifact} --pretty"
    )


def _resource_plan_refresh_command(
    *,
    output_artifact: Path = DEFAULT_RESOURCE_ARTIFACT,
) -> str:
    return (
        "python3 scripts/stepfun_gguf_load_smoke.py --dry-run-plan "
        "--kv-context-pages 1 --kv-page-size 512 --pretty "
        f"> {output_artifact}"
    )


def _next_action_commands(
    *,
    oracle_progress: dict[str, object],
    prompt_artifact: Path,
    oracle_artifact: Path,
    resource_artifact: Path,
    docs_path: Path,
) -> dict[str, object]:
    """Return copy/pasteable blocker reproduction and refresh commands."""

    status_refresh = _status_refresh_command(
        prompt_artifact=prompt_artifact,
        oracle_artifact=oracle_artifact,
        resource_artifact=resource_artifact,
        docs_path=docs_path,
    )
    return {
        "oracle_parity_blocked": {
            "rerun_command_shell": oracle_progress.get("command_shell"),
            "status_refresh_command": status_refresh,
            "success_criteria": [
                "oracle_progress.status is executed",
                "oracle_parity is true",
                "readiness_gates.oracle_parity.ready is true",
            ],
        },
        "kv_backed_decode_not_wired": {
            "resource_plan_refresh_command": _resource_plan_refresh_command(
                output_artifact=resource_artifact
            ),
            "status_refresh_command": status_refresh,
            "success_criteria": [
                "kv_backed_decode_ready is true",
                "readiness_gates.kv_backed_decode.ready is true",
                "e2e_inference_ready is true only after oracle_parity is also true",
            ],
        },
    }


def _readiness_gates(
    *,
    oracle_parity: bool,
    kv_decode_dispatch_ready: bool,
    kv_backed_decode_ready: bool,
    oracle_progress: dict[str, object],
    kv_decode_dispatch_progress: dict[str, object],
) -> dict[str, object]:
    """Return explicit readiness gates for the remaining StepFun blockers."""

    return {
        "oracle_parity": {
            "ready": oracle_parity,
            "blocked_by": None if oracle_parity else oracle_progress.get("oracle_blocker_kind"),
            "required_evidence": (
                "A StepFun/step35-capable oracle must generate a comparable token and match the "
                "expected next-token text/logit policy recorded in oracle_progress."
            ),
            "expected_next_token_id": oracle_progress.get("expected_next_token_id"),
            "expected_next_token_text": oracle_progress.get("expected_next_token_text"),
            "current_oracle_status": oracle_progress.get("status"),
        },
        "kv_backed_decode": {
            "ready": kv_backed_decode_ready,
            "blocked_by": None if kv_backed_decode_ready else "kv_backed_decode_not_wired",
            "dispatch_ready": kv_decode_dispatch_ready,
            "required_evidence": (
                "A streaming runner must materialize prompt KV rows, launch one-token decode KV "
                "write and gated paged attention from the resident cache, then feed final logits "
                "without host-composed layer outputs."
            ),
            "current_evidence": {
                "dispatch_ready": kv_decode_dispatch_ready,
                "decode_span_shape_compatible": kv_decode_dispatch_progress.get(
                    "decode_span_shape_compatible"
                ),
                "prompt_span_shape_compatible": kv_decode_dispatch_progress.get(
                    "prompt_span_shape_compatible"
                ),
                "resident_prompt_smoke": "host_composed_layer_prefix",
            },
        },
        "e2e_inference": {
            "ready": oracle_parity and kv_backed_decode_ready,
            "blocked_by": [
                name
                for name, ready in (
                    ("oracle_parity", oracle_parity),
                    ("kv_backed_decode", kv_backed_decode_ready),
                )
                if not ready
            ],
            "required_evidence": (
                "Both oracle parity and KV-backed decode readiness must be true before StepFun "
                "text-only GGUF inference is marked ready."
            ),
        },
    }


def build_status(
    prompt_artifact: Path,
    oracle_artifact: Path,
    docs_path: Path = DEFAULT_DOCS_PATH,
    resource_artifact: Path = DEFAULT_RESOURCE_ARTIFACT,
) -> dict[str, object]:
    prompt = _load(prompt_artifact)
    oracle = _load(oracle_artifact)
    resource = _load(resource_artifact)
    docs_status = _docs_checklist_status(docs_path)
    all_layer_prompt_smoke = (
        prompt.get("status") == "partial_prompt_smoke"
        and prompt.get("execution_mode") == "chunked"
        and prompt.get("layer_count") == 45
        and prompt.get("skipped_layers") == []
        and prompt.get("no_vision_projector_mtp_slots") is True
        and prompt.get("memory_stats_after_free", {}).get("active_allocations") == 0
        and prompt.get("memory_stats_after_free", {}).get("current_allocated_bytes") == 0
    )
    oracle_parity = (
        oracle.get("status") == "executed"
        and oracle.get("returncode") == 0
        and oracle.get("text_matches_expected_exact") is True
    )
    oracle_progress = _oracle_progress(oracle)
    kv_decode_dispatch_progress = _kv_decode_dispatch_progress(resource)
    kv_decode_dispatch_ready = kv_decode_dispatch_progress["all_registered"] is True
    kv_backed_decode_ready = False
    e2e_inference_ready = oracle_parity and kv_backed_decode_ready
    readiness_gates = _readiness_gates(
        oracle_parity=oracle_parity,
        kv_decode_dispatch_ready=kv_decode_dispatch_ready,
        kv_backed_decode_ready=kv_backed_decode_ready,
        oracle_progress=oracle_progress,
        kv_decode_dispatch_progress=kv_decode_dispatch_progress,
    )
    next_action_commands = _next_action_commands(
        oracle_progress=oracle_progress,
        prompt_artifact=prompt_artifact,
        oracle_artifact=oracle_artifact,
        resource_artifact=resource_artifact,
        docs_path=docs_path,
    )
    blockers: list[dict[str, object]] = []
    if not oracle_parity:
        blockers.append(
            {
                "kind": "oracle_parity_blocked",
                "detail": oracle.get("oracle_blocker_detail")
                or "llama.cpp/CPU oracle result has not matched the StepFun artifact yet",
                "artifact": str(oracle_artifact),
                "oracle_blocker_kind": oracle.get("oracle_blocker_kind"),
                "expected_next_token_id": oracle_progress.get("expected_next_token_id"),
                "expected_next_token_text": oracle_progress.get("expected_next_token_text"),
                "elapsed_s": oracle_progress.get("elapsed_s"),
                "timeout_s": oracle_progress.get("timeout_s"),
            }
        )
    blockers.append(
        {
            "kind": "kv_backed_decode_not_wired",
            "detail": (
                "KV dispatch registry keys are recorded as ready in the text resource plan; "
                "current all-layer prompt smoke is still host-composed prefill/logits, so the "
                "final KV-backed one-token decode runner remains open."
            ),
            "artifact": str(prompt_artifact),
            "resource_artifact": str(resource_artifact),
            "kv_decode_dispatch_ready": kv_decode_dispatch_ready,
        }
    )
    next_actions = [
        {
            "blocker_kind": "oracle_parity_blocked",
            "action": (
                "Build or locate a StepFun/step35-capable llama.cpp or CPU oracle, rerun "
                "scripts/stepfun_llamacpp_oracle.py --execute, and record exact token/logit comparison."
            ),
        },
        {
            "blocker_kind": "kv_backed_decode_not_wired",
            "action": (
                "Replace the host-composed layer-prefix prompt smoke with a KV-backed one-token decode runner "
                "using StepFunResidentSession weight/KV ownership and the validated layer probes."
            ),
        },
    ]
    return {
        "status": "blocked" if blockers else "ready",
        "model": "Step-3.7-flash-Q3_K_L",
        "source_artifacts": _source_artifacts(
            prompt_artifact=prompt_artifact,
            oracle_artifact=oracle_artifact,
            resource_artifact=resource_artifact,
            docs_path=docs_path,
        ),
        "backend": prompt.get("backend", "hip_gfx1151"),
        "prompt_artifact": str(prompt_artifact),
        "oracle_artifact": str(oracle_artifact),
        "text_resource_artifact": str(resource_artifact),
        "all_layer_prompt_smoke": all_layer_prompt_smoke,
        "all_layer_prompt_next_token_id": prompt.get("next_token_id"),
        "all_layer_prompt_next_token_text": prompt.get("next_token_text"),
        "all_layer_prompt_peak_resident_weight_nbytes": prompt.get("peak_resident_weight_nbytes"),
        "oracle_parity": oracle_parity,
        "oracle_status": oracle.get("status"),
        "oracle_elapsed_s": oracle.get("elapsed_s"),
        "oracle_llama_cpp_version": oracle.get("llama_cpp_version"),
        "oracle_stdout_len": len(str(oracle.get("stdout", ""))),
        "oracle_stderr_len": len(str(oracle.get("stderr", ""))),
        "oracle_blocker_kind": oracle.get("oracle_blocker_kind"),
        "step35_supported_by_local_llama_cpp": oracle.get("step35_supported"),
        "oracle_progress": oracle_progress,
        "linear_projection_progress": _linear_projection_progress(prompt),
        "kv_decode_dispatch_progress": kv_decode_dispatch_progress,
        "kv_decode_dispatch_ready": kv_decode_dispatch_ready,
        "kv_backed_decode_ready": kv_backed_decode_ready,
        "e2e_inference_ready": e2e_inference_ready,
        "readiness_gates": readiness_gates,
        "blockers": blockers,
        "next_actions": next_actions,
        "next_action_commands": next_action_commands,
        "docs_checklist": docs_status,
        "note": (
            "Host-composed all-layer prompt smoke is present; true e2e inference still needs "
            "oracle parity and KV-backed decode."
        ),
    }


def _emit_json(result: dict[str, object], *, pretty: bool, output: Path | None) -> None:
    text = json.dumps(result, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    status = build_status(
        args.prompt_artifact,
        args.oracle_artifact,
        args.docs,
        resource_artifact=args.resource_artifact,
    )
    _emit_json(
        status,
        pretty=args.pretty,
        output=args.output,
    )
    if args.fail_on_blocked and status["status"] != "ready":
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
