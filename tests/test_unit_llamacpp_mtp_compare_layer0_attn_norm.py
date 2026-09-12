from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.llamacpp_mtp_compare_input_embed import f32_to_bf16_roundtrip
from scripts.llamacpp_mtp_compare_layer0_attn_norm import (
    annotate_effective_attn_norm_tap,
    classify_layer0_attn_norm,
    compare_layer0_attn_norm,
    next_action,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import pack_float32


def test_compare_layer0_attn_norm_matches_exactly_with_fake_harness(
    tmp_path: Path,
) -> None:
    values = [1.0, -2.0, 3.5]
    compile_artifact = _write_compile_artifact(
        tmp_path,
        _write_fake_harness(tmp_path, values),
    )
    plan = _write_plan_reference(tmp_path)

    artifact = compare_layer0_attn_norm(
        compile_artifact_path=compile_artifact,
        plan_artifact_path=plan,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(1, 2, 3),
        position=2,
        layer_id=0,
        output_prefix=tmp_path / "capture" / "attn-norm-pos2",
        all_rows=True,
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture(values),
    )

    assert artifact["status"] == "matched"
    assert artifact["classification"] == "layer0_attn_norm_exact_match"
    assert artifact["numeric_delta"]["exact_match"] is True
    assert artifact["bf16_rounded_delta"]["available"] is True
    assert artifact["llamacpp_capture"]["effective_tap"] == "h_nextn_layer0_attn_norm"
    assert artifact["numeric_delta"]["all_rows_scan"]["matches"][0]["row"] == 0
    assert artifact["next_action"] == (
        "continue_layer0_subboundary_bisect_inside_linear_attention"
    )
    json.dumps(artifact)


def test_compare_layer0_attn_norm_classifies_bf16_roundtrip_match(
    tmp_path: Path,
) -> None:
    llama = [1.1, -2.3, 3.7]
    hip = [f32_to_bf16_roundtrip(value) for value in llama]
    compile_artifact = _write_compile_artifact(
        tmp_path,
        _write_fake_harness(tmp_path, llama),
    )
    plan = _write_plan_reference(tmp_path)

    artifact = compare_layer0_attn_norm(
        compile_artifact_path=compile_artifact,
        plan_artifact_path=plan,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(7, 8),
        position=1,
        layer_id=0,
        output_prefix=tmp_path / "capture" / "attn-norm-pos1",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture(hip),
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "layer0_attn_norm_matches_after_bf16_roundtrip"
    assert artifact["numeric_delta"]["exact_match"] is False
    assert artifact["bf16_rounded_delta"]["exact_match"] is True
    assert artifact["next_action"] == (
        "continue_layer0_subboundary_bisect_inside_linear_attention"
    )


def test_compare_layer0_attn_norm_classifies_bf16_mismatch(
    tmp_path: Path,
) -> None:
    llama = [1.1, -2.3]
    hip = [f32_to_bf16_roundtrip(1.1) + 0.25, f32_to_bf16_roundtrip(-2.3)]
    compile_artifact = _write_compile_artifact(
        tmp_path,
        _write_fake_harness(tmp_path, llama),
    )
    plan = _write_plan_reference(tmp_path)

    artifact = compare_layer0_attn_norm(
        compile_artifact_path=compile_artifact,
        plan_artifact_path=plan,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(7, 8),
        position=1,
        layer_id=0,
        output_prefix=tmp_path / "capture" / "attn-norm-pos1",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture(hip),
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "layer0_attn_norm_mismatch_after_bf16_roundtrip"
    assert artifact["bf16_rounded_delta"]["max_abs_diff"] == 0.25
    assert artifact["next_action"] == "audit_layer0_attn_norm_rmsnorm_or_weight_materialization"


def test_compare_layer0_attn_norm_requires_layer_zero(tmp_path: Path) -> None:
    compile_artifact = _write_compile_artifact(
        tmp_path,
        _write_fake_harness(tmp_path, [1.0]),
    )
    plan = _write_plan_reference(tmp_path)

    try:
        compare_layer0_attn_norm(
            compile_artifact_path=compile_artifact,
            plan_artifact_path=plan,
            model_path=tmp_path / "model.gguf",
            prompt_tokens=(1,),
            position=0,
            layer_id=1,
            output_prefix=tmp_path / "capture" / "attn-norm-pos0",
            hip_capture_fn=_hip_capture([1.0]),
        )
    except ValueError as exc:
        assert "layer_id=0" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")


def test_attn_norm_helpers_and_annotation() -> None:
    capture = {"metadata": {"tap": "h_nextn_post_output_norm"}}

    annotate_effective_attn_norm_tap(capture, layer_id=0)

    assert capture["effective_tap"] == "h_nextn_layer0_attn_norm"
    assert capture["effective_layer_id"] == 0
    assert "generic hidden-seed harness" in capture["metadata_tap_note"]
    assert classify_layer0_attn_norm(
        "mismatched",
        {"shape_match": True},
        {"available": True, "exact_match": True},
    ) == "layer0_attn_norm_matches_after_bf16_roundtrip"
    assert next_action("llamacpp_capture_failed", "layer0_attn_norm_exact_match") == (
        "inspect_layer0_attn_norm_llamacpp_capture_logs"
    )


def _write_compile_artifact(tmp_path: Path, executable: Path) -> Path:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(exist_ok=True)
    path = tmp_path / "compile.json"
    path.write_text(
        json.dumps({"outputs": {"executable": str(executable)}, "lib_dir": str(lib_dir)})
    )
    return path


def _write_plan_reference(tmp_path: Path) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "status": "ready",
                "conclusion": "layer0_attn_norm_capture_plan_ready",
                "comparison_plan": {
                    "first_probe": "layer0_attn_norm_vs_linear_boundary_attn_norm_f32",
                    "llamacpp_effective_tap": "h_nextn_layer0_attn_norm",
                    "hipengine_value_field": "attn_norm_f32",
                },
                "prior_input_embed_result": {"bf16_exact_match": True},
                "prior_layer0_result": {"rmse": 0.012},
                "next_action": "build_layer0_attn_norm_capture_and_compare",
            }
        )
    )
    return path


def _write_fake_harness(tmp_path: Path, values: list[float]) -> Path:
    exe = tmp_path / f"fake-attn-norm-{len(list(tmp_path.glob('fake-attn-norm-*')))}"
    raw_literal = repr(pack_float32(values))
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "prefix = pathlib.Path(args[args.index('--output-prefix') + 1])\n"
        "prefix.parent.mkdir(parents=True, exist_ok=True)\n"
        f"raw = {raw_literal}\n"
        "prefix.with_suffix('.f32').write_bytes(raw)\n"
        "if '--all-rows' in args:\n"
        "    prefix.with_suffix('.all.f32').write_bytes(raw * 2)\n"
        "meta = {\n"
        "    'kind': 'llamacpp_hidden_seed_capture',\n"
        "    'tap': 'h_nextn_post_output_norm',\n"
        "    'binary': str(prefix.with_suffix('.f32')),\n"
        f"    'n_embd': {len(values)},\n"
        "}\n"
        "if '--all-rows' in args:\n"
        "    meta['all_rows_binary'] = str(prefix.with_suffix('.all.f32'))\n"
        "prefix.with_suffix('.json').write_text(json.dumps(meta))\n"
        "print('{\"captured_layer0_attn_norm\":true}')\n"
    )
    exe.chmod(0o755)
    return exe


def _hip_capture(values: list[float]):
    def capture(
        _model: Path,
        prompt_tokens: tuple[int, ...],
        position: int,
        layer_id: int,
        _max_seq: int | None,
    ):
        return {
            "status": "captured",
            "mode": "capture_linear_attention_boundary_attn_norm",
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "layer_id": int(layer_id),
            "dtype": "BF16_to_F32",
            "provenance": "capture_linear_attention_boundary.attn_norm_f32",
            "values": [float(value) for value in values],
        }

    return capture


def _prepend_path(path: Path) -> str:
    return str(path) + os.pathsep + os.environ.get("PATH", "")
