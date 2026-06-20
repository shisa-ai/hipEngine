from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.llamacpp_mtp_compare_pre_output_norm import (
    classify_pre_vs_post,
    compare_pre_output_norm,
    next_action,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import pack_float32, sha256_bytes


def test_compare_pre_output_norm_matches_with_fake_harness(tmp_path: Path) -> None:
    values = [1.0, -2.0, 3.5, 4.25]
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, values))
    post_artifact = _write_post_artifact(tmp_path, rmse=3.0)

    artifact = compare_pre_output_norm(
        compile_artifact_path=compile_artifact,
        post_output_artifact_path=post_artifact,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(1, 2, 3),
        position=2,
        output_prefix=tmp_path / "capture" / "pre-pos2",
        all_rows=True,
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture(values),
    )

    assert artifact["status"] == "matched"
    assert artifact["classification"] == "pre_output_matches_post_output_mismatch_is_output_norm"
    assert artifact["numeric_delta"]["exact_match"] is True
    assert artifact["numeric_delta"]["all_rows_scan"]["matches"][0]["row"] == 0
    assert artifact["hipengine_capture"]["summary"]["sha256"] == sha256_bytes(
        pack_float32(values)
    )
    assert artifact["next_action"] == "fix_or_match_output_norm_precision_for_mtp_seed"
    json.dumps(artifact)


def test_compare_pre_output_norm_much_closer_classifies_output_norm_suspect(tmp_path: Path) -> None:
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, [1.0, 2.0]))
    post_artifact = _write_post_artifact(tmp_path, rmse=10.0)

    artifact = compare_pre_output_norm(
        compile_artifact_path=compile_artifact,
        post_output_artifact_path=post_artifact,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(7, 8),
        position=1,
        output_prefix=tmp_path / "capture" / "pre-pos1",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture([1.5, 2.0]),
    )

    assert artifact["status"] == "mismatched"
    assert artifact["numeric_delta"]["max_abs_diff"] == 0.5
    assert artifact["classification"] == "pre_output_much_closer_output_norm_suspect"
    assert artifact["next_action"] == "audit_output_norm_kernel_precision_against_llamacpp"


def test_compare_pre_output_norm_reports_pre_mismatch_already_present(tmp_path: Path) -> None:
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, [1.0, 2.0]))
    post_artifact = _write_post_artifact(tmp_path, rmse=1.0)

    artifact = compare_pre_output_norm(
        compile_artifact_path=compile_artifact,
        post_output_artifact_path=post_artifact,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(7, 8),
        position=1,
        output_prefix=tmp_path / "capture" / "pre-pos1",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture([3.0, 2.0]),
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "pre_output_mismatch_already_present"
    assert artifact["next_action"] == "bisect_final_decoder_layer_output_before_output_norm"


def test_compare_pre_output_norm_requires_final_position(tmp_path: Path) -> None:
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, [1.0]))
    post_artifact = _write_post_artifact(tmp_path, rmse=1.0)

    try:
        compare_pre_output_norm(
            compile_artifact_path=compile_artifact,
            post_output_artifact_path=post_artifact,
            model_path=tmp_path / "model.gguf",
            prompt_tokens=(1, 2, 3),
            position=1,
            output_prefix=tmp_path / "capture" / "pre-pos1",
            hip_capture_fn=_hip_capture([1.0]),
        )
    except ValueError as exc:
        assert "final prompt token" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")


def test_classification_detects_overwritten_llamacpp_patch() -> None:
    classification = classify_pre_vs_post(
        pre_delta={
            "available": True,
            "shape_match": True,
            "llamacpp_sha256": "same-sha",
            "rmse": 2.0,
        },
        post_artifact={"numeric_delta": {"llamacpp_sha256": "same-sha", "rmse": 3.0}},
    )

    assert classification == "llamacpp_pre_output_patch_overwritten_by_post_output_h_nextn"
    assert next_action("mismatched", classification) == (
        "move_or_replace_post_output_h_nextn_assignment_in_llamacpp_patch"
    )


def test_classification_and_next_action_unavailable() -> None:
    classification = classify_pre_vs_post(
        pre_delta={"available": False},
        post_artifact={"numeric_delta": {"rmse": 3.0}},
    )

    assert classification == "pre_output_comparison_unavailable"
    assert next_action("llamacpp_capture_failed", classification) == (
        "inspect_pre_output_norm_llamacpp_capture_logs"
    )


def _write_compile_artifact(tmp_path: Path, executable: Path) -> Path:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(exist_ok=True)
    path = tmp_path / "compile.json"
    path.write_text(
        json.dumps({"outputs": {"executable": str(executable)}, "lib_dir": str(lib_dir)})
    )
    return path


def _write_post_artifact(tmp_path: Path, *, rmse: float) -> Path:
    path = tmp_path / "post.json"
    path.write_text(
        json.dumps(
            {
                "status": "mismatched",
                "numeric_delta": {
                    "rmse": rmse,
                    "max_abs_diff": rmse + 1.0,
                    "mean_abs_diff": rmse / 2.0,
                },
            }
        )
    )
    return path


def _write_fake_harness(tmp_path: Path, values: list[float]) -> Path:
    exe = tmp_path / "fake-pre-output"
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
        "    'tap': 'h_nextn_pre_output_norm',\n"
        "    'binary': str(prefix.with_suffix('.f32')),\n"
        f"    'n_embd': {len(values)},\n"
        "}\n"
        "if '--all-rows' in args:\n"
        "    meta['all_rows_binary'] = str(prefix.with_suffix('.all.f32'))\n"
        "prefix.with_suffix('.json').write_text(json.dumps(meta))\n"
        "print('{\"captured_pre_output_norm\":true}')\n"
    )
    exe.chmod(0o755)
    return exe


def _hip_capture(values: list[float]):
    def capture(_model: Path, prompt_tokens: tuple[int, ...], position: int, _max_seq: int | None):
        return {
            "status": "captured",
            "mode": "step-serial-pre-output_norm",
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "dtype": "BF16_to_F32",
            "provenance": "final_decoder_output_before_output_norm",
            "values": [float(value) for value in values],
        }

    return capture


def _prepend_path(path: Path) -> str:
    return str(path) + os.pathsep + os.environ.get("PATH", "")
