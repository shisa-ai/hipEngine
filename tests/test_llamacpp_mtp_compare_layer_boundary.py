from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.llamacpp_mtp_compare_layer_boundary import (
    annotate_effective_layer_tap,
    classify_layer_boundary,
    classify_prior_alignment,
    compare_layer_boundary,
    next_action,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import pack_float32, sha256_bytes


def test_compare_layer_boundary_matches_with_fake_harness(tmp_path: Path) -> None:
    values = [1.0, -2.0, 3.5]
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, values))
    prior = _write_prior_pre_output(tmp_path, llama=[9.0], hip=[8.0])

    artifact = compare_layer_boundary(
        compile_artifact_path=compile_artifact,
        prior_pre_output_path=prior,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(1, 2, 3),
        position=2,
        layer_id=39,
        output_prefix=tmp_path / "capture" / "layer39-pos2",
        all_rows=True,
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture(values),
    )

    assert artifact["status"] == "matched"
    assert artifact["classification"] == "layer_boundary_matches"
    assert artifact["numeric_delta"]["exact_match"] is True
    assert artifact["llamacpp_capture"]["effective_tap"] == "h_nextn_layer_out"
    assert artifact["llamacpp_capture"]["effective_layer_id"] == 39
    assert artifact["numeric_delta"]["all_rows_scan"]["matches"][0]["row"] == 0
    assert artifact["next_action"] == "continue_bisect_with_earlier_midpoint_layer"
    json.dumps(artifact)


def test_compare_layer_boundary_reproduces_prior_pre_output_mismatch(tmp_path: Path) -> None:
    llama = [1.0, 2.0]
    hip = [2.0, 2.0]
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, llama))
    prior = _write_prior_pre_output(tmp_path, llama=llama, hip=hip)

    artifact = compare_layer_boundary(
        compile_artifact_path=compile_artifact,
        prior_pre_output_path=prior,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(7, 8),
        position=1,
        layer_id=39,
        output_prefix=tmp_path / "capture" / "layer39-pos1",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture(hip),
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "final_layer_reproduces_pre_output_mismatch"
    assert artifact["prior_alignment"]["both_match_prior_pre_output"] is True
    assert artifact["numeric_delta"]["max_abs_diff"] == 1.0
    assert artifact["next_action"] == "continue_bisect_with_layer_19"


def test_compare_layer_boundary_detects_capture_alignment_problem(tmp_path: Path) -> None:
    llama = [1.0, 2.0]
    hip = [3.0, 2.0]
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, llama))
    prior = _write_prior_pre_output(tmp_path, llama=llama, hip=[2.0, 2.0])

    artifact = compare_layer_boundary(
        compile_artifact_path=compile_artifact,
        prior_pre_output_path=prior,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(7, 8),
        position=1,
        layer_id=39,
        output_prefix=tmp_path / "capture" / "layer39-pos1",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture(hip),
    )

    assert artifact["classification"] == "hipengine_layer_out_differs_from_serial_pre_output"
    assert artifact["prior_alignment"]["llamacpp_matches_prior_pre_output"] is True
    assert artifact["prior_alignment"]["hipengine_matches_prior_pre_output"] is False
    assert artifact["next_action"] == "audit_hipengine_capture_attention_layer_vs_serial_loop"


def test_compare_layer_boundary_requires_valid_position(tmp_path: Path) -> None:
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, [1.0]))
    prior = _write_prior_pre_output(tmp_path, llama=[1.0], hip=[1.0])

    try:
        compare_layer_boundary(
            compile_artifact_path=compile_artifact,
            prior_pre_output_path=prior,
            model_path=tmp_path / "model.gguf",
            prompt_tokens=(1,),
            position=1,
            layer_id=39,
            output_prefix=tmp_path / "capture" / "layer39-pos1",
            hip_capture_fn=_hip_capture([1.0]),
        )
    except ValueError as exc:
        assert "position" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")


def test_prior_alignment_and_next_action_helpers() -> None:
    delta = {"llamacpp_sha256": "a", "hipengine_sha256": "b", "rmse": 1.0}
    prior = {"llamacpp_sha256": "a", "hipengine_sha256": "c", "rmse": 1.0}

    alignment = classify_prior_alignment(delta, prior)
    classification = classify_layer_boundary("mismatched", delta, alignment)

    assert alignment["llamacpp_matches_prior_pre_output"] is True
    assert alignment["hipengine_matches_prior_pre_output"] is False
    assert alignment["rmse_matches_prior_pre_output"] is True
    assert classification == "hipengine_layer_out_differs_from_serial_pre_output"
    assert next_action("llamacpp_capture_failed", classification) == (
        "inspect_layer_boundary_llamacpp_capture_logs"
    )


def test_effective_layer_tap_annotation_records_stale_generic_metadata() -> None:
    capture = {"metadata": {"tap": "h_nextn_post_output_norm"}}

    annotate_effective_layer_tap(capture, layer_id=19)

    assert capture["effective_tap"] == "h_nextn_layer_out"
    assert capture["effective_layer_id"] == 19
    assert "generic hidden-seed harness" in capture["metadata_tap_note"]


def _write_compile_artifact(tmp_path: Path, executable: Path) -> Path:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(exist_ok=True)
    path = tmp_path / "compile.json"
    path.write_text(
        json.dumps({"outputs": {"executable": str(executable)}, "lib_dir": str(lib_dir)})
    )
    return path


def _write_prior_pre_output(tmp_path: Path, *, llama: list[float], hip: list[float]) -> Path:
    diffs = [left - right for left, right in zip(llama, hip)]
    abs_diffs = [abs(value) for value in diffs]
    rmse = (sum(value * value for value in diffs) / len(diffs)) ** 0.5
    path = tmp_path / f"prior-{len(list(tmp_path.glob('prior-*.json')))}.json"
    path.write_text(
        json.dumps(
            {
                "status": "mismatched" if max(abs_diffs) else "matched",
                "classification": "pre_output_mismatch_already_present",
                "numeric_delta": {
                    "rmse": rmse,
                    "max_abs_diff": max(abs_diffs) if abs_diffs else 0.0,
                    "mean_abs_diff": sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0,
                    "llamacpp_sha256": sha256_bytes(pack_float32(llama)),
                    "hipengine_sha256": sha256_bytes(pack_float32(hip)),
                },
            }
        )
    )
    return path


def _write_fake_harness(tmp_path: Path, values: list[float]) -> Path:
    exe = tmp_path / f"fake-layer-{len(list(tmp_path.glob('fake-layer-*')))}"
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
        "print('{\"captured_layer_boundary\":true}')\n"
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
            "mode": "capture_attention_layer_layer_out",
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "layer_id": int(layer_id),
            "preceding_layer_count": int(layer_id),
            "dtype": "BF16_to_F32",
            "provenance": "capture_attention_layer.layer_out_f32",
            "values": [float(value) for value in values],
        }

    return capture


def _prepend_path(path: Path) -> str:
    return str(path) + os.pathsep + os.environ.get("PATH", "")
