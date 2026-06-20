from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.llamacpp_mtp_compare_input_embed import (
    annotate_effective_input_tap,
    classify_input_embed,
    compare_input_embed,
    f32_to_bf16_roundtrip,
    next_action,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import pack_float32, sha256_bytes


def test_compare_input_embed_matches_exactly_with_fake_harness(tmp_path: Path) -> None:
    values = [1.0, -2.0, 3.5]
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, values))
    layer0 = _write_layer0_reference(tmp_path)

    artifact = compare_input_embed(
        compile_artifact_path=compile_artifact,
        layer0_reference_path=layer0,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(1, 2, 3),
        position=2,
        output_prefix=tmp_path / "capture" / "input-pos2",
        all_rows=True,
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture(values),
    )

    assert artifact["status"] == "matched"
    assert artifact["classification"] == "input_embed_exact_match"
    assert artifact["numeric_delta"]["exact_match"] is True
    assert artifact["bf16_rounded_delta"]["available"] is True
    assert artifact["llamacpp_capture"]["effective_tap"] == "h_nextn_input_embed"
    assert artifact["numeric_delta"]["all_rows_scan"]["matches"][0]["row"] == 0
    assert artifact["next_action"] == "investigate_layer0_implementation_after_embedding"
    json.dumps(artifact)


def test_compare_input_embed_classifies_bf16_roundtrip_match(tmp_path: Path) -> None:
    llama = [1.1, -2.3, 3.7]
    hip = [f32_to_bf16_roundtrip(value) for value in llama]
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, llama))
    layer0 = _write_layer0_reference(tmp_path)

    artifact = compare_input_embed(
        compile_artifact_path=compile_artifact,
        layer0_reference_path=layer0,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(7, 8),
        position=1,
        output_prefix=tmp_path / "capture" / "input-pos1",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture(hip),
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "input_embed_matches_after_bf16_roundtrip"
    assert artifact["numeric_delta"]["exact_match"] is False
    assert artifact["bf16_rounded_delta"]["exact_match"] is True
    assert artifact["next_action"] == "investigate_layer0_implementation_after_embedding"


def test_compare_input_embed_classifies_bf16_mismatch(tmp_path: Path) -> None:
    llama = [1.1, -2.3]
    hip = [f32_to_bf16_roundtrip(1.1) + 0.5, f32_to_bf16_roundtrip(-2.3)]
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, llama))
    layer0 = _write_layer0_reference(tmp_path)

    artifact = compare_input_embed(
        compile_artifact_path=compile_artifact,
        layer0_reference_path=layer0,
        model_path=tmp_path / "model.gguf",
        prompt_tokens=(7, 8),
        position=1,
        output_prefix=tmp_path / "capture" / "input-pos1",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
        hip_capture_fn=_hip_capture(hip),
    )

    assert artifact["status"] == "mismatched"
    assert artifact["classification"] == "input_embed_mismatch_after_bf16_roundtrip"
    assert artifact["bf16_rounded_delta"]["max_abs_diff"] == 0.5
    assert artifact["next_action"] == "audit_token_embedding_lookup_or_bf16_conversion"


def test_compare_input_embed_requires_valid_position(tmp_path: Path) -> None:
    compile_artifact = _write_compile_artifact(tmp_path, _write_fake_harness(tmp_path, [1.0]))
    layer0 = _write_layer0_reference(tmp_path)

    try:
        compare_input_embed(
            compile_artifact_path=compile_artifact,
            layer0_reference_path=layer0,
            model_path=tmp_path / "model.gguf",
            prompt_tokens=(1,),
            position=1,
            output_prefix=tmp_path / "capture" / "input-pos1",
            hip_capture_fn=_hip_capture([1.0]),
        )
    except ValueError as exc:
        assert "position" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")


def test_effective_input_tap_annotation_records_stale_generic_metadata() -> None:
    capture = {"metadata": {"tap": "h_nextn_post_output_norm"}}

    annotate_effective_input_tap(capture)

    assert capture["effective_tap"] == "h_nextn_input_embed"
    assert "generic hidden-seed harness" in capture["metadata_tap_note"]


def test_bf16_roundtrip_helper_and_next_action() -> None:
    assert f32_to_bf16_roundtrip(1.1) == 1.1015625
    assert classify_input_embed(
        "mismatched",
        {"shape_match": True},
        {"available": True, "exact_match": True},
    ) == "input_embed_matches_after_bf16_roundtrip"
    assert next_action("llamacpp_capture_failed", "input_embed_exact_match") == (
        "inspect_input_embed_llamacpp_capture_logs"
    )


def _write_compile_artifact(tmp_path: Path, executable: Path) -> Path:
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(exist_ok=True)
    path = tmp_path / "compile.json"
    path.write_text(
        json.dumps({"outputs": {"executable": str(executable)}, "lib_dir": str(lib_dir)})
    )
    return path


def _write_layer0_reference(tmp_path: Path) -> Path:
    path = tmp_path / "layer0.json"
    path.write_text(
        json.dumps(
            {
                "status": "mismatched",
                "classification": "layer_boundary_mismatch",
                "layer_id": 0,
                "numeric_delta": {
                    "rmse": 0.012,
                    "max_abs_diff": 0.045,
                    "mean_abs_diff": 0.009,
                    "llamacpp_sha256": "llama-layer0",
                    "hipengine_sha256": "hip-layer0",
                },
            }
        )
    )
    return path


def _write_fake_harness(tmp_path: Path, values: list[float]) -> Path:
    exe = tmp_path / f"fake-input-{len(list(tmp_path.glob('fake-input-*')))}"
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
        "print('{\"captured_input_embed\":true}')\n"
    )
    exe.chmod(0o755)
    return exe


def _hip_capture(values: list[float]):
    def capture(
        _model: Path,
        prompt_tokens: tuple[int, ...],
        position: int,
        _max_seq: int | None,
    ):
        return {
            "status": "captured",
            "mode": "capture_attention_layer_hidden_in",
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "layer_id": 0,
            "preceding_layer_count": 0,
            "dtype": "BF16_to_F32",
            "provenance": "capture_attention_layer.hidden_in_f32",
            "values": [float(value) for value in values],
        }

    return capture


def _prepend_path(path: Path) -> str:
    return str(path) + os.pathsep + os.environ.get("PATH", "")
