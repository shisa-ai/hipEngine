from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.llamacpp_mtp_audit_layer0_attn_norm_formula import (
    audit_layer0_attn_norm_formula,
    bf16_roundtrip_array,
    build_formula_candidates,
    classify_formula_audit,
    delta_summary,
    pack_float32,
    rmsnorm_f32,
    summarize_candidate,
    unpack_float32,
)


def test_rmsnorm_f32_matches_manual_formula() -> None:
    x = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    weight = np.asarray([1.0, 0.5, 2.0, -1.0], dtype=np.float32)
    eps = 1.0e-6

    actual = rmsnorm_f32(x, weight, eps)

    expected = x * np.float32(1.0 / np.sqrt(np.mean(x * x) + eps)) * weight
    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=1.0e-6, atol=1.0e-6)


def test_formula_candidates_explain_bf16_input_contraction() -> None:
    input_f32 = np.asarray([0.1253, -0.0627, 0.0311, -0.0189], dtype=np.float32)
    weight = np.asarray([1.1, 0.9, 1.25, 0.75], dtype=np.float32)
    eps = 1.0e-6
    llama_attn = rmsnorm_f32(input_f32, weight, eps)
    llama_bf16 = bf16_roundtrip_array(llama_attn)
    hip_attn = bf16_roundtrip_array(rmsnorm_f32(bf16_roundtrip_array(input_f32), weight, eps))
    candidates = build_formula_candidates(
        input_f32=input_f32,
        weight=weight,
        eps=eps,
        eps_alternates=(0.0, 1.0e-5),
    )
    records = [summarize_candidate(item, llama_attn, llama_bf16, hip_attn) for item in candidates]
    best_llama = min(records, key=lambda item: item["delta_vs_llamacpp_attn_norm"]["rmse"])
    best_hip = min(records, key=lambda item: item["delta_vs_hipengine_attn_norm"]["rmse"])
    best = {
        "vs_llamacpp_attn_norm": {
            "available": True,
            "name": best_llama["name"],
            "eps_source": best_llama["eps_source"],
            "weight_source": best_llama["weight_source"],
            "delta": best_llama["delta_vs_llamacpp_attn_norm"],
        },
        "vs_hipengine_attn_norm": {
            "available": True,
            "name": best_hip["name"],
            "eps_source": best_hip["eps_source"],
            "weight_source": best_hip["weight_source"],
            "delta": best_hip["delta_vs_hipengine_attn_norm"],
        },
    }

    assert best_llama["name"] == "input_f32_weight_f32_eps_model_f32_out"
    assert best_llama["delta_vs_llamacpp_attn_norm"]["exact_match"] is True
    assert best_hip["name"] == "input_bf16_weight_f32_eps_model_bf16_out"
    assert best_hip["delta_vs_hipengine_attn_norm"]["exact_match"] is True
    assert classify_formula_audit(best, {"status": "captured"}) == (
        "attn_norm_mismatch_explained_by_input_activation_bf16_contraction"
    )


def test_delta_summary_reports_shape_and_exactness() -> None:
    actual = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    reference = np.asarray([1.0, 1.5, 3.25], dtype=np.float32)

    delta = delta_summary(actual, reference)

    assert delta["available"] is True
    assert delta["shape_match"] is True
    assert delta["exact_match"] is False
    assert delta["max_abs_diff"] == 0.5
    assert delta["top_abs_diff"][0]["index"] == 1


def test_pack_unpack_float32_roundtrip() -> None:
    values = np.asarray([1.0, -2.5, 3.25], dtype=np.float32)

    assert np.array_equal(unpack_float32(pack_float32(values)), values)


def test_audit_layer0_attn_norm_formula_from_synthetic_artifacts(tmp_path: Path) -> None:
    input_f32 = np.asarray([0.1253, -0.0627, 0.0311, -0.0189], dtype=np.float32)
    weight = np.asarray([1.1, 0.9, 1.25, 0.75], dtype=np.float32)
    eps = 1.0e-6
    llama_attn = rmsnorm_f32(input_f32, weight, eps)
    hip_attn = bf16_roundtrip_array(rmsnorm_f32(bf16_roundtrip_array(input_f32), weight, eps))
    input_compare = _write_compare_artifact(
        tmp_path,
        name="input",
        values=input_f32,
        model=tmp_path / "model.gguf",
    )
    attn_compare = _write_compare_artifact(
        tmp_path,
        name="attn",
        values=llama_attn,
        model=tmp_path / "model.gguf",
        extra={"layer_id": 0, "token_id": 9},
    )

    artifact = audit_layer0_attn_norm_formula(
        input_compare_path=input_compare,
        attn_norm_compare_path=attn_compare,
        model_path=tmp_path / "model.gguf",
        hip_capture_fn=_hip_capture(hip_attn),
        weight_loader=_weight_loader(weight, eps),
    )

    assert artifact["status"] == "ready"
    assert artifact["conclusion"] == (
        "attn_norm_mismatch_explained_by_input_activation_bf16_contraction"
    )
    assert artifact["best_candidates"]["vs_llamacpp_attn_norm"]["name"] == (
        "input_f32_weight_f32_eps_model_f32_out"
    )
    assert artifact["best_candidates"]["vs_hipengine_attn_norm"]["name"] == (
        "input_bf16_weight_f32_eps_model_bf16_out"
    )
    assert artifact["best_candidates"]["vs_llamacpp_attn_norm"]["delta"]["exact_match"] is True
    assert artifact["best_candidates"]["vs_hipengine_attn_norm"]["delta"]["exact_match"] is True
    assert artifact["external_checkout_modified"] is False
    assert artifact["next_action"] == (
        "decide_whether_to_add_f32_activation_path_or_adjust_llamacpp_oracle_dtype"
    )
    json.dumps(artifact)


def _write_compare_artifact(
    tmp_path: Path,
    *,
    name: str,
    values: np.ndarray,
    model: Path,
    extra: dict[str, object] | None = None,
) -> Path:
    binary = tmp_path / f"{name}.f32"
    binary.write_bytes(pack_float32(values))
    artifact = {
        "model": str(model),
        "prompt_tokens": [3, 5, 9],
        "position": 2,
        "token_id": 9,
        "llamacpp_capture": {
            "binary_path": str(binary),
            "binary_exists": True,
        },
    }
    if extra:
        artifact.update(extra)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(artifact))
    return path


def _hip_capture(values: np.ndarray):
    def capture(
        _model: Path,
        prompt_tokens: tuple[int, ...],
        position: int,
        layer_id: int,
        _max_seq: int | None,
    ):
        return {
            "status": "captured",
            "position": int(position),
            "token_id": int(prompt_tokens[position]),
            "layer_id": int(layer_id),
            "values": [float(value) for value in values.tolist()],
        }

    return capture


def _weight_loader(weight: np.ndarray, eps: float):
    def load(_model: Path, layer_id: int):
        return weight, eps, {
            "tensor_name": f"blk.{layer_id}.attn_norm.weight",
            "ggml_type": "F32",
            "shape": list(weight.shape),
            "summary": {"count": int(weight.size)},
            "config_eps": float(eps),
            "metadata_eps": float(eps),
            "materialization_layout": "dense_f32",
            "materialization_quant_key": "f32",
        }

    return load
