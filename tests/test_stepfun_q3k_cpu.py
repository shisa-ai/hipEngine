from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFReader, scan_gguf_splits
from hipengine.loading.stepfun_gguf import build_stepfun_gguf_tensor_map
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data, dequantization_supported


DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")
DEFAULT_LLAMA_GGUF_PY = Path("/home/lhl/llama.cpp/llama.cpp-lhl/gguf-py")


def _stepfun_gguf_paths() -> tuple[Path, ...]:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return paths


def _llama_gguf_modules() -> tuple[Any, Any]:
    root = Path(os.environ.get("HIPENGINE_LLAMA_CPP_GGUF_PY", DEFAULT_LLAMA_GGUF_PY))
    if not root.is_dir():
        pytest.skip("llama.cpp gguf-py reference not found; set HIPENGINE_LLAMA_CPP_GGUF_PY")
    sys.path.insert(0, str(root))
    try:
        quants = importlib.import_module("gguf.quants")
        constants = importlib.import_module("gguf.constants")
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    return quants, constants


def _raw_slice(tensor, rows: int = 2) -> np.ndarray:
    raw = GGUFReader(tensor.source_path).tensor_data(tensor.name)
    if len(tensor.shape) == 1:
        return np.asarray(raw[: min(tensor.shape[0], 64)]).copy()
    return np.asarray(raw[:rows]).copy()


def test_q3_k_cpu_dequant_matches_llama_cpp_reference_on_step_slice() -> None:
    info = scan_gguf_splits(_stepfun_gguf_paths())
    model_map = build_stepfun_gguf_tensor_map(info)
    tensor = model_map.layer(0).tensor("attn_q")
    raw = _raw_slice(tensor)
    quants, constants = _llama_gguf_modules()

    ours = dequantize_gguf_data(raw, GGMLQuantizationType.Q3_K)
    reference = quants.dequantize(raw, constants.GGMLQuantizationType.Q3_K)

    assert ours.shape == (2, 4096)
    assert ours.dtype == np.float32
    np.testing.assert_allclose(ours, reference, rtol=0, atol=0)


def test_step_mixed_quant_slices_match_llama_cpp_reference() -> None:
    info = scan_gguf_splits(_stepfun_gguf_paths())
    model_map = build_stepfun_gguf_tensor_map(info)
    quants, constants = _llama_gguf_modules()
    samples = {
        "q3": model_map.layer(0).tensor("attn_q"),
        "q5": model_map.layer(0).tensor("ffn_down"),
        "q8": model_map.root("token_embedding"),
        "f32": model_map.layer(3).tensor("exp_probs_bias"),
    }

    for label, tensor in samples.items():
        raw = _raw_slice(tensor)
        qtype = GGMLQuantizationType(tensor.ggml_type)
        reference_qtype = constants.GGMLQuantizationType[tensor.ggml_type_name]
        ours = dequantize_gguf_data(raw, qtype)
        reference = quants.dequantize(raw, reference_qtype)

        assert dequantization_supported(qtype), label
        assert ours.dtype == np.float32
        np.testing.assert_allclose(ours, reference, rtol=0, atol=0)


def test_step_mixed_quant_keys_are_exposed_per_tensor() -> None:
    info = scan_gguf_splits(_stepfun_gguf_paths())
    model_map = build_stepfun_gguf_tensor_map(info)

    assert model_map.layer(0).tensor("attn_q").quant_key == "gguf_q3_k"
    assert model_map.layer(0).tensor("ffn_down").quant_key == "gguf_q5_k"
    assert model_map.root("token_embedding").quant_key == "gguf_q8_0"
    assert model_map.layer(3).tensor("exp_probs_bias").quant_key == "gguf_f32"
