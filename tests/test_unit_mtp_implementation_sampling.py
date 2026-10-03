"""Processed MTP admission follows implementation, not measured artifact identity."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.llm import LLM
from hipengine.models.qwen35 import Qwen35GGUFModel
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.speculative.serving import SpeculativeMTPServingKey


def key(**overrides):
    values = dict(artifact_sha256=None, artifact_size_bytes=123456, content_verified=False,
                  backend="hip_gfx1151", target_arch="gfx1151",
                  weight_quant="gguf_q4_k_m", kv_storage="bf16", kv_layout="uniform",
                  realized_group_rows=1, resident_capacity=4, candidate_budget=3,
                  sampling_mode="sampled", memory_fit=True)
    values.update(overrides)
    return SpeculativeMTPServingKey(**values)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_unmeasured_dense_bf16_processed_mtp_is_implemented(width):
    plugin = replace(Qwen35GGUFModel(), speculative_mtp_serving_evidence=())
    decision = plugin.resolve_speculative_mtp_serving_plan(key=key(realized_group_rows=width))
    assert decision.admitted
    assert decision.as_dict()["admission_basis"] == "implementation"


def test_implementation_keeps_resource_and_shape_refusals():
    plugin = replace(Qwen35GGUFModel(), speculative_mtp_serving_evidence=())
    for fields in ({"memory_fit": False}, {"realized_group_rows": 5},
                   {"candidate_budget": 8}, {"kv_storage": "fp16"},
                   {"backend": "cpu_reference"}, {"sampling_mode": "unknown"}):
        assert not plugin.resolve_speculative_mtp_serving_plan(key=key(**fields)).admitted


def test_llm_advertises_implementation_modes_without_evidence():
    plugin = replace(Qwen35GGUFModel(), speculative_mtp_serving_evidence=())
    llm = object.__new__(LLM)
    llm.kv_storage = "bf16"
    llm._load_model_metadata = lambda: (None, plugin)
    llm.resolve_speculative_mtp_serving_plan = lambda **kw: plugin.resolve_speculative_mtp_serving_plan(
        key=key(**{k: v for k, v in kw.items() if k != "request_mode"}))
    assert "sampled" in llm.speculative_mtp_sampling_modes


@pytest.mark.parametrize("backend,expected", [("hip_gfx1151", True), ("hip_gfx1100", False)])
def test_adapter_advertises_implemented_mode_without_artifact_row(backend, expected):
    plugin = replace(Qwen35GGUFModel(), speculative_mtp_serving_evidence=())
    generator = SimpleNamespace(backend=backend, target_arch="gfx1151", model_plugin=plugin,
                                execution_profile="production")
    adapter = Qwen35GGUFMTP2Adapter(SimpleNamespace(generator=generator, capacity=4),
                                   enabled=True, target_verify_mode="native",
                                   candidate_budget=3, quant="gguf_q4_k_m")
    assert adapter._sampled_route_qualified() is expected
