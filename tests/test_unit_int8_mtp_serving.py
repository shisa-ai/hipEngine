from dataclasses import replace
from types import SimpleNamespace
from collections import deque

import pytest

from hipengine.models.qwen35 import Qwen35GGUFModel
from hipengine.models.kv_capabilities import ModelArtifactIdentity
from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.speculative.serving import SpeculativeMTPServingKey, SpeculativeMTPStaticEligibility


_PLAIN_FINGERPRINT = "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"


def _weight_index(path):
    """A scanned-inventory stand-in for the serving key's execution identity.

    The serving key binds the GGUF execution identity (tensor table plus
    routing-relevant metadata), so a placeholder object is not enough: the
    generator reads the inventory the loader would have produced.
    """

    from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo

    tensor = GGUFTensorInfo(
        name="token_embd.weight",
        shape=(2048, 248320),
        ggml_shape=(248320, 2048),
        ggml_type=12,
        ggml_type_name="Q4_K",
        n_elements=2048 * 248320,
        nbytes=0,
        offset=0,
        data_offset=0,
        byte_shape=(248320, 2048),
    )
    return GGUFModelInfo(
        path=path,
        version=3,
        alignment=32,
        metadata={
            "general.architecture": "qwen35",
            "general.file_type": 15,
            "qwen35.block_count": 48,
        },
        tensors=(tensor,),
        tensor_data_offset=0,
    )


def _key(**changes):
    return replace(SpeculativeMTPServingKey(
        artifact_sha256="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
        artifact_size_bytes=17_106_775_008,
        artifact_execution_fingerprint=_PLAIN_FINGERPRINT,
        content_verified=True,
        backend="hip_gfx1151", target_arch="gfx1151", weight_quant="gguf_q4_k_m",
        kv_storage="int8_per_token_head", kv_layout="uniform",
        realized_group_rows=1, resident_capacity=4, candidate_budget=3,
        sampling_mode="greedy_fast", memory_fit=True,
    ), **changes)


@pytest.mark.parametrize("capacity", [1, 4, 8])
@pytest.mark.parametrize("budget", [1, 3, 7])
def test_dense_int8_mtp_uses_implementation_admission(capacity, budget):
    decision = Qwen35GGUFModel().resolve_speculative_mtp_serving_plan(
        key=_key(resident_capacity=capacity, candidate_budget=budget),
    )
    assert decision.admitted
    assert decision.automatic_eligible
    assert decision.selected_candidate_count == budget
    payload = decision.as_dict()
    assert payload["admission_basis"] == "implementation"
    assert payload["evidence_key"] is None
    assert payload["evidence_artifacts"] == []
    eligibility = SpeculativeMTPStaticEligibility.from_mapping(payload["static_eligibility"])
    assert eligibility.eligible
    assert eligibility.max_realized_group_rows == 1
    assert eligibility.implementation_key == "gguf_dense_int8_native_chain"


@pytest.mark.parametrize("changes,reason", [
    ({"realized_group_rows": 2}, "packed_int8_mtp_not_implemented"),
    ({"kv_layout": "tail4_hadamard_group32"}, "mtp_kv_layout_unsupported"),
    ({"sampling_mode": "processed_argmax"}, "mtp_sampling_unsupported"),
    ({"memory_fit": False}, "insufficient_memory"),
    ({"candidate_budget": 8}, "mtp_candidate_depth_unsupported"),
    ({"backend": "cpu_reference"}, "mtp_backend_unsupported"),
    ({"kv_scale_dtype": "int8"}, "mtp_kv_scale_dtype_unsupported"),
    ({"kv_scale_granularity": "block16"}, "mtp_kv_scale_granularity_unsupported"),
])
def test_int8_mtp_structural_refusals_name_actual_cause(changes, reason):
    decision = Qwen35GGUFModel().resolve_speculative_mtp_serving_plan(key=_key(**changes))
    assert not decision.admitted
    assert decision.reason == reason
    assert decision.selected_candidate_count == 0


@pytest.mark.parametrize("requested,effective", [
    ("auto", "int8_per_token_head"),
    ("int8_per_token_head", "bf16"),
    ("bf16", "bf16"),
])
def test_serving_key_uses_prepared_effective_storage(tmp_path, requested, effective):
    path = tmp_path / "model.gguf"
    with path.open("wb") as handle:
        handle.truncate(17_106_775_008)
    generator = Qwen35GGUFBringupGenerator.__new__(Qwen35GGUFBringupGenerator)
    generator.model_path = path
    generator.weight_index = _weight_index(path)
    generator.model_plugin = Qwen35GGUFModel()
    generator.backend = "hip_gfx1151"
    generator._kv_artifact_identity = ModelArtifactIdentity(
        path=str(path), size_bytes=17_106_775_008,
        sha256=_key().artifact_sha256, content_verified=True,
    )
    generator._prepared_kv_signature = (effective, "uniform", "fp32", "per_token_head")
    explicit = generator.resolve_speculative_mtp_serving_plan(
        realized_group_rows=1, resident_capacity=4, candidate_budget=3,
        sampling_mode="greedy_fast", kv_storage=requested, memory_fit=True,
        request_mode="explicit",
    )
    assert explicit.key.kv_storage == effective
    # An explicitly requested run executes on implementation capability even
    # when no retained row describes the resident artifact.
    assert explicit.admitted

    automatic = generator.resolve_speculative_mtp_serving_plan(
        realized_group_rows=1, resident_capacity=4, candidate_budget=3,
        sampling_mode="greedy_fast", kv_storage=requested, memory_fit=True,
    )
    assert automatic.key.kv_storage == effective
    # Automatic intent takes the same capability route as an explicit request.
    # Neither chain has a retained row for this artifact, and that never
    # withholds a path the kernels implement.
    assert automatic.admitted
    assert automatic.as_dict()["admission_basis"] == "implementation"


def test_implementation_c1_depth_does_not_require_a_benchmark_policy_cell(monkeypatch):
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    eligibility = Qwen35GGUFModel().resolve_speculative_mtp_serving_plan(
        key=_key(candidate_budget=7),
    ).static_eligibility
    adapter._static_eligibility_by_request = {12: eligibility}
    monkeypatch.setattr(adapter, "_physical_width_depth_admitted", lambda *_: False)
    assert adapter._physical_width_depth_admitted_for_group(1, 7, (12,))
    assert not adapter._physical_width_depth_admitted_for_group(2, 7, (12, 13))


def test_resident_int8_capacity_uses_layout_contract_not_evidence_width():
    checked = []
    owner = SimpleNamespace(
        kv_attention_source="int8_direct",
        resident_slot_view=lambda index: SimpleNamespace(slot=index),
        _resident_ar_kv_layout_for_sessions=lambda sessions: checked.append(len(sessions)),
        close=lambda: None,
    )
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner.capacity = 4
    runner._shared_runner = object()
    runner._available = deque()
    runner.generator = SimpleNamespace(
        _defer_resident_session_policy_resolution=True,
        _acquire_shared_session=lambda *args, **kwargs: (owner, "pool", False),
        kv_capability_provenance={"runtime_action": "diagnostic_override"},
    )
    runner._reserve_sessions()
    assert checked == [4]
    assert len(runner._available) == 4
