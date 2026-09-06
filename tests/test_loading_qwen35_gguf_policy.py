"""Guards for the pure GGUF dense-weight policy API.

``hipengine.loading.qwen35_gguf_policy`` is the single home of the backend
capability + environment-override semantics used by
``materialize_qwen35_gguf_weights`` (runtime reader) and by the quant-route
audit (source-reading reader). These tests pin the pure resolution behavior
with stub readers and dict environments; cross-caller parity against the real
backend packages is guarded separately in
``tests/test_qwen35_gguf_policy_capability_parity.py`` because it imports
kernel packages and needs the HIP runtime.
"""

from __future__ import annotations

import pytest

from hipengine.loading.qwen35_gguf_policy import (
    HIPENGINE_C8_Q5_PLANAR_DP4A_ENV,
    HIPENGINE_GGUF_C8_Q5_RAW_MMQ_ENV,
    gguf_ar_raw_iq_contract,
    gguf_fp16_recurrent_state_default,
    resolve_gguf_dense_flags,
)
from hipengine.quant.gguf import GGMLQuantizationType


def _stub_reader(values: dict[str, object]):
    def reader(backend: str, name: str, default):
        return values.get(name, default)

    return reader


def test_resolve_flags_defaults_with_empty_capabilities():
    flags = resolve_gguf_dense_flags(
        "hip_gfx1100", "MOSTLY_Q4_K_M", capability_reader=_stub_reader({}), environ={}
    )
    assert flags == {
        "dense_q4_t16": False,
        "dense_q4_qmicro_t16_gate_up": False,
        "dense_q4_t16_attn_q_08b": False,
        "dense_q5_t16_ssm_out": False,
        "dense_q5_raw_mmq_ssm_out": False,
        "dense_q5_qmicro_planar_ssm_out": False,
        "dense_q5_t16_ssm_out_08b": False,
        "dense_q5_t16_qkv": False,
        "dense_q5_t16_h5120": False,
        "dense_q6_qmicro_planar": False,
        "dense_q6_qmicro_planar_excluded_slots": (),
    }


def test_resolve_flags_qmicro_gate_is_case_sensitive_on_the_file_type_stamp():
    values = {
        "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP": True,
        "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES": ("MOSTLY_Q4_K_S",),
    }
    reader = _stub_reader(values)
    assert (
        resolve_gguf_dense_flags("b", "MOSTLY_Q4_K_S", capability_reader=reader, environ={})[
            "dense_q4_qmicro_t16_gate_up"
        ]
        is True
    )
    assert (
        resolve_gguf_dense_flags("b", "mostly_q4_k_s", capability_reader=reader, environ={})[
            "dense_q4_qmicro_t16_gate_up"
        ]
        is False
    )
    assert (
        resolve_gguf_dense_flags("b", None, capability_reader=reader, environ={})[
            "dense_q4_qmicro_t16_gate_up"
        ]
        is False
    )


def test_resolve_flags_q5_raw_mmq_env_override_gates_capability():
    values = {"GGUF_C8_Q5_RAW_MMQ_SSM_OUT": True}
    reader = _stub_reader(values)
    # Default (env absent): raw-MMQ sidecar on where the backend declares it.
    assert (
        resolve_gguf_dense_flags("b", None, capability_reader=reader, environ={})[
            "dense_q5_raw_mmq_ssm_out"
        ]
        is True
    )
    for off in ("0", "false", "off", "no", ""):
        assert (
            resolve_gguf_dense_flags(
                "b", None, capability_reader=reader, environ={HIPENGINE_GGUF_C8_Q5_RAW_MMQ_ENV: off}
            )["dense_q5_raw_mmq_ssm_out"]
            is False
        )
    for on in ("1", "true", "yes", "on"):
        assert (
            resolve_gguf_dense_flags(
                "b", None, capability_reader=reader, environ={HIPENGINE_GGUF_C8_Q5_RAW_MMQ_ENV: on}
            )["dense_q5_raw_mmq_ssm_out"]
            is True
        )
    # The env override cannot enable a backend that lacks the capability.
    assert (
        resolve_gguf_dense_flags(
            "b",
            None,
            capability_reader=_stub_reader({}),
            environ={HIPENGINE_GGUF_C8_Q5_RAW_MMQ_ENV: "1"},
        )["dense_q5_raw_mmq_ssm_out"]
        is False
    )


def test_resolve_flags_planar_sidecar_env_is_default_off():
    values = {"GGUF_C8_Q5_RAW_MMQ_SSM_OUT": True}
    reader = _stub_reader(values)
    assert (
        resolve_gguf_dense_flags("b", None, capability_reader=reader, environ={})[
            "dense_q5_qmicro_planar_ssm_out"
        ]
        is False
    )
    assert (
        resolve_gguf_dense_flags(
            "b", None, capability_reader=reader, environ={HIPENGINE_C8_Q5_PLANAR_DP4A_ENV: "1"}
        )["dense_q5_qmicro_planar_ssm_out"]
        is True
    )
    assert (
        resolve_gguf_dense_flags(
            "b", None, capability_reader=_stub_reader({}), environ={HIPENGINE_C8_Q5_PLANAR_DP4A_ENV: "1"}
        )["dense_q5_qmicro_planar_ssm_out"]
        is False
    )


def test_resolve_flags_q6_excluded_slots_normalize_to_string_tuples():
    values = {"GGUF_DENSE_Q6_T16_QMICRO_PLANAR_EXCLUDED_SLOTS": ("attn_qkv",)}
    flags = resolve_gguf_dense_flags("b", None, capability_reader=_stub_reader(values), environ={})
    assert flags["dense_q6_qmicro_planar_excluded_slots"] == ("attn_qkv",)
    # A non-container capability resolves to the runtime default (empty tuple).
    flags = resolve_gguf_dense_flags("b", None, capability_reader=_stub_reader({}), environ={})
    assert flags["dense_q6_qmicro_planar_excluded_slots"] == ()


@pytest.mark.parametrize(
    ("ggml_type", "expected"),
    [
        (GGMLQuantizationType.IQ2_XS, True),
        (GGMLQuantizationType.IQ3_XXS, True),
        (GGMLQuantizationType.IQ4_XS, True),
        (GGMLQuantizationType.Q4_K, False),
        (GGMLQuantizationType.Q3_K, False),
        (GGMLQuantizationType.F32, False),
    ],
)
def test_raw_iq_contract_predicate_matches_production_set(ggml_type, expected):
    assert gguf_ar_raw_iq_contract([int(ggml_type)]) is expected


def test_raw_iq_contract_ignores_root_and_draft_scopes_by_construction():
    # The caller passes AR layer types only; an empty AR scope never contracts.
    assert gguf_ar_raw_iq_contract([]) is False


def test_fp16_recurrent_state_default_normalizes_like_the_runner():
    values = {"GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES": frozenset({"mostly_q4_k_s"})}
    reader = _stub_reader(values)
    assert gguf_fp16_recurrent_state_default("b", "MOSTLY_Q4_K_S", capability_reader=reader) is True
    assert gguf_fp16_recurrent_state_default("b", "  mostly_q4_k_s ", capability_reader=reader) is True
    assert gguf_fp16_recurrent_state_default("b", "MOSTLY_Q4_K_M", capability_reader=reader) is False
    # A backend without the constant has no default.
    assert gguf_fp16_recurrent_state_default("b", "MOSTLY_Q4_K_S", capability_reader=_stub_reader({})) is False
    assert gguf_fp16_recurrent_state_default("b", None, capability_reader=reader) is False
    assert gguf_fp16_recurrent_state_default(None, "MOSTLY_Q4_K_S", capability_reader=reader) is False
