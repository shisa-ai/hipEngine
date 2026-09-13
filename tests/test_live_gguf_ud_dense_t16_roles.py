"""Dense T16 role coverage for the published UD artifacts.

`GGUF_DENSE_Q5_T16_H5120` was originally scoped to the three Q5 roles that
existed in the plain Qwen3.8-27B `Q4_K_S` file (`ffn_down`, `attn_qkv`,
`attn_v`); that file has no Q5 tensors at the other roles. The UD files carry
Q5_K at additional dense roles whose shapes are already qualified for Q4_K in
`_DENSE_Q4_T16_SIDECAR_POLICY`, so they must reach the same T16 family instead
of falling to the raw GEMV layout.

The same applies to Q6_K (`ffn_up`, `attn_output`, `ssm_out`, `attn_k`) and to
Q4_K `ssm_out`, which carries the same (5120, 6144) geometry as `attn_output`.

Pure metadata tests: they plan real artifacts and never touch the GPU.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.kernels.backends import backend_package_capability, load_backend_kernel_package
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q5_K_T16,
    gguf_decode_repack_enabled,
    plan_qwen35_gguf_materialization,
)

MODELS = {
    "K_M": Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf"),
    "K_S": Path("/models/gguf/Qwen3.8-27B-UD-Q4_K_S.gguf"),
    "plain_K_S": Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf"),
}

# Roles whose Q5_K tensors previously fell to raw_gguf, with the shape each
# carries in this H5120 geometry. Every one is T16-alignable and every shape is
# already admitted for Q4_K by _DENSE_Q4_T16_SIDECAR_POLICY.
NEWLY_ADMITTED = {
    "attn_gate": (6_144, 5_120),
    "attn_k": (1_024, 5_120),
    "attn_output": (5_120, 6_144),
    "attn_q": (12_288, 5_120),
    "ffn_gate": (17_408, 5_120),
    "ffn_up": (17_408, 5_120),
}
# Roles the original H5120 scope already covered; they must not regress.
ALREADY_ADMITTED = {
    "ffn_down": (5_120, 17_408),
    "attn_qkv": (10_240, 5_120),
    "attn_v": (1_024, 5_120),
}


def _plan(model_path):
    load_backend_kernel_package("hip_gfx1151")
    reader = GGUFReader(model_path)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    from hipengine.loading.qwen35_gguf_policy import resolve_gguf_dense_flags
    flags = resolve_gguf_dense_flags(
        "hip_gfx1151", reader.info.file_type_name,
        capability_reader=backend_package_capability,
    )
    return plan_qwen35_gguf_materialization(
        model_map, decode_repack=gguf_decode_repack_enabled(None), **flags)


def _q5_layer_specs(plan):
    for layer in plan.layer_specs:
        for slot, spec in layer.items():
            if spec.source.ggml_type_name == "Q5_K" and len(spec.source.shape) == 2:
                yield slot, spec


def _require(name):
    path = MODELS[name]
    if not path.exists():
        pytest.skip(f"local GGUF fixture not found: {path}")
    return path


@pytest.mark.parametrize("model", ("K_M", "K_S"))
@pytest.mark.parametrize("role", sorted(NEWLY_ADMITTED))
def test_ud_q5_dense_roles_reach_t16(model, role):
    """Every Q5_K tensor at a qualified dense role plans to the T16 family."""
    plan = _plan(_require(model))
    matched = [
        (slot, spec) for slot, spec in _q5_layer_specs(plan)
        if slot == role
        and tuple(map(int, spec.source.shape)) == NEWLY_ADMITTED[role]
    ]
    if not matched:
        pytest.skip(f"{model} has no Q5_K {role} at {NEWLY_ADMITTED[role]}")
    offenders = [(slot, spec.layout) for slot, spec in matched
                 if spec.layout != LAYOUT_GGUF_Q5_K_T16]
    assert not offenders, (
        f"{len(offenders)}/{len(matched)} Q5_K {role} tensors missed the T16 "
        f"family: {offenders[:3]}")


@pytest.mark.parametrize("model", ("K_M", "K_S", "plain_K_S"))
@pytest.mark.parametrize("role", sorted(ALREADY_ADMITTED))
def test_previously_admitted_q5_roles_do_not_regress(model, role):
    """The original H5120 scope keeps its T16 owners."""
    plan = _plan(_require(model))
    matched = [
        (slot, spec) for slot, spec in _q5_layer_specs(plan)
        if slot == role
        and tuple(map(int, spec.source.shape)) == ALREADY_ADMITTED[role]
    ]
    if not matched:
        pytest.skip(f"{model} has no Q5_K {role} at {ALREADY_ADMITTED[role]}")
    offenders = [(slot, spec.layout) for slot, spec in matched
                 if spec.layout != LAYOUT_GGUF_Q5_K_T16]
    assert not offenders, f"regressed Q5_K {role} owners: {offenders[:3]}"


def test_no_dense_q5_layer_tensor_is_left_raw_in_ud_files():
    """After the role extension no rank-2 Q5_K layer tensor stays on raw."""
    for model in ("K_M", "K_S"):
        plan = _plan(_require(model))
        raw = [(slot, tuple(map(int, spec.source.shape)))
               for slot, spec in _q5_layer_specs(plan)
               if spec.layout == "raw_gguf"]
        assert not raw, f"{model} still has raw Q5_K layer tensors: {raw[:5]}"


def test_fixture_slots_actually_match_the_role_names():
    """Guard: the role filters must select tensors, not silently skip."""
    plan = _plan(_require("K_M"))
    slots = {slot for slot, _ in _q5_layer_specs(plan)}
    missing = [r for r in NEWLY_ADMITTED if r not in slots]
    assert not missing, (
        f"role filter matched nothing for {missing}; layer slot keys are bare "
        f"names like {sorted(slots)[:4]}")


def test_plain_q4ks_q5_plan_is_unchanged_by_the_extension():
    """The extension is a no-op for the file the original scope targeted."""
    plan = _plan(_require("plain_K_S"))
    layouts = {spec.layout for _, spec in _q5_layer_specs(plan)}
    assert layouts <= {LAYOUT_GGUF_Q5_K_T16}, (
        f"plain Q4_K_S Q5_K layer layouts changed: {layouts}")


# --- Q6_K and Q4_K role coverage ------------------------------------------

Q6_NEWLY_ADMITTED = {
    "attn_k": (1_024, 5_120),
    "attn_output": (5_120, 6_144),
    "ffn_up": (17_408, 5_120),
    "ssm_out": (5_120, 6_144),
}
_Q6_FAST = {"gguf_q6_k_t16_v1", "gguf_q6_k_t16_qmicro_planar_v1"}


def _layer_specs_of(plan, quant):
    for layer in plan.layer_specs:
        for slot, spec in layer.items():
            if spec.source.ggml_type_name == quant and len(spec.source.shape) == 2:
                yield slot, spec


@pytest.mark.parametrize("model", ("K_M", "K_S"))
@pytest.mark.parametrize("role", sorted(Q6_NEWLY_ADMITTED))
def test_ud_q6_dense_roles_reach_t16(model, role):
    plan = _plan(_require(model))
    matched = [
        (slot, spec) for slot, spec in _layer_specs_of(plan, "Q6_K")
        if slot == role
        and tuple(map(int, spec.source.shape)) == Q6_NEWLY_ADMITTED[role]
    ]
    if not matched:
        pytest.skip(f"{model} has no Q6_K {role} at {Q6_NEWLY_ADMITTED[role]}")
    offenders = [(slot, spec.layout) for slot, spec in matched
                 if spec.layout not in _Q6_FAST]
    assert not offenders, f"Q6_K {role} missed the T16 family: {offenders[:3]}"


@pytest.mark.parametrize("model", ("K_M", "K_S"))
def test_ud_q4_ssm_out_reaches_t16(model):
    """Q4_K ssm_out shares attn_output's (5120, 6144) geometry."""
    plan = _plan(_require(model))
    matched = [
        (slot, spec) for slot, spec in _layer_specs_of(plan, "Q4_K")
        if slot == "ssm_out" and tuple(map(int, spec.source.shape)) == (5_120, 6_144)
    ]
    if not matched:
        pytest.skip(f"{model} has no Q4_K ssm_out at (5120, 6144)")
    offenders = [(slot, spec.layout) for slot, spec in matched
                 if spec.layout != "gguf_q4_k_t16_v1"]
    assert not offenders, f"Q4_K ssm_out still on pack8: {offenders[:3]}"


@pytest.mark.parametrize("quant", ("Q4_K", "Q5_K", "Q6_K", "Q8_0"))
def test_plain_q4ks_layouts_are_unchanged_by_the_role_extensions(quant):
    """Every extension targets a role the plain Q4_K_S file does not carry."""
    plan = _plan(_require("plain_K_S"))
    fallback = [
        (slot, tuple(map(int, spec.source.shape)))
        for slot, spec in _layer_specs_of(plan, quant)
        if spec.layout in ("raw_gguf", "q4_k_pack8")
    ]
    assert not fallback, (
        f"plain Q4_K_S {quant} layer tensors regressed to a fallback layout: "
        f"{fallback[:5]}")
