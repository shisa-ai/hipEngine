from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.loading.qwen4_exp_gguf import GDN, build_qwen4_exp_gguf_tensor_map
from hipengine.loading.qwen4_exp_materialize import plan_qwen4_exp_residency
from hipengine.runtime.qwen4_exp_runner import (
    bind_qwen4_exp_gdn_layer,
    bind_qwen4_exp_qsa_layer,
)
from tests.test_qwen4_exp_gguf_mapping import _infos


class _Weight:
    def __init__(self, spec, ptr: int) -> None:
        self.spec = spec
        self.backend = "hip_gfx1151"
        self._allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=ptr))

    def allocation(self, name=None):
        assert name in (None, "raw")
        return self._allocation


def _resident():
    model_map = build_qwen4_exp_gguf_tensor_map(_infos())
    plan = plan_qwen4_exp_residency(model_map)
    weights = {
        spec.slot_path: _Weight(spec, index + 1000)
        for index, spec in enumerate(plan.device_specs)
    }
    return SimpleNamespace(
        plan=plan,
        weight=weights.__getitem__,
        device_weights=weights,
    )


def test_bind_qwen4_exp_gdn_layer_maps_every_physical_slot_without_quant_branches() -> None:
    resident = _resident()

    binding = bind_qwen4_exp_gdn_layer(resident, 1)

    assert binding.layer_id == 1
    assert binding.layer_type == GDN
    assert binding.gdn_state_index == 1
    assert binding.has_ple
    assert set(binding.mixer.projections) == {
        "attn_qkv",
        "attn_gate",
        "ssm_alpha",
        "ssm_beta",
        "ssm_out",
    }
    assert set(binding.moe) == {
        "router",
        "expert_gate",
        "expert_up",
        "expert_down",
        "shared_gate",
        "shared_up",
        "shared_down",
        "shared_gate_weight",
    }
    assert binding.attention_gr.norm_weight_ptr > 0
    assert binding.ffn_gr.norm_weight_ptr > 0
    assert binding.mixer.conv_weight_ptr > 0
    assert binding.mixer.dt_bias_ptr > 0
    assert binding.mixer.a_log_ptr > 0
    assert binding.mixer.norm_weight_ptr > 0


def test_bind_qwen4_exp_gdn_layer_tracks_dense_state_indices() -> None:
    resident = _resident()

    assert bind_qwen4_exp_gdn_layer(resident, 0).gdn_state_index == 0
    assert bind_qwen4_exp_gdn_layer(resident, 2).gdn_state_index == 2
    assert bind_qwen4_exp_gdn_layer(resident, 4).gdn_state_index == 3
    assert bind_qwen4_exp_gdn_layer(resident, 46).gdn_state_index == 35


def test_bind_qwen4_exp_qsa_layer_maps_roles_and_state_indices() -> None:
    resident = _resident()

    binding = bind_qwen4_exp_qsa_layer(resident, 3)
    assert binding.layer_id == 3
    assert binding.layer_type == "qsa"
    assert binding.qsa_state_index == 0
    assert set(binding.mixer.projections) == {
        "attn_q", "attn_k", "attn_v", "attn_output"
    }
    assert binding.mixer.q_norm_weight_ptr > 0
    assert binding.mixer.k_norm_weight_ptr > 0
    assert bind_qwen4_exp_qsa_layer(resident, 47).qsa_state_index == 11


def test_bind_qwen4_exp_gdn_layer_rejects_qsa_and_bad_ids() -> None:
    resident = _resident()

    with pytest.raises(ValueError, match="not GDN"):
        bind_qwen4_exp_gdn_layer(resident, 3)
    with pytest.raises(ValueError, match="layer_id"):
        bind_qwen4_exp_gdn_layer(resident, 48)
    with pytest.raises(ValueError, match="not QSA"):
        bind_qwen4_exp_qsa_layer(resident, 2)
