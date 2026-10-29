"""Retired T16 residency cannot replace the raw-weight loading path."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.generation import qwen4_exp_gguf as generation
from hipengine.loading import qwen4_exp_materialize as loading
from hipengine.runtime import qwen4_exp_runner as runtime


@pytest.mark.parametrize("name", [
    "_t16_resident_enabled", "_t16_resident_max_layers",
    "_t16_resident_slot_suffixes", "_populate_t16_resident_stages",
    "_populate_t16_resident_stages_from_env",
])
def test_retired_residency_helpers_are_removed(name):
    assert not hasattr(generation, name)


def test_retired_generator_populate_method_is_removed():
    assert not hasattr(generation.Qwen4ExpGGUFTextGenerator,
                       "populate_t16_resident_stages")


@pytest.mark.parametrize("name", [
    "attach_t16_allocation", "detach_raw_allocation",
    "materialize_qwen4_exp_t16_stages",
])
def test_retired_loading_stages_are_removed(name):
    assert not hasattr(loading, name)


def test_retired_routes_and_candidate_controls_are_removed():
    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_t16_selected_prefill as kernels
    from hipengine.quant import gguf_t16

    for name in ("_T16_GATE_UP_TILES", "_T16_DOWN_TILES",
                 "_t16_gate_up_enabled", "_t16_down_enabled"):
        assert not hasattr(runtime, name)
    assert not {"t16_gate_up", "t16_down"} & CANDIDATES.keys()
    assert not hasattr(kernels, "gguf_q5_1_t16_selected_expert_major_wmma_comp_bf16_bf16_out")
    assert not hasattr(gguf_t16, "repack_gguf_q5_1_tile16_packed")
    # The distinct, expanded Q5_1 layout still serves Gemma4.
    assert callable(gguf_t16.repack_gguf_q5_1_tile16)
    root = Path(__file__).resolve().parents[1]
    for directory in ("hipengine", "scripts"):
        for path in (root / directory).rglob("*.py"):
            assert "HIPENGINE_QWEN4_EXP_T16_" not in path.read_text(), path


def test_weight_replacement_and_layer_rebinding_survive(monkeypatch):
    old, new = object(), object()
    resident = loading.Qwen4ExpResidentWeights(
        plan=None, device_weights={"weight": old}, ple_table=None,
        ple_staging=None,
    )
    owner = SimpleNamespace(
        resident=resident, config=SimpleNamespace(layer_types=("gdn", "qsa")),
    )
    def bind(current, layer):
        return (layer, current.weight("weight"))
    monkeypatch.setattr(runtime, "bind_qwen4_exp_gdn_layer", bind)
    monkeypatch.setattr(runtime, "bind_qwen4_exp_qsa_layer", bind)
    rebind = runtime.Qwen4ExpGGUFResidentModelRunner.rebind_resident_weights
    rebind(owner)
    assert owner.gdn_bindings == {0: (0, old)}
    assert owner.qsa_bindings == {1: (1, old)}
    resident.swap_weight("weight", new)
    rebind(owner)
    assert owner.gdn_bindings == {0: (0, new)}
    assert owner.qsa_bindings == {1: (1, new)}
    with pytest.raises(KeyError, match="unknown resident weight slot"):
        resident.swap_weight("missing", new)
    resident.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        resident.swap_weight("weight", old)
