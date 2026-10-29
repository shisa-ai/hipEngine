"""Host ownership contracts for the fused selected-SiLU pack comparison."""

from types import SimpleNamespace

import pytest

from hipengine.runtime import laguna_moe as moe
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from scripts import laguna_fused_silu_pack_ab as comparison


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("gate", sorted(moe._FUSED_SELECTED_SILU_PACK_GATE_UP_MODES) + ["mmq64x32_d4x3_f32"])
@pytest.mark.parametrize("down", sorted(moe._FUSED_SELECTED_SILU_PACK_DOWN_MODES) + ["mmq64x32_d4x3_f32"])
def test_fused_silu_requires_compatible_one_plane_consumers(requested, gate, down):
    assert moe._should_fuse_selected_silu_pack(
        requested=requested, selected_gate_up_mode=gate, selected_down_mode=down,
    ) is (requested and gate != "mmq64x32_d4x3_f32" and down != "mmq64x32_d4x3_f32")


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("down_bytes", [1023, 1024, 3077])
@pytest.mark.parametrize("compact", [False, True])
def test_real_caller_defers_silu_only_with_compatible_scratch_and_gate(monkeypatch, requested, down_bytes, compact):
    calls = []
    plan = SimpleNamespace(
        hidden_size=8, expert_count=11, top_k=3, routed_scaling_factor=2.5,
        router_logits_functions={"token_tile_4": lambda *a, **kw: None},
        router_select=lambda *a, **kw: None,
        grouped_weighted_sum_shared_add=lambda *a, **kw: calls.append(("combine", a, kw)),
        routed_sum_rows=lambda *a, **kw: calls.append(("routed_sum", a, kw)),
        add=lambda *a, **kw: calls.append(("add", a, kw)),
    )
    scratch = SimpleNamespace(plan=plan, max_rows=32)
    for index, name in enumerate((
        "router_logits", "routing_scores", "selection_scores", "selected_experts",
        "routing_weights", "scaled_routing_weights", "expert_down", "expert_gate_up",
        "grouped_sorted_weights", "grouped_sorted_lanes", "grouped_lane_to_row", "shared_output", "routed_output", "output",
    )):
        setattr(scratch, name, SimpleNamespace(ptr=0x2000 + index * 0x100, nbytes=1024))
    scratch.expert_down.nbytes = down_bytes
    layer = SimpleNamespace(weight=lambda name: SimpleNamespace(allocation=lambda key: SimpleNamespace(tensor=SimpleNamespace(ptr=0x1000))))
    monkeypatch.setattr(moe, "validate_laguna_moe_layer", lambda *a: None)
    for name, label, returned in (
        ("_launch_selected_gate_up_mmq32_d4x3", "gate", compact),
        ("_launch_selected_gate_up", "gate_fallback", None),
        ("_launch_selected_down_mmq64x32_d4x3_f32", "down", True),
        ("_launch_selected_down", "down_fallback", None),
        ("_launch_laguna_shared_rows", "shared", None),
    ):
        monkeypatch.setattr(moe, name, lambda *a, _label=label, _returned=returned, **kw: calls.append((_label, a, kw)) or _returned)
    runtime = object()
    result = moe.run_laguna_moe_rows(
        0x9000, layer, scratch, rows=17,
        selected_gate_up_mode="mmq128x32_d8_f32_wavecols_direct_doublebuf",
        selected_down_mode="mmq64x64_d4_f32_q6_wavecols_direct_q4",
        fuse_selected_silu_pack=requested, stream=7, runtime=runtime,
    )
    by_label = {label: (args, kwargs) for label, args, kwargs in calls}
    assert by_label["gate"][1]["defer_silu_pack"] is (requested and down_bytes >= 1024)
    if compact:
        assert "gate_fallback" not in by_label and "down_fallback" not in by_label
        assert by_label["down"][1]["fused_silu_pack"] is (requested and down_bytes >= 1024)
        assert by_label["down"][1]["lanes"] == 17 * plan.top_k
        assert by_label["combine"][0][0] == scratch.expert_down.ptr
        assert by_label["combine"][0][5] == scratch.output.ptr
    else:
        assert "gate_fallback" in by_label and "down" not in by_label
        assert "down_fallback" in by_label and "combine" not in by_label
        assert by_label["routed_sum"][0][:3] == (
            scratch.expert_down.ptr, scratch.scaled_routing_weights.ptr, scratch.routed_output.ptr,
        )
        assert by_label["add"][0][:3] == (
            scratch.routed_output.ptr, scratch.shared_output.ptr, scratch.output.ptr,
        )
    assert result is scratch.output
    assert all(kw["stream"] == 7 and kw["runtime"] is runtime for _, _, kw in calls)


def test_comparison_child_borrows_weights_and_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(comparison, "_compiler_version", lambda path: "synthetic")
    monkeypatch.setattr(comparison, "LagunaGGUFResidentSession", lambda **kw: calls.append(kw))
    owner = SimpleNamespace(weights=object(), runtime=object())
    comparison._child_session(owner)
    kw, = calls
    assert kw["resident_weights"] is owner.weights and kw["runtime"] is owner.runtime
    assert kw["context_length"] == kw["prefill_chunk_size"] == 512
    assert kw["prefill_attention_chunk_size"] == 128
    assert kw["require_cached_build"] is True


@pytest.mark.parametrize("mode", ["unfused_rollback", "fused_candidate"])
@pytest.mark.parametrize("raises", [False, True])
def test_comparison_captures_complete_state_and_closes_after_prefill(monkeypatch, mode, raises):
    events = []
    child = object.__new__(LagunaGGUFResidentSession)
    child.position = 511
    child.runtime = SimpleNamespace(device_synchronize=lambda: events.append("sync"))

    def prefill(tokens):
        assert child.fuse_selected_silu_pack is (mode == "fused_candidate")
        assert tokens == [1, 2, 3]
        if raises:
            raise RuntimeError("synthetic prefill failure")
        return SimpleNamespace(next_token_id=42, next_token_logit=1.5, logits="logits", final_hidden="final", post_layer_hidden="post")

    child.prefill = prefill
    child.close = lambda: events.append("close")
    monkeypatch.setattr(comparison, "_child_session", lambda owner: child)
    monkeypatch.setattr(comparison, "_device_digest", lambda runtime, value, dtype: value + "-digest")
    monkeypatch.setattr(comparison, "_kv_digest", lambda selected: "kv-digest" if selected is child else pytest.fail("wrong child"))
    ticks = iter((10.0, 11.0))
    monkeypatch.setattr(comparison.time, "perf_counter", lambda: next(ticks))
    if raises:
        with pytest.raises(RuntimeError, match="synthetic prefill failure"):
            comparison._run(object(), mode, [1, 2, 3])
        assert events == ["close"]
    else:
        record = comparison._run(object(), mode, [1, 2, 3])
        assert record == {
            "tok_s": 512.0, "wall_ms": 1000.0, "next_token": 42,
            "next_token_logit_hex": float(1.5).hex(), "logits_sha256": "logits-digest",
            "final_hidden_sha256": "final-digest", "post_layer_hidden_sha256": "post-digest",
            "kv_sha256": "kv-digest", "final_position": 511,
        }
        assert events == ["sync", "close"]
