"""Prerequisite contracts for historical Q6 activation-staging comparisons."""

import json
from types import SimpleNamespace

import pytest

from hipengine.runtime import laguna_gguf_runner as runner
from hipengine.runtime import laguna_moe as moe
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out as launch_q6,
)
from scripts import (
    laguna_q6_compact_activation_ab as compact,
    laguna_q6_half_row_activation_ab as half,
    laguna_q6_skip_padded_activation_ab as padded,
)


@pytest.mark.parametrize("module,flag", [
    (compact, "q6_compact_activation"),
    (half, "q6_half_row_activation"),
    (padded, "q6_skip_padded_activation"),
])
@pytest.mark.parametrize("enabled", [False, True])
def test_comparison_children_reach_kernel_planning_with_interleaved_resident(monkeypatch, module, flag, enabled):
    config = SimpleNamespace(context_length=4096)
    weights = SimpleNamespace(
        config=config, backend="hip_gfx1151", q6_qmicro=True, q6_qmicro_planar=False,
    )
    owner = SimpleNamespace(weights=weights, runtime=object())
    monkeypatch.setattr(module, "_compiler_version", lambda path: "synthetic")
    sessions = []

    def stop_before_libraries(config, **kwargs):
        raise RuntimeError("comparison reached kernel planning")

    monkeypatch.setattr(runner, "resolve_laguna_eager_kernel_plan", stop_before_libraries)

    def construct(**kwargs):
        session = object.__new__(runner.LagunaGGUFResidentSession)
        sessions.append(session)
        return session.__init__(**kwargs)

    monkeypatch.setattr(module, "LagunaGGUFResidentSession", construct)
    with pytest.raises(RuntimeError, match="comparison reached kernel planning"):
        module._child_session(owner, **{flag: enabled})
    session, = sessions
    assert getattr(session, flag) is enabled
    assert session.q6_qmicro and not session.q6_qmicro_planar
    assert not session.q6_qmicro_permute
    assert not session.q6_wmma_prefetch_weight
    assert not session.q6_wmma_prefetch_activation
    assert not session.q6_precomputed_activation_sums
    if module is compact:
        assert not session.q6_half_row_activation and not session.q6_skip_padded_activation
    elif module is half:
        assert session.q6_compact_activation and not session.q6_skip_padded_activation
    else:
        assert session.q6_compact_activation and session.q6_half_row_activation
    assert session._closed and session.weights is None


@pytest.mark.parametrize("module", [compact, half, padded])
def test_comparison_owner_loads_same_layout_and_records_policy(monkeypatch, module, capsys):
    runtime = SimpleNamespace(device_synchronize=lambda: None)
    monkeypatch.setattr(module, "_parse_args", lambda: SimpleNamespace(repetitions=1, require_cached_build=False, output=None))
    monkeypatch.setattr(module, "GGUFReader", lambda path: SimpleNamespace(info=object()))
    monkeypatch.setattr(module.LagunaGGUFTokenizer, "from_gguf_info", lambda info: object())
    monkeypatch.setattr(module, "_load_prompts", lambda *a: [])
    monkeypatch.setattr(module, "_profile_token_stream", lambda *a: ([1] * 512, None))
    monkeypatch.setattr(module, "get_hip_runtime", lambda: runtime)
    monkeypatch.setattr(module, "_compiler_version", lambda path: "synthetic")
    constructors = []
    closed = []

    def construct(*args, **kwargs):
        constructors.append(kwargs)
        return SimpleNamespace(close=lambda: closed.append("owner"))

    monkeypatch.setattr(module, "LagunaGGUFResidentSession", construct)
    monkeypatch.setattr(module, "_child_session", lambda *a, **kw: SimpleNamespace(prefill=lambda tokens: None, close=lambda: closed.append("child")))
    monkeypatch.setattr(module, "_run", lambda *a: {
        "tok_s": 1.0, "next_token": 2, "next_token_logit_hex": "synthetic",
        "logits_sha256": "l", "final_hidden_sha256": "f", "post_layer_hidden_sha256": "p",
        "kv_sha256": "k", "final_position": 511,
    })
    assert module.main() == 0
    kw, = constructors
    expected = {
        "q6_qmicro": True, "q6_qmicro_planar": False, "q6_qmicro_permute": False,
        "q6_wmma_prefetch_weight": False, "q6_wmma_prefetch_activation": False,
        "q6_precomputed_activation_sums": False,
    }
    if module is compact:
        expected.update(q6_half_row_activation=False, q6_skip_padded_activation=False)
    elif module is half:
        expected.update(q6_skip_padded_activation=False)
    assert {key: kw.get(key) for key in expected} == expected
    output = capsys.readouterr().out
    record = json.loads(output[output.index("{"):])
    assert record["comparison_policy"] == expected
    assert record["comparison"]["complete_state_exact"]
    assert closed == ["child", "child", "owner"]


@pytest.mark.parametrize("lanes", [31, 32, 33, 97])
@pytest.mark.parametrize("compact_flag,half_flag,padded_flag,symbol_part", [
    (False, False, False, "qmicro_selected"),
    (True, False, False, "qmicro_compact_activation_selected"),
    (True, True, False, "qmicro_half_row_activation_selected"),
    (True, True, True, "qmicro_skip_padded_activation_selected"),
])
def test_activation_peer_reaches_matching_primitive_with_owned_pointers(lanes, compact_flag, half_flag, padded_flag, symbol_part):
    launches = []

    class Function:
        def __call__(self, *args):
            launches.append(tuple(arg.value for arg in args))
            return 0

    class Library:
        def __getattr__(self, name):
            assert symbol_part in name
            return Function()

    ptrs = [0x1000 + index * 0x100 for index in range(8)]
    scratch = SimpleNamespace(**{
        name: SimpleNamespace(ptr=ptr)
        for name, ptr in zip((
            "grouped_expert_start", "grouped_expert_start_mmq32", "grouped_sorted_experts",
            "grouped_mmq_total", "expert_down", "expert_intermediate", "expert_gate_up", "unused",
        ), ptrs, strict=True)
    })
    quant_calls = []
    scratch.plan = SimpleNamespace(
        expert_count=3, expert_ffn_size=256, hidden_size=64,
        q6_qmicro=True, q6_compact_activation=compact_flag,
        q6_half_row_activation=half_flag, q6_skip_padded_activation=padded_flag,
        q6_qmicro_planar=False, q6_qmicro_permute=False,
        down_activation_quant=lambda *a, **kw: quant_calls.append((a, kw)),
        mmq64_tile_map=lambda *a, **kw: None,
        selected_down_prefill=launch_q6,
    )
    weight = SimpleNamespace(
        spec=SimpleNamespace(quant_key="gguf_q6_k_t16_v1"),
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=0x9000)),
    )
    runtime = object()
    assert moe._launch_selected_down_mmq64x32_d4x3_f32(
        SimpleNamespace(weight=lambda name: weight), scratch, lanes=lanes,
        stream=7, runtime=runtime, libraries={"selected_gate_up_prefill": Library()},
        residual_passes=1, rowvec=True, q6_tile_rows=64,
    )
    args, kwargs = quant_calls.pop()
    assert args == (scratch.expert_intermediate.ptr, scratch.expert_gate_up.ptr, lanes, 256)
    assert kwargs["runtime"] is runtime and kwargs["stream"] == 7
    args, = launches
    assert args[:6] == (
        scratch.expert_gate_up.ptr, scratch.grouped_expert_start.ptr,
        scratch.grouped_expert_start_mmq32.ptr, scratch.grouped_sorted_experts.ptr,
        0x9000, scratch.expert_down.ptr,
    )
    assert args[6:10] == (lanes, 256, 64, 3)
    assert args[10] % 64 == 0 and args[11] == 7
