from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.speculative import mtp_resident_draft as draft


class _ReachedSharedSilu(Exception):
    pass


def _run_to_shared_silu(monkeypatch, *, dp4a: bool, dual_error: Exception | None = None):
    runner = object.__new__(draft.Qwen35GGUFResidentMTPDraftRunner)
    runner.runtime = SimpleNamespace(memset_async=lambda *args: None)
    runner.hidden_size = 64
    runner.num_heads = 2
    runner.num_kv_heads = 1
    runner.qk_head_dim = 32
    runner.experts_used = 2
    runner.inter_dim = 32
    runner.sync_stage_timings = False
    runner.eps = 1e-6
    runner._router_row_parallel_enabled = False
    for index, name in enumerate((
        "attn_normed", "wq", "wk", "wv", "q_full", "key_cur",
        "value_cur", "query", "gate", "q_norm", "k_norm", "attn", "gated", "wo",
        "wo_out", "projected", "attended", "post_norm_weight", "post_norm",
        "router_weight_f32", "router_logits", "selected", "routing", "post_norm_bf16",
        "gate_exps", "up_exps", "gate_bf16", "up_bf16", "shared_gate", "shared_up",
        "shared_gate_out", "shared_up_out", "shared_inter",
    )):
        setattr(runner, name, SimpleNamespace(ptr=0x1000 + index * 0x100, nbytes=256))
    runner._mtp_lib = object()
    runner._k_lib = object()
    runner._q4_lib = object()
    runner._cast_lib = object()
    runner._router_lib = object()
    runner._project_current_to_attn_normed = lambda *args, **kwargs: None
    runner._try_dense_q8_dp4a_triple_f32 = lambda *args, **kwargs: False
    runner._try_dense_q8_dp4a_f32 = lambda *args, **kwargs: False
    calls = []

    def try_dp4a(*args, **kwargs):
        calls.append(("dp4a", args, kwargs))
        return dp4a

    def dual(*args, **kwargs):
        calls.append(("dual", args, kwargs))
        if dual_error is not None:
            raise dual_error

    def single(*args, **kwargs):
        calls.append(("single", args, kwargs))

    def silu(*args, **kwargs):
        calls.append(("silu", args, kwargs))
        raise _ReachedSharedSilu

    runner._try_dense_q8_dp4a_dual_f32 = try_dp4a
    for name in (
        "mtp_split_q_gate_f32", "mtp_rmsnorm_f32", "mtp_rope_f32",
        "mtp_dense_attn_f32", "mtp_sigmoid_gate_mul_f32", "mtp_add_f32",
        "mtp_linear_f32", "qwen35_router_select", "f32_to_bf16",
        "gguf_q4_k_selected_dual_gemv_bf16_bf16_out",
    ):
        monkeypatch.setattr(draft, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(draft, "gguf_q8_0_gemv_f32_f32_out", single)
    monkeypatch.setattr(draft, "gguf_q8_0_dual_gemv_f32_f32_out", dual)
    monkeypatch.setattr(draft, "mtp_silu_mul_f32", silu)
    expected_error = type(dual_error) if dual_error is not None else _ReachedSharedSilu
    with pytest.raises(expected_error) as caught:
        runner._run_one(
            SimpleNamespace(ptr=0xF000), SimpleNamespace(ptr=0xF100),
            cos_ptr=1, sin_ptr=2, pos_ptr=3, ctx_ptr=4,
            dense_key_cache=None, dense_value_cache=None, dense_cache_len=0,
            rotary_dim=32, stream=7,
        )
    if dual_error is not None:
        assert caught.value is dual_error
    return runner, calls


@pytest.mark.parametrize("retired", [None, "0", "false", "1", "true"])
@pytest.mark.parametrize("dp4a", [False, True])
def test_shared_gate_up_uses_dual_unless_separate_dp4a_route_handles_it(
    monkeypatch, retired, dp4a: bool,
) -> None:
    if retired is None:
        monkeypatch.delenv("HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL", retired)
    runner, calls = _run_to_shared_silu(monkeypatch, dp4a=dp4a)
    shared_singles = [c for c in calls if c[0] == "single" and c[1][1] in {
        runner.shared_gate.ptr, runner.shared_up.ptr,
    }]
    assert shared_singles == []
    route_calls = [c for c in calls if c[0] != "single"]
    assert [c[0] for c in route_calls] == (["dp4a", "silu"] if dp4a else ["dp4a", "dual", "silu"])
    if not dp4a:
        _, args, kwargs = route_calls[1]
        assert args == (
            runner.post_norm.ptr, runner.shared_gate.ptr, runner.shared_up.ptr,
            runner.shared_gate_out.ptr, runner.shared_up_out.ptr, 1, 64, 32,
        )
        assert kwargs == {"stream": 7, "library": runner._k_lib, "runtime": runner.runtime}


def test_dual_launch_failure_propagates_without_single_resubmission(monkeypatch) -> None:
    error = RuntimeError("shared dual launch failed")
    runner, calls = _run_to_shared_silu(monkeypatch, dp4a=False, dual_error=error)
    assert [c[0] for c in calls if c[0] != "single"] == ["dp4a", "dual"]
    assert not any(c[0] == "single" and c[1][1] in {
        runner.shared_gate.ptr, runner.shared_up.ptr,
    } for c in calls)
