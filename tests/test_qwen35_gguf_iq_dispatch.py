"""Registry-driven UD-Q3_K_M resident-MoE dispatch policy tests."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.generation import qwen35_gguf as qwen35_gguf_generation
from hipengine.generation.registry import resolve_text_generator
from hipengine.kernels import registry as kernel_registry
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen35_gguf_runner as qgr
from tests.test_qwen35_gguf_compact_moe_gemv_routing import (
    _FakeWeight,
    _fail_if_called,
    _fake_runner_and_scratch,
    _patch_common_moe_kernels,
)

_IQ3_SINGLE_KEY = KernelKey(
    "hip_gfx1100",
    "moe_linear",
    "gguf_iq3_xxs",
    "selected_gemv_decode_bf16_bf16_out",
)
_IQ3_DUAL_SILU_KEY = KernelKey(
    "hip_gfx1100",
    "moe_linear",
    "gguf_iq3_xxs",
    "selected_dual_silu_gemv_decode_bf16_bf16_out",
)
_IQ4_SINGLE_KEY = KernelKey(
    "hip_gfx1100",
    "moe_linear",
    "gguf_iq4_xs",
    "selected_gemv_decode_bf16_bf16_out",
)
_IQ4_WEIGHTED_KEY = KernelKey(
    "hip_gfx1100",
    "moe_linear",
    "gguf_iq4_xs",
    "selected_weighted_down_gemv_decode_bf16_bf16_out",
)


def test_ud_q3_k_m_public_generator_key_is_registered() -> None:
    for model in ("qwen3_5_gguf", "qwen3_5_moe_gguf"):
        assert resolve_text_generator(
            model=model,
            backend="hip_gfx1100",
            quant="gguf_ud_q3_k_m",
        ) is qwen35_gguf_generation.make_qwen35_gguf_ud_q3_k_m_generator


def test_ud_q3_k_m_generator_plugin_selects_native_bulk_correctness_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qwen35_gguf_generation.Qwen35GGUFTokenizer,
        "from_gguf_info",
        staticmethod(lambda weight_index: SimpleNamespace(weight_index=weight_index)),
    )

    generator = qwen35_gguf_generation.make_qwen35_gguf_ud_q3_k_m_generator(
        model_path="/tmp/q3.gguf",
        weight_index=object(),
        model_plugin=object(),
    )
    session = SimpleNamespace(default_bulk_attention_mode="bulk")
    generator._configure_session(session)

    assert generator.bulk_prefill_attention_mode == "native"
    assert session.default_bulk_attention_mode == "native"


def test_resident_session_honors_plugin_selected_native_bulk_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object.__new__(qgr.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=4))
    )
    session.default_bulk_attention_mode = "native"
    calls: list[str] = []

    def fake_bulk_prefill(self, token_ids, *, bulk_attention_mode, return_logits):
        del self, token_ids, return_logits
        calls.append(bulk_attention_mode)
        return SimpleNamespace(token_id=11)

    monkeypatch.setattr(
        qgr.Qwen35GGUFResidentSession,
        "_run_bulk_prefill_and_sample",
        fake_bulk_prefill,
    )

    result = session.prefill([9419, 11, 271, 40], return_logits=False)

    assert result.token_id == 11
    assert calls == ["native"]


def test_iq_helpers_resolve_exact_four_axis_registry_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple, dict]] = []

    def fake(label: str):
        def launch(*args, **kwargs) -> None:
            calls.append((label, args, kwargs))

        return launch

    monkeypatch.setitem(kernel_registry._KERNELS, _IQ3_SINGLE_KEY, fake("iq3_single"))
    monkeypatch.setitem(kernel_registry._KERNELS, _IQ3_DUAL_SILU_KEY, fake("iq3_dual"))
    monkeypatch.setitem(kernel_registry._KERNELS, _IQ4_SINGLE_KEY, fake("iq4_single"))
    monkeypatch.setitem(kernel_registry._KERNELS, _IQ4_WEIGHTED_KEY, fake("iq4_weighted"))
    gate = _expert_weight("gate", "gguf_iq3_xxs", 12, out_features=512, in_features=2048)
    up = _expert_weight("up", "gguf_iq3_xxs", 13, out_features=512, in_features=2048)
    down = _expert_weight("down", "gguf_iq4_xs", 14, out_features=2048, in_features=512)

    assert qgr._launch_selected_raw_gguf_moe_pair_silu(
        gate,
        up,
        100,
        110,
        120,
        x_rows=1,
        rows=8,
        num_experts=256,
        in_features=2048,
        out_features=512,
        stream=7,
        runtime="runtime",
    )
    qgr._launch_selected_raw_gguf_moe_linear(
        gate,
        130,
        140,
        150,
        x_rows=1,
        rows=8,
        num_experts=256,
        in_features=2048,
        out_features=512,
        stream=7,
        runtime="runtime",
    )
    qgr._launch_selected_raw_gguf_moe_linear(
        down,
        200,
        210,
        220,
        x_rows=8,
        rows=8,
        num_experts=256,
        in_features=512,
        out_features=2048,
        stream=7,
        runtime="runtime",
    )
    assert qgr._launch_weighted_selected_raw_gguf_moe_linear(
        down,
        300,
        310,
        320,
        340,
        tokens=1,
        top_k=8,
        num_experts=256,
        in_features=512,
        out_features=2048,
        stream=7,
        runtime="runtime",
    )

    assert [label for label, _, _ in calls] == [
        "iq3_dual",
        "iq3_single",
        "iq4_single",
        "iq4_weighted",
    ]
    assert calls[0][1] == (100, 110, 12, 13, 120)
    assert calls[0][2] == {
        "x_rows": 1,
        "rows": 8,
        "num_experts": 256,
        "in_features": 2048,
        "out_features": 512,
        "stream": 7,
        "runtime": "runtime",
    }
    assert calls[1][1] == (130, 140, 12, 150)
    assert calls[2][1] == (200, 210, 14, 220)
    assert calls[3][1] == (300, 310, 320, 14, 340)
    assert calls[3][2] == {
        "tokens": 1,
        "top_k": 8,
        "num_experts": 256,
        "in_features": 512,
        "out_features": 2048,
        "stream": 7,
        "runtime": "runtime",
    }


def test_iq_fused_and_weighted_registry_misses_preserve_unfused_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(kernel_registry._KERNELS, _IQ3_DUAL_SILU_KEY, raising=False)
    monkeypatch.delitem(kernel_registry._KERNELS, _IQ4_WEIGHTED_KEY, raising=False)
    gate = _expert_weight("gate", "gguf_iq3_xxs", 12, out_features=512, in_features=2048)
    up = _expert_weight("up", "gguf_iq3_xxs", 13, out_features=512, in_features=2048)
    down = _expert_weight("down", "gguf_iq4_xs", 14, out_features=2048, in_features=512)

    assert not qgr._launch_selected_raw_gguf_moe_pair_silu(
        gate,
        up,
        100,
        110,
        120,
        x_rows=1,
        rows=8,
        num_experts=256,
        in_features=2048,
        out_features=512,
        stream=0,
        runtime="runtime",
    )
    assert not qgr._launch_weighted_selected_raw_gguf_moe_linear(
        down,
        200,
        210,
        220,
        240,
        tokens=1,
        top_k=8,
        num_experts=256,
        in_features=512,
        out_features=2048,
        stream=0,
        runtime="runtime",
    )


def test_missing_iq_selected_single_key_raises_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(kernel_registry._KERNELS, _IQ3_SINGLE_KEY, raising=False)
    weight = _expert_weight("gate", "gguf_iq3_xxs", 12, out_features=512, in_features=2048)

    with pytest.raises(ValueError, match="unsupported selected GGUF MoE quant 'gguf_iq3_xxs'"):
        qgr._launch_selected_raw_gguf_moe_linear(
            weight,
            100,
            110,
            120,
            x_rows=1,
            rows=8,
            num_experts=256,
            in_features=2048,
            out_features=512,
            stream=0,
            runtime="runtime",
        )


def test_mixed_iq_gate_up_pair_never_selects_a_fused_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        kernel_registry._KERNELS,
        _IQ3_DUAL_SILU_KEY,
        _fail_if_called("mixed fused IQ pair"),
    )
    gate = _expert_weight("gate", "gguf_iq3_xxs", 12, out_features=512, in_features=2048)
    up = _expert_weight("up", "gguf_iq4_xs", 13, out_features=512, in_features=2048)

    assert not qgr._launch_selected_raw_gguf_moe_pair_silu(
        gate,
        up,
        100,
        110,
        120,
        x_rows=1,
        rows=8,
        num_experts=256,
        in_features=2048,
        out_features=512,
        stream=0,
        runtime="runtime",
    )


def test_iq_registry_dispatch_helpers_do_not_branch_on_quant_names() -> None:
    for name in (
        "_launch_selected_raw_gguf_moe_pair_silu",
        "_launch_selected_raw_gguf_moe_linear",
        "_launch_weighted_selected_raw_gguf_moe_linear",
    ):
        source = inspect.getsource(getattr(qgr, name))
        assert "gguf_iq3_xxs" not in source
        assert "gguf_iq4_xs" not in source


def test_main_iq_layer_uses_fused_gate_up_and_weighted_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_runner_and_scratch()
    _set_expert_weights(runner, gate_quant="gguf_iq3_xxs", up_quant="gguf_iq3_xxs", down_quant="gguf_iq4_xs")
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.delenv("HIPENGINE_GGUF_COMPACT_MOE_C1", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_FUSED_MOE_FFN", raising=False)
    monkeypatch.setattr(qgr, "launch_gguf_linear_pair_concat", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_pair_silu",
        lambda *args, **kwargs: calls.append(("iq3_fused", kwargs["x_rows"])) or True,
    )
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair", _fail_if_called("split gate/up"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_linear", _fail_if_called("selected single"))
    monkeypatch.setattr(
        qgr,
        "_launch_weighted_selected_raw_gguf_moe_linear",
        lambda *args, **kwargs: calls.append(("iq4_weighted", (kwargs["tokens"], kwargs["top_k"]))) or True,
    )
    monkeypatch.setattr(
        qgr,
        "weighted_sum_shared_gate_combine_residual_out_bf16_f32w",
        _fail_if_called("weighted fallback combine"),
    )
    monkeypatch.setattr(
        qgr,
        "shared_gate_combine_residual_out_bf16",
        lambda *args, **kwargs: calls.append(("shared_only", None)),
    )

    runner._run_post_attention_moe_c1(0, out_ptr=9000, scratch=scratch, stream=7)

    assert ("iq3_fused", 1) in calls
    assert ("iq4_weighted", (1, 2)) in calls
    assert ("shared_only", None) in calls


def test_main_iq_layer_fuses_already_weighted_tail_with_next_rms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_runner_and_scratch()
    runner.weights.config.rms_norm_eps = 1e-6
    _set_expert_weights(runner, gate_quant="gguf_iq3_xxs", up_quant="gguf_iq3_xxs", down_quant="gguf_iq4_xs")
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.delenv("HIPENGINE_GGUF_COMPACT_MOE_C1", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_FUSED_MOE_FFN", raising=False)
    monkeypatch.setattr(qgr, "launch_gguf_linear_pair_concat", lambda *args, **kwargs: False)
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair_silu", lambda *args, **kwargs: True)
    monkeypatch.setattr(qgr, "_launch_weighted_selected_raw_gguf_moe_linear", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        qgr,
        "shared_gate_combine_residual_out_bf16",
        _fail_if_called("unfused shared combine"),
    )
    monkeypatch.setattr(
        qgr,
        "shared_gate_combine_residual_rmsnorm_gguf_bf16_out",
        lambda *args, **kwargs: calls.append(("shared_next_rms", (args, kwargs))),
        raising=False,
    )

    runner._run_post_attention_moe_c1(
        0,
        out_ptr=9000,
        scratch=scratch,
        next_norm_weight_ptr=8000,
        next_norm_out_ptr=8100,
        stream=7,
    )

    fused_args, fused_kwargs = next(payload for name, payload in calls if name == "shared_next_rms")
    assert fused_args[4:10] == (8000, 8100, 9000, 1, 256, 1)
    assert fused_kwargs["stream"] == 7


def test_missing_iq4_weighted_composite_uses_selected_and_weighted_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_runner_and_scratch()
    _set_expert_weights(runner, gate_quant="gguf_iq3_xxs", up_quant="gguf_iq3_xxs", down_quant="gguf_iq4_xs")
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.delenv("HIPENGINE_GGUF_COMPACT_MOE_C1", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_FUSED_MOE_FFN", raising=False)
    monkeypatch.setattr(qgr, "launch_gguf_linear_pair_concat", lambda *args, **kwargs: False)
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair_silu", lambda *args, **kwargs: True)
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair", _fail_if_called("split gate/up"))
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("selected_single", weight.spec.quant_key)),
    )
    monkeypatch.setattr(qgr, "_launch_weighted_selected_raw_gguf_moe_linear", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        qgr,
        "shared_gate_combine_residual_out_bf16",
        _fail_if_called("already-weighted combine"),
    )

    runner._run_post_attention_moe_c1(0, out_ptr=9000, scratch=scratch, stream=7)

    assert [payload for name, payload in calls if name == "selected_single"] == ["gguf_iq4_xs"]
    assert ("weighted_shared", None) in calls


def test_slot_weighted_tail_keeps_feature_parallel_combine_before_next_rms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_runner_and_scratch()
    runner.weights.config.rms_norm_eps = 1e-6
    _set_expert_weights(runner, gate_quant="gguf_iq3_xxs", up_quant="gguf_iq3_xxs", down_quant="gguf_iq4_xs")
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.delenv("HIPENGINE_GGUF_COMPACT_MOE_C1", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_FUSED_MOE_FFN", raising=False)
    monkeypatch.setattr(qgr, "launch_gguf_linear_pair_concat", lambda *args, **kwargs: False)
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair_silu", lambda *args, **kwargs: True)
    monkeypatch.setattr(qgr, "_launch_weighted_selected_raw_gguf_moe_linear", lambda *args, **kwargs: False)
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_linear", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        qgr,
        "weighted_sum_shared_gate_combine_residual_out_bf16_f32w",
        lambda *args, **kwargs: calls.append(("weighted_combine", (args, kwargs))),
    )
    monkeypatch.setattr(
        qgr,
        "gguf_rmsnorm_bf16_f32_weight",
        lambda *args, **kwargs: calls.append(("next_rms", (args, kwargs))),
    )

    runner._run_post_attention_moe_c1(
        0,
        out_ptr=9000,
        scratch=scratch,
        next_norm_weight_ptr=8000,
        next_norm_out_ptr=8100,
        stream=7,
    )

    combine_args, combine_kwargs = next(payload for name, payload in calls if name == "weighted_combine")
    norm_args, norm_kwargs = next(payload for name, payload in calls if name == "next_rms")
    assert combine_args[4:8] == (scratch.residual.ptr, 9000, 2, 256)
    assert combine_kwargs["stream"] == 7
    assert norm_args[:3] == (9000, 8000, 8100)
    assert norm_kwargs["rows"] == 1
    assert norm_kwargs["hidden_size"] == 256
    assert norm_kwargs["stream"] == 7


def test_bulk_iq_layer_uses_direct_x_rows_fused_and_weighted_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_runner_and_scratch()
    _set_expert_weights(runner, gate_quant="gguf_iq3_xxs", up_quant="gguf_iq3_xxs", down_quant="gguf_iq4_xs")
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "_try_run_post_attention_moe_rows_compact_wmma", lambda *args, **kwargs: False)
    monkeypatch.setattr(qgr, "launch_gguf_linear_pair_concat", lambda *args, **kwargs: False)

    def fused(*args, **kwargs):
        calls.append(("iq3_fused_bulk", (kwargs["x_rows"], kwargs["rows"], kwargs["allow_legacy"])))
        return True

    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair_silu", fused)
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair", _fail_if_called("split gate/up"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_linear", _fail_if_called("selected single"))
    monkeypatch.setattr(
        qgr,
        "_launch_weighted_selected_raw_gguf_moe_linear",
        lambda *args, **kwargs: calls.append(("iq4_weighted_bulk", (kwargs["tokens"], kwargs["top_k"]))) or True,
    )
    monkeypatch.setattr(
        qgr,
        "weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w",
        _fail_if_called("weighted batch fallback combine"),
    )
    monkeypatch.setattr(
        qgr,
        "shared_gate_combine_residual_batch_out_bf16",
        lambda *args, **kwargs: calls.append(("shared_only_batch", args[5:8])),
    )

    runner._run_post_attention_moe_rows(0, out_ptr=9000, scratch=scratch, rows=2, stream=7)

    assert ("iq3_fused_bulk", (2, 4, False)) in calls
    assert ("iq4_weighted_bulk", (2, 2)) in calls
    assert ("shared_only_batch", (2, 256, 1)) in calls


def test_blk39_iq4_gate_up_and_q6_down_keep_single_kernel_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_runner_and_scratch()
    _set_expert_weights(runner, gate_quant="gguf_iq4_xs", up_quant="gguf_iq4_xs", down_quant="gguf_q6_k")
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.delenv("HIPENGINE_GGUF_COMPACT_MOE_C1", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_FUSED_MOE_FFN", raising=False)
    monkeypatch.setattr(qgr, "launch_gguf_linear_pair_concat", lambda *args, **kwargs: False)
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair_silu", lambda *args, **kwargs: False)
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("selected_single", weight.spec.quant_key)),
    )
    monkeypatch.setattr(
        qgr,
        "_launch_weighted_selected_raw_gguf_moe_linear",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        qgr,
        "shared_gate_combine_residual_out_bf16",
        _fail_if_called("already-weighted combine"),
        raising=False,
    )

    runner._run_post_attention_moe_c1(0, out_ptr=9000, scratch=scratch, stream=7)

    assert [payload for name, payload in calls if name == "selected_single"] == [
        "gguf_iq4_xs",
        "gguf_iq4_xs",
        "gguf_q6_k",
    ]
    assert ("weighted_shared", None) in calls


def _expert_weight(
    name: str,
    quant_key: str,
    ptr: int,
    *,
    out_features: int,
    in_features: int,
) -> _FakeWeight:
    return _FakeWeight(
        name,
        quant_key,
        ptr,
        experts=256,
        out_features=out_features,
        in_features=in_features,
    )


def _set_expert_weights(
    runner,
    *,
    gate_quant: str,
    up_quant: str,
    down_quant: str,
) -> None:
    layer = runner.weights.layer(0)
    layer._weights["ffn_gate_exps"] = _FakeWeight(
        "ffn_gate_exps", gate_quant, 12, experts=4, out_features=256, in_features=256
    )
    layer._weights["ffn_up_exps"] = _FakeWeight(
        "ffn_up_exps", up_quant, 13, experts=4, out_features=256, in_features=256
    )
    layer._weights["ffn_down_exps"] = _FakeWeight(
        "ffn_down_exps", down_quant, 14, experts=4, out_features=256, in_features=256
    )
