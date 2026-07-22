from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen35_gguf_runner as qgr
from hipengine.runtime.gguf_linear import set_wmma_prefill_enabled


@pytest.fixture(autouse=True)
def _reset_wmma_prefill_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HIPENGINE_GGUF_WMMA_PREFILL", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_T16_DS4_PREFILL", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS", raising=False)
    set_wmma_prefill_enabled(None)
    yield
    set_wmma_prefill_enabled(None)


def test_qwen35moe_compact_wmma_off_by_default_routes_raw_selected_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", _fail_if_called("group_count"))
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_pair",
        lambda *args, **kwargs: calls.append(("raw_pair", None)) or False,
    )
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("raw_linear", weight.spec.source.name)),
    )

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    assert ("raw_pair", None) in calls
    assert [payload for name, payload in calls if name == "raw_linear"] == [
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
    ]
    assert "compact_gate_up" not in [name for name, _ in calls]


def test_qwen35moe_compact_wmma_opt_in_routes_grouped_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    _patch_compact_scheduler(monkeypatch, calls)
    _patch_compact_registry(monkeypatch, calls, down_quant="gguf_q6_k")
    monkeypatch.setattr(qgr, "_read_i64_device_scalar", lambda *args, **kwargs: 16)
    monkeypatch.setenv("HIPENGINE_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS", "0")
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair", _fail_if_called("raw_pair"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_linear", _fail_if_called("raw_linear"))
    set_wmma_prefill_enabled(True)

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    names = [name for name, _ in calls]
    assert names.index("group_count") < names.index("group_scatter_gather") < names.index("tile_map")
    assert ("compact_gate_up", (6, 256, 256, 256, 4, 16)) in calls
    assert ("silu_dual", (6, 256)) in calls
    assert ("compact_down", (6, 256, 256, 4, 16)) in calls
    assert ("weighted_lanes", (3, 2, 256)) in calls
    assert ("shared_batch", (3, 256, 1)) in calls


def test_qwen35moe_iq_grouped_prefill_policy_defaults_on_with_optout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qgr._iq_grouped_prefill_enabled() is True
    assert qgr._COMPACT_MOE_IQ_GROUPED_DUAL_KEYS[("gguf_iq3_xxs", "gguf_iq3_xxs")].variant == (
        "selected_dual_grouped_prefill_compact_auto_bf16_bf16_out"
    )
    monkeypatch.setenv("HIPENGINE_GGUF_IQ_GROUPED_PREFILL", "0")
    assert qgr._iq_grouped_prefill_enabled() is False


def test_qwen35moe_iq_grouped_scalar_routes_without_tile_map_or_d2h(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_runner_and_scratch()
    _set_iq_weights(runner, gate_quant="gguf_iq3_xxs", down_quant="gguf_iq4_xs")
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    _patch_compact_scheduler(monkeypatch, calls)
    _patch_iq_compact_registry(monkeypatch, calls, down_quant="gguf_iq4_xs")
    monkeypatch.setattr(qgr, "qwen35_moe_wmma_tile_map", _fail_if_called("tile_map"))
    monkeypatch.setattr(qgr, "_read_i64_device_scalar", _fail_if_called("scalar_d2h"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair", _fail_if_called("raw_pair"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_linear", _fail_if_called("raw_linear"))

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    assert ("iq_grouped_gate_up", (6, 256, 256, 4)) in calls
    assert ("silu_dual", (6, 256)) in calls
    assert ("iq_grouped_down", (6, 256, 256, 4)) in calls
    assert ("weighted_lanes", (3, 2, 256)) in calls
    assert "tile_map" not in [name for name, _ in calls]


def test_qwen35moe_iq_compact_wmma_is_not_admitted_to_runtime() -> None:
    runner, _scratch = _fake_runner_and_scratch()
    _set_iq_weights(runner, gate_quant="gguf_iq3_xxs", down_quant="gguf_iq4_xs")
    layer = runner.weights.layer(0)

    assert (
        qgr._resolve_compact_moe_wmma_kernels(
            layer.weight("ffn_gate_exps"),
            layer.weight("ffn_up_exps"),
            layer.weight("ffn_down_exps"),
        )
        is None
    )


def test_qwen35moe_iq_grouped_scalar_uses_direct_q6_compact_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_runner_and_scratch()
    _set_iq_weights(runner, gate_quant="gguf_iq4_xs", down_quant="gguf_q6_k")
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    _patch_compact_scheduler(monkeypatch, calls)
    _patch_iq_compact_registry(monkeypatch, calls, down_quant="gguf_q6_k")
    monkeypatch.setattr(qgr, "qwen35_moe_wmma_tile_map", _fail_if_called("tile_map"))
    monkeypatch.setattr(qgr, "_read_i64_device_scalar", _fail_if_called("scalar_d2h"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair", _fail_if_called("raw_pair"))
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("raw_linear", weight.spec.source.name)),
    )
    monkeypatch.setenv("HIPENGINE_GGUF_IQ_GROUPED_PREFILL", "1")

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    assert ("iq_grouped_gate_up", (6, 256, 256, 4)) in calls
    assert [payload for name, payload in calls if name == "raw_linear"] == [
        "ffn_down_exps"
    ]


def test_qwen35moe_iq_grouped_small_assignment_count_stays_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, scratch = _fake_runner_and_scratch()
    runner.weights.config.expert_count = 16
    _set_iq_weights(runner, gate_quant="gguf_iq3_xxs", down_quant="gguf_iq4_xs")
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", _fail_if_called("group_count"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair_silu", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_pair",
        lambda *args, **kwargs: calls.append(("raw_pair", None)) or False,
    )
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("raw_linear", weight.spec.source.name)),
    )
    monkeypatch.setattr(qgr, "_launch_weighted_selected_raw_gguf_moe_linear", lambda *args, **kwargs: False)
    monkeypatch.setenv("HIPENGINE_GGUF_IQ_GROUPED_PREFILL", "1")
    set_wmma_prefill_enabled(True)

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    assert ("raw_pair", None) in calls
    assert "group_count" not in [name for name, _ in calls]


def test_qwen35moe_compact_wmma_missing_selected_kernel_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", _fail_if_called("group_count"))
    monkeypatch.setattr(qgr, "resolve", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_pair",
        lambda *args, **kwargs: calls.append(("raw_pair", None)) or False,
    )
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("raw_linear", weight.spec.source.name)),
    )
    set_wmma_prefill_enabled(True)

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    assert [payload for name, payload in calls if name == "raw_linear"] == [
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
    ]
    assert "compact_gate_up" not in [name for name, _ in calls]


def test_qwen35moe_compact_wmma_t16_ds4_flag_packs_then_routes_gate_up(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    layer = runner.weights.layer(0)
    layer._weights["ffn_gate_exps"] = _FakeWeight(
        "ffn_gate_exps", "gguf_q4_k_t16_v1", 1200, experts=4, out_features=256, in_features=256
    )
    layer._weights["ffn_up_exps"] = _FakeWeight(
        "ffn_up_exps", "gguf_q4_k_t16_v1", 1300, experts=4, out_features=256, in_features=256
    )
    layer._weights["ffn_down_exps"] = _FakeWeight(
        "ffn_down_exps", "gguf_q6_k_t16_v1", 1400, experts=4, out_features=256, in_features=256
    )
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    _patch_compact_scheduler(monkeypatch, calls)
    _patch_compact_registry(monkeypatch, calls, down_quant="gguf_q6_k_t16_v1", use_ds4=True)
    monkeypatch.setattr(qgr, "_read_i64_device_scalar", lambda *args, **kwargs: 16)
    monkeypatch.setenv("HIPENGINE_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS", "0")
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair", _fail_if_called("raw_pair"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_linear", _fail_if_called("raw_linear"))
    monkeypatch.setattr(
        qgr,
        "gguf_q8_1_mmq_ds4_pack_bf16",
        lambda x_ptr, out_ptr, rows, hidden, **kwargs: calls.append(("ds4_pack", (x_ptr, out_ptr, rows, hidden))),
    )
    monkeypatch.setenv("HIPENGINE_GGUF_T16_DS4_PREFILL", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE", "baseline")
    set_wmma_prefill_enabled(True)

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    names = [name for name, _ in calls]
    assert names.index("group_scatter_gather") < names.index("ds4_pack") < names.index("compact_gate_up_ds4")
    assert ("ds4_pack", (scratch.moe_down_out.ptr, scratch.moe_q8_1_ds4.ptr, 6, 256)) in calls
    assert ("compact_gate_up_ds4", (scratch.moe_q8_1_ds4.ptr, 6, 256, 256, 256, 4, 16)) in calls
    assert ("compact_down", (6, 256, 256, 4, 16)) in calls


@pytest.mark.parametrize(
    ("selected_rows", "num_experts", "expected_rows", "expected_tiles"),
    [
        (1, 4, 16, 1),
        (6, 4, 64, 4),
        (20, 4, 80, 5),
        (4096, 256, 7936, 496),
    ],
)
def test_compact_wmma_static_upper_bound_is_tight(
    selected_rows: int,
    num_experts: int,
    expected_rows: int,
    expected_tiles: int,
) -> None:
    assert qgr._compact_wmma_static_upper_bound(selected_rows, num_experts) == (
        expected_rows,
        expected_tiles,
    )


def test_compact_wmma_no_read_scope_is_backend_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    assert qgr._gguf_compact_wmma_no_read_max_selected_rows("hip_gfx1100") == 4096
    assert qgr._gguf_compact_wmma_no_read_max_selected_rows("hip_gfx1151") == 0

    monkeypatch.setenv("HIPENGINE_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS", "7")
    assert qgr._gguf_compact_wmma_no_read_max_selected_rows("hip_gfx1100") == 7
    assert qgr._gguf_compact_wmma_no_read_max_selected_rows("hip_gfx1151") == 7


def test_compact_wmma_small_rows_skips_host_wmma_total_read(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    scratch.moe_wmma_rows_capacity = 96
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    _patch_compact_scheduler(monkeypatch, calls)
    _patch_compact_registry(monkeypatch, calls, down_quant="gguf_q6_k")
    monkeypatch.setattr(qgr, "_read_i64_device_scalar", _fail_if_called("read_wmma_total"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair", _fail_if_called("raw_pair"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_linear", _fail_if_called("raw_linear"))
    monkeypatch.setenv("HIPENGINE_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS", "6")
    set_wmma_prefill_enabled(True)

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    assert ("tile_map", 4) in calls
    assert ("compact_gate_up", (6, 256, 256, 256, 4, 64)) in calls
    assert ("compact_down", (6, 256, 256, 4, 64)) in calls


def test_q4k_selected_dual_dp4a_off_by_default_keeps_raw_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", _fail_if_called("group_count"))
    monkeypatch.setattr(qgr, "gguf_q4_k_quantize_bf16_q8_1", _fail_if_called("q8_quantize"))
    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out",
        _fail_if_called("q8_dp4a_pair"),
    )
    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_selected_dual_gemv_bf16_bf16_out",
        lambda *args, **kwargs: calls.append(("raw_pair", args[:11])),
    )
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("raw_linear", weight.spec.source.name)),
    )

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    assert ("raw_pair", (100, 130, 12, 13, 150, 3222, 3, 6, 4, 256, 256)) in calls
    assert [payload for name, payload in calls if name == "raw_linear"] == ["ffn_down_exps"]


def test_q4k_selected_dual_dp4a_env_uses_q8_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A", "1")
    runner, scratch = _fake_runner_and_scratch()
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", _fail_if_called("group_count"))
    monkeypatch.setattr(qgr, "gguf_q4_k_selected_dual_gemv_bf16_bf16_out", _fail_if_called("raw_pair"))
    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_quantize_bf16_q8_1",
        lambda *args, **kwargs: calls.append(("q8_quantize", args[:4])),
    )
    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out",
        lambda *args, **kwargs: calls.append(("q8_dp4a_pair", args[:11])),
    )
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("raw_linear", weight.spec.source.name)),
    )

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    assert ("q8_quantize", (100, 360, 3, 256)) in calls
    assert ("q8_dp4a_pair", (360, 130, 12, 13, 150, 3222, 3, 6, 4, 256, 256)) in calls
    assert [payload for name, payload in calls if name == "raw_linear"] == ["ffn_down_exps"]


def test_shared_q8_dp4a_env_routes_shared_expert_projections(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "_gguf_dense_q8_dp4a_shared_enabled", lambda: True)
    monkeypatch.setattr(
        qgr,
        "_try_launch_dense_q8_pair_dp4a",
        lambda gate, up, *args, **kwargs: calls.append(
            ("shared_pair_dp4a", (gate.spec.source.name, up.spec.source.name))
        )
        or True,
    )
    monkeypatch.setattr(
        qgr,
        "_try_launch_dense_q8_single_dp4a",
        lambda weight, *args, **kwargs: calls.append(("shared_down_dp4a", weight.spec.source.name)) or True,
    )
    monkeypatch.setattr(qgr, "launch_gguf_linear_pair_concat", _fail_if_called("shared_pair_fallback"))
    monkeypatch.setattr(qgr, "launch_gguf_linear", _fail_if_called("shared_down_fallback"))
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_pair",
        lambda *args, **kwargs: calls.append(("raw_pair", None)) or False,
    )
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("raw_linear", weight.spec.source.name)),
    )

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    assert ("shared_pair_dp4a", ("ffn_gate_shexp", "ffn_up_shexp")) in calls
    assert ("shared_down_dp4a", "ffn_down_shexp") in calls
    assert ("silu_separate", None) in calls


def _fake_runner_and_scratch():
    cfg = SimpleNamespace(
        is_moe=True,
        expert_used_count=2,
        expert_count=4,
        hidden_size=256,
        expert_feed_forward_length=256,
        expert_shared_feed_forward_length=16,
    )
    layer = _FakeLayer()
    weights = SimpleNamespace(config=cfg, layer=lambda layer_id: layer)
    runner = object.__new__(qgr.Qwen35GGUFFullStackRunner)
    runner.backend = "hip_gfx1100"
    runner.weights = weights
    runner.runtime = "runtime-sentinel"
    # object.__new__ bypasses __post_init__, which normally resolves `auto`.
    runner.backend = "hip_gfx1100"
    scratch = SimpleNamespace(
        post_norm=_buf(100),
        moe_router_logits=_buf(110),
        moe_shared_gate_logits=_buf(120),
        moe_selected_experts=_buf(130),
        moe_routing_weights=_buf(140),
        ffn_gate_up=_buf(150),
        ffn_intermediate=_buf(160),
        ffn_down=_buf(170),
        moe_q8_1=_buf(360, nbytes=3 * (256 // 32) * 36),
        moe_down_out=_buf(180),
        moe_q8_1_ds4=_buf(185, nbytes=4096),
        moe_group_counts=_buf(190),
        moe_padded_counts=_buf(200),
        moe_scatter_offsets=_buf(210),
        moe_expert_start_compact=_buf(220),
        moe_expert_start_wmma=_buf(230),
        moe_total_compact=_buf(240),
        moe_wmma_total=_buf(250),
        moe_tile_expert=_buf(260),
        moe_sorted_lanes=_buf(270),
        moe_sorted_experts=_buf(280),
        moe_sorted_weights=_buf(290),
        moe_lane_to_row=_buf(300),
        moe_group_counts_zero=np.zeros((4,), dtype=np.int32),
        moe_scatter_offsets_zero=np.zeros((4,), dtype=np.int32),
        moe_wmma_total_host=np.empty((1,), dtype=np.int64),
        moe_selected_rows_capacity=6,
        moe_wmma_rows_capacity=70,
        moe_shared_gate=_buf(310),
        moe_shared_up=_buf(320),
        moe_shared_intermediate=_buf(330),
        moe_shared_out=_buf(340),
        residual=_buf(350),
    )
    return runner, scratch


class _FakeWeight:
    def __init__(self, name: str, quant_key: str, ptr: int, *, experts: int, out_features: int, in_features: int):
        row_bytes = max(1, in_features // 2)
        self.spec = SimpleNamespace(
            quant_key=quant_key,
            layout="dense_bf16" if quant_key == "dense" else "raw_gguf",
            source=SimpleNamespace(
                name=name,
                shape=(experts, out_features, in_features),
                byte_shape=(experts, out_features, row_bytes),
            ),
        )
        self._allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=ptr), buffer=SimpleNamespace(nbytes=1))

    def allocation(self, name: str = "raw"):
        return self._allocation


class _FakeLayer:
    def __init__(self):
        self._weights = {
            "ffn_gate_inp": _FakeWeight("ffn_gate_inp", "dense", 10, experts=1, out_features=1, in_features=1),
            "ffn_gate_inp_shexp": _FakeWeight("ffn_gate_inp_shexp", "dense", 11, experts=1, out_features=1, in_features=1),
            "ffn_gate_exps": _FakeWeight("ffn_gate_exps", "gguf_q4_k", 12, experts=4, out_features=256, in_features=256),
            "ffn_up_exps": _FakeWeight("ffn_up_exps", "gguf_q4_k", 13, experts=4, out_features=256, in_features=256),
            "ffn_down_exps": _FakeWeight("ffn_down_exps", "gguf_q6_k", 14, experts=4, out_features=256, in_features=256),
            "ffn_gate_shexp": _FakeWeight("ffn_gate_shexp", "dense", 15, experts=1, out_features=1, in_features=1),
            "ffn_up_shexp": _FakeWeight("ffn_up_shexp", "dense", 16, experts=1, out_features=1, in_features=1),
            "ffn_down_shexp": _FakeWeight("ffn_down_shexp", "dense", 17, experts=1, out_features=1, in_features=1),
        }

    def weight(self, slot: str):
        return self._weights[slot]


def _buf(ptr: int, *, nbytes: int = 8):
    return SimpleNamespace(ptr=ptr, nbytes=nbytes)


def _set_iq_weights(runner, *, gate_quant: str, down_quant: str) -> None:
    layer = runner.weights.layer(0)
    layer._weights["ffn_gate_exps"] = _FakeWeight(
        "ffn_gate_exps", gate_quant, 1200, experts=4, out_features=256, in_features=256
    )
    layer._weights["ffn_up_exps"] = _FakeWeight(
        "ffn_up_exps", gate_quant, 1300, experts=4, out_features=256, in_features=256
    )
    layer._weights["ffn_down_exps"] = _FakeWeight(
        "ffn_down_exps", down_quant, 1400, experts=4, out_features=256, in_features=256
    )


def _patch_common_moe_kernels(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, object]]) -> None:
    monkeypatch.setattr(
        qgr,
        "_launch_qwen35_router_logits_bf16_hidden",
        lambda *args, **kwargs: calls.append(("router", (args[1].spec.source.name, args[3], args[5]))),
    )
    monkeypatch.setattr(qgr, "qwen35_router_select", lambda *args, **kwargs: calls.append(("router_select", None)))
    monkeypatch.setattr(qgr, "copy_host_to_device", lambda *args, **kwargs: calls.append(("zero", None)))
    monkeypatch.setattr(qgr, "silu_mul_separate_out_bf16", lambda *args, **kwargs: calls.append(("silu_separate", None)))
    monkeypatch.setattr(
        qgr,
        "weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w",
        lambda *args, **kwargs: calls.append(("weighted_shared_batch", None)),
    )
    monkeypatch.setattr(qgr, "launch_gguf_linear_pair", lambda *args, **kwargs: calls.append(("linear_pair", None)) or False)
    monkeypatch.setattr(qgr, "launch_gguf_linear", lambda *args, **kwargs: calls.append(("linear", None)))


def _patch_compact_scheduler(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, object]]) -> None:
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", lambda *args, **kwargs: calls.append(("group_count", None)))
    monkeypatch.setattr(qgr, "qwen35_moe_group_prefix", lambda *args, **kwargs: calls.append(("group_prefix", None)))
    monkeypatch.setattr(
        qgr,
        "qwen35_moe_group_scatter_gather_lowp",
        lambda *args, **kwargs: calls.append(("group_scatter_gather", None)),
    )
    monkeypatch.setattr(
        qgr,
        "qwen35_moe_wmma_tile_map",
        lambda *args, **kwargs: calls.append(("tile_map", kwargs.get("tile_capacity"))),
    )
    monkeypatch.setattr(
        qgr,
        "silu_mul_dual_out_bf16",
        lambda gate_up, out, *, rows, features, **kwargs: calls.append(("silu_dual", (rows, features))),
    )
    monkeypatch.setattr(
        qgr,
        "weighted_lanes_sum_out_bf16_f32w",
        lambda *args, **kwargs: calls.append(("weighted_lanes", args[5:8])),
    )
    monkeypatch.setattr(
        qgr,
        "shared_gate_combine_residual_batch_out_bf16",
        lambda *args, **kwargs: calls.append(("shared_batch", args[5:8])),
    )


def _patch_iq_compact_registry(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, object]],
    *,
    down_quant: str,
) -> None:
    gate_quant = "gguf_iq4_xs" if down_quant == "gguf_q6_k" else "gguf_iq3_xxs"
    gate_key = qgr._COMPACT_MOE_IQ_GROUPED_DUAL_KEYS[(gate_quant, gate_quant)]
    down_key = KernelKey(
        "hip_gfx1100",
        "moe_linear",
        down_quant,
        "selected_grouped_prefill_compact_auto_bf16_bf16_out",
    )

    def fake_gate_up(*args, **kwargs):
        calls.append(
            (
                "iq_grouped_gate_up",
                (
                    kwargs["compact_rows"],
                    kwargs["in_features"],
                    kwargs["out_features"],
                    kwargs["num_experts"],
                ),
            )
        )

    def fake_down(*args, **kwargs):
        calls.append(
            (
                "iq_grouped_down",
                (
                    kwargs["compact_rows"],
                    kwargs["in_features"],
                    kwargs["out_features"],
                    kwargs["num_experts"],
                ),
            )
        )

    available = {
        gate_key: fake_gate_up,
        **{
            key: (lambda *args, **kwargs: None)
            for key in qgr._COMPACT_MOE_GROUPED_SCHEDULER_KEYS
        },
        **{key: (lambda *args, **kwargs: None) for key in qgr._COMPACT_MOE_FUSED_KEYS},
    }
    if down_quant != "gguf_q6_k":
        available[down_key] = fake_down

    def fake_resolve(*, backend: str, layer: str, quant: str, variant: str = "", missing: str = "error"):
        key = KernelKey(backend, layer, quant, variant)
        fn = available.get(key)
        if fn is not None or missing == "none":
            return fn
        raise AssertionError(f"unexpected resolve miss for {key}")

    monkeypatch.setattr(qgr, "resolve", fake_resolve)


def _patch_compact_registry(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, object]],
    *,
    down_quant: str,
    use_ds4: bool = False,
) -> None:
    gate_key = (
        qgr._COMPACT_MOE_Q4_DUAL_DS4_KEYS[("gguf_q4_k_t16_v1", "gguf_q4_k_t16_v1")]
        if use_ds4
        else qgr._COMPACT_MOE_Q4_DUAL_KEYS[("gguf_q4_k", "gguf_q4_k")]
    )
    down_key = qgr._COMPACT_MOE_DOWN_KEYS[down_quant]

    def fake_gate_up(*args, **kwargs):
        name = "compact_gate_up_ds4" if use_ds4 else "compact_gate_up"
        calls.append((name, (args[0], *args[7:13]) if use_ds4 else args[7:13]))

    def fake_down(*args, **kwargs):
        calls.append(("compact_down", args[6:11]))

    available = {
        gate_key: fake_gate_up,
        down_key: fake_down,
        **{key: (lambda *args, **kwargs: None) for key in qgr._COMPACT_MOE_SCHEDULER_KEYS},
        **{key: (lambda *args, **kwargs: None) for key in qgr._COMPACT_MOE_FUSED_KEYS},
    }

    def fake_resolve(*, backend: str, layer: str, quant: str, variant: str = "", missing: str = "error"):
        key = KernelKey(backend, layer, quant, variant)
        fn = available.get(key)
        if fn is not None or missing == "none":
            return fn
        raise AssertionError(f"unexpected resolve miss for {key}")

    monkeypatch.setattr(qgr, "resolve", fake_resolve)


def _fail_if_called(name: str):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} should not be called")

    return fail
