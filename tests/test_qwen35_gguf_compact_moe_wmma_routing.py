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
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair", _fail_if_called("raw_pair"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_linear", _fail_if_called("raw_linear"))
    monkeypatch.setattr(
        qgr,
        "gguf_q8_1_mmq_ds4_pack_bf16",
        lambda x_ptr, out_ptr, rows, hidden, **kwargs: calls.append(("ds4_pack", (x_ptr, out_ptr, rows, hidden))),
    )
    monkeypatch.setenv("HIPENGINE_GGUF_T16_DS4_PREFILL", "1")
    set_wmma_prefill_enabled(True)

    runner._run_post_attention_moe_rows(0, 9000, scratch, rows=3, stream=7)

    names = [name for name, _ in calls]
    assert names.index("group_scatter_gather") < names.index("ds4_pack") < names.index("compact_gate_up_ds4")
    assert ("ds4_pack", (scratch.moe_down_out.ptr, scratch.moe_q8_1_ds4.ptr, 6, 256)) in calls
    assert ("compact_gate_up_ds4", (scratch.moe_q8_1_ds4.ptr, 6, 256, 256, 256, 4, 16)) in calls
    assert ("compact_down", (6, 256, 256, 4, 16)) in calls


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
    runner.weights = weights
    runner.runtime = "runtime-sentinel"
    scratch = SimpleNamespace(
        post_norm=_buf(100),
        moe_router_logits=_buf(110),
        moe_shared_gate_logits=_buf(120),
        moe_selected_experts=_buf(130),
        moe_routing_weights=_buf(140),
        ffn_gate_up=_buf(150),
        ffn_intermediate=_buf(160),
        ffn_down=_buf(170),
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


def _patch_common_moe_kernels(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, object]]) -> None:
    monkeypatch.setattr(qgr, "qwen35_router_logits_bf16", lambda *args, **kwargs: calls.append(("router", None)))
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
    monkeypatch.setattr(qgr, "qwen35_moe_wmma_tile_map", lambda *args, **kwargs: calls.append(("tile_map", None)))
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
