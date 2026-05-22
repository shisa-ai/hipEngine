"""Routing tests for the P9.B6 compact MoE c=1 decode dispatch.

These tests cover the new ``_try_run_post_attention_moe_c1_compact_gemv``
opt-in branch in ``_run_post_attention_moe_c1``:

* Off by default: c=1 decode keeps using the legacy per-row selected GEMV
  via ``_launch_selected_raw_gguf_moe_pair`` + ``_linear``.
* Opt-in routes through the compact scheduler (P8.6 ``group_count`` /
  ``group_prefix`` / ``group_scatter_gather``; no ``wmma_tile_map`` because
  GEMV does not consume the WMMA tile space) and the new P9.B1 / P9.B2
  selected GEMV decode kernels.
* When any registry key for the new GEMV decode chain is missing, the
  c=1 path falls back to the legacy decoder without raising.
* The shared-expert and combine paths reuse the same primitives as the
  bulk WMMA compact path.

Mirrors :mod:`tests/test_qwen35_gguf_compact_moe_wmma_routing.py`. The
fake-scratch / fake-layer / fake-resolve harness keeps the tests no-GPU
and focused on the dispatch decision tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen35_gguf_runner as qgr
from hipengine.runtime.gguf_linear import set_gemv_decode_enabled


@pytest.fixture(autouse=True)
def _reset_gemv_decode_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HIPENGINE_GGUF_GEMV_DECODE", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_REPACK", raising=False)
    set_gemv_decode_enabled(None)
    yield
    set_gemv_decode_enabled(None)


def test_compact_gemv_off_by_default_uses_legacy_selected_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", _fail_if_called("group_count"))
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_pair",
        lambda *args, **kwargs: calls.append(("legacy_pair", None)) or False,
    )
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("legacy_linear", weight.spec.source.name)),
    )

    runner._run_post_attention_moe_c1(0, out_ptr=9000, scratch=scratch, stream=7)

    assert ("legacy_pair", None) in calls
    assert [payload for name, payload in calls if name == "legacy_linear"] == [
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
    ]
    assert "compact_gate_up" not in [name for name, _ in calls]


def test_c1_decode_uses_split_router_coop_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", _fail_if_called("group_count"))
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_pair",
        lambda *args, **kwargs: calls.append(("legacy_pair", None)) or False,
    )
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("legacy_linear", weight.spec.source.name)),
    )

    runner._run_post_attention_moe_c1(0, out_ptr=9000, scratch=scratch, stream=7)

    names = [name for name, _ in calls]
    assert names.count("router_split_coop") == 1
    assert "router" not in names
    assert "router_select" not in names
    assert ("router_split_coop", (10, 11, 110, 1, 256, 4, 2)) in calls
    assert ("router_split_coop_threads", 256) in calls



def test_compact_gemv_opt_in_routes_grouped_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_COMPACT_MOE_C1", "1")
    runner, scratch = _fake_runner_and_scratch()
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    _patch_compact_scheduler(monkeypatch, calls)
    _patch_compact_gemv_registry(monkeypatch, calls, down_quant="gguf_q6_k")
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_pair", _fail_if_called("legacy_pair"))
    monkeypatch.setattr(qgr, "_launch_selected_raw_gguf_moe_linear", _fail_if_called("legacy_linear"))
    set_gemv_decode_enabled(True)

    runner._run_post_attention_moe_c1(0, out_ptr=9000, scratch=scratch, stream=7)

    names = [name for name, _ in calls]
    # Compact scheduler runs in order, and `tile_map` (WMMA-only) is NOT in the
    # GEMV decode chain even when registered.
    assert names.index("group_count") < names.index("group_scatter_gather")
    assert "tile_map" not in names
    # Compact GEMV kernels are called with rows=top_k=2.
    assert ("compact_gate_up", (2, 256, 256, 256, 4)) in calls
    assert ("silu_dual", (2, 256)) in calls
    assert ("compact_down", (2, 256, 256, 4)) in calls
    # Weighted lane combine and shared-gate residual combine fire at rows=1.
    assert ("weighted_lanes", (1, 2, 256)) in calls
    assert ("shared_batch", (1, 256, 1)) in calls


def test_t16_weights_route_direct_selected_tiles_allocations(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    layer = runner.weights.layer(0)
    layer._weights["ffn_gate_exps"] = _FakeWeight(
        "ffn_gate_exps", "gguf_q4_k_t16_v1", 12, experts=4, out_features=256, in_features=256
    )
    layer._weights["ffn_up_exps"] = _FakeWeight(
        "ffn_up_exps", "gguf_q4_k_t16_v1", 13, experts=4, out_features=256, in_features=256
    )
    layer._weights["ffn_down_exps"] = _FakeWeight(
        "ffn_down_exps", "gguf_q6_k_t16_v1", 14, experts=4, out_features=256, in_features=256
    )
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", _fail_if_called("group_count"))
    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out",
        lambda *args, **kwargs: calls.append(("t16_pair_silu", (args[2], args[3], args[4], args[5:10]))),
    )
    monkeypatch.setattr(
        qgr,
        "gguf_q6_k_t16_selected_gemv_bf16_bf16_out",
        lambda *args, **kwargs: calls.append(("t16_down", (args[2], args[3], args[4:9]))),
    )
    set_gemv_decode_enabled(True)

    runner._run_post_attention_moe_c1(0, out_ptr=9000, scratch=scratch, stream=7)

    assert ("t16_pair_silu", (1012, 1013, 160, (1, 2, 4, 256, 256))) in calls
    assert ("t16_down", (1014, 180, (2, 2, 4, 256, 256))) in calls
    assert ("weighted_shared", None) in calls



def test_row_bulk_t16_direct_selected_prefill_routes_without_compact_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    layer = runner.weights.layer(0)
    layer._weights["ffn_gate_exps"] = _FakeWeight(
        "ffn_gate_exps", "gguf_q4_k_t16_v1", 12, experts=4, out_features=256, in_features=256
    )
    layer._weights["ffn_up_exps"] = _FakeWeight(
        "ffn_up_exps", "gguf_q4_k_t16_v1", 13, experts=4, out_features=256, in_features=256
    )
    layer._weights["ffn_down_exps"] = _FakeWeight(
        "ffn_down_exps", "gguf_q5_k_t16_v1", 14, experts=4, out_features=256, in_features=256
    )
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", _fail_if_called("group_count"))
    monkeypatch.setattr(qgr, "_launch_selected_expert_pack8_moe_pair", _fail_if_called("sidecar_pair"))
    monkeypatch.setattr(qgr, "_launch_selected_expert_pack8_moe_linear", _fail_if_called("sidecar_linear"))
    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out",
        lambda *args, **kwargs: calls.append(("t16_pair", (args[2], args[3], args[4], args[5], args[6:11]))),
    )
    monkeypatch.setattr(
        qgr,
        "gguf_q5_k_t16_selected_gemv_bf16_bf16_out",
        lambda *args, **kwargs: calls.append(("t16_down", (args[2], args[3], args[4:9]))),
    )
    set_gemv_decode_enabled(True)

    runner._run_post_attention_moe_rows(0, rows=2, out_ptr=9000, scratch=scratch, stream=7)

    names = [name for name, _ in calls]
    assert "group_count" not in names
    assert "tile_map" not in names
    assert "weighted_lanes" not in names
    assert ("t16_pair", (1012, 1013, 150, 150 + 4 * 256 * 2, (2, 4, 4, 256, 256))) in calls
    assert ("t16_down", (1014, 180, (4, 4, 4, 256, 256))) in calls
    assert ("weighted_shared_batch", None) in calls



def test_compact_gemv_missing_kernel_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch()
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", _fail_if_called("group_count"))
    # The registry resolver returns ``None`` for every compact key -> the
    # opt-in branch should detect the registry miss and fall back.
    monkeypatch.setattr(qgr, "resolve", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_pair",
        lambda *args, **kwargs: calls.append(("legacy_pair", None)) or False,
    )
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("legacy_linear", weight.spec.source.name)),
    )
    set_gemv_decode_enabled(True)

    runner._run_post_attention_moe_c1(0, out_ptr=9000, scratch=scratch, stream=7)

    assert [payload for name, payload in calls if name == "legacy_linear"] == [
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
    ]
    assert "compact_gate_up" not in [name for name, _ in calls]


def test_compact_gemv_missing_compact_scratch_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, scratch = _fake_runner_and_scratch(strip_compact_scratch=True)
    calls: list[tuple[str, object]] = []
    _patch_common_moe_kernels(monkeypatch, calls)
    monkeypatch.setattr(qgr, "qwen35_moe_group_count", _fail_if_called("group_count"))
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_pair",
        lambda *args, **kwargs: calls.append(("legacy_pair", None)) or False,
    )
    monkeypatch.setattr(
        qgr,
        "_launch_selected_raw_gguf_moe_linear",
        lambda weight, *args, **kwargs: calls.append(("legacy_linear", weight.spec.source.name)),
    )
    set_gemv_decode_enabled(True)

    runner._run_post_attention_moe_c1(0, out_ptr=9000, scratch=scratch, stream=7)

    assert ("legacy_pair", None) in calls
    assert "compact_gate_up" not in [name for name, _ in calls]


# ---------------------------------------------------------------------------
# Fake fixtures (mirrors test_qwen35_gguf_compact_moe_wmma_routing.py).
# ---------------------------------------------------------------------------


def _fake_runner_and_scratch(*, strip_compact_scratch: bool = False):
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
    fields = dict(
        post_norm=_buf(100),
        moe_router_logits=_buf(110),
        moe_shared_gate_logits=_buf(120),
        moe_selected_experts=_buf(130),
        moe_routing_weights=_buf(140),
        ffn_gate_up=_buf(150),
        ffn_intermediate=_buf(160),
        ffn_down=_buf(170),
        moe_down_out=_buf(180),
        moe_shared_gate=_buf(310),
        moe_shared_up=_buf(320),
        moe_shared_intermediate=_buf(330),
        moe_shared_out=_buf(340),
        residual=_buf(350),
        moe_selected_host=np.empty((4,), dtype=np.int64),
    )
    if not strip_compact_scratch:
        fields.update(
            moe_group_counts=_buf(190),
            moe_padded_counts=_buf(200),
            moe_scatter_offsets=_buf(210),
            moe_expert_start_compact=_buf(220),
            moe_total_compact=_buf(240),
            moe_sorted_lanes=_buf(270),
            moe_sorted_experts=_buf(280),
            moe_sorted_weights=_buf(290),
            moe_lane_to_row=_buf(300),
            moe_group_counts_zero=np.zeros((4,), dtype=np.int32),
            moe_scatter_offsets_zero=np.zeros((4,), dtype=np.int32),
            moe_selected_rows_capacity=4,
        )
    return runner, SimpleNamespace(**fields)


class _FakeWeight:
    def __init__(
        self,
        name: str,
        quant_key: str,
        ptr: int,
        *,
        experts: int,
        out_features: int,
        in_features: int,
    ):
        row_bytes = max(1, in_features // 2)
        layout = "gguf_q8_0_t16_v1" if quant_key.endswith("_t16_v1") else ("raw_gguf" if quant_key.startswith("gguf_") else "dense_bf16")
        self.spec = SimpleNamespace(
            quant_key=quant_key,
            layout=layout,
            source=SimpleNamespace(
                name=name,
                shape=(experts, out_features, in_features),
                byte_shape=(experts, out_features, row_bytes),
            ),
        )
        self._allocations = {
            "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=ptr), buffer=SimpleNamespace(nbytes=1)),
            "tiles": SimpleNamespace(tensor=SimpleNamespace(ptr=ptr + 1000), buffer=SimpleNamespace(nbytes=1)),
        }

    def allocation(self, name: str = "raw"):
        return self._allocations[name]


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


def _buf(ptr: int):
    return SimpleNamespace(ptr=ptr, nbytes=8)


def _patch_common_moe_kernels(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, object]]) -> None:
    monkeypatch.setattr(qgr, "qwen35_router_logits_bf16", lambda *args, **kwargs: calls.append(("router", None)))
    monkeypatch.setattr(qgr, "qwen35_router_select", lambda *args, **kwargs: calls.append(("router_select", None)))
    monkeypatch.setattr(
        qgr,
        "qwen35_router_topk_split_shared_coop_out_bf16",
        lambda *args, **kwargs: (
            calls.append(("router_split_coop", args[1:4] + args[6:10])),
            calls.append(("router_split_coop_threads", kwargs.get("threads"))),
        ),
    )
    monkeypatch.setattr(qgr, "copy_host_to_device", lambda *args, **kwargs: calls.append(("zero", None)))
    monkeypatch.setattr(qgr, "silu_mul_separate_out_bf16", lambda *args, **kwargs: calls.append(("silu_separate", None)))
    monkeypatch.setattr(
        qgr,
        "weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w",
        lambda *args, **kwargs: calls.append(("weighted_shared_batch", None)),
    )
    monkeypatch.setattr(
        qgr,
        "weighted_sum_shared_gate_combine_residual_out_bf16_f32w",
        lambda *args, **kwargs: calls.append(("weighted_shared", None)),
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
        _fail_if_called("tile_map"),
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
        lambda *args, **kwargs: (
            calls.append(("shared_batch", args[5:8])),
            calls.append(("shared_batch_gate_ptr", args[2])),
        ),
    )


def _patch_compact_gemv_registry(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, object]],
    *,
    down_quant: str,
    gate_quant: str = "gguf_q4_k",
    up_quant: str = "gguf_q4_k",
) -> None:
    gate_key = qgr._COMPACT_MOE_Q4_DUAL_GEMV_KEYS[(gate_quant, up_quant)]
    down_key = qgr._COMPACT_MOE_DOWN_GEMV_KEYS[down_quant]
    scheduler_keys = (
        KernelKey("hip_gfx1100", "moe_group_count", "w4_paro", "qwen35"),
        KernelKey("hip_gfx1100", "moe_group_prefix", "w4_paro", "qwen35"),
        KernelKey("hip_gfx1100", "moe_group_scatter_gather", "w4_paro", "qwen35_lowp"),
    )

    def fake_gate_up(*args, **kwargs):
        # compact_dual_*_gemv signature:
        # (x, expert_start, qa, qb, out, compact_rows, in_f, out_a, out_b, num_experts)
        calls.append(("compact_gate_up", args[5:10]))
        calls.append(("compact_gate_up_ptrs", args[2:4]))

    def fake_down(*args, **kwargs):
        # compact_*_gemv signature:
        # (x, expert_start, qweight, out, compact_rows, in_f, out_f, num_experts)
        calls.append(("compact_down", args[4:8]))
        calls.append(("compact_down_ptr", args[2]))

    available = {
        gate_key: fake_gate_up,
        down_key: fake_down,
        **{key: (lambda *args, **kwargs: None) for key in scheduler_keys},
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
