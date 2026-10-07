"""Focused contract tests for the DE-2 shared-expert decode attribution.

The DE-2 row needs the shared expert separated from the rest of the replayed MoE
graph. These tests pin the classification, the aggregation contract, and the
role-marker scope/cleanup that the attribution relies on. No GPU is required;
the end-to-end trace test skips when the git-ignored captures are absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
ATTR_SCRIPT = ROOT / "scripts" / "qwen4exp_de2_shared_expert_attribution.py"
PROFILE_SCRIPT = ROOT / "scripts" / "qwen4exp_profile_gap.py"
MODEL_ROOT = Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
RENAMED_TRACE = (
    ROOT / "benchmarks/results/2026-09-29-qwen4exp-x4-decode-split/rocprof"
)
NAME_TRACE = ROOT / "benchmarks/results/2026-09-29-qwen4exp-de1-dp4a-e2e/rocprof"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def attr():
    return _load_script(ATTR_SCRIPT, "qwen4exp_de2_shared_expert_attribution")


def _sig(grid_x: str, wg_x: str = "128", vgpr: str = "48", lds: str = "512") -> tuple[str, ...]:
    return (grid_x, "1", "1", wg_x, "1", "1", vgpr, lds)


def test_classify_signature_maps_every_moe_owner(attr) -> None:
    cases = [
        ("gguf_q4_k_selected_dual_silu_logical128_t64_kernel", _sig("40960", "64", "24"), "routed_gate_up"),
        ("q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_kernel", _sig("163840", "64", "56"), "routed_down"),
        ("gguf_k_selected_prefill_out_kernel", _sig("327680", "128", "16"), "routed_selected_misc"),
        ("qwen35_router_select_kernel", _sig("512", "512", "24"), "router_select"),
        # pack8 q8_0: out_features = (grid_x / workgroup_x) * 8
        ("gguf_k_pack8_prefill_out_kernel<float, float, 8>", _sig("10240", "128", "48"), "shared_gate_up"),
        ("gguf_k_pack8_prefill_out_kernel<float, float, 8>", _sig("40960", "128", "48"), "shared_down"),
        ("dense_gemv_out_kernel<float, float, false>", _sig("256", "256", "24", "1024"), "shared_gate_weight"),
        ("dense_gemv_out_kernel<float, float, false>", _sig("131072", "256", "24", "1024"), "router_gemv"),
        ("silu_mul_f32_kernel", _sig("768", "256", "16", "0"), "shared_silu_mul"),
        ("shared_gate_combine_out_kernel<unsigned short>", _sig("2560", "256", "16", "0"), "shared_combine"),
        ("f32_to_bf16_kernel", _sig("2560", "256", "8", "0"), "moe_convert_mixed"),
        ("weighted_sum_out_kernel<unsigned short, float>", _sig("2560", "256", "8", "0"), "moe_convert_mixed"),
    ]
    for real_name, signature, expected in cases:
        assert attr.classify_signature(real_name, signature) == expected, real_name


def test_classify_with_gguf_rejects_shared_shape_mismatch(attr) -> None:
    gguf = {
        "gate_shexp_out": 640,
        "down_shexp_out": 2560,
        "router_out": 512,
        "gate_inp_shexp_out": 1,
    }
    # Correct shapes pass.
    assert (
        attr.classify_with_gguf(
            "gguf_k_pack8_prefill_out_kernel<float, float, 8>",
            _sig("10240", "128", "48"),
            gguf,
        )
        == "shared_gate_up"
    )
    # A signature that does not match a shared-expert payload shape is not
    # silently absorbed into a shared role.
    assert (
        attr.classify_with_gguf(
            "gguf_k_pack8_prefill_out_kernel<float, float, 8>",
            _sig("20480", "128", "48"),  # out=1280, not a shared-expert shape
            gguf,
        )
        == "pack8_out1280"
    )
    # A GGUF table that disagrees with the identified shared role is a hard error.
    with pytest.raises(ValueError):
        attr.classify_with_gguf(
            "gguf_k_pack8_prefill_out_kernel<float, float, 8>",
            _sig("10240", "128", "48"),
            {**gguf, "gate_shexp_out": 1280},
        )
    with pytest.raises(ValueError):
        attr.classify_with_gguf(
            "dense_gemv_out_kernel<float, float, false>",
            _sig("256", "256", "24", "1024"),
            {**gguf, "gate_inp_shexp_out": 4},
        )


def test_select_windows_is_inclusive_and_rejects_outside(attr) -> None:
    rows = [
        {"kernel": "a", "start_ns": 100, "end_ns": 200, "duration_ns": 100},
        {"kernel": "b", "start_ns": 150, "end_ns": 250, "duration_ns": 100},
        {"kernel": "c", "start_ns": 90, "end_ns": 199, "duration_ns": 109},
    ]
    selected = attr.select_windows(rows, [(100, 200)])
    assert [row["kernel"] for row in selected] == ["a"]


def test_summarize_splits_shared_expert_and_sums_to_moe_total(attr) -> None:
    rows = [
        # routed gate+up
        {"signature": _sig("40960", "64", "24"), "duration_ns": 3_000_000},
        # routed down
        {"signature": _sig("163840", "64", "56"), "duration_ns": 2_000_000},
        # shared gate+up (2 kernels)
        {"signature": _sig("10240", "128", "48"), "duration_ns": 600_000},
        {"signature": _sig("10240", "128", "48"), "duration_ns": 600_000},
        # shared down
        {"signature": _sig("40960", "128", "48"), "duration_ns": 400_000},
        # shared gate weight
        {"signature": _sig("256", "256", "24", "1024"), "duration_ns": 100_000},
        # shared combine
        {"signature": _sig("2560", "256", "16", "0"), "duration_ns": 50_000},
        # shared silu
        {"signature": _sig("768", "256", "16", "0"), "duration_ns": 50_000},
    ]
    name_by_signature = {
        _sig("40960", "64", "24"): "gguf_q4_k_selected_dual_silu_kernel",
        _sig("163840", "64", "56"): "q5_1_selected_weighted_sum_kernel",
        _sig("10240", "128", "48"): "gguf_k_pack8_prefill_out_kernel<float, float, 8>",
        _sig("40960", "128", "48"): "gguf_k_pack8_prefill_out_kernel<float, float, 8>",
        _sig("256", "256", "24", "1024"): "dense_gemv_out_kernel<float, float, false>",
        _sig("2560", "256", "16", "0"): "shared_gate_combine_out_kernel<unsigned short>",
        _sig("768", "256", "16", "0"): "silu_mul_f32_kernel",
    }
    gguf = {
        "gate_shexp_out": 640,
        "down_shexp_out": 2560,
        "router_out": 512,
        "gate_inp_shexp_out": 1,
    }
    summary = attr.summarize(
        rows, name_by_signature, gguf, steps=1, total_decode_ms_per_step=20.0
    )
    assert summary["moe_graph_ms_per_step"] == pytest.approx(6.8)
    # shared = gate_up + down + gate_weight + combine + silu
    assert summary["shared_expert_ms_per_step"] == pytest.approx(1.8)
    assert summary["routed_ms_per_step"] == pytest.approx(5.0)
    # Every moe row lands in exactly one bucket.
    accounted = (
        summary["shared_expert_ms_per_step"]
        + summary["routed_ms_per_step"]
        + summary["router_ms_per_step"]
        + summary["mixed_ms_per_step"]
        + summary["unclassified_ms_per_step"]
    )
    assert accounted == pytest.approx(summary["moe_graph_ms_per_step"])
    assert summary["shared_expert_pct_of_decode_kernel_sum"] == pytest.approx(9.0)


def test_role_markers_scope_shared_linear_and_clean_up() -> None:
    module = _load_script(PROFILE_SCRIPT, "qwen4exp_profile_gap")

    class _Spec:
        def __init__(self, slot_path: str) -> None:
            self.slot_path = slot_path

    class _Weight:
        def __init__(self, slot_path: str) -> None:
            self.spec = _Spec(slot_path)

    class _Marker:
        def __init__(self) -> None:
            self.pushed: list[str] = []
            self.depth = 0

        def push(self, name: str) -> None:
            self.pushed.append(name)
            self.depth += 1

        def pop(self) -> None:
            self.depth -= 1

    class _Module:
        def __init__(self) -> None:
            self.calls = 0

        def launch_gguf_linear(self, *a: Any, **k: Any) -> str:
            self.calls += 1
            return "launched"

        def run_qwen4_exp_gr_read(self, *a: Any, **k: Any) -> None:
            return None

        def run_qwen4_exp_moe(self, *a: Any, **k: Any) -> None:
            return None

        def run_qwen4_exp_gdn_token_mixer(self, *a: Any, **k: Any) -> None:
            return None

        def run_qwen4_exp_qsa_prefill_token_mixer(self, *a: Any, **k: Any) -> None:
            return None

        def run_qwen4_exp_dense_qsa_token_mixer(self, *a: Any, **k: Any) -> None:
            return None

        def run_qwen4_exp_ple(self, *a: Any, **k: Any) -> None:
            return None

    module_under_test = _Module()
    marker = _Marker()
    roles = module.RoleMarkers(module_under_test, marker)
    roles.install()
    try:
        # A shared-expert weight is scoped by its slot path, request/layer exact.
        result = module_under_test.launch_gguf_linear(_Weight("layers.7.shared_gate"), 0, 0, 1, 2560, 640)
        assert result == "launched"
        assert marker.pushed == ["qwen4exp_role:linear:layers.7.shared_gate"]
        assert marker.depth == 0  # popped in the finally block
        # A routed projection is scoped separately, not folded into shared.
        module_under_test.launch_gguf_linear(_Weight("layers.7.expert_gate"), 0, 0, 1, 2560, 640)
        assert marker.pushed[-1] == "qwen4exp_role:linear:layers.7.expert_gate"
    finally:
        roles.close()
    assert (
        module_under_test.launch_gguf_linear.__func__
        is _Module.launch_gguf_linear
    )
    assert module_under_test.calls == 2


def test_gguf_role_shapes_from_shipped_model(attr) -> None:
    if not MODEL_ROOT.exists():
        pytest.skip("shipped GGUF model root is unavailable")
    shapes = attr.gguf_role_shapes(MODEL_ROOT)
    assert shapes == {
        "gate_shexp_out": 640,
        "down_shexp_out": 2560,
        "router_out": 512,
        "gate_inp_shexp_out": 1,
    }


def test_end_to_end_attribution_on_committed_capture(attr) -> None:
    renamed = RENAMED_TRACE / "x4full_kernel_trace.csv"
    marker = RENAMED_TRACE / "x4full_marker_api_trace.csv"
    name_kernel = NAME_TRACE / "e2ebase_kernel_trace.csv"
    name_marker = NAME_TRACE / "e2ebase_marker_api_trace.csv"
    if not all(p.exists() for p in (renamed, marker, name_kernel, name_marker)):
        pytest.skip("git-ignored decode captures are absent")
    if not MODEL_ROOT.exists():
        pytest.skip("shipped GGUF model root is unavailable")

    from scripts.gguf_decode_rocprof import _read_marker_windows

    windows = _read_marker_windows(marker, "qwen4exp_decode_live4097_rep")
    step_pairs = [(s, e) for _i, s, e in windows]
    steps = len(step_pairs)
    assert steps == 3

    selected = attr.select_windows(attr.read_dispatch_rows(renamed), step_pairs)
    moe_rows = [r for r in selected if str(r["kernel"]).startswith("qwen4exp_role:moe:")]
    total_decode = sum(int(r["duration_ns"]) for r in selected) / 1e6 / steps

    name_windows = _read_marker_windows(name_marker, "qwen4exp_decode_live4097_rep")
    name_selected = attr.select_windows(
        attr.read_dispatch_rows(name_kernel),
        [(s, e) for _i, s, e in name_windows],
    )
    launched = {
        launch["correlation_id"]
        for launch in attr._read_hip_launches(NAME_TRACE / "e2ebase_hip_api_trace.csv")
    }
    names_by_signature: dict[tuple[str, ...], set[str]] = {}
    for row in name_selected:
        if row.get("correlation_id") in launched:
            continue
        names_by_signature.setdefault(tuple(row["signature"]), set()).add(str(row["kernel"]))
    name_by_signature = {sig: sorted(names)[0] for sig, names in names_by_signature.items()}

    summary = attr.summarize(
        moe_rows,
        name_by_signature,
        attr.gguf_role_shapes(MODEL_ROOT),
        steps=steps,
        total_decode_ms_per_step=total_decode,
    )
    # The MoE family total and the two routed owners are published self-checks.
    assert summary["moe_graph_ms_per_step"] == pytest.approx(24.5504, abs=5e-3)
    assert summary["roles"]["routed_gate_up"]["ms_per_step"] == pytest.approx(14.013, abs=5e-3)
    assert summary["roles"]["routed_down"]["ms_per_step"] == pytest.approx(5.444, abs=5e-3)
    assert summary["roles"]["shared_combine"]["ms_per_step"] == pytest.approx(0.0671, abs=1e-3)
    # The DE-2 bracket: shared alone is inside [1.36, 2.544] ms/step.
    assert 1.36 <= summary["shared_expert_ms_per_step"] <= 2.544
