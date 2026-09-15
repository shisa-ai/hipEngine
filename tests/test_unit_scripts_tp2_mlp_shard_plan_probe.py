"""Tests for the TP2 MLP shard-plan probe.

The probe is a **host-payload** prerequisite check: it establishes that the
rank-local bytes for one MLP are the right bytes in the right layout. It does not
qualify device execution, and these tests pin that scope so the artifact cannot be
read as more than it is.

The layout assertions matter because the layouts are not guessable: the Q6_K down
projection resolves to the planar t16 layout under the incumbent capability set,
not to raw storage, so a hardcoded table would benchmark a different TP1 path.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tp2_mlp_shard_plan_probe.py"
GGUF_PATH = pathlib.Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")

_model_available = GGUF_PATH.exists()
requires_model = pytest.mark.skipif(
    not _model_available, reason=f"GGUF model file not available: {GGUF_PATH}"
)


def _load():
    spec = importlib.util.spec_from_file_location("tp2_mlp_shard_plan_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_half_intermediate_row_block_is_t16_admissible(mod) -> None:
    """8704 rows is half of 17408 and still a whole number of t16 tiles."""

    admissible = mod._t16_admissible(
        out_features=8704, bytes_per_row=5120 * 144 // 256, block_bytes=144
    )
    assert admissible["admissible"] is True
    assert admissible["out_features_divisible"] is True
    assert admissible["bytes_per_row_divisible"] is True


def test_an_unaligned_row_block_is_rejected(mod) -> None:
    """The probe must not report an inadmissible local shape as usable."""

    admissible = mod._t16_admissible(out_features=8705, bytes_per_row=2880, block_bytes=144)
    assert admissible["admissible"] is False
    assert admissible["out_features_divisible"] is False


def test_every_layout_maps_to_a_repack_or_is_raw(mod) -> None:
    """Layouts drive the repack decision, not source quant types."""

    assert mod._t16_repack_for("gguf_q4_k_t16_v1", "Q4_K") is mod.repack_gguf_q4_k_tile16
    assert (
        mod._t16_repack_for("gguf_q6_k_t16_qmicro_planar_v1", "Q6_K")
        is mod.repack_gguf_q6_k_tile16_qmicro_planar
    )
    assert mod._t16_repack_for("raw_gguf", "Q6_K") is None


def test_an_unknown_layout_is_refused_rather_than_guessed(mod) -> None:
    """A layout the probe cannot materialize must fail, not fall back."""

    with pytest.raises(ValueError, match="no repack is recorded"):
        mod._t16_repack_for("gguf_q5_k_t16_v1", "Q5_K")


def test_a_missing_repack_is_refused(mod) -> None:
    with pytest.raises(ValueError, match="no t16 repack"):
        mod._t16_repack_tiles(b"", rows=1, bytes_per_row=1, quant_type="Q2_K", repack=None)


def test_the_mlp_tensor_names_are_the_three_projections(mod) -> None:
    assert mod._mlp_tensor_names(7) == (
        "blk.7.ffn_gate.weight",
        "blk.7.ffn_up.weight",
        "blk.7.ffn_down.weight",
    )


# ---------------------------------------------------------------------------
# Incumbent-plan resolution
# ---------------------------------------------------------------------------


@requires_model
def test_the_plan_resolver_reports_the_environment_it_resolved_under(mod) -> None:
    """A layout claim is only meaningful with the flags that produced it."""

    info = mod.scan_gguf(str(GGUF_PATH))
    plan, context = mod.resolve_incumbent_plan(info)
    assert context["backend"] == "hip_gfx1100"
    assert isinstance(context["decode_repack"], bool)
    assert "dense_flags" in context
    assert len(plan.layer_specs) >= 64


@requires_model
def test_an_unknown_slot_is_named_rather_than_silently_skipped(mod) -> None:
    info = mod.scan_gguf(str(GGUF_PATH))
    plan, _ = mod.resolve_incumbent_plan(info)
    with pytest.raises(SystemExit, match="no slot"):
        mod._spec_for_slot(plan, layer=0, slot="ffn_nonexistent")
    with pytest.raises(SystemExit, match="no layer"):
        mod._spec_for_slot(plan, layer=999, slot="ffn_gate")


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------


@requires_model
def test_probe_answers_every_question_with_evidence(mod) -> None:
    """The gate questions must all be answered yes on the real model."""

    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2)
    assert set(report["questions"]) == {
        "rank_local_materialization_exists",
        "rank_local_bytes_round_trip",
        "local_shapes_are_layout_admissible",
        "repack_commutes_with_the_split",
        "layouts_resolved_from_the_incumbent_planner",
    }
    for question, answer in report["questions"].items():
        assert answer["answer"] is True, question
        assert answer["evidence"]


@requires_model
def test_probe_reports_a_halved_weight_budget_and_one_reduction(mod) -> None:
    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2)
    exists = report["questions"]["rank_local_materialization_exists"]
    assert exists["shard_to_tp1_ratio"] == pytest.approx(0.5)
    assert report["reduction_point"]["per_layer_count"] == 1
    # The down projection is row-parallel over the full hidden size, so the
    # reduction carries one partial per hidden element.
    assert report["reduction_point"]["elements"] == report["hidden_size"]


@requires_model
def test_every_repack_commutes_with_its_split_axis(mod) -> None:
    """A rank must be able to repack its own slice, bit-identically.

    If this failed, the shard arm would run on weights that differ from the
    corresponding half of the TP1 repack while still producing plausible output.
    All three MLP tensors are repacked in the incumbent plan.
    """

    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2)
    question = report["questions"]["repack_commutes_with_the_split"]
    assert question["answer"] is True
    assert question["tensors_checked"] == [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ]
    assert question["not_applicable"] == []
    for name in question["tensors_checked"]:
        entry = report["tensors"][name]
        assert entry["repack_commutes_with_split"] is True, name
        for rank, shard in entry["ranks"].items():
            assert shard["repack_commutes_with_split"] is True, f"{name} rank {rank}"
            assert shard["repack_slice_compared"]


@requires_model
def test_the_incumbent_layouts_come_from_the_planner_not_a_table(mod) -> None:
    """The Q6_K down projection is not raw; guessing it benchmarks the wrong path."""

    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2)
    assert report["plan_context"]["decode_repack"] is True
    layouts = {
        name: entry["incumbent_spec"]["layout"] for name, entry in report["tensors"].items()
    }
    assert layouts["blk.0.ffn_gate.weight"] == "gguf_q4_k_t16_v1"
    assert layouts["blk.0.ffn_up.weight"] == "gguf_q4_k_t16_v1"
    # Resolved by the planner under the incumbent capability set: the planar Q6_K
    # t16 layout, not raw_gguf.
    assert layouts["blk.0.ffn_down.weight"] == "gguf_q6_k_t16_qmicro_planar_v1"
    for entry in report["tensors"].values():
        assert entry["incumbent_spec"]["allocation_names"] == ["tiles"]


@requires_model
def test_the_probe_scopes_itself_to_host_payloads(mod) -> None:
    """Passing this probe must not be readable as device qualification."""

    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2)
    assert "does not qualify device" in report["scope"]
    assert "shard_dispatch_needs_no_new_kernel" not in report["questions"]


@requires_model
def test_the_reduction_dtype_is_left_explicitly_unresolved(mod) -> None:
    """The exchange payload must match the down projection's real output dtype."""

    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2)
    point = report["reduction_point"]
    assert point["selected"] is None
    assert "unresolved" in point["selection_reason"]
    names = {option["name"] for option in point["options"]}
    assert names == {"fp32_partials", "bf16_partials_then_fp32_sum"}
    for option in point["options"]:
        assert option["note"]


@requires_model
def test_the_inventory_excludes_the_mtp_block(mod) -> None:
    """The AR cost model is 64 blocks: 16 full-attention plus 48 GDN."""

    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2)
    inventory = report["inventory"]
    assert inventory["ar_blocks"] == 64
    assert inventory["full_attention_blocks"] == 16
    assert inventory["linear_attention_gdn_blocks"] == 48
    assert inventory["excluded_blocks"] == [64]
    assert inventory["full_attention_blocks"] + inventory["linear_attention_gdn_blocks"] == 64


@requires_model
def test_the_model_identity_is_content_bound(mod) -> None:
    """A placeholder hash would let another file inherit this evidence."""

    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2)
    assert report["model_hash"] not in {"mlp-probe", "", None}
    assert len(report["model_hash"]) >= 16
    assert report["manifest_hash"]


@requires_model
def test_every_rank_local_slice_covers_half_the_split_axis(mod) -> None:
    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2)
    for name, entry in report["tensors"].items():
        axis = entry["split_axis"]
        total = entry["source_shape"][axis]
        covered = 0
        for shard in entry["ranks"].values():
            start, stop = shard["axis_ranges"][0]
            covered += stop - start
            assert (stop - start) == total // 2, name
        assert covered == total, name


# ---------------------------------------------------------------------------
# Device stage
# ---------------------------------------------------------------------------


def _rocm_available() -> bool:
    import ctypes

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


requires_rocm = pytest.mark.skipif(not _rocm_available(), reason="no ROCm HIP runtime")


@requires_rocm
@requires_model
def test_the_device_stage_matches_the_incumbent_resident_bytes(mod) -> None:
    """The shard must be a sub-range of what the engine actually puts on the card.

    Without this, a shard could be constructed correctly in isolation yet not be
    a slice of the resident weight the TP1 path uses, and every segment time
    measured on it would describe a different tensor.
    """

    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2, device=True)
    stage = report["device_stage"]
    assert stage["ran"] is not False
    for name, entry in stage["slots"].items():
        assert entry["resident_matches_host_repack"] is True, name
        assert entry["device_nbytes"] == entry["host_repack_nbytes"], name
        assert entry["allocation"] == "tiles"
        for rank, shard in entry["ranks"].items():
            assert shard["device_slice_equals_rank_payload"] is True, f"{name} rank {rank}"
            assert shard["device_slice_compared"]
    assert report["questions"]["device_resident_bytes_match_the_host_repack"]["answer"] is True
    assert (
        report["questions"]["rank_shard_is_a_sub_range_of_the_resident_weight"]["answer"]
        is True
    )


@requires_rocm
@requires_model
def test_the_device_stage_does_not_claim_kernel_execution(mod) -> None:
    """Reading back bytes is not running the segment; the scope must say so."""

    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2, device=True)
    assert "does not qualify device" in report["scope"]
    # No question may assert anything about kernel acceptance or execution.
    for question in report["questions"]:
        assert "kernel" not in question
        assert "execute" not in question


def test_the_device_stage_is_opt_in(mod) -> None:
    """A host-only run must record that the device stage did not run."""

    source = SCRIPT.read_text(encoding="utf-8")
    assert '"ran": False' in source
    assert "run with --device on a ROCm host" in source


# ---------------------------------------------------------------------------
# Fused-route admission
# ---------------------------------------------------------------------------


def test_the_fused_decode_route_is_shape_keyed(mod) -> None:
    """The gap is admission, not the kernel: record both facts."""

    admission = mod.fused_path_admission(hidden_size=5120, intermediate=17408)
    assert admission["lookup_key"] == "(rows, in_features, out_features)"
    assert admission["tp1_key"] == [1, 5120, 17408]
    assert admission["tp1_admitted"] is True
    assert admission["tp1_variant"] == "dense_dual_local32_bf16_bf16_out"
    # The shard's half-intermediate shape is not in the table.
    assert admission["shard_key"] == [1, 5120, 8704]
    assert admission["shard_admitted"] is False
    assert admission["gap"] is not None
    assert "inside the kernel's shape contract" in admission["gap"]
    # Both shapes are inside the kernel's contract, so the gap is purely
    # admission: no kernel work and no shape work is required.
    assert admission["tp1_shape_error"] is None
    assert admission["shard_shape_error"] is None


def test_the_shard_shape_satisfies_the_pure_shape_contract() -> None:
    """A half-intermediate shard is inside the kernel launcher's shape contract.

    This is why the gap is a one-line policy admission rather than a new kernel.
    The contract is the kernel module's pure validator, asked directly: no
    pointers, no library, no device.
    """

    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        dense_t16_pair_decode_shape_error,
    )

    # The shapes this probe actually reports.
    assert dense_t16_pair_decode_shape_error(rows=1, in_features=5120, out_features=17408) is None
    assert dense_t16_pair_decode_shape_error(rows=1, in_features=5120, out_features=8704) is None
    # The rejecting boundary, with the reason each condition produces.
    assert "rows == 1" in dense_t16_pair_decode_shape_error(rows=2, in_features=5120, out_features=8704)
    assert "in_features" in dense_t16_pair_decode_shape_error(rows=1, in_features=5118, out_features=8704)
    assert "in_features" in dense_t16_pair_decode_shape_error(rows=1, in_features=0, out_features=8704)
    assert "out_features" in dense_t16_pair_decode_shape_error(rows=1, in_features=5120, out_features=8700)
    assert "out_features" in dense_t16_pair_decode_shape_error(rows=1, in_features=5120, out_features=0)


def test_the_launcher_refuses_exactly_what_the_contract_refuses(monkeypatch) -> None:
    """The launcher must consult the shared contract, not its own copy.

    Invalid shapes are safe to pass to a launcher with null pointers because the
    launcher raises before any HIP contact; the loaders are patched to raise
    anyway, so a regression that launched before validating would fail this test
    instead of enqueueing an invalid device launch.
    """

    import inspect

    from hipengine.kernels.hip_gfx1100.quant import gguf_t16_selected_gemv as gemv

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("HIP contact during a shape-contract test")

    monkeypatch.setattr(gemv, "get_hip_runtime", _forbidden)
    monkeypatch.setattr(gemv, "_t16_selected_gemv_library", _forbidden)

    # One shared contract, consulted by both pair decode launchers.
    for launcher in (
        gemv.gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
        gemv.gguf_q5_k_t16_dense_dual_silu_gemv_bf16_bf16_out,
    ):
        assert "dense_t16_pair_decode_shape_error" in inspect.getsource(launcher)
        for rows, in_features, out_features in ((2, 5120, 8704), (1, 5118, 8704), (1, 5120, 8700)):
            reason = gemv.dense_t16_pair_decode_shape_error(
                rows=rows, in_features=in_features, out_features=out_features
            )
            with pytest.raises(ValueError, match=reason.split("requires ")[1]):
                launcher(0, 0, 0, 0, rows, in_features, out_features)


def test_the_probe_answers_the_admission_question_without_hip(monkeypatch) -> None:
    """fused_path_admission must stay pure: no runtime, no library, no launch.

    This is the structural guard for the review finding: an earlier revision
    called the launcher with null pointers, which enqueues an invalid device
    launch on any host where the library is loaded. With the loaders patched to
    raise, the probe can only pass by never touching them.
    """

    from hipengine.kernels.hip_gfx1100.quant import gguf_t16_selected_gemv as gemv

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fused_path_admission touched HIP or the kernel library")

    monkeypatch.setattr(gemv, "get_hip_runtime", _forbidden)
    monkeypatch.setattr(gemv, "_t16_selected_gemv_library", _forbidden)

    probe = sys.modules.get("tp2_mlp_shard_plan_probe")
    if probe is None:
        spec = importlib.util.spec_from_file_location("tp2_mlp_shard_plan_probe", SCRIPT)
        assert spec is not None and spec.loader is not None
        probe = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = probe
        spec.loader.exec_module(probe)
    admission = probe.fused_path_admission(hidden_size=5120, intermediate=17408)
    assert admission["shard_admitted"] is False
    assert admission["shard_shape_error"] is None


def test_an_odd_intermediate_reports_a_non_integral_shard(mod) -> None:
    """The probe must not silently floor a shape that does not split evenly."""

    admission = mod.fused_path_admission(hidden_size=5120, intermediate=17409)
    assert admission["shard_key"] == [1, 5120, 8704]
    assert admission["shard_admitted"] is False
    assert admission["gap"] is not None
