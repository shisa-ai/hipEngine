"""Tests for the TP2 MLP shard-plan probe.

The probe exists to answer one question that decides whether a shard segment
needs a new kernel: does the engine's linear dispatch key a weight's own row
count? It does not, so a shard-shaped MLP weight resolves to the same registered
kernel as the TP1 shape. These tests pin that property, the layout admissibility
of a half-intermediate row block, and the artifact contract.
"""

from __future__ import annotations

import importlib.util
import json
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


def test_half_intermediate_row_block_is_t16_admissible(mod) -> None:
    """8704 rows is half of 17408 and still a whole number of t16 tiles."""

    admissible = mod._t16_admissible(out_features=8704, bytes_per_row=5120 * 144 // 256, block_bytes=144)
    assert admissible["admissible"] is True
    assert admissible["out_features_divisible"] is True
    assert admissible["bytes_per_row_divisible"] is True


def test_an_unaligned_row_block_is_rejected(mod) -> None:
    """The probe must not report an inadmissible local shape as usable."""

    admissible = mod._t16_admissible(out_features=8705, bytes_per_row=2880, block_bytes=144)
    assert admissible["admissible"] is False
    assert admissible["out_features_divisible"] is False


def test_q4_k_resolves_to_t16_and_q6_k_to_raw(mod) -> None:
    """The resident layout per source type comes from the engine's own tables."""

    q4_layout, q4_quant = mod._resident_layout("Q4_K")
    q6_layout, q6_quant = mod._resident_layout("Q6_K")
    assert q4_layout == "gguf_q4_k_t16_v1"
    assert q4_quant == "gguf_q4_k_t16_v1"
    assert q6_layout == "raw_gguf"
    assert q6_quant == "gguf_q6_k"


def test_an_unknown_source_quant_is_refused_rather_than_guessed(mod) -> None:
    with pytest.raises(ValueError, match="no resident layout"):
        mod._resident_layout("Q2_K_UNKNOWN")


def test_the_mlp_tensor_names_are_the_three_projections(mod) -> None:
    assert mod._mlp_tensor_names(7) == (
        "blk.7.ffn_gate.weight",
        "blk.7.ffn_up.weight",
        "blk.7.ffn_down.weight",
    )


@requires_model
def test_probe_answers_every_question_with_evidence(mod) -> None:
    """The four gate questions must all be answered yes on the real model."""

    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2)
    assert set(report["questions"]) == {
        "rank_local_materialization_exists",
        "rank_local_bytes_round_trip",
        "local_shapes_are_layout_admissible",
        "shard_dispatch_needs_no_new_kernel",
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
    # reduction payload is hidden_size * 4 and matches the transport screen.
    assert report["reduction_point"]["elements"] == report["hidden_size"]
    assert report["reduction_point"]["payload_bytes"] == 20480


@requires_model
def test_the_shard_resolves_to_the_same_kernel_as_tp1(mod) -> None:
    """A shape-keyed dispatch would break the shard path; catch it here."""

    report = mod.probe(model=GGUF_PATH, layer=0, world_size=2)
    for name, entry in report["tensors"].items():
        assert entry["round_trip_bit_exact"] is True, name
        for rank, shard in entry["ranks"].items():
            assert shard["dispatch_matches_tp1"] is True, f"{name} rank {rank}"
            assert shard["dispatch_key"] == entry["tp1_dispatch"]["key"]


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
