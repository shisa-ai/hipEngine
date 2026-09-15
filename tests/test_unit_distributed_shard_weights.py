"""Tests for the runtime MLP shard-payload materialization.

The pure-helper tests run anywhere. The materialization tests are gated on the
target GGUF and compare against the probe script's independently validated
path: the runtime payloads must be byte-identical to
``scripts/tp2_mlp_slice_e2e.rank_shard_payload`` (TP2-A's oracle-validated
composition), because a runtime that materializes different bytes than the
validated slice would silently change the arithmetic the shard plan was
qualified with.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from hipengine.distributed.shard_weights import (
    MLP_ROLES,
    MLPShardError,
    MlpShardLayer,
    _t16_repack_for_layout,
    materialize_mlp_shards,
    resolve_mlp_shard_context,
    shard_bytes,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GGUF_PATH = pathlib.Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
PROBE = REPO_ROOT / "scripts" / "tp2_mlp_shard_plan_probe.py"
SLICE = REPO_ROOT / "scripts" / "tp2_mlp_slice_e2e.py"

_model_available = GGUF_PATH.exists()
requires_model = pytest.mark.skipif(
    not _model_available, reason=f"GGUF model file not available: {GGUF_PATH}"
)


def _load_script(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_the_mlp_roles_are_the_three_projections() -> None:
    assert MLP_ROLES == ("ffn_gate", "ffn_up", "ffn_down")


def test_a_raw_layout_has_no_repack_and_an_unknown_layout_is_refused() -> None:
    assert _t16_repack_for_layout("raw_gguf_q4_k", "gguf_q4_k") is None
    # A recorded layout resolves to its repack callable.
    assert callable(_t16_repack_for_layout("gguf_q4_k_t16_v1", "gguf_q4_k"))
    with pytest.raises(MLPShardError, match="no t16 repack"):
        _t16_repack_for_layout("not_a_layout", "gguf_q4_k")


def test_a_nonpositive_world_size_is_rejected() -> None:
    with pytest.raises(MLPShardError, match="world_size"):
        resolve_mlp_shard_context(str(GGUF_PATH) if _model_available else "x", world_size=0)


# ---------------------------------------------------------------------------
# Model-gated materialization
# ---------------------------------------------------------------------------


@requires_model
def test_the_context_resolves_from_the_engine_planner_chain() -> None:
    materialization, context, plans, config = resolve_mlp_shard_context(
        str(GGUF_PATH), world_size=2
    )
    assert int(config.block_count) == len(plans)
    assert context["backend"] == "hip_gfx1100"
    assert set(plans[0].keys()) == set(MLP_ROLES)
    # The layouts come from the planner, never from a table here.
    for role, plan in plans[0].items():
        spec_layer = materialization.layer_specs[0]
        spec = spec_layer[role]
        assert plan.kind in {"column", "row"}


@requires_model
def test_one_layer_shards_split_each_axis_in_half() -> None:
    shards = materialize_mlp_shards(str(GGUF_PATH), world_size=2, layer_ids=(0,))
    layer = shards[0]
    assert isinstance(layer, MlpShardLayer)
    for rank in (0, 1):
        payloads = layer.rank_payloads(rank)
        assert set(payloads.keys()) == set(MLP_ROLES)
        gate = payloads["ffn_gate"]
        up = payloads["ffn_up"]
        down = payloads["ffn_down"]
        # Column-parallel gate/up: each rank owns half the output rows.
        assert gate.local_shape[0] * 2 == 17408
        assert up.local_shape[0] * 2 == 17408
        # Row-parallel down: each rank owns half the input columns.
        assert down.local_shape[1] * 2 == 17408
        # The payloads are flat byte arrays sized by the planner.
        for payload in payloads.values():
            assert payload.payload.dtype == np.uint8
            assert payload.payload.ndim == 1
            assert payload.nbytes == payload.payload.size


@requires_model
def test_runtime_payloads_are_byte_identical_to_the_validated_probe_path() -> None:
    probe = _load_script(PROBE, "tp2_mlp_shard_plan_probe")
    slice_e2e = _load_script(SLICE, "tp2_mlp_slice_e2e")
    from hipengine.loading.gguf import GGUFReader, scan_gguf
    from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
    from hipengine.loading.qwen35_gguf_admission import build_qwen35_gguf_role_manifest
    from hipengine.loading.qwen35_gguf_shards import build_shard_manifest

    reader = GGUFReader(str(GGUF_PATH))
    info = scan_gguf(str(GGUF_PATH))
    probe_materialization, _ = probe.resolve_incumbent_plan(info)
    model_map = build_qwen35_gguf_tensor_map(info)
    manifest = build_shard_manifest(
        info,
        world_size=2,
        model_hash=build_qwen35_gguf_role_manifest(model_map).fingerprint,
    )
    by_name = {plan.name: plan for plan in manifest.tensors}
    runtime_shards = materialize_mlp_shards(str(GGUF_PATH), world_size=2, layer_ids=(0,))
    for rank in (0, 1):
        for name in slice_e2e.mlp_tensor_names(0):
            plan = by_name[name]
            shard = plan.slice_for(rank)
            ref, ref_layout, ref_quant = slice_e2e.rank_shard_payload(
                reader, probe_materialization, plan, shard, layer=0
            )
            role = name.split(".")[2]
            mine = runtime_shards[0].rank_payloads(rank)[role]
            assert np.array_equal(np.asarray(mine.payload), np.asarray(ref))
            assert mine.layout == ref_layout
            assert mine.quant_key == ref_quant


@requires_model
def test_shard_bytes_accounts_for_every_layer_and_rank() -> None:
    shards = materialize_mlp_shards(str(GGUF_PATH), world_size=2, layer_ids=(0, 1))
    per_layer = shard_bytes({0: shards[0]}, rank=0)
    assert shard_bytes(shards, rank=0) == 2 * per_layer
    assert shard_bytes(shards, rank=1) == shard_bytes(shards, rank=0)


@requires_model
def test_an_out_of_range_layer_is_refused() -> None:
    with pytest.raises(MLPShardError, match="outside"):
        materialize_mlp_shards(str(GGUF_PATH), world_size=2, layer_ids=(10_000,))


@requires_model
def test_a_degree_that_does_not_split_the_mlp_is_refused_before_allocation() -> None:
    # 17408 splits across 2 and 4; a third rank is refused before any byte is
    # read, mirroring the shard manifest's own admission rule.
    with pytest.raises(MLPShardError, match="does not split"):
        resolve_mlp_shard_context(str(GGUF_PATH), world_size=3)
