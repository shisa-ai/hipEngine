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
    attention_sharded_config,
    family_slot_names,
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


# ---------------------------------------------------------------------------
# The attention (head-sharding) family
# ---------------------------------------------------------------------------


def _source_row_runs(plan, rank: int) -> list[tuple[int, int]]:
    """Contiguous ``(start_row, row_count)`` runs of source rows one rank owns."""

    row_bytes = int(plan.source_row_bytes)
    runs: list[tuple[int, int]] = []
    for segment in plan.slice_for(int(rank)).iter_segments():
        start = int(segment.source_offset) // row_bytes
        rows = int(segment.nbytes) // row_bytes
        if runs and runs[-1][0] + runs[-1][1] == start:
            runs[-1] = (runs[-1][0], runs[-1][1] + rows)
        else:
            runs.append((start, rows))
    return runs


def test_the_attention_family_slots_cover_both_layer_types() -> None:
    from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION
    from hipengine.distributed.shard_weights import (
        ATTENTION_FAMILY,
        FULL_ATTENTION_SLOTS,
        LINEAR_ATTENTION_SLOTS,
        MLP_FAMILY,
    )

    assert ATTENTION_FAMILY.slots_for(FULL_ATTENTION) == FULL_ATTENTION_SLOTS
    assert ATTENTION_FAMILY.slots_for(LINEAR_ATTENTION) == LINEAR_ATTENTION_SLOTS
    # The GDN set names materialization slots, not GGUF tensor names: the source
    # of ``ssm_dt_bias`` is ``blk.N.ssm_dt.bias``, which has no ``.weight``.
    assert "ssm_dt_bias" in LINEAR_ATTENTION_SLOTS
    assert "ssm_a" in LINEAR_ATTENTION_SLOTS
    # A layer type the family does not describe is a refusal, not a silent
    # fallback to the other set.
    with pytest.raises(MLPShardError, match="no tensor set"):
        ATTENTION_FAMILY.slots_for("mtp_block")
    # The dense-MLP family keeps its own contract, including refusing raw
    # layouts; attention needs them because its GDN parameters are dense_f32.
    assert MLP_FAMILY.allow_raw is False
    assert ATTENTION_FAMILY.allow_raw is True
    assert MLP_FAMILY.allowed_kinds == frozenset({"column", "row"})
    assert ATTENTION_FAMILY.allowed_kinds == frozenset({"group", "row"})


def test_the_q5_k_t16_repack_and_the_dense_layouts_resolve() -> None:
    from hipengine.distributed.shard_weights import _t16_repack_for_layout

    # ssm_out resolves to gguf_q5_k_t16_v1 on the target artifact; before this
    # unit that layout had no entry and the payload builder refused it.
    assert callable(_t16_repack_for_layout("gguf_q5_k_t16_v1", "gguf_q5_k"))
    # Dense/raw layouts have no repack: the rank payload is its slice's bytes.
    for layout in ("dense_f32", "dense_bf16", "raw_gguf"):
        assert _t16_repack_for_layout(layout, "f32") is None
    # A t16-family layout with no recorded repack is still a named refusal
    # rather than a silently wrong payload.
    with pytest.raises(MLPShardError, match="no t16 repack"):
        _t16_repack_for_layout("gguf_q8_0_t16_v1", "gguf_q8_0")


@requires_model
def test_the_attention_family_refuses_a_degree_that_does_not_split_the_heads() -> None:
    from hipengine.distributed.shard_weights import resolve_attention_shard_context

    # head_count 24 does not split across 5, and neither does the GDN group
    # count or time-step rank; the refusal must name the axis.
    with pytest.raises(MLPShardError, match="does not split across 5 ranks"):
        resolve_attention_shard_context(str(GGUF_PATH), world_size=5)


@requires_model
def test_attention_shards_materialize_each_layer_types_own_slots() -> None:
    from hipengine.distributed.shard_weights import (
        FULL_ATTENTION_SLOTS,
        LINEAR_ATTENTION_SLOTS,
        materialize_attention_shards,
    )

    shards = materialize_attention_shards(str(GGUF_PATH), world_size=2, layer_ids=(0, 3))
    for rank in (0, 1):
        gdn = shards[0].rank_payloads(rank)
        full = shards[3].rank_payloads(rank)
        assert set(gdn) == set(LINEAR_ATTENTION_SLOTS)
        assert set(full) == set(FULL_ATTENTION_SLOTS)
        # Local shapes are the rank's own slices, not the full tensors.
        assert gdn["attn_qkv"].local_shape == (5120, 5120)
        assert gdn["attn_gate"].local_shape == (3072, 5120)
        assert gdn["ssm_alpha"].local_shape == (24, 5120)
        assert gdn["ssm_a"].local_shape == (24,)
        assert gdn["ssm_conv1d"].local_shape == (5120, 4)
        assert gdn["ssm_out"].local_shape == (5120, 3072)
        assert full["attn_q"].local_shape == (6144, 5120)
        assert full["attn_k"].local_shape == (512, 5120)
        assert full["attn_v"].local_shape == (512, 5120)
        assert full["attn_output"].local_shape == (5120, 3072)
        # Raw slots carry their slice bytes verbatim; t16 slots are tiled.
        assert gdn["ssm_alpha"].layout == "dense_f32"
        assert gdn["ssm_alpha"].payload.nbytes == 24 * 5120 * 4
        assert gdn["ssm_a"].payload.nbytes == 24 * 4
        assert "t16" in full["attn_q"].layout
        assert "t16" in gdn["ssm_out"].layout
    # A halved config's derived widths must equal those slices: this is what
    # lets the existing attention helpers run sharded attention unchanged.
    _, _, _, config = resolve_mlp_shard_context(str(GGUF_PATH), world_size=2)
    assert config.head_count // 2 * config.key_length == 3072          # q_width
    assert config.head_count // 2 * 2 * config.key_length == 6144      # 2*q_width
    assert config.head_count_kv // 2 * config.key_length == 512        # kv_width
    assert config.ssm_inner_size // 2 == 3072                          # attn_gate rows
    assert config.ssm_time_step_rank // 2 == 24                        # alpha rows


@requires_model
def test_attention_local_head_indices_are_identity_under_a_halved_config() -> None:
    """The materialized rows must match a plain halved config's local indexing.

    If they did not, a rank would read another rank's heads and the sharded
    route would be silently wrong. Full attention is a contiguous prefix, so
    local head j is global head ``rank * local_heads + j``. GDN value heads are
    tile-major (llama.cpp order, ``k = v % k_heads``), which is exactly what
    makes ``local_k = local_v % local_k_heads`` hold - the identity mapping the
    runner's own per-head loops use.
    """

    from hipengine.loading.gguf import scan_gguf
    from hipengine.loading.qwen35_gguf import qwen35_gguf_config_from_metadata
    from hipengine.loading.qwen35_gguf_shards import (
        build_shard_manifest,
        gdn_head_map,
    )

    info = scan_gguf(str(GGUF_PATH))
    config = qwen35_gguf_config_from_metadata(info)
    manifest = build_shard_manifest(info, world_size=2, model_hash="head-identity")
    by_name = {plan.name: plan for plan in manifest.tensors}
    local_q_heads = config.head_count // 2
    local_kv_heads = config.head_count_kv // 2

    for rank in (0, 1):
        # Full attention: contiguous prefix on every axis.
        for role, rows_per_head, local_heads in (
            ("attn_q", 2 * config.key_length, local_q_heads),
            ("attn_k", config.key_length, local_kv_heads),
            ("attn_v", config.value_length, local_kv_heads),
        ):
            runs = _source_row_runs(by_name[f"blk.3.{role}.weight"], rank)
            assert runs == [(rank * local_heads * rows_per_head, local_heads * rows_per_head)]
        # Grouped-query ratio is preserved, so local q head j pairs with local
        # kv head j // (local_q_heads // local_kv_heads), same as globally.
        assert config.head_count // config.head_count_kv == local_q_heads // local_kv_heads

        # GDN: the value-head runs are the head map's own local order, which is
        # tile-major, so a rank's rows come in one run per value tile.
        head_map = gdn_head_map(config, rank, 2)
        owned_v = head_map.v_heads_for()
        expected_runs: list[tuple[int, int]] = []
        for global_v in owned_v:
            if expected_runs and expected_runs[-1][0] + expected_runs[-1][1] == global_v:
                expected_runs[-1] = (expected_runs[-1][0], expected_runs[-1][1] + 1)
            else:
                expected_runs.append((global_v, 1))
        assert len(expected_runs) == head_map.tiles
        assert sum(rows for _, rows in expected_runs) == head_map.local_v_heads
        alpha_runs = _source_row_runs(by_name["blk.0.ssm_alpha.weight"], rank)
        assert alpha_runs == expected_runs
        for local_v in range(head_map.local_v_heads):
            global_v = owned_v[local_v]
            # Global pairing is k = v % k_heads; local pairing must agree.
            assert global_v % head_map.k_heads == head_map.k_heads_for()[
                local_v % head_map.local_k_heads
            ]
            assert head_map.local_k_head(local_v) == global_v % head_map.k_heads
        # The gate projection has head_v_dim rows per value head, same order.
        gate_runs = _source_row_runs(by_name["blk.0.attn_gate.weight"], rank)
        assert gate_runs == [
            (start * head_map.head_v_dim, rows * head_map.head_v_dim)
            for start, rows in expected_runs
        ]


def _synthetic_attention_config(**overrides):
    """A frozen config with only the axes head sharding touches, for pure tests."""

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Config:
        head_count: int = 48
        head_count_kv: int = 8
        ssm_group_count: int = 16
        ssm_inner_size: int = 6144
        ssm_time_step_rank: int = 48
        hidden_size: int = 5120
        key_length: int = 128
        value_length: int = 128
        ssm_state_size: int = 128
        feed_forward_length: int = 17408

    return _Config(**overrides)


def test_attention_sharded_config_halves_only_the_head_axes() -> None:
    """Head sharding moves the head axes and leaves the replicated ones alone.

    The residual stream, the per-head key/value widths, the SSM state width and
    the FFN width are not head counts, so halving any of them would misdescribe
    the rank's payloads.
    """

    config = _synthetic_attention_config()
    sharded = attention_sharded_config(config, world_size=2)

    assert sharded.head_count == config.head_count // 2
    assert sharded.head_count_kv == config.head_count_kv // 2
    assert sharded.ssm_group_count == config.ssm_group_count // 2
    assert sharded.ssm_inner_size == config.ssm_inner_size // 2
    assert sharded.ssm_time_step_rank == config.ssm_time_step_rank // 2
    # The derived widths the runner reads follow from those axes.
    assert sharded.head_count * sharded.key_length == 3072
    assert sharded.head_count_kv * sharded.value_length == 512
    assert 2 * sharded.ssm_group_count * sharded.ssm_state_size + sharded.ssm_inner_size == 5120
    # Grouped-query pairing is preserved, so local q head j still pairs with the
    # same kv head it did globally.
    assert sharded.head_count // sharded.head_count_kv == config.head_count // config.head_count_kv

    for replicated in (
        "hidden_size",
        "key_length",
        "value_length",
        "ssm_state_size",
        "feed_forward_length",
    ):
        assert getattr(sharded, replicated) == getattr(config, replicated), replicated
    assert sharded != config, "a halved config must not be the unsharded one"

    assert attention_sharded_config(config, world_size=1) == config


def test_attention_sharded_config_refuses_an_axis_that_does_not_split() -> None:
    """A degree that cannot divide a head axis must refuse, not truncate."""

    config = _synthetic_attention_config(ssm_time_step_rank=45)
    with pytest.raises(MLPShardError, match="ssm.time_step_rank 45 does not split"):
        attention_sharded_config(config, world_size=2)
    with pytest.raises(MLPShardError, match="world_size must be positive"):
        attention_sharded_config(_synthetic_attention_config(), world_size=0)


def test_family_slot_names_are_the_tables_own_leaves() -> None:
    """The allowlist's leaf names come from the family tables, not a copy."""

    from hipengine.distributed.shard_weights import MLP_FAMILY, SHARD_FAMILIES

    assert family_slot_names(MLP_FAMILY) == frozenset(MLP_ROLES)
    assert sorted(SHARD_FAMILIES) == ["attention", "mlp"]
    assert family_slot_names(SHARD_FAMILIES["attention"]) >= {
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "attn_qkv",
        "ssm_out",
    }


@requires_model
def test_attention_sharded_config_widths_reproduce_every_payload_shape() -> None:
    """The halved config's derived widths must equal the payloads' own shapes.

    This is the property that makes the substitution safe: the runner derives
    every attention width from its config, so if the halved config described a
    different geometry than the shards hold, the kernels would read the wrong
    rows - silently, since both would be internally consistent.
    """

    from hipengine.distributed.shard_weights import materialize_attention_shards
    from hipengine.loading.gguf import scan_gguf
    from hipengine.loading.qwen35_gguf import (
        FULL_ATTENTION,
        LINEAR_ATTENTION,
        qwen35_gguf_config_from_metadata,
    )

    config = qwen35_gguf_config_from_metadata(scan_gguf(str(GGUF_PATH)))
    sharded = attention_sharded_config(config, world_size=2)
    q_width = sharded.head_count * sharded.key_length
    kv_width = sharded.head_count_kv * sharded.value_length
    qkv_width = (
        2 * sharded.ssm_group_count * sharded.ssm_state_size + sharded.ssm_inner_size
    )
    hidden = sharded.hidden_size
    full_layer = config.layer_types.index(FULL_ATTENTION)
    linear_layer = config.layer_types.index(LINEAR_ATTENTION)
    shards = materialize_attention_shards(
        str(GGUF_PATH), world_size=2, layer_ids=(full_layer, linear_layer)
    )

    # attn_q carries the query and its output gate, so it is 2 * q_width rows.
    expected = {
        full_layer: {
            "attn_q": (2 * q_width, hidden),
            "attn_k": (kv_width, hidden),
            "attn_v": (kv_width, hidden),
            "attn_output": (hidden, q_width),
        },
        linear_layer: {
            "attn_qkv": (qkv_width, hidden),
            "attn_gate": (sharded.ssm_inner_size, hidden),
            "ssm_alpha": (sharded.ssm_time_step_rank, hidden),
            "ssm_beta": (sharded.ssm_time_step_rank, hidden),
            "ssm_a": (sharded.ssm_time_step_rank,),
            "ssm_dt_bias": (sharded.ssm_time_step_rank,),
            "ssm_conv1d": (qkv_width, config.ssm_conv_kernel),
            "ssm_out": (hidden, sharded.ssm_inner_size),
        },
    }
    for layer_id, roles in expected.items():
        payloads = shards[layer_id].rank_payloads(0)
        assert set(payloads) == set(roles), f"layer {layer_id} slot set"
        for role, shape in roles.items():
            assert payloads[role].local_shape == shape, (
                f"layer {layer_id} {role}: payload is {payloads[role].local_shape}, "
                f"the halved config says {shape}"
            )
    # The unsharded widths are twice the payloads', so the assertion above is
    # not vacuously true of any config.
    assert config.head_count * config.key_length == 2 * q_width


@requires_model
def test_head_sharding_halves_the_resident_scratch_plan() -> None:
    """The scratch plan follows the halved widths, so residency becomes per-rank.

    ``_FullStackScratch.allocate`` sizes the KV cache, conv state, recurrent
    state and rope tables from this plan, and the plan is a pure function of the
    config plus the widths. Substituting the rank's config before allocation is
    therefore what makes the whole scratch stack per-rank instead of replicated,
    which is why the substitution has to happen before ``allocate``.
    """

    from hipengine.loading.gguf import scan_gguf
    from hipengine.loading.qwen35_gguf import qwen35_gguf_config_from_metadata
    from hipengine.runtime.qwen35_gguf_runner import _full_stack_scratch_plan

    config = qwen35_gguf_config_from_metadata(scan_gguf(str(GGUF_PATH)))
    sharded = attention_sharded_config(config, world_size=2)

    def plan(cfg):
        return _full_stack_scratch_plan(
            cfg,
            hidden_size=cfg.hidden_size,
            ffn_size=cfg.feed_forward_length,
            q_width=cfg.head_count * cfg.key_length,
            kv_width=cfg.head_count_kv * cfg.value_length,
            linear_qkv_width=(
                2 * cfg.ssm_group_count * cfg.ssm_state_size + cfg.ssm_inner_size
            ),
            max_sequence_length=4096,
            materialize=False,
        )

    full_plan = plan(config)
    sharded_plan = plan(sharded)
    full_total = sum(int(size) for size in full_plan.owner_sizes)
    sharded_total = sum(int(size) for size in sharded_plan.owner_sizes)

    # The KV payload is the head-proportional term, so it must halve exactly.
    assert sharded_plan.kv_payload_bytes * 2 == full_plan.kv_payload_bytes
    # The whole stack shrinks but does not halve: the fixed-width buffers and the
    # block/position counts do not follow the head axes.
    assert sharded_plan.owner_sizes != full_plan.owner_sizes
    assert 0.5 < sharded_total / full_total < 1.0, (
        f"scratch plan went {full_total} -> {sharded_total}, which is not a partial reduction"
    )


@requires_model
def test_raw_attention_payloads_are_the_owned_heads_rows() -> None:
    """A dense slot's rank payload must decode to its own heads' values.

    Checked against the head map rather than against the slice machinery, so a
    mis-strided or repacked dense payload cannot pass by agreeing with itself.
    """

    from hipengine.distributed.shard_weights import materialize_attention_shards
    from hipengine.loading.gguf import GGUFReader, scan_gguf
    from hipengine.loading.qwen35_gguf import qwen35_gguf_config_from_metadata
    from hipengine.loading.qwen35_gguf_shards import gdn_head_map

    reader = GGUFReader(str(GGUF_PATH))
    info = scan_gguf(str(GGUF_PATH))
    config = qwen35_gguf_config_from_metadata(info)
    shards = materialize_attention_shards(str(GGUF_PATH), world_size=2, layer_ids=(0,))

    # ssm_alpha: one f32 row per value head, so payload row j must be the
    # source row of the j-th global value head this rank owns.
    alpha_info = reader.tensor_info("blk.0.ssm_alpha.weight")
    alpha_source = np.memmap(
        reader.path,
        dtype=np.float32,
        mode="r",
        offset=int(alpha_info.data_offset),
        shape=(config.ssm_time_step_rank, config.hidden_size),
    )
    # ssm_a: one f32 scalar per value head.
    a_info = reader.tensor_info("blk.0.ssm_a")
    a_source = np.memmap(
        reader.path,
        dtype=np.float32,
        mode="r",
        offset=int(a_info.data_offset),
        shape=(config.ssm_time_step_rank,),
    )
    for rank in (0, 1):
        owned = list(gdn_head_map(config, rank, 2).v_heads_for())
        payloads = shards[0].rank_payloads(rank)
        alpha = np.frombuffer(
            np.asarray(payloads["ssm_alpha"].payload).tobytes(), dtype=np.float32
        ).reshape(len(owned), config.hidden_size)
        assert np.array_equal(alpha, np.asarray(alpha_source[owned]))
        decay = np.frombuffer(
            np.asarray(payloads["ssm_a"].payload).tobytes(), dtype=np.float32
        )
        assert np.array_equal(decay, np.asarray(a_source[owned]))
        # The two ranks partition the source rather than overlapping it.
        assert payloads["ssm_alpha"].nbytes == int(alpha_info.nbytes) // 2
        assert payloads["ssm_a"].nbytes == int(a_info.nbytes) // 2
