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


# ---------------------------------------------------------------------------
# The runtime payloads, checked against the file's own value-head order
# ---------------------------------------------------------------------------
#
# The GGUF value-head order is a property of the file, so the expectation below
# comes from the two references that define it, not from this repository's shard
# rule or ``gdn_head_map``:
#
# * the converter that wrote the file (llama.cpp ``convert_hf_to_gguf.py``,
#   ``_LinearAttentionVReorderBase``: HF stores V heads grouped by K head as
#   ``[G0_v0..v{r-1}, G1_v0..v{r-1}, ...]`` and GGUF stores them tiled as
#   ``[K0, K1, ..., K0, K1, ...]``), and
# * the runtime that consumes it (llama.cpp ``src/models/qwen35.cpp``: the fused
#   qkv rows are ``key_dim`` q rows, then ``key_dim`` k rows, then the value
#   rows, and ``ggml_repeat_4d(q_conv, ..., num_v_heads, ...)`` tiles the K heads
#   across the V axis, so V head ``v`` pairs with K head ``v % n_k_heads``).
#
# The check is on bytes: the expected local slice is built from the source rows
# (or columns) the rule assigns, put through the same repack the runtime uses,
# and compared with the payload the runtime actually hands the kernels.


def _tiled_local_heads(config, rank: int, world_size: int) -> list[int]:
    """Global value heads a rank owns, in the local slot order the kernels use."""

    k_heads = int(config.ssm_group_count)
    local_k = k_heads // int(world_size)
    return [
        tile * k_heads + rank * local_k + index
        for tile in range(int(config.ssm_time_step_rank) // k_heads)
        for index in range(local_k)
    ]


def _repacked(local_bytes, rows: int, row_bytes: int, layout: str, source_type: str) -> np.ndarray:
    """The runtime's own repack of a local slice, or the bytes for raw layouts."""

    repack = _t16_repack_for_layout(str(layout), source_type)
    flat = np.ascontiguousarray(local_bytes, dtype=np.uint8).reshape(-1)
    if repack is None:
        return flat
    expert = flat.reshape(1, int(rows), int(row_bytes))
    return np.ascontiguousarray(np.asarray(repack(expert).tiles)).reshape(-1)


def _source_rows(reader, name: str, row_indexes, *, row_bytes: int, rows: int) -> np.ndarray:
    """Concatenate whole source rows, so the caller's row choice is the only input."""

    from hipengine.loading.qwen35_gguf_shards import source_payload

    info = reader.tensor_info(name)
    view = np.asarray(
        source_payload(reader.path, data_offset=info.data_offset, nbytes=info.nbytes)
    ).reshape(int(rows), int(row_bytes))
    return np.ascontiguousarray(np.concatenate([view[int(r)] for r in row_indexes])).reshape(-1)


@requires_model
def test_attention_ssm_a_payload_is_the_kernel_a_log_abi_not_the_source() -> None:
    """A rank's ``ssm_a`` payload must be the converted A_log, not the source.

    GGUF stores the negative decay coefficient and the GDN kernels take
    ``A_log``, computing ``exp(-exp(A_log) * softplus(alpha + dt))``. The
    replicated materializer converts the tensor before it uploads, so a shard
    payload that kept the source coefficient would have the kernel read it as an
    ``A_log`` instead: every head's decay becomes ``exp(-exp(A) * ...)`` rather
    than ``exp(-A * ...)``. Token 0 is unaffected because the state is zero
    there, which is what makes this a silent failure rather than an obvious one.
    """

    from hipengine.distributed.shard_weights import materialize_attention_shards
    from hipengine.loading.gguf import GGUFReader, scan_gguf
    from hipengine.loading.qwen35_gguf import qwen35_gguf_config_from_metadata
    from hipengine.loading.qwen35_gguf_materialize import _gguf_ssm_a_to_kernel_a_log
    from hipengine.loading.qwen35_gguf_shards import (
        build_shard_manifest,
        iter_rank_payloads,
        source_payload,
    )

    config = qwen35_gguf_config_from_metadata(scan_gguf(str(GGUF_PATH)))
    reader = GGUFReader(str(GGUF_PATH))
    info = reader.tensor_info("blk.0.ssm_a")
    source = np.asarray(
        source_payload(reader.path, data_offset=info.data_offset, nbytes=info.nbytes)
    ).reshape(-1)
    raw = source.view("<f4")
    a_log = _gguf_ssm_a_to_kernel_a_log(raw)
    assert np.all(raw < 0.0) and np.all(a_log < 0.0)
    # The conversion is a different magnitude per head, so the two cannot be
    # confused by a tolerance: exp() of them differs by orders of magnitude.
    assert not np.allclose(np.exp(a_log), np.exp(raw), rtol=1e-3)

    layer = materialize_attention_shards(str(GGUF_PATH), world_size=2, layer_ids=(0,))[0]
    manifest = build_shard_manifest(scan_gguf(str(GGUF_PATH)), world_size=2)
    streamed: dict[int, np.ndarray] = {}
    for plan, payload in iter_rank_payloads(reader, manifest, rank=0):
        if plan.name == "blk.0.ssm_a":
            streamed[0] = np.asarray(payload).reshape(-1)
    for rank in range(2):
        owned = _tiled_local_heads(config, rank, 2)
        want = np.ascontiguousarray(a_log[owned], dtype="<f4")
        got = np.asarray(layer.rank_payloads(rank)["ssm_a"].payload).reshape(-1)
        assert np.array_equal(got.view("<f4"), want), (
            f"rank {rank} ssm_a payload is not the kernel A_log ABI"
        )
        assert not np.array_equal(got.view("<f4"), raw[owned])
    assert streamed, "the loader path did not yield blk.0.ssm_a"
    assert np.array_equal(
        streamed[0].view("<f4"),
        np.ascontiguousarray(a_log[_tiled_local_heads(config, 0, 2)], dtype="<f4"),
    ), "the streaming loader path must apply the same ABI"


@requires_model
def test_attention_runtime_payloads_carry_the_tiled_local_slices() -> None:
    """Every slot the sharded route uploads must hold the heads it claims.

    The manifest path and this runtime path are separate code paths, and this is
    the one the head-sharded route uploads from. A rank that held another rank's
    heads, or held its own heads in a different order than ``local_k = local_v %
    local_k_heads`` assumes, would compute a plausible-looking wrong answer.
    """

    from hipengine.distributed.shard_weights import materialize_attention_shards
    from hipengine.loading.gguf import GGUFReader, scan_gguf
    from hipengine.loading.qwen35_gguf import qwen35_gguf_config_from_metadata
    from hipengine.loading.qwen35_gguf_shards import source_payload

    config = qwen35_gguf_config_from_metadata(scan_gguf(str(GGUF_PATH)))
    k_heads = int(config.ssm_group_count)
    head_k = int(config.ssm_state_size)
    v_heads = int(config.ssm_time_step_rank)
    head_v = int(config.ssm_inner_size) // v_heads
    tiles = v_heads // k_heads
    world_size = 2
    local_k = k_heads // world_size
    key_width = k_heads * head_k
    reader = GGUFReader(str(GGUF_PATH))
    layer = materialize_attention_shards(str(GGUF_PATH), world_size=world_size, layer_ids=(0,))[0]

    # (slot, source tensor, source row count, rows per value head)
    value_axes = (
        ("attn_gate", "blk.0.attn_gate.weight", v_heads * head_v, head_v),
        ("ssm_alpha", "blk.0.ssm_alpha.weight", v_heads, 1),
        ("ssm_beta", "blk.0.ssm_beta.weight", v_heads, 1),
        ("ssm_a", "blk.0.ssm_a", v_heads, 1),
        ("ssm_dt_bias", "blk.0.ssm_dt.bias", v_heads, 1),
    )
    for rank in range(world_size):
        owned = _tiled_local_heads(config, rank, world_size)
        assert len(owned) == v_heads // world_size
        # The pairing the rank-local kernel relies on: local slot j pairs with
        # local K head ``j % local_k_heads``, which must be the global K head of
        # the value head it holds.
        for local_slot, global_v in enumerate(owned):
            assert global_v % k_heads == rank * local_k + local_slot % local_k
        for slot, name, rows, per_head in value_axes:
            payload = layer.rank_payloads(rank)[slot]
            info = reader.tensor_info(name)
            row_bytes = int(info.nbytes) // rows
            want = _source_rows(
                reader,
                name,
                [global_v * per_head + i for global_v in owned for i in range(per_head)],
                row_bytes=row_bytes,
                rows=rows,
            )
            if slot == "ssm_a":
                # GGUF ``ssm_a`` is the negative decay coefficient and the GDN
                # kernels take ``A_log``, so the uploaded slice is the converted
                # one. The rows are the same rows; only the ABI differs.
                want = np.log(-want.view("<f4")).astype("<f4").view(np.uint8)
            want = _repacked(
                want, len(owned) * per_head, row_bytes, payload.layout, info.ggml_type_name
            )
            got = np.asarray(payload.payload).reshape(-1)
            assert got.size == want.size, (slot, got.size, want.size)
            assert np.array_equal(got, want), f"{slot} rank {rank} holds the wrong heads"

        # The fused qkv's q and k blocks, then its value block per tile.
        qkv = layer.rank_payloads(rank)["attn_qkv"]
        info = reader.tensor_info("blk.0.attn_qkv.weight")
        qkv_rows = 2 * key_width + v_heads * head_v
        qkv_row_bytes = int(info.nbytes) // qkv_rows
        want_rows = (
            [rank * local_k * head_k + i for i in range(local_k * head_k)]
            + [key_width + rank * local_k * head_k + i for i in range(local_k * head_k)]
            + [
                2 * key_width + tile * k_heads * head_v + global_v * head_v + i
                for tile in range(tiles)
                for global_v in range(rank * local_k, (rank + 1) * local_k)
                for i in range(head_v)
            ]
        )
        want = _repacked(
            _source_rows(
                reader,
                "blk.0.attn_qkv.weight",
                want_rows,
                row_bytes=qkv_row_bytes,
                rows=qkv_rows,
            ),
            len(want_rows),
            qkv_row_bytes,
            qkv.layout,
            info.ggml_type_name,
        )
        got = np.asarray(qkv.payload).reshape(-1)
        assert got.size == want.size, (got.size, want.size)
        assert np.array_equal(got, want), f"attn_qkv rank {rank} holds the wrong rows"

        # ssm_out splits the value-head axis of its input, and a value head is
        # narrower than one quant block, so the unit is the tile's own run.
        out = layer.rank_payloads(rank)["ssm_out"]
        info = reader.tensor_info("blk.0.ssm_out.weight")
        out_rows = int(config.hidden_size)
        out_row_bytes = int(info.nbytes) // out_rows
        block_size = 256
        type_size = out_row_bytes * block_size // int(config.ssm_inner_size)
        segment_bytes = local_k * head_v // block_size * type_size
        view = np.asarray(
            source_payload(reader.path, data_offset=info.data_offset, nbytes=info.nbytes)
        ).reshape(out_rows, out_row_bytes)
        pieces = []
        for tile in range(tiles):
            start = (tile * k_heads + rank * local_k) * head_v // block_size * type_size
            pieces.append(view[:, start : start + segment_bytes])
        want = _repacked(
            np.ascontiguousarray(np.concatenate(pieces, axis=1)),
            out_rows,
            tiles * segment_bytes,
            out.layout,
            info.ggml_type_name,
        )
        got = np.asarray(out.payload).reshape(-1)
        assert got.size == want.size, (got.size, want.size)
        assert np.array_equal(got, want), f"ssm_out rank {rank} holds the wrong columns"


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
    # ssm_a: one f32 scalar per value head, uploaded as the kernel's ``A_log``
    # (``log(-coefficient)``) rather than as the GGUF coefficient.
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
        assert np.array_equal(decay, np.log(-np.asarray(a_source[owned])))
        assert not np.array_equal(decay, np.asarray(a_source[owned]))
        # The two ranks partition the source rather than overlapping it.
        assert payloads["ssm_alpha"].nbytes == int(alpha_info.nbytes) // 2
        assert payloads["ssm_a"].nbytes == int(a_info.nbytes) // 2
