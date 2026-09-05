from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from hipengine.runtime.gguf_packed_manifest import build_packed_decode_execution_manifest
from scripts.gguf_packed_ar_rocprof import (
    KernelTraceRow,
    _artifact_kind,
    _correctness_gate,
    _marker_name,
    _q6_rowtile_dispatch_count,
    build_arg_parser,
    build_c3_family_census,
    build_execution_census,
    classify_decode_kernel_family,
    classify_packed_execution_bucket,
    execution_census_closure_level,
    summarize_decode_kernel_families,
)


def test_packed_profiler_counts_exact_q6_rowtile_groups() -> None:
    assert _q6_rowtile_dispatch_count(4) == 1
    assert _q6_rowtile_dispatch_count(6) == 1
    assert _q6_rowtile_dispatch_count(7) == 2
    assert _q6_rowtile_dispatch_count(8) == 2
    assert _q6_rowtile_dispatch_count(13) == 3


def test_packed_rocprof_accepts_declared_c2_c4_c8_targets_and_markers() -> None:
    parser = build_arg_parser()
    assert parser.parse_args([]).packed_concurrency == 4

    for width in (2, 4, 8):
        args = parser.parse_args(
            ["--backend", "hip_gfx1151", "--packed-concurrency", str(width)]
        )
        assert args.backend == "hip_gfx1151"
        assert args.packed_concurrency == width
        assert _marker_name(args.packed_concurrency) == (
            f"hipengine_gguf_packed_c1_profile_c{width}_steady_decode_step"
        )


def test_packed_rocprof_scopes_kind_and_correctness_gate_to_backend() -> None:
    assert _artifact_kind("gfx1100", closure_level="c4", packed_concurrency=8) == (
        "gfx1100_gguf_concurrency_e2_native_c8_graph_profiler_census"
    )
    assert _artifact_kind("gfx1151", closure_level="c4", packed_concurrency=8) == (
        "gfx1151_gguf_concurrency_e1_native_c8_graph_profiler_census"
    )
    assert _artifact_kind("gfx1151", closure_level="c4", packed_concurrency=4) == (
        "gfx1151_gguf_concurrency_c4_graph_replay_census"
    )
    assert _artifact_kind("gfx1151", closure_level="c4", packed_concurrency=2) == (
        "gfx1151_gguf_concurrency_c2_graph_replay_census"
    )
    assert _correctness_gate("hip_gfx1100").endswith(
        "2026-07-16-gfx1100-gguf-concurrency-b4-category-lifecycle.json"
    )
    assert _correctness_gate("hip_gfx1151").endswith(
        "2026-07-17-gfx1151-gguf-concurrency-e1-direct-correctness.json"
    )


def _layer_types() -> tuple[str, ...]:
    return ("linear_attention",) * 30 + ("full_attention",) * 10


def _c3_routes(
    *,
    full_attention: bool = True,
    metadata_prepare_path: str = "host_upload",
) -> dict[str, object]:
    return {
        "full_attention_decode_path": (
            "kv_live_spans_batch" if full_attention else "not_applicable"
        ),
        "moe_decode_path": "selected_rows_batch",
        "moe_top_k": 8,
        "lm_head_decode_path": "q6_rowtile_f32_logits",
        "sampler_decode_path": "argmax_i32_rows",
        "metadata_prepare_path": metadata_prepare_path,
    }


def test_packed_decode_manifest_counts_steady_c4_hybrid_boundary() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 513, 513, 513),
        scatter_state=False,
        blocks_per_slot=4,
        **_c3_routes(),
    )

    assert manifest["schema"] == 1
    assert manifest["kind"] == "gguf_packed_ar_execution_manifest"
    assert manifest["mode"] == "decode"
    assert manifest["rows"] == 4
    assert manifest["linear_attention_decode_path"] == "exact_row_local"
    assert manifest["model_step"] == {
        "complete_c1_session_replays": 0,
        "complete_c1_layer_replays": 0,
        "host_model_row_loop_sites": 30,
        "host_model_row_iterations": 120,
        "per_row_model_subgraph_invocations": 120,
        "expected_exact_row_local_kernel_launches": 840,
    }

    families = manifest["layer_families"]
    assert families["projection"]["execution"] == "hybrid"
    assert families["projection"]["exact_row_local_kernel_launches"] == 480
    assert families["conv_gdn"]["execution"] == "exact_row_local"
    assert families["conv_gdn"]["exact_row_local_kernel_launches"] == 240
    assert families["normalization"]["exact_row_local_kernel_launches"] == 120
    for name in ("full_attention", "moe_ffn", "lm_head", "sampler"):
        assert families[name]["execution"] == "packed_native"
        assert families[name]["exact_row_local_kernel_launches"] == 0

    movement = manifest["host_device_movement"]
    assert movement["host_to_device_metadata_copies"] == 8
    assert movement["host_to_device_metadata_bytes"] == 200
    assert movement["host_to_device_input_copies"] == 1
    assert movement["host_to_device_input_bytes"] == 32
    assert movement["host_to_device_total_copies"] == 9
    assert movement["host_to_device_total_bytes"] == 232
    assert movement["device_to_device_state_import_copies"] == 0
    assert movement["device_to_device_state_scatter_copies"] == 0
    assert movement["device_to_host_vector_copies"] == 1
    assert movement["device_to_host_vector_values"] == 4
    assert movement["device_to_host_vector_bytes"] == 16
    assert manifest["synchronizations"] == 2
    assert manifest["scalar_fallbacks"] == 0
    assert manifest["steady_packed_state_reused"] is True


def test_packed_decode_manifest_accounts_direct_resident_linear_state() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(0, 1, 2, 3),
        import_positions=(513, 513, 513, 513),
        scatter_state=True,
        blocks_per_slot=4,
        direct_resident_linear_state=True,
        **_c3_routes(),
    )

    assert manifest["linear_state_storage"] == "resident_slot_slab_direct"
    movement = manifest["host_device_movement"]
    assert movement["device_to_device_state_import_copies"] == 80
    assert movement["device_to_device_state_scatter_copies"] == 80


def test_packed_decode_manifest_accepts_registered_int8_kv_batch_route() -> None:
    routes = _c3_routes()
    routes["full_attention_decode_path"] = "kv_live_spans_int8_batch"

    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 513, 513, 513),
        scatter_state=False,
        blocks_per_slot=4,
        **routes,
    )

    assert manifest["full_attention_decode_path"] == "kv_live_spans_int8_batch"
    assert manifest["layer_families"]["full_attention"]["execution"] == "packed_native"


def test_packed_decode_manifest_accounts_indexed_recurrent_closure() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 513, 513, 513),
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        gdn_recurrent_decode_path="indexed_singleton",
        **_c3_routes(),
    )

    assert manifest["linear_attention_decode_path"] == "indexed_batch"
    assert manifest["gdn_recurrent_decode_path"] == "indexed_singleton"
    assert manifest["claim_level"] == "exact_hybrid"
    assert manifest["model_step"] == {
        "complete_c1_session_replays": 0,
        "complete_c1_layer_replays": 0,
        "host_model_row_loop_sites": 0,
        "host_model_row_iterations": 0,
        "per_row_model_subgraph_invocations": 0,
        "expected_exact_row_local_kernel_launches": 0,
    }
    families = manifest["layer_families"]
    for name in ("projection", "conv_gdn", "normalization"):
        assert families[name]["execution"] == "packed_native"
        assert families[name]["host_row_loop_sites"] == 0
        assert families[name]["host_row_iterations"] == 0
        assert families[name]["exact_row_local_kernel_launches"] == 0
        assert families[name]["exact_row_local_work"] == []
    assert families["conv_gdn"]["packed_native_work"] == [
        "conv_decode_indexed",
        "gdn_recurrent_decode_indexed_fp32_out",
    ]
    assert manifest["profiler_contract"] == {
        "expected_execution_buckets": ["packed_native"],
        "expected_exact_row_local_kernel_launches": 0,
        "require_cached_build": True,
    }


def test_packed_decode_manifest_reports_sparse_physical_bucket() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        active_mask=(True, False, True, False),
        layer_types=_layer_types(),
        imported_slot_indices=(0, 2),
        import_positions=(513, -1, 521, -1),
        scatter_state=True,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        **_c3_routes(),
    )

    assert manifest["rows"] == 4
    assert manifest["physical_rows"] == 4
    assert manifest["active_rows"] == 2
    assert manifest["active_mask"] == [True, False, True, False]
    assert manifest["state_import_slot_indices"] == [0, 2]
    assert manifest["layer_families"]["full_attention"]["live_counts"] == [514, 0, 522, 0]
    movement = manifest["host_device_movement"]
    assert movement["device_to_device_state_import_copies"] == 160
    assert movement["device_to_device_state_scatter_copies"] == 160
    assert movement["device_to_host_vector_values"] == 4


def test_packed_decode_manifest_requires_explicit_c3_family_routes() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 517, 521, 525),
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        full_attention_decode_path="kv_live_spans_batch",
        moe_decode_path="selected_rows_batch",
        moe_top_k=8,
        lm_head_decode_path="q6_rowtile_f32_logits",
        sampler_decode_path="argmax_i32_rows",
        metadata_prepare_path="host_upload",
    )

    assert manifest["full_attention_decode_path"] == "kv_live_spans_batch"
    assert manifest["moe_decode_path"] == "selected_rows_batch"
    assert manifest["lm_head_decode_path"] == "q6_rowtile_f32_logits"
    assert manifest["sampler_decode_path"] == "argmax_i32_rows"
    assert manifest["metadata_prepare_path"] == "host_upload"

    families = manifest["layer_families"]
    assert families["full_attention"]["kv_abi"] == "KVLiveSpans"
    assert families["full_attention"]["row_positions"] == 4
    assert families["full_attention"]["live_counts"] == [514, 518, 522, 526]
    assert families["moe_ffn"]["router_rows"] == 4
    assert families["moe_ffn"]["selected_lanes"] == 32
    assert families["moe_ffn"]["lane_to_row"] == "selected_lane // top_k"
    assert families["lm_head"]["output_rows"] == 4
    assert families["lm_head"]["full_vocab_host_readback"] is False
    assert families["sampler"]["device_result"] == "argmax_i32_rows"
    assert families["sampler"]["host_readback"] == "one_i32_vector"


def test_packed_decode_manifest_accounts_copy_free_device_metadata() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 517, 521, 525),
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        **_c3_routes(metadata_prepare_path="device_prepare_persistent"),
    )

    movement = manifest["host_device_movement"]
    assert manifest["metadata_prepare_path"] == "device_prepare_persistent"
    assert movement["host_to_device_metadata_copies"] == 0
    assert movement["host_to_device_metadata_bytes"] == 0
    assert movement["device_metadata_prepare_launches"] == 1
    assert movement["host_to_device_total_copies"] == 1
    assert movement["host_to_device_total_bytes"] == 32

    census = build_c3_family_census(
        [
            KernelTraceRow(
                kernel="prepare_packed_decode_metadata_kernel",
                duration_ns=10,
            ),
            KernelTraceRow(kernel="__amd_rocclr_copyBuffer", duration_ns=10),
            KernelTraceRow(kernel="__amd_rocclr_copyBuffer", duration_ns=10),
        ],
        manifest=manifest,
    )
    assert census["host_device_movement"]["passed"] is True
    assert census["host_device_movement"]["expected_copy_dispatches"] == 2
    assert census["host_device_movement"]["observed_metadata_prepare_dispatches"] == 1

    replay_manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 517, 521, 525),
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        **_c3_routes(metadata_prepare_path="device_positions_persistent"),
    )
    replay_movement = replay_manifest["host_device_movement"]
    assert replay_movement["host_to_device_metadata_copies"] == 0
    assert replay_movement["device_metadata_prepare_launches"] == 1


def test_profiler_artifact_classifies_the_highest_closed_boundary() -> None:
    recurrent_manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 517, 521, 525),
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        **_c3_routes(),
    )
    device_manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 517, 521, 525),
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        **_c3_routes(metadata_prepare_path="device_prepare_persistent"),
    )
    replay_manifest = deepcopy(device_manifest)
    replay_manifest["mode"] = "decode_graph_replay"
    replay_manifest["host_device_movement"].update(
        {
            "host_to_device_total_copies": 0,
            "device_to_host_vector_copies": 0,
        }
    )
    replay_manifest["synchronizations"] = 0
    replay_manifest["graph"] = {
        "captured": True,
        "replay_count": 1,
        "replayed_steps": 1,
    }
    complete_families = {"c3_family_census": {"route_check_passed": True}}

    assert execution_census_closure_level(recurrent_manifest, complete_families) == "c2"
    assert execution_census_closure_level(device_manifest, complete_families) == "c3"
    assert execution_census_closure_level(replay_manifest, complete_families) == "c4"
    replay_manifest["graph"]["replay_count"] = 0
    assert execution_census_closure_level(replay_manifest, complete_families) == "c3"
    assert (
        execution_census_closure_level(
            device_manifest,
            {"c3_family_census": {"route_check_passed": False}},
        )
        == "c2"
    )


def test_packed_decode_manifest_separates_import_and_scatter_from_steady_step() -> None:
    imported = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(0, 1, 2, 3),
        import_positions=(512, 512, 512, 512),
        scatter_state=False,
        blocks_per_slot=4,
        **_c3_routes(),
    )
    scattered = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 513, 513, 513),
        scatter_state=True,
        blocks_per_slot=4,
        **_c3_routes(),
    )

    assert imported["host_device_movement"]["device_to_device_state_import_copies"] == 320
    assert imported["host_device_movement"]["device_to_device_state_scatter_copies"] == 0
    assert imported["steady_packed_state_reused"] is False
    assert scattered["host_device_movement"]["device_to_device_state_import_copies"] == 0
    assert scattered["host_device_movement"]["device_to_device_state_scatter_copies"] == 320
    assert scattered["steady_packed_state_reused"] is False


def test_packed_c3_profiler_census_requires_each_caware_family() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 517, 521, 525),
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        **_c3_routes(),
    )
    rows = [
        *[
            KernelTraceRow(
                kernel="qwen35_paged_full_attn_decode_context_tensor_batch_kernel",
                duration_ns=10,
                grid_y=4,
            )
            for _ in range(10)
        ],
        *[
            KernelTraceRow(
                kernel="qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel",
                duration_ns=10,
                grid_y=4,
            )
            for _ in range(10)
        ],
        *[
            KernelTraceRow(
                kernel="q4_k_t16_selected_dual_direct_gemv_kernel",
                duration_ns=10,
                grid_y=32,
            )
            for _ in range(40)
        ],
        *[
            KernelTraceRow(
                kernel="qk_t16_selected_direct_gemv_kernel",
                duration_ns=10,
                grid_y=32,
            )
            for _ in range(40)
        ],
        *[
            KernelTraceRow(
                kernel="weighted_sum_shared_gate_combine_residual_batch_out_kernel",
                duration_ns=10,
                grid_y=4,
            )
            for _ in range(40)
        ],
        KernelTraceRow(
            kernel="q6_k_t16_gemv_rowtile_col8_kernel",
            duration_ns=10,
            grid_y=1,
        ),
        KernelTraceRow(
            kernel="argmax_rows_stage1_i32_kernel",
            duration_ns=10,
            grid_y=4,
        ),
        KernelTraceRow(
            kernel="argmax_rows_stage2_i32_kernel",
            duration_ns=10,
            grid_y=1,
        ),
        *[
            KernelTraceRow(kernel="__amd_rocclr_copyBuffer", duration_ns=10)
            for _ in range(10)
        ],
    ]

    census = build_c3_family_census(rows, manifest=manifest)

    assert census["route_check_passed"] is True
    assert census["full_attention"]["context_dispatches"] == 10
    assert census["full_attention"]["kv_write_dispatches"] == 10
    assert census["moe_ffn"]["selected_gate_up_dispatches"] == 40
    assert census["moe_ffn"]["selected_down_dispatches"] == 40
    assert census["moe_ffn"]["combine_dispatches"] == 40
    assert census["lm_head_sampler"]["expected_lm_head_dispatches"] == 1
    assert census["lm_head_sampler"]["lm_head_dispatches"] == 1
    assert census["lm_head_sampler"]["argmax_stage1_dispatches"] == 1
    assert census["host_device_movement"]["observed_copy_dispatches"] == 10

    c8_manifest = build_packed_decode_execution_manifest(
        rows=8,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513,) * 8,
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        **_c3_routes(),
    )
    row_extent_eight = (
        "qwen35_paged_full_attn_decode_context_tensor_batch_kernel",
        "qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel",
        "weighted_sum_shared_gate_combine_residual_batch_out_kernel",
        "argmax_rows_stage1_i32_kernel",
    )
    selected_extent = (
        "selected_dual_direct_gemv_kernel",
        "qk_t16_selected_direct_gemv_kernel",
    )
    c8_rows = [
        replace(
            row,
            grid_y=(
                8
                if any(name in row.kernel for name in row_extent_eight)
                else 64
                if any(name in row.kernel for name in selected_extent)
                else row.grid_y
            ),
        )
        for row in rows
    ]
    c8_rows.append(
        KernelTraceRow(
            kernel="q6_k_t16_gemv_rowtile_kernel",
            duration_ns=10,
            grid_y=1,
        )
    )

    c8_census = build_c3_family_census(c8_rows, manifest=c8_manifest)
    assert c8_census["route_check_passed"] is True
    assert c8_census["lm_head_sampler"]["expected_lm_head_dispatches"] == 2
    assert c8_census["lm_head_sampler"]["lm_head_dispatches"] == 2


def test_packed_profiler_classifier_separates_exact_row_local_from_native() -> None:
    exact_rows = [
        KernelTraceRow(
            kernel="qwen35_linear_attn_conv_decode_lowp_kernel<unsigned short>",
            duration_ns=10,
            grid_y=1,
        ),
        KernelTraceRow(
            kernel="qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<unsigned short>",
            duration_ns=20,
            grid_y=1,
        ),
        KernelTraceRow(
            kernel="q8_0_t16_dual_split_gemv_kernel<unsigned short>",
            duration_ns=30,
            grid_y=1,
        ),
        KernelTraceRow(
            kernel="dense_gemv_bf16_f32w_bf16_out_kernel(unsigned short const*)",
            duration_ns=40,
            grid_y=1,
        ),
        KernelTraceRow(
            kernel="q8_0_t16_gemv_kernel<float>(float const*)",
            duration_ns=50,
            grid_y=1,
        ),
        KernelTraceRow(
            kernel="gguf_rmsnorm_bf16_f32_weight_kernel",
            duration_ns=60,
            grid_y=1,
        ),
    ]
    native_rows = [
        KernelTraceRow(
            kernel="q8_0_t16_dual_split_gemv_kernel<unsigned short>",
            duration_ns=61,
            grid_y=4,
        ),
        KernelTraceRow(
            kernel="dense_gemv_bf16_f32w_bf16_out_kernel(unsigned short const*)",
            duration_ns=62,
            grid_y=4,
        ),
        KernelTraceRow(
            kernel="q8_0_t16_gemv_kernel<float>(float const*)",
            duration_ns=63,
            grid_y=4,
        ),
        KernelTraceRow(
            kernel="qwen35_paged_full_attn_decode_context_tensor_batch_kernel",
            duration_ns=70,
            grid_y=4,
        ),
        KernelTraceRow(
            kernel="qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_kernel<unsigned short>",
            duration_ns=75,
            grid_y=32,
        ),
        KernelTraceRow(
            kernel="argmax_rows_stage1_i32_kernel",
            duration_ns=80,
            grid_y=4,
        ),
    ]

    assert all(classify_packed_execution_bucket(row) == "exact_row_local" for row in exact_rows)
    assert all(classify_packed_execution_bucket(row) == "packed_native" for row in native_rows)


def test_packed_profiler_attributes_every_decode_kernel_family() -> None:
    rows = [
        KernelTraceRow(kernel="prepare_packed_decode_metadata_kernel", duration_ns=10),
        KernelTraceRow(kernel="q6_k_t16_gemv_rowtile_kernel", duration_ns=20),
        KernelTraceRow(kernel="gguf_q8_0_embedding_bf16_out_kernel", duration_ns=30),
        KernelTraceRow(kernel="qwen35_router_select_kernel", duration_ns=40),
        KernelTraceRow(kernel="qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_kernel", duration_ns=50),
        KernelTraceRow(kernel="qwen35_paged_full_attn_decode_context_tensor_batch_kernel", duration_ns=60),
        KernelTraceRow(kernel="q4_k_t16_selected_dual_direct_gemv_kernel", duration_ns=70),
        KernelTraceRow(kernel="gguf_add_rmsnorm_bf16_f32_weight_kernel", duration_ns=80),
        KernelTraceRow(kernel="q8_0_t16_dual_split_gemv_kernel", duration_ns=90),
        KernelTraceRow(kernel="future_unclassified_kernel", duration_ns=100),
    ]
    expected = [
        "metadata_lifecycle",
        "lm_head_sampler",
        "embedding",
        "moe_router",
        "linear_attention_state",
        "full_attention_core",
        "moe_selected_combine",
        "norm_residual",
        "dense_projection",
        "other",
    ]

    assert [classify_decode_kernel_family(row.kernel) for row in rows] == expected
    summary = summarize_decode_kernel_families(rows)
    by_name = {record["family"]: record for record in summary["families"]}
    assert summary["total_dispatches"] == 10
    assert summary["total_duration_ns"] == 550
    assert sum(record["dispatches"] for record in summary["families"]) == 10
    assert sum(record["total_duration_ns"] for record in summary["families"]) == 550
    assert by_name["dense_projection"]["dispatches"] == 1
    assert by_name["dense_projection"]["share_of_gpu_duration"] == 90 / 550
    assert summary["unclassified"] == {
        "dispatches": 1,
        "total_duration_ns": 100,
        "kernel_names": ["future_unclassified_kernel"],
    }


def test_profiler_census_accepts_zero_row_local_indexed_boundary() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=("linear_attention",),
        imported_slot_indices=(),
        import_positions=(1, 1, 1, 1),
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        **_c3_routes(full_attention=False),
    )
    c1 = [KernelTraceRow(kernel="c1_kernel", duration_ns=5, grid_y=1)]
    c4 = [
        KernelTraceRow(
            kernel="q8_0_t16_dual_split_gemv_kernel<unsigned short>",
            duration_ns=10,
            grid_y=4,
        ),
        KernelTraceRow(
            kernel="qwen35_linear_attn_conv_decode_indexed_lowp_kernel<unsigned short>",
            duration_ns=20,
            grid_y=4,
        ),
    ]

    census = build_execution_census(c1, c4, manifest=manifest)

    assert census["route_check_passed"] is True
    assert census["packed_concurrency"] == 4
    assert census["c4"]["buckets"]["exact_row_local"]["dispatches"] == 0
    assert census["c4"]["buckets"]["packed_native"]["dispatches"] == 2
    assert census["family_attribution"]["c1"]["total_dispatches"] == 1
    assert census["family_attribution"]["c4"]["total_dispatches"] == 2

    c8_manifest = deepcopy(manifest)
    c8_manifest.update({"rows": 8, "physical_rows": 8, "active_rows": 8})
    c8_manifest["active_mask"] = [True] * 8
    c8_census = build_execution_census(
        c1,
        c4,
        manifest=c8_manifest,
        packed_concurrency=8,
    )
    assert c8_census["packed_concurrency"] == 8
    assert "c8" in c8_census
    assert "c4" not in c8_census
    assert (
        c8_census["c8"]["buckets"]["packed_native"][
            "share_of_c8_gpu_duration"
        ]
        == 1.0
    )


def test_profiler_census_requires_runtime_manifest_launch_accounting() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=("linear_attention",),
        imported_slot_indices=(),
        import_positions=(1, 1, 1, 1),
        scatter_state=False,
        blocks_per_slot=4,
        **_c3_routes(full_attention=False),
    )
    exact_count = manifest["model_step"]["expected_exact_row_local_kernel_launches"]
    c1 = [KernelTraceRow(kernel="c1_kernel", duration_ns=5, grid_y=1)]
    c4 = [
        *[
            KernelTraceRow(
                kernel="qwen35_linear_attn_conv_decode_lowp_kernel<unsigned short>",
                duration_ns=10,
                grid_y=1,
            )
            for _ in range(exact_count)
        ],
        KernelTraceRow(
            kernel="argmax_rows_stage1_i32_kernel",
            duration_ns=20,
            grid_y=4,
        ),
    ]

    census = build_execution_census(c1, c4, manifest=manifest)

    assert census["route_check_passed"] is True
    assert census["c1_reference"]["dispatches"] == 1
    assert census["c4"]["buckets"]["exact_row_local"]["dispatches"] == exact_count
    assert census["c4"]["buckets"]["packed_native"]["dispatches"] == 1


def _dense_routes() -> dict[str, object]:
    return {
        "full_attention_decode_path": "kv_live_spans_batch",
        "moe_decode_path": "dense_ffn_rows",
        "moe_top_k": 0,
        "lm_head_decode_path": "q6_rowtile_f32_logits",
        "sampler_decode_path": "argmax_i32_rows",
        "metadata_prepare_path": "host_upload",
    }


def _dense_census_rows(*, selected_leak: bool = False) -> list[KernelTraceRow]:
    rows: list[KernelTraceRow] = [
        *[
            KernelTraceRow(
                kernel="qwen35_paged_full_attn_decode_context_tensor_batch_kernel",
                duration_ns=10,
                grid_y=8,
            )
            for _ in range(10)
        ],
        *[
            KernelTraceRow(
                kernel="qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel",
                duration_ns=10,
                grid_y=8,
            )
            for _ in range(10)
        ],
        *[
            KernelTraceRow(
                kernel=(
                    "void (anonymous namespace)::q4_k_t16_dense_dual_rowtile_"
                    "silu_gemv_kernel<...>(unsigned short const*, unsigned char "
                    "const*, unsigned short*, long, long, long)"
                ),
                duration_ns=10,
            )
            for _ in range(40)
        ],
        *[
            KernelTraceRow(
                kernel=(
                    "void (anonymous namespace)::q6_k_t16_qmicro_planar_gemv_"
                    "rowtile_col8_kernel<...>(unsigned short const*, unsigned "
                    "char const*, unsigned short*, long, long, long)"
                ),
                duration_ns=10,
            )
            for _ in range(40)
        ],
        KernelTraceRow(
            kernel=(
                "void (anonymous namespace)::q6_k_t16_qmicro_planar_gemv_"
                "rowtile_col8_kernel<...>(unsigned short const*, unsigned char "
                "const*, float*, long, long, long)"
            ),
            duration_ns=10,
        ),
        KernelTraceRow(kernel="argmax_rows_stage1_i32_kernel", duration_ns=10, grid_y=8),
        KernelTraceRow(kernel="argmax_rows_stage2_i32_kernel", duration_ns=10),
        *[
            KernelTraceRow(kernel="__amd_rocclr_copyBuffer", duration_ns=10)
            for _ in range(10)
        ],
    ]
    if selected_leak:
        rows.append(
            KernelTraceRow(
                kernel="q4_k_t16_selected_dual_direct_gemv_kernel",
                duration_ns=10,
                grid_y=64,
            )
        )
    return rows


def test_packed_c3_profiler_census_accepts_dense_ffn_with_qmicro_lm_head() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=8,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513,) * 8,
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        **_dense_routes(),
    )

    census = build_c3_family_census(
        _dense_census_rows(),
        manifest=manifest,
        lm_head_max_chunk=8,
    )

    assert census["route_check_passed"] is True
    assert census["moe_ffn"]["passed"] is True
    assert census["moe_ffn"]["dense_gate_up_dispatches"] == 40
    assert census["moe_ffn"]["dense_down_dispatches"] == 40
    assert census["moe_ffn"]["selected_gate_up_dispatches"] == 0
    assert census["moe_ffn"]["selected_down_dispatches"] == 0
    assert census["moe_ffn"]["combine_dispatches"] == 0
    assert census["lm_head_sampler"]["passed"] is True
    assert census["lm_head_sampler"]["expected_lm_head_dispatches"] == 1
    assert census["lm_head_sampler"]["lm_head_dispatches"] == 1


def test_packed_c3_profiler_census_rejects_selected_leak_on_dense_route() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=8,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513,) * 8,
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        **_dense_routes(),
    )

    census = build_c3_family_census(
        _dense_census_rows(selected_leak=True),
        manifest=manifest,
        lm_head_max_chunk=8,
    )

    assert census["moe_ffn"]["passed"] is False
    assert census["route_check_passed"] is False


def test_packed_c3_profiler_census_default_chunk_keeps_legacy_partition() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=8,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513,) * 8,
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
        **_dense_routes(),
    )

    census = build_c3_family_census(_dense_census_rows(), manifest=manifest)

    # Default chunk 6 expects the 6+2 partition; the rows-8 qmicro owner
    # launches once, so the stale expectation must fail rather than pass.
    assert census["lm_head_sampler"]["expected_lm_head_dispatches"] == 2
    assert census["lm_head_sampler"]["lm_head_dispatches"] == 1
    assert census["lm_head_sampler"]["passed"] is False
