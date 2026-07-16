from __future__ import annotations

from hipengine.runtime.gguf_packed_manifest import build_packed_decode_execution_manifest
from scripts.gguf_packed_ar_rocprof import (
    KernelTraceRow,
    build_execution_census,
    classify_packed_execution_bucket,
)


def _layer_types() -> tuple[str, ...]:
    return ("linear_attention",) * 30 + ("full_attention",) * 10


def test_packed_decode_manifest_counts_steady_c4_hybrid_boundary() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 513, 513, 513),
        scatter_state=False,
        blocks_per_slot=4,
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


def test_packed_decode_manifest_accounts_indexed_recurrent_closure() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 513, 513, 513),
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
    )

    assert manifest["linear_attention_decode_path"] == "indexed_batch"
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
        "gdn_recurrent_decode_segments_fp32_out",
    ]
    assert manifest["profiler_contract"] == {
        "expected_execution_buckets": ["packed_native"],
        "expected_exact_row_local_kernel_launches": 0,
        "require_cached_build": True,
    }


def test_packed_decode_manifest_separates_import_and_scatter_from_steady_step() -> None:
    imported = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(0, 1, 2, 3),
        import_positions=(512, 512, 512, 512),
        scatter_state=False,
        blocks_per_slot=4,
    )
    scattered = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=_layer_types(),
        imported_slot_indices=(),
        import_positions=(513, 513, 513, 513),
        scatter_state=True,
        blocks_per_slot=4,
    )

    assert imported["host_device_movement"]["device_to_device_state_import_copies"] == 320
    assert imported["host_device_movement"]["device_to_device_state_scatter_copies"] == 0
    assert imported["steady_packed_state_reused"] is False
    assert scattered["host_device_movement"]["device_to_device_state_import_copies"] == 0
    assert scattered["host_device_movement"]["device_to_device_state_scatter_copies"] == 320
    assert scattered["steady_packed_state_reused"] is False


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
            kernel="argmax_rows_stage1_i32_kernel",
            duration_ns=80,
            grid_y=4,
        ),
    ]

    assert all(classify_packed_execution_bucket(row) == "exact_row_local" for row in exact_rows)
    assert all(classify_packed_execution_bucket(row) == "packed_native" for row in native_rows)


def test_profiler_census_accepts_zero_row_local_indexed_boundary() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=("linear_attention",),
        imported_slot_indices=(),
        import_positions=(1, 1, 1, 1),
        scatter_state=False,
        blocks_per_slot=4,
        linear_attention_decode_path="indexed_batch",
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
    assert census["c4"]["buckets"]["exact_row_local"]["dispatches"] == 0
    assert census["c4"]["buckets"]["packed_native"]["dispatches"] == 2


def test_profiler_census_requires_runtime_manifest_launch_accounting() -> None:
    manifest = build_packed_decode_execution_manifest(
        rows=4,
        layer_types=("linear_attention",),
        imported_slot_indices=(),
        import_positions=(1, 1, 1, 1),
        scatter_state=False,
        blocks_per_slot=4,
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
