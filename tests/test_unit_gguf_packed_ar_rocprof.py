from __future__ import annotations

from pathlib import Path

from scripts.gguf_packed_ar_rocprof import (
    _child_command,
    _profiler_concurrency_supported,
    build_arg_parser,
    classify_decode_kernel_family,
)


def test_packed_rocprof_accepts_every_direct_width_c2_c8() -> None:
    parser = build_arg_parser()

    assert _profiler_concurrency_supported(None) is False
    assert _profiler_concurrency_supported(0) is False
    for width in range(1, 9):
        assert _profiler_concurrency_supported(width) is True
    assert _profiler_concurrency_supported(9) is False

    for width in range(2, 9):
        assert parser.parse_args(
            ["--packed-concurrency", str(width)]
        ).packed_concurrency == width


def test_packed_rocprof_classifies_quant_wmma_projection_and_cast() -> None:
    assert classify_decode_kernel_family(
        "gguf_q5_t16_dense_wmma_prefill_bf16_kernel"
    ) == "dense_projection"
    assert classify_decode_kernel_family(
        "q6_k_t16_qmicro_planar_wmma_prefill_bf16_kernel"
    ) == "dense_projection"
    assert classify_decode_kernel_family(
        "f32_to_bf16_kernel"
    ) == "dense_projection"


def test_packed_rocprof_forwards_explicit_graph_submission_transport() -> None:
    args = build_arg_parser().parse_args(
        ["--decode-mode", "graph", "--submission-transport", "aql"]
    )

    command = _child_command(
        args,
        mode="profile",
        concurrency=4,
        child_json=Path("/tmp/child.json"),
        require_cached=True,
    )

    index = command.index("--submission-transport")
    assert command[index + 1] == "aql"
    assert args.submission_transport == "aql"


def test_packed_rocprof_census_counts_residual_dense_down_variant() -> None:
    # Regression: the dense down owner appears as both col8 and col8_residual
    # variants (the residual one carries an extra `unsigned short const*`
    # input). The census must count both, or a c2 dense burst undercounts the
    # down dispatches and false-fails the route check.
    from scripts.gguf_packed_ar_rocprof import (
        KernelTraceRow,
        build_c3_family_census,
    )

    total_layers = 64
    rows = [
        KernelTraceRow(
            kernel="q4_k_t16_dense_dual_rowtile_silu_gemv_kernel",
            duration_ns=1,
        )
        for _ in range(total_layers)
    ]
    rows += [
        KernelTraceRow(
            kernel=(
                "q6_k_t16_qmicro_planar_gemv_rowtile_col8_kernel"
                "(unsigned short const*, unsigned char const*, unsigned short*,"
                " long, long, long)"
            ),
            duration_ns=1,
        )
        for _ in range(32)
    ]
    rows += [
        KernelTraceRow(
            kernel=(
                "q6_k_t16_qmicro_planar_gemv_rowtile_col8_residual_kernel"
                "(unsigned short const*, unsigned char const*, unsigned short"
                " const*, unsigned short*, long, long, long)"
            ),
            duration_ns=1,
        )
        for _ in range(32)
    ]
    # float-output lm_head rowtile must not be counted as a down owner.
    rows.append(
        KernelTraceRow(
            kernel=(
                "q6_k_t16_qmicro_planar_gemv_rowtile_kernel"
                "(unsigned short const*, unsigned char const*, float*, long,"
                " long, long)"
            ),
            duration_ns=1,
        )
    )
    rows.append(KernelTraceRow(kernel="top1_gather", duration_ns=1))
    manifest = {
        "rows": 2,
        "layers": {"full_attention": 0, "total": total_layers},
        "layer_families": {
            "moe_ffn": {"selected_lanes": 0, "decode_path": "dense_ffn_rows"},
            "lm_head": {"full_vocab_host_readback": False},
        },
        "full_attention_decode_path": "kv_live_spans_batch",
        "lm_head_decode_path": "direct_top1_rows",
        "moe_decode_path": "dense_ffn_rows",
        "sampler_decode_path": "fused_top1_i32_rows",
        "host_device_movement": {
            "host_to_device_total_copies": 0,
            "device_to_device_state_import_copies": 0,
            "device_to_device_state_scatter_copies": 0,
            "device_to_host_vector_copies": 0,
            "device_metadata_prepare_launches": 0,
        },
    }
    census = build_c3_family_census(rows, manifest=manifest)
    assert census["moe_ffn"]["dense_down_dispatches"] == total_layers
    assert census["moe_ffn"]["passed"] is True
    assert census["route_check_passed"] is True
