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
