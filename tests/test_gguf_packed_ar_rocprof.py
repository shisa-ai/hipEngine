from __future__ import annotations

from pathlib import Path

from scripts.gguf_packed_ar_rocprof import _child_command, build_arg_parser


def test_packed_rocprof_accepts_every_direct_width_c2_c8() -> None:
    parser = build_arg_parser()

    for width in range(2, 9):
        assert parser.parse_args(
            ["--packed-concurrency", str(width)]
        ).packed_concurrency == width


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
