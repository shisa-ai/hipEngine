from __future__ import annotations

from scripts.gguf_packed_ar_state_oracle import (
    _compare_state_rows,
    _session_build_policy,
    build_parser,
)


def _state(*, position: int = 4, conv: str = "c", recurrent: str = "r", key: str = "k", value: str = "v"):
    return {
        "position": position,
        "linear": [{"layer": 0, "conv": conv, "recurrent": recurrent}],
        "kv": [{"layer": 3, "key": key, "value": value, "checked_nbytes": 16}],
    }


def test_gguf_packed_ar_state_oracle_compares_every_state_part() -> None:
    assert _compare_state_rows([_state(), _state()], [_state(), _state()]) == []

    mismatches = _compare_state_rows(
        [_state(position=5, conv="bad-c", value="bad-v"), _state(recurrent="bad-r", key="bad-k")],
        [_state(), _state()],
    )

    assert [(row["row"], row["component"], row["layer"], row["part"]) for row in mismatches] == [
        (0, "position", None, None),
        (0, "linear", 0, "conv"),
        (0, "kv", 3, "value"),
        (1, "linear", 0, "recurrent"),
        (1, "kv", 3, "key"),
    ]


def test_gguf_packed_ar_state_oracle_cached_build_policy_reads_compiler_file(tmp_path) -> None:
    compiler_version_file = tmp_path / "hipcc-version.txt"
    compiler_version_file.write_text("HIP version: test\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--model",
            "/tmp/model.gguf",
            "--compiler-version-file",
            str(compiler_version_file),
            "--require-cached-build",
        ]
    )

    assert _session_build_policy(args) == {
        "compiler_version": "HIP version: test",
        "require_cached_build": True,
    }


def test_gguf_packed_ar_state_oracle_defaults_to_decode_isolation() -> None:
    args = build_parser().parse_args(["--model", "/tmp/model.gguf"])

    assert args.prefill_mode == "independent_c1"
    assert args.rows == 2
    assert args.lifecycle == "steady"
    assert args.prompt_length == 16
    assert args.alternate_prompt_length is None
    assert args.decode_steps == 4
    assert args.backend == "hip_gfx1151"
