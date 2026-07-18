from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.gguf_packed_ar_state_oracle import (
    _compare_layer_hidden_sessions,
    _compare_state_rows,
    _session_build_policy,
    build_parser,
    run,
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


def test_gguf_packed_ar_state_oracle_compares_layer_hidden_rows() -> None:
    packed = SimpleNamespace(
        last_layer_output_hidden={
            0: np.asarray([[1.0, 2.0]], dtype=np.float32),
            1: np.asarray([[3.0, 4.0]], dtype=np.float32),
        }
    )
    reference = SimpleNamespace(
        last_layer_output_hidden={
            0: np.asarray([[1.0, 2.0]], dtype=np.float32),
            1: np.asarray([[3.0, 4.5]], dtype=np.float32),
        }
    )

    comparisons, mismatches = _compare_layer_hidden_sessions(
        (packed,),
        (reference,),
        row_indices=(3,),
        layer_ids=(0, 1),
        phase="decode_hidden",
        step=2,
    )

    assert comparisons == 2
    assert len(mismatches) == 1
    assert mismatches[0]["row"] == 3
    assert mismatches[0]["layer"] == 1
    assert mismatches[0]["phase"] == "decode_hidden"
    assert mismatches[0]["step"] == 2
    assert mismatches[0]["mismatch_elements"] == 1
    assert mismatches[0]["max_abs"] == 0.5


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
    assert args.gdn_prefill_mode == "exact"
    assert not args.capture_layer_hidden
    assert args.rows == 2
    assert args.lifecycle == "steady"
    assert args.prompt_length == 16
    assert args.alternate_prompt_length is None
    assert args.decode_steps == 4
    assert args.backend == "hip_gfx1151"


def test_gguf_packed_ar_state_oracle_requires_grouped_decode_isolation_above_c8() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "/missing/model.gguf",
            "--rows",
            "13",
            "--prefill-mode",
            "packed",
        ]
    )

    with pytest.raises(ValueError, match="packed prefill above 8"):
        run(args)
