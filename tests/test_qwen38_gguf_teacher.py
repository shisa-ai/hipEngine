from __future__ import annotations

from pathlib import Path

from scripts.qwen38_gguf_teacher import PROTOCOL_ID, TEACHER_STEPS, _sha256_json, build_parser


def test_qwen38_gguf_teacher_reuses_the_established_ninety_row_schema() -> None:
    assert PROTOCOL_ID == "qwen36-bf16-teacher-mtpbench-v1"
    assert TEACHER_STEPS == 9


def test_qwen38_gguf_teacher_parser_requires_model_identity(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--model",
            str(tmp_path / "teacher.gguf"),
            "--model-sha256",
            "a" * 64,
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert args.model.name == "teacher.gguf"
    assert args.model_sha256 == "a" * 64
    assert args.backend == "hip_gfx1151"
    assert args.allow_unqualified_diagnostic is False


def test_qwen38_teacher_prompt_hash_is_canonical() -> None:
    assert _sha256_json((1, 2, 3)) == _sha256_json([1, 2, 3])
    assert _sha256_json([1, 2, 3]) != _sha256_json([3, 2, 1])
