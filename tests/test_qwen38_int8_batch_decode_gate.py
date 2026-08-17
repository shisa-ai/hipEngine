from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hipengine.core.dtype import DType

from scripts import qwen38_int8_batch_decode_gate as gate


class _Tokenizer:
    @staticmethod
    def encode(text: str) -> list[int]:
        return [ord(char) for char in text]


def test_gate_uses_one_canonical_prompt_from_every_required_category() -> None:
    rows = gate._load_prompt_rows(gate.DEFAULT_PROMPTS)

    assert tuple(row["category"] for row in rows) == gate._REQUIRED_CATEGORIES
    assert len({row["id"] for row in rows}) == 4


def test_gate_builds_exact_requested_lengths_and_stable_prompt_manifest() -> None:
    rows = [
        {"id": f"p{index}", "category": category, "content": "abc"}
        for index, category in enumerate(gate._REQUIRED_CATEGORIES)
    ]

    prompts, manifest = gate._build_prompts(
        _Tokenizer(),
        rows,
        (31, 32, 33, 64),
    )

    assert tuple(map(len, prompts)) == (31, 32, 33, 64)
    assert [row["tokens"] for row in manifest] == [31, 32, 33, 64]
    assert all(len(row["token_ids_sha256"]) == 64 for row in manifest)


def test_gate_counts_tensor_bytes_from_numel_and_dtype() -> None:
    assert gate._tensor_nbytes(SimpleNamespace(numel=7, dtype=DType.FP32)) == 28


def test_gate_logit_and_state_checks_fail_closed() -> None:
    same = gate._logit_metrics(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )
    changed = gate._logit_metrics(
        np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )

    assert same == {
        "shape_match": True,
        "max_abs": 0.0,
        "kl": 0.0,
        "top1_match": True,
    }
    assert changed["kl"] > 0.0
    assert changed["top1_match"] is False
    assert gate._state_mismatches(
        [{"position": 3, "linear": [], "kv": []}],
        [{"position": 4, "linear": [], "kv": []}],
    ) == [
        {
            "row": 0,
            "actual_sha256": gate._sha256_bytes(
                b'{"kv": [], "linear": [], "position": 3}'
            ),
            "expected_sha256": gate._sha256_bytes(
                b'{"kv": [], "linear": [], "position": 4}'
            ),
        }
    ]


def test_gate_parser_requires_explicit_pre_promotion_width_override() -> None:
    args = gate.build_parser().parse_args(
        [
            "--model",
            str(Path("/tmp/model.gguf")),
            "--prompt-lengths",
            "8,9",
            "--diagnostic-direct-rows",
            "2",
        ]
    )

    assert args.prompt_lengths == "8,9"
    assert args.diagnostic_direct_rows == 2
