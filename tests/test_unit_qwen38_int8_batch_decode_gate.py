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


def test_gate_expands_prompts_beyond_the_legacy_repeat_limit() -> None:
    rows = [{"id": "long", "category": "code", "content": "abc"}]

    prompts, manifest = gate._build_prompts(_Tokenizer(), rows, (1000,))

    assert len(prompts[0]) == 1000
    assert manifest[0]["tokens"] == 1000


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


def test_gate_parser_supports_rejected_artifact_mirror_free_diagnostics() -> None:
    args = gate.build_parser().parse_args(
        [
            "--allow-rejected-artifact",
            "--max-sequence-length",
            "9216",
        ]
    )

    assert args.allow_rejected_artifact is True
    assert args.max_sequence_length == 9216


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


def test_capability_preparation_uses_declaration_without_evidence():
    capability = _capability_payload()
    prepared, admitted, diagnostic = gate._prepare_capability(
        capability, rows=4, allow_rejected=False, diagnostic_rows=0
    )
    assert prepared == capability
    assert admitted == 8
    assert diagnostic is False


def _capability_payload():
    return {
        "status": "unmeasured", "runtime_action": "admit",
        "effective_kv_storage": "int8_per_token_head",
        "max_direct_rows": 8, "max_serial_resident_rows": 8,
        "persistent_bf16_mirror": False, "decode_batch_variant": "direct",
        "evidence": None,
        "declaration": {"max_direct_rows": 8, "max_serial_resident_rows": 8,
                        "persistent_bf16_mirror": False, "decode_batch_variant": "direct"},
    }


def test_capability_preparation_rejected_is_explicit_and_uses_declaration():
    import pytest
    capability = _capability_payload()
    capability.update(status="rejected", runtime_action="fallback_bf16",
                      effective_kv_storage="bf16", max_direct_rows=0,
                      evidence={"decision": "rejected", "max_direct_rows": 0})
    with pytest.raises(ValueError, match="rejected"):
        gate._prepare_capability(capability, rows=4, allow_rejected=False, diagnostic_rows=4)
    prepared, admitted, diagnostic = gate._prepare_capability(
        capability, rows=4, allow_rejected=True, diagnostic_rows=4)
    assert prepared["runtime_action"] == "diagnostic_override"
    assert prepared["effective_kv_storage"] == "int8_per_token_head"
    assert prepared["evidence"] == capability["evidence"]
    assert capability["runtime_action"] == "fallback_bf16"
    assert diagnostic and admitted == 0


def test_capability_preparation_never_overrides_missing_implementation():
    import pytest
    capability = _capability_payload()
    capability.update(status="unsupported", declaration=None)
    with pytest.raises(ValueError, match="implementation"):
        gate._prepare_capability(capability, rows=4, allow_rejected=True, diagnostic_rows=4)


def test_mirror_free_audit_rejects_bf16_payload_or_mirror():
    import pytest
    audit = {"persistent_int8_payload_bytes": 1024,
             "persistent_bf16_payload_bytes": 0, "persistent_bf16_mirror_bytes": 0}
    gate._assert_mirror_free(audit)
    for key in ("persistent_bf16_payload_bytes", "persistent_bf16_mirror_bytes"):
        with pytest.raises(RuntimeError, match="mirror-free"):
            gate._assert_mirror_free({**audit, key: 256})


def test_resident_audit_counts_legacy_scratch_buffers():
    scratch = SimpleNamespace(
        full_key_caches=(SimpleNamespace(nbytes=128),),
        full_value_caches=(SimpleNamespace(nbytes=128),),
        full_scale_metadata=lambda layer: object(),
        int8_kv_value_bf16=False,
        full_bf16_mirror_key_caches=(None,),
        full_bf16_mirror_value_caches=(None,),
    )
    session = SimpleNamespace(scratch=scratch)
    audit = gate._resident_layout_audit(session)
    gate._assert_mirror_free(audit)
    assert audit["persistent_int8_payload_bytes"] == 256
    scratch.int8_kv_value_bf16 = True
    assert gate._resident_layout_audit(session)["persistent_bf16_payload_bytes"] == 128
    scratch.full_scale_metadata = lambda layer: None
    assert gate._resident_layout_audit(session)["persistent_bf16_payload_bytes"] == 256
    scratch.full_bf16_mirror_key_caches = (SimpleNamespace(nbytes=256),)
    assert gate._resident_layout_audit(session)["persistent_bf16_mirror_bytes"] == 256
