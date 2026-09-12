from __future__ import annotations

import numpy as np
import pytest

from hipengine.speculative import DFlashDraftRequest, compile_dflash_chain, TargetVerifyBatch
from hipengine.speculative.verify_graph import (
    DFlashVerifyGraphAddresses,
    DFlashVerifyGraphBucketKey,
    DFlashVerifyGraphValidation,
    dflash_verify_graph_decision,
    fingerprint_int_arrays,
)


def _target(depth: int = 4) -> TargetVerifyBatch:
    draft = compile_dflash_chain(
        [DFlashDraftRequest(request_id=1, root_position=10, candidate_tokens=tuple(range(100, 100 + depth)))],
        candidate_budget=depth,
    )
    return TargetVerifyBatch.from_draft(draft, root_tokens=(99,), root_positions=(10,))


def test_dflash_verify_graph_bucket_key_includes_shape_axes() -> None:
    batch = _target(4)

    key = DFlashVerifyGraphBucketKey.from_batch(
        batch,
        backend="hip_gfx1151",
        context_bucket=4096,
        page_bucket=128,
        top_k=1,
        experts_per_token=8,
        replay_steps=3,
    )

    assert key.supported
    assert key.fallback_reason is None
    assert key.as_dict() == {
        "backend": "hip_gfx1151",
        "active_c": 1,
        "context_bucket": 4096,
        "page_bucket": 128,
        "mode": "verify_chain",
        "draft_depth": 4,
        "tree_shape": [0, 1, 2, 3],
        "top_k": 1,
        "experts_per_token": 8,
        "replay_steps": 3,
    }


def test_dflash_verify_graph_bucket_falls_back_for_rare_page_bucket() -> None:
    key = DFlashVerifyGraphBucketKey.from_batch(
        _target(2),
        backend="hip_gfx1151",
        context_bucket=123,
        page_bucket=17,
    )

    validation = dflash_verify_graph_decision(key)

    assert not key.supported
    assert "unsupported page_bucket 17" == key.fallback_reason
    assert validation.status == "direct_fallback"
    assert validation.direct_match is True
    assert validation.as_artifact_row()["fallback_reason"] == "unsupported page_bucket 17"


def test_dflash_verify_graph_addresses_are_stable_and_fingerprinted() -> None:
    addresses = DFlashVerifyGraphAddresses.from_mapping({"token_ids": 0x1000, "target_top1": 0x2000})

    assert addresses.as_dict() == {"target_top1": "0x2000", "token_ids": "0x1000"}
    assert len(addresses.fingerprint) == 64
    with pytest.raises(ValueError, match="non-zero"):
        DFlashVerifyGraphAddresses.from_mapping({"bad": 0})


def test_dflash_verify_graph_validation_requires_addresses_for_capture() -> None:
    key = DFlashVerifyGraphBucketKey.from_batch(_target(2), backend="hip_gfx1151", context_bucket=1, page_bucket=1)
    addresses = DFlashVerifyGraphAddresses.from_mapping({"token_ids": 0x1000})

    validation = DFlashVerifyGraphValidation(
        bucket_key=key,
        status="captured",
        replay_steps=2,
        fixed_addresses=addresses,
        direct_match=True,
        graph_validation_passed=True,
        direct_output_fingerprint="a" * 64,
        graph_output_fingerprint="a" * 64,
    )

    row = validation.as_artifact_row()
    assert row["status"] == "captured"
    assert row["graph_validation_passed"] is True
    assert row["fixed_buffer_addresses"] == {"token_ids": "0x1000"}
    with pytest.raises(ValueError, match="requires fixed addresses"):
        DFlashVerifyGraphValidation(bucket_key=key, status="captured", replay_steps=1)


def test_fingerprint_int_arrays_is_stable() -> None:
    arr = np.array([1, 2, 3], dtype=np.int32)

    assert fingerprint_int_arrays([arr]) == fingerprint_int_arrays([arr.copy()])
    assert fingerprint_int_arrays([arr]) != fingerprint_int_arrays([arr.astype(np.int64)])
