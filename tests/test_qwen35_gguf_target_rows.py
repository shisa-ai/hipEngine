"""Row-shaped resident GGUF target execution contracts."""

from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFResidentSession,
    Qwen35GGUFResidentTargetLayout,
)
from hipengine.speculative import TargetVerifyBatch

_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")
_MIN_FREE_VRAM_BYTES = 18 * 1024**3


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.fixture
def require_q3_model_vram() -> None:
    from hipengine.core.hip import get_hip_runtime

    free_bytes, _ = get_hip_runtime().mem_get_info()
    if free_bytes < _MIN_FREE_VRAM_BYTES:
        pytest.skip(
            f"UD-Q3_K_M target-row parity needs 18 GiB free VRAM; "
            f"only {free_bytes / 1024**3:.2f} GiB available"
        )


def test_gguf_resident_target_layout_is_row_shaped() -> None:
    layout = Qwen35GGUFResidentTargetLayout(
        max_batch_size=2,
        hidden_size=2048,
        vocab_size=248320,
        max_sequence_length=300,
        block_size=256,
    )

    assert layout.token_shape == (2,)
    assert layout.position_shape == (2,)
    assert layout.hidden_shape == (2, 2048)
    assert layout.logits_shape == (2, 248320)
    assert layout.blocks_per_slot == 2
    assert layout.block_table_shape == (2, 2)
    assert layout.slot0_hidden_shape == (1, 2048)


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_c2_target_rows_match_independent_c1_boundaries_and_logits(
    require_q3_model_vram: None,
) -> None:
    """RED: C=2 must preserve the independent c=1 layer/logit contract."""

    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=64,
        max_batch_size=2,
    ) as batch:
        batch.select_prefill_quant("gguf_ud_q3_k_m")
        assert batch.runner is not None
        with ExitStack() as stack:
            scalar_refs = tuple(
                stack.enter_context(
                    Qwen35GGUFResidentSession(
                        _MODEL,
                        max_sequence_length=64,
                        shared_runner=batch.runner,
                    )
                )
                for _ in range(2)
            )
            boundary_refs = tuple(
                stack.enter_context(
                    Qwen35GGUFResidentSession(
                        _MODEL,
                        max_sequence_length=64,
                        shared_runner=batch.runner,
                    )
                )
                for _ in range(2)
            )
            for tokens in ((9707, 11), (220, 264)):
                actual = batch.step_rows(
                    tokens,
                    return_logits=True,
                    capture_layer_ids=(0, 3, 39),
                )
                expected = [
                    ref.step(token, return_logits=True)
                    for ref, token in zip(scalar_refs, tokens, strict=True)
                ]
                expected_boundaries = [
                    ref.step_rows(
                        (token,),
                        return_logits=False,
                        capture_layer_ids=(0, 3, 39),
                    )
                    for ref, token in zip(boundary_refs, tokens, strict=True)
                ]

                assert actual.slot_indices == (0, 1)
                assert actual.span_role == "decode"
                assert actual.token_ids == tuple(result.token_id for result in expected)
                np.testing.assert_array_equal(
                    actual.logits,
                    np.concatenate([result.logits for result in expected], axis=0),
                )
                for layer_id in (0, 3, 39):
                    np.testing.assert_array_equal(
                        actual.layer_hidden_bits[layer_id],
                        np.concatenate(
                            [
                                result.layer_hidden_bits[layer_id]
                                for result in expected_boundaries
                            ],
                            axis=0,
                        ),
                    )

            assert batch.row_positions == (2, 2)
            assert tuple(ref.position for ref in scalar_refs) == (2, 2)
            assert tuple(ref.position for ref in boundary_refs) == (2, 2)
            spans = batch.target_spans(slot_indices=(0, 1), span_role="decode")
            assert spans.span_role == "decode"
            assert spans.base_offsets.shape == (2, 1)
            assert spans.live_counts.shape == (2,)
            assert spans.row_positions is not None
            assert spans.row_positions.shape == (2,)


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_v2_verify_chain_uses_same_target_rows_as_c1(
    require_q3_model_vram: None,
) -> None:
    """RED: a root+candidate V=2 chain uses the shared target-row executor."""

    batch = TargetVerifyBatch(
        request_ids=(17,),
        tokens=(9707, 11),
        positions=(0, 1),
        row_to_request=(17, 17),
        parent_rows=(-1, 0),
        root_rows=(0,),
        candidate_rows=(1,),
        draft_depths=(0, 1),
        active_mask=(True, True),
        mode="verify_chain",
    )

    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=64,
        max_batch_size=1,
    ) as verifier:
        verifier.select_prefill_quant("gguf_ud_q3_k_m")
        assert verifier.runner is not None
        with Qwen35GGUFResidentSession(
            _MODEL,
            max_sequence_length=64,
            shared_runner=verifier.runner,
        ) as scalar_reference, Qwen35GGUFResidentSession(
            _MODEL,
            max_sequence_length=64,
            shared_runner=verifier.runner,
        ) as boundary_reference:
            actual = verifier.verify_rows(
                batch,
                return_logits=True,
                capture_layer_ids=(0, 3, 39),
            )
            expected = [
                scalar_reference.step(token, return_logits=True)
                for token in batch.tokens
            ]
            expected_boundaries = [
                boundary_reference.step_rows(
                    (token,),
                    return_logits=False,
                    capture_layer_ids=(0, 3, 39),
                )
                for token in batch.tokens
            ]

            assert actual.span_role == "verify_chain"
            assert actual.positions == batch.positions
            assert actual.token_ids == tuple(result.token_id for result in expected)
            np.testing.assert_array_equal(
                actual.logits,
                np.concatenate([result.logits for result in expected], axis=0),
            )
            for layer_id in (0, 3, 39):
                np.testing.assert_array_equal(
                    actual.layer_hidden_bits[layer_id],
                    np.concatenate(
                        [
                            result.layer_hidden_bits[layer_id]
                            for result in expected_boundaries
                        ],
                        axis=0,
                    ),
                )

            spans = verifier.target_spans(slot_indices=(0,), span_role="verify_chain")
            assert spans.span_role == "verify_chain"
