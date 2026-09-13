"""Row-shaped resident GGUF target execution contracts."""

from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.runtime import qwen35_gguf_runner as runner_mod
from hipengine.runtime.qwen35_gguf_runner import (
    _GGUFPackedVerifySlotBlock,
    _build_gguf_packed_verify_layout,
    _capture_packed_verify_norm_rows,
    _stage_gguf_packed_verify_token_ids,
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


def test_packed_target_stages_device_candidates_without_host_materialization(
    monkeypatch,
) -> None:
    calls = []
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock((11, 0), 5),
            _GGUFPackedVerifySlotBlock((22, 0, 0), 8),
        )
    )
    jobs = (
        {
            "candidate_token_ids_device": Tensor.from_handle(
                0x1000,
                (1,),
                DType.INT32,
                Device("hip", 0),
            )
        },
        {
            "candidate_token_ids_device": Tensor.from_handle(
                0x2000,
                (2,),
                DType.INT32,
                Device("hip", 0),
            )
        },
    )
    monkeypatch.setattr(
        runner_mod,
        "copy_host_to_device",
        lambda destination, source, nbytes, *, runtime: calls.append(
            ("roots", destination.ptr, int(nbytes))
        ),
    )
    monkeypatch.setattr(
        runner_mod,
        "copy_i32_to_i64",
        lambda source, destination, rows, **kwargs: calls.append(
            ("candidates", int(source), int(destination), int(rows))
        ),
    )

    _stage_gguf_packed_verify_token_ids(
        layout,
        jobs,
        DeviceBuffer(0x3000, layout.rows * DType.INT64.itemsize),
        runtime=object(),
        stream=0,
        runtime_state_library=object(),
    )

    assert calls == [
        ("roots", 0x3000, 5 * DType.INT64.itemsize),
        ("candidates", 0x1000, 0x3000 + DType.INT64.itemsize, 1),
        ("candidates", 0x2000, 0x3000 + 3 * DType.INT64.itemsize, 2),
    ]


def test_packed_verify_norm_capture_writes_aligned_diagnostic_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    layout = _build_gguf_packed_verify_layout(
        (
            _GGUFPackedVerifySlotBlock((11, 12), 5),
            _GGUFPackedVerifySlotBlock((22, 23, 24), 8),
        )
    )
    source = np.arange(layout.rows * 4, dtype=np.uint16).reshape(layout.rows, 4)

    def fake_copy(destination, source_buffer, nbytes, *, runtime):
        assert source_buffer.ptr == 0x5000
        ctypes.memmove(destination, source.ctypes.data, nbytes)

    monkeypatch.setenv("HIPENGINE_GGUF_PACKED_VERIFY_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(runner_mod, "copy_device_to_host", fake_copy)
    _capture_packed_verify_norm_rows(
        layout,
        (
            {"request_id": 10, "transaction_id": 100},
            {"request_id": 20, "transaction_id": 200},
        ),
        DeviceBuffer(0x5000, source.nbytes),
        hidden_size=4,
        runtime=object(),
        sequence=3,
    )

    with np.load(tmp_path / "packed-verify-000003.npz") as capture:
        np.testing.assert_array_equal(capture["norm_bf16_bits"], source)
        np.testing.assert_array_equal(
            capture["input_token_ids"], layout.input_token_ids
        )
        np.testing.assert_array_equal(capture["row_positions"], layout.row_positions)
        np.testing.assert_array_equal(
            capture["slot_active_mask"], layout.active_mask
        )
        np.testing.assert_array_equal(capture["request_ids"], (10, 20))
        np.testing.assert_array_equal(capture["transaction_ids"], (100, 200))


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
def test_ud_q3_k_m_c2_native_rows_match_independent_c1_boundaries_and_logits(
    require_q3_model_vram: None,
) -> None:
    """Native indexed-state/attention/MoE rows must be full-logit exact."""

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
                actual = batch.step_rows_native(
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
                assert actual.execution_paths == {
                    "linear_attention": "indexed_conv_gdn",
                    "full_attention": "kv_live_spans_batch_c1_exact",
                    "moe": "selected_rows_batch",
                    "lm_head": "row_linear_f32",
                    "sampler": "host_argmax_full_logits",
                }

            assert batch.row_positions == (2, 2)


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_c2_slot_prefill_then_native_decode_matches_c1(
    require_q3_model_vram: None,
) -> None:
    """Slot-local bulk prefill must seed the exact native batch decode state."""

    prompts = ((9707, 11, 220, 264), (11, 220, 264, 9707))
    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=64,
        max_batch_size=2,
    ) as batch:
        batch.select_prefill_quant("gguf_ud_q3_k_m")
        assert batch.runner is not None
        with ExitStack() as stack:
            refs = tuple(
                stack.enter_context(
                    Qwen35GGUFResidentSession(
                        _MODEL,
                        max_sequence_length=64,
                        shared_runner=batch.runner,
                    )
                )
                for _ in range(2)
            )
            first_tokens: list[int] = []
            for slot, (prompt, ref) in enumerate(zip(prompts, refs, strict=True)):
                actual_prefill = batch.prefill_slot(
                    prompt,
                    slot=slot,
                    return_logits=True,
                )
                expected_prefill = ref.prefill(prompt, return_logits=True)
                assert actual_prefill.token_id == expected_prefill.token_id
                np.testing.assert_array_equal(
                    actual_prefill.logits,
                    expected_prefill.logits,
                )
                first_tokens.append(actual_prefill.token_id)

            actual = batch.step_rows_native(first_tokens, return_logits=True)
            expected = [
                ref.step(token, return_logits=True)
                for ref, token in zip(refs, first_tokens, strict=True)
            ]
            assert actual.token_ids == tuple(result.token_id for result in expected)
            np.testing.assert_array_equal(
                actual.logits,
                np.concatenate([result.logits for result in expected], axis=0),
            )
            assert batch.row_positions == (5, 5)


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_c2_split_attention_after_prefill_matches_c1_full_logits(
    require_q3_model_vram: None,
) -> None:
    """The long-context native reducer must use contiguous row/head gate strides."""

    seed = (9707, 11, 220, 264)
    prompts = tuple(
        tuple(seed[(index + slot) % len(seed)] for index in range(1024))
        for slot in range(2)
    )
    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=1280,
        max_batch_size=2,
    ) as batch:
        batch.select_prefill_quant("gguf_ud_q3_k_m")
        assert batch.runner is not None
        with ExitStack() as stack:
            refs = tuple(
                stack.enter_context(
                    Qwen35GGUFResidentSession(
                        _MODEL,
                        max_sequence_length=1280,
                        shared_runner=batch.runner,
                    )
                )
                for _ in range(2)
            )
            first_tokens: list[int] = []
            for slot, (prompt, ref) in enumerate(zip(prompts, refs, strict=True)):
                actual_prefill = batch.prefill_slot(prompt, slot=slot, return_logits=True)
                expected_prefill = ref.prefill(prompt, return_logits=True)
                assert actual_prefill.token_id == expected_prefill.token_id
                np.testing.assert_array_equal(actual_prefill.logits, expected_prefill.logits)
                first_tokens.append(actual_prefill.token_id)

            actual = batch.step_rows_native(first_tokens, return_logits=True)
            expected = [
                ref.step(token, return_logits=True)
                for ref, token in zip(refs, first_tokens, strict=True)
            ]
            assert actual.execution_paths["full_attention"] == "kv_live_spans_batch_split_gqa"
            assert actual.token_ids == tuple(result.token_id for result in expected)
            np.testing.assert_array_equal(
                actual.logits,
                np.concatenate([result.logits for result in expected], axis=0),
            )
            assert batch.row_positions == (1025, 1025)


@pytest.mark.parametrize("rows", (4, 8))
@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_native_rows_c4_c8_match_independent_c1_full_logits(
    rows: int,
    require_q3_model_vram: None,
) -> None:
    """C=4/8 native rows must preserve every logit and generated id."""

    tokens = (9707, 11, 220, 264, 9707, 11, 220, 264)[:rows]
    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=64,
        max_batch_size=rows,
    ) as batch:
        batch.select_prefill_quant("gguf_ud_q3_k_m")
        assert batch.runner is not None
        actual = batch.step_rows_native(tokens, return_logits=True)
        expected_logits: list[np.ndarray] = []
        expected_tokens: list[int] = []
        for token in tokens:
            with Qwen35GGUFResidentSession(
                _MODEL,
                max_sequence_length=64,
                shared_runner=batch.runner,
            ) as scalar:
                expected = scalar.step(token, return_logits=True)
                expected_tokens.append(expected.token_id)
                expected_logits.append(expected.logits)

        assert actual.token_ids == tuple(expected_tokens)
        np.testing.assert_array_equal(
            actual.logits,
            np.concatenate(expected_logits, axis=0),
        )
        assert "fallback" not in " ".join(actual.execution_paths.values())
        assert actual.execution_paths["linear_attention"] == "indexed_conv_gdn"
        assert actual.execution_paths["moe"] == "selected_rows_batch"


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_variable_short_slot_prefill_matches_independent_c1(
    require_q3_model_vram: None,
) -> None:
    """Physical cache ids and slot-local cache pointers must not be mixed."""

    prompts = (
        (760, 4087, 369),
        (657, 799, 1829, 13),
        (17, 10, 17, 28),
        (9419,),
    )
    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=256,
        max_batch_size=4,
    ) as batch:
        batch.select_prefill_quant("gguf_ud_q3_k_m")
        first_tokens = tuple(
            batch.prefill_slot(prompt, slot=slot, return_logits=False).token_id
            for slot, prompt in enumerate(prompts)
        )
        assert batch.row_positions == (3, 4, 4, 1)
        actual = batch.step_rows_native(first_tokens, return_logits=True)
        assert batch.runner is not None
        expected_tokens: list[int] = []
        expected_logits: list[np.ndarray] = []
        for prompt, first_token in zip(prompts, first_tokens, strict=True):
            with Qwen35GGUFResidentSession(
                _MODEL,
                max_sequence_length=256,
                shared_runner=batch.runner,
            ) as reference:
                reference_first = reference.prefill(prompt, return_logits=False)
                assert reference_first.token_id == first_token
                expected = reference.step(first_token, return_logits=True)
                expected_tokens.append(expected.token_id)
                expected_logits.append(expected.logits)
        assert actual.token_ids == tuple(expected_tokens)
        np.testing.assert_array_equal(actual.logits, np.concatenate(expected_logits, axis=0))
        assert batch.row_positions == (4, 5, 5, 2)


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_reclaim_compact_and_readmit_matches_c1_full_logits(
    require_q3_model_vram: None,
) -> None:
    """A surviving row keeps exact state/KV after a hole is reclaimed."""

    prompts = (
        (9707, 11, 220, 264),
        (11, 220, 264, 9707),
        (220, 264, 9707, 11),
    )
    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=64,
        max_batch_size=2,
    ) as batch:
        batch.select_prefill_quant("gguf_ud_q3_k_m")
        assert batch.runner is not None
        with Qwen35GGUFResidentSession(
            _MODEL,
            max_sequence_length=64,
            shared_runner=batch.runner,
        ) as survivor_ref, Qwen35GGUFResidentSession(
            _MODEL,
            max_sequence_length=64,
            shared_runner=batch.runner,
        ) as admitted_ref:
            first_a = batch.prefill_slot(prompts[0], slot=0, return_logits=False).token_id
            first_b = batch.prefill_slot(prompts[1], slot=1, return_logits=True)
            expected_first_b = survivor_ref.prefill(prompts[1], return_logits=True)
            assert first_b.token_id == expected_first_b.token_id
            np.testing.assert_array_equal(first_b.logits, expected_first_b.logits)

            before_reclaim = batch.step_rows_native(
                (first_a, first_b.token_id),
                return_logits=True,
            )
            expected_survivor = survivor_ref.step(first_b.token_id, return_logits=True)
            assert before_reclaim.token_ids[1] == expected_survivor.token_id
            np.testing.assert_array_equal(before_reclaim.logits[1], expected_survivor.logits[0])

            assert batch.compact_target_slots((1,)) == ((1, 0),)
            assert batch.row_positions == (5, 0)
            first_c = batch.prefill_slot(prompts[2], slot=1, return_logits=True)
            expected_first_c = admitted_ref.prefill(prompts[2], return_logits=True)
            assert first_c.token_id == expected_first_c.token_id
            np.testing.assert_array_equal(first_c.logits, expected_first_c.logits)

            after_readmit = batch.step_rows_native(
                (before_reclaim.token_ids[1], first_c.token_id),
                return_logits=True,
            )
            expected_after_readmit = (
                survivor_ref.step(before_reclaim.token_ids[1], return_logits=True),
                admitted_ref.step(first_c.token_id, return_logits=True),
            )
            assert after_readmit.token_ids == tuple(
                result.token_id for result in expected_after_readmit
            )
            np.testing.assert_array_equal(
                after_readmit.logits,
                np.concatenate(
                    [result.logits for result in expected_after_readmit],
                    axis=0,
                ),
            )
            assert batch.row_positions == (6, 5)
            assert after_readmit.execution_paths["linear_attention"] == "indexed_conv_gdn"
            assert after_readmit.execution_paths["moe"] == "selected_rows_batch"


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
