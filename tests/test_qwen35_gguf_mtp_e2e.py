"""End-to-end GGUF NextN target verify/accept/commit gates."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.kernels.hip_gfx1100.speculative.dflash_accept import dflash_accept_chain_i32
from hipengine.kernels.registry import resolve
from hipengine.kvcache import KVTransaction
import hipengine.runtime.gguf_linear as gguf_linear_module
from hipengine.runtime.qwen35_gguf_mtp import (
    Qwen35GGUFMTPDecodeSession,
    Qwen35GGUFTransactionalVerifier,
)
from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNDraftProvider
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.speculative import (
    DraftBatch,
    TargetCommitPlan,
    TargetVerifyBatch,
)

_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")
_DENSE_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _require_free_vram(gib: float) -> None:
    from hipengine.core.hip import get_hip_runtime

    free_bytes, _ = get_hip_runtime().mem_get_info()
    required = int(gib * 1024**3)
    if free_bytes < required:
        pytest.skip(
            f"GGUF MTP gate needs {gib:.1f} GiB free VRAM; "
            f"only {free_bytes / 1024**3:.2f} GiB available"
        )


def _target_batch(root: int, position: int, candidates: tuple[int, ...]) -> TargetVerifyBatch:
    request_id = 17
    draft = DraftBatch(
        request_ids=(request_id,),
        candidate_tokens=candidates,
        parent_positions=tuple(position + depth for depth in range(len(candidates))),
        draft_depths=tuple(range(1, len(candidates) + 1)),
        row_to_request=(request_id,) * len(candidates),
        mode="verify_chain",
    )
    return TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(root,),
        root_positions=(position,),
    )


def _wrong_token(token: int, vocab_size: int) -> int:
    return (int(token) + 1) % int(vocab_size)


def _copy_buffer(buffer: DeviceBuffer, *, nbytes: int | None = None) -> np.ndarray:
    count = int(buffer.nbytes if nbytes is None else nbytes)
    out = np.empty((count,), dtype=np.uint8)
    copy_device_to_host(host_array_ptr(out), DeviceBuffer(buffer.ptr, count), count)
    return out


def _assert_committed_state_matches(
    actual: Qwen35GGUFResidentSession,
    expected: Qwen35GGUFResidentSession,
) -> None:
    actual_owner = actual._target_scratch_owner
    expected_owner = expected._target_scratch_owner
    assert actual_owner is not None and expected_owner is not None
    assert actual.position == expected.position
    for left, right in zip(
        (*actual_owner.layer_conv_states, *actual_owner.layer_recurrent_states),
        (*expected_owner.layer_conv_states, *expected_owner.layer_recurrent_states),
        strict=True,
    ):
        if left is None or right is None:
            assert left is None and right is None
            continue
        np.testing.assert_array_equal(_copy_buffer(left), _copy_buffer(right))

    live_positions = int(actual.position)
    for left, right in zip(
        (*actual_owner.full_key_caches, *actual_owner.full_value_caches),
        (*expected_owner.full_key_caches, *expected_owner.full_value_caches),
        strict=True,
    ):
        if left is None or right is None:
            assert left is None and right is None
            continue
        row_nbytes = int(left.nbytes) // int(actual_owner.max_positions)
        live_nbytes = live_positions * row_nbytes
        np.testing.assert_array_equal(
            _copy_buffer(left, nbytes=live_nbytes),
            _copy_buffer(right, nbytes=live_nbytes),
        )
    hidden_nbytes = actual.last_target_hidden.shape[1] * 2
    np.testing.assert_array_equal(
        _copy_buffer(DeviceBuffer(actual.last_target_hidden.ptr, hidden_nbytes)),
        _copy_buffer(DeviceBuffer(expected.last_target_hidden.ptr, hidden_nbytes)),
    )


@pytest.mark.parametrize("quant", ("gguf_ud_q3_k_m", "gguf_q4_k_m"))
def test_gguf_mtp_uses_registered_shared_gpu_accept_route(quant: str) -> None:
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dflash_accept_chain",
            quant=quant,
            variant="i32",
        )
        is dflash_accept_chain_i32
    )


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_b1_b2_b3_target_logits_match_scalar_and_gpu_accept() -> None:
    """B=1/2/3 rows keep scalar logits and GPU/CPU accept summaries exact."""

    _require_free_vram(18.0)
    prompt = (9707, 11, 220, 264)
    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=64,
    ) as target:
        target.select_prefill_quant("gguf_ud_q3_k_m")
        assert target.runner is not None
        with Qwen35GGUFResidentSession(
            _MODEL,
            max_sequence_length=64,
            shared_runner=target.runner,
        ) as scalar, Qwen35GGUFTransactionalVerifier(
            target,
            max_candidate_budget=3,
        ) as verifier:
            for budget in (1, 2, 3):
                root = target.prefill(prompt, return_logits=False).token_id
                scalar_root = scalar.prefill(prompt, return_logits=False).token_id
                assert root == scalar_root
                oracle_tokens: list[int] = []
                scalar_logits: list[np.ndarray] = []
                current = int(root)
                scalar_step = scalar.step(current, return_logits=True)
                scalar_logits.append(scalar_step.logits)
                for _ in range(budget):
                    current = int(scalar_step.token_id)
                    oracle_tokens.append(current)
                    scalar_step = scalar.step(current, return_logits=True)
                    scalar_logits.append(scalar_step.logits)

                batch = _target_batch(int(root), len(prompt), tuple(oracle_tokens))
                key = ("verify_chain", budget)
                bucket = verifier.graph_bucket(key, batch)
                prepared = verifier.prepare(
                    batch,
                    transaction_id=budget,
                    graph_bucket=bucket,
                    remaining_decode=(budget + 1,),
                    return_logits=True,
                )
                try:
                    assert prepared.buffers.mode == "verify_chain"
                    assert prepared.gpu_accept_match_cpu
                    assert prepared.summary.accepted_counts == (budget,)
                    assert prepared.summary.full_accept == (True,)
                    assert prepared.target_top1 == tuple(
                        int(np.argmax(row[0])) for row in scalar_logits
                    )
                    np.testing.assert_array_equal(
                        prepared.target_logits,
                        np.concatenate(scalar_logits, axis=0),
                    )
                    spans = target.target_spans(slot_indices=(0,), span_role="verify_chain")
                    assert spans.span_role == "verify_chain"
                    assert spans.row_positions is not None
                finally:
                    verifier.rollback(prepared)
                assert target.position == len(prompt)


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_reject_partial_full_commit_state_and_kv_match_ar() -> None:
    """Reject/partial/full accept publishes only the selected target prefix."""

    _require_free_vram(18.0)
    prompt = (9707, 11, 220, 264)
    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=64,
    ) as target:
        target.select_prefill_quant("gguf_ud_q3_k_m")
        assert target.runner is not None
        with Qwen35GGUFResidentSession(
            _MODEL,
            max_sequence_length=64,
            shared_runner=target.runner,
        ) as reference, Qwen35GGUFTransactionalVerifier(
            target,
            max_candidate_budget=3,
        ) as verifier:
            for case, accepted_count in (("reject", 0), ("partial", 1), ("full", 3)):
                root = int(target.prefill(prompt, return_logits=False).token_id)
                assert root == int(reference.prefill(prompt, return_logits=False).token_id)
                oracle: list[int] = []
                current = root
                for _ in range(3):
                    current = int(reference.step(current, return_logits=False).token_id)
                    oracle.append(current)
                candidates = list(oracle)
                if accepted_count < 3:
                    candidates[accepted_count] = _wrong_token(
                        oracle[accepted_count],
                        target.runner.vocab_size,
                    )
                batch = _target_batch(root, len(prompt), tuple(candidates))
                bucket = verifier.graph_bucket(("commit", case), batch)
                prepared = verifier.prepare(
                    batch,
                    transaction_id=100 + accepted_count,
                    graph_bucket=bucket,
                    remaining_decode=(4,),
                    return_logits=False,
                )
                assert prepared.summary.accepted_counts == (accepted_count,)
                assert prepared.summary.full_accept == (accepted_count == 3,)
                plan = TargetCommitPlan(
                    transaction_id=100 + accepted_count,
                    request_ids=batch.request_ids,
                    accepted_counts=prepared.summary.accepted_counts,
                    commit_rows=prepared.summary.commit_rows,
                    commit_tokens=prepared.summary.commit_tokens,
                    commit_positions=prepared.summary.commit_positions,
                    next_tokens=prepared.summary.next_tokens,
                    candidate_counts=batch.candidate_counts,
                    draft_depth=batch.draft_depth,
                    tree_shape=batch.tree_shape,
                    mode=batch.mode,
                )
                state_buffers = verifier.commit(prepared, plan)
                assert state_buffers.has_hidden_taps
                verifier.finish(prepared)

                reference.prefill(prompt, return_logits=False)
                reference.step(root, return_logits=False)
                for token in candidates[:accepted_count]:
                    reference.step(token, return_logits=False)
                _assert_committed_state_matches(target, reference)
                assert target.position == len(prompt) + 1 + accepted_count

                correction = prepared.summary.next_tokens[0]
                assert correction is not None
                actual_next = target.step(int(correction), return_logits=True)
                expected_next = reference.step(int(correction), return_logits=True)
                assert actual_next.token_id == expected_next.token_id
                np.testing.assert_array_equal(actual_next.logits, expected_next.logits)
                target.reset()
                reference.reset()


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_real_nextn_chain_matches_mtp_disabled_greedy_output() -> None:
    """Real blk.40 proposal stays exact through scheduler verify/commit cycles."""

    _require_free_vram(21.0)
    prompt = (
        7734,
        264,
        12654,
        709,
        421,
        4523,
        279,
        307,
        12,
        337,
        76938,
        1324,
        1608,
        20781,
        1954,
        13,
        28763,
        264,
        4479,
        889,
        13,
    )
    decode_tokens = 16
    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=128,
    ) as target:
        target.select_prefill_quant("gguf_ud_q3_k_m")
        root = int(target.prefill(prompt, use_bulk=False, return_logits=False).token_id)
        expected = [root]
        while len(expected) < decode_tokens:
            expected.append(int(target.step(expected[-1], return_logits=False).token_id))

        provider = Qwen35GGUFNextNDraftProvider.from_model(
            _MODEL,
            max_positions=128,
            max_requests=1,
            runtime=target.runtime,
            require_cached_build=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
        )
        try:
            with Qwen35GGUFMTPDecodeSession(
                target,
                provider,
                candidate_budget=1,
            ) as decoder:
                actual = decoder.generate(
                    prompt,
                    max_new_tokens=decode_tokens,
                    use_bulk_prefill=False,
                )
        finally:
            provider.close()

    assert actual.token_ids == tuple(expected)
    assert actual.gpu_accept_match_cpu
    assert actual.accepted_draft_tokens >= 1
    assert actual.visible_tokens_per_cycle > 1.0
    assert actual.graph_stats["entries"] == 1
    assert actual.graph_stats["hits"] >= 1
    assert all(record["span_role"] == "verify_chain" for record in actual.cycle_records)
    assert all(int(record["transaction_id"]) >= 0 for record in actual.cycle_records)


@pytest.mark.skipif(not _DENSE_MODEL.exists(), reason=f"local GGUF fixture not found: {_DENSE_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_dense_q4_k_m_nextn_transaction_and_provider_match_scalar_ar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense B1-B3 rows and reject/partial/full commits stay target-exact."""

    _require_free_vram(32.0)
    rowtile_rows: set[int] = set()
    original_rowtile = gguf_linear_module.gguf_q4_k_pack8_rowtile_bf16_bf16_out

    def counted_rowtile(*args, **kwargs):
        rowtile_rows.add(int(args[5]))
        return original_rowtile(*args, **kwargs)

    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_pack8_rowtile_bf16_bf16_out",
        counted_rowtile,
    )
    prompt = (9707, 9707, 9707, 9707)
    require_cached = os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1"
    with Qwen35GGUFResidentSession(
        _DENSE_MODEL,
        max_sequence_length=64,
        require_cached_build=require_cached,
    ) as target:
        target.select_prefill_quant("gguf_q4_k_m")
        assert target.runner is not None
        with Qwen35GGUFResidentSession(
            _DENSE_MODEL,
            max_sequence_length=64,
            shared_runner=target.runner,
            require_cached_build=require_cached,
        ) as reference, Qwen35GGUFTransactionalVerifier(
            target,
            max_candidate_budget=3,
            quant="gguf_q4_k_m",
            target_verify_mode="native",
        ) as verifier:
            for budget in (1, 2, 3):
                target.reset()
                reference.reset()
                root = int(target.prefill(prompt, use_bulk=False, return_logits=False).token_id)
                assert root == int(
                    reference.prefill(prompt, use_bulk=False, return_logits=False).token_id
                )
                oracle_tokens: list[int] = []
                scalar_logits: list[np.ndarray] = []
                current = root
                scalar_step = reference.step(current, return_logits=True)
                scalar_logits.append(scalar_step.logits)
                for _ in range(budget):
                    current = int(scalar_step.token_id)
                    oracle_tokens.append(current)
                    scalar_step = reference.step(current, return_logits=True)
                    scalar_logits.append(scalar_step.logits)

                batch = _target_batch(root, len(prompt), tuple(oracle_tokens))
                bucket = verifier.graph_bucket(("dense-logits", budget), batch)
                prepared = verifier.prepare(
                    batch,
                    transaction_id=budget,
                    graph_bucket=bucket,
                    remaining_decode=(budget + 1,),
                    return_logits=True,
                )
                assert prepared.gpu_accept_match_cpu
                assert prepared.target_verify_mode == "native"
                assert prepared.summary.accepted_counts == (budget,)
                assert prepared.summary.full_accept == (True,)
                np.testing.assert_array_equal(
                    prepared.target_logits,
                    np.concatenate(scalar_logits, axis=0),
                )
                verifier.rollback(prepared)
                assert target.position == len(prompt)
                reference.reset()
                reference.prefill(prompt, use_bulk=False, return_logits=False)
                _assert_committed_state_matches(target, reference)

            for case, accepted_count in (("reject", 0), ("partial", 1), ("full", 3)):
                target.reset()
                reference.reset()
                root = int(target.prefill(prompt, use_bulk=False, return_logits=False).token_id)
                assert root == int(
                    reference.prefill(prompt, use_bulk=False, return_logits=False).token_id
                )
                oracle: list[int] = []
                current = root
                for _ in range(3):
                    current = int(reference.step(current, return_logits=False).token_id)
                    oracle.append(current)
                candidates = list(oracle)
                if accepted_count < 3:
                    candidates[accepted_count] = _wrong_token(
                        oracle[accepted_count],
                        target.runner.vocab_size,
                    )
                batch = _target_batch(root, len(prompt), tuple(candidates))
                transaction_id = 100 + accepted_count
                bucket = verifier.graph_bucket(("dense-commit", case), batch)
                prepared = verifier.prepare(
                    batch,
                    transaction_id=transaction_id,
                    graph_bucket=bucket,
                    remaining_decode=(4,),
                    return_logits=False,
                )
                assert prepared.summary.accepted_counts == (accepted_count,)
                assert prepared.summary.full_accept == (accepted_count == 3,)
                plan = TargetCommitPlan(
                    transaction_id=transaction_id,
                    request_ids=batch.request_ids,
                    accepted_counts=prepared.summary.accepted_counts,
                    commit_rows=prepared.summary.commit_rows,
                    commit_tokens=prepared.summary.commit_tokens,
                    commit_positions=prepared.summary.commit_positions,
                    next_tokens=prepared.summary.next_tokens,
                    candidate_counts=batch.candidate_counts,
                    draft_depth=batch.draft_depth,
                    tree_shape=batch.tree_shape,
                    mode=batch.mode,
                )
                state_buffers = verifier.commit(prepared, plan)
                assert state_buffers.has_hidden_taps
                verifier.finish(prepared)

                reference.reset()
                reference.prefill(prompt, use_bulk=False, return_logits=False)
                reference.step(root, return_logits=False)
                for token in candidates[:accepted_count]:
                    reference.step(token, return_logits=False)
                _assert_committed_state_matches(target, reference)
                assert target.position == len(prompt) + 1 + accepted_count

                correction = prepared.summary.next_tokens[0]
                assert correction is not None
                actual_next = target.step(int(correction), return_logits=True)
                expected_next = reference.step(int(correction), return_logits=True)
                assert actual_next.token_id == expected_next.token_id
                np.testing.assert_array_equal(actual_next.logits, expected_next.logits)

        natural_prompt = (
            7734,
            264,
            12654,
            709,
            421,
            4523,
            279,
            307,
            7324,
            76938,
            1324,
            1608,
            20781,
            1954,
            13,
            28763,
            264,
            4479,
            889,
            13,
        )
        target.reset()
        root = int(target.prefill(natural_prompt, use_bulk=False, return_logits=False).token_id)
        expected = [root]
        while len(expected) < 8:
            expected.append(int(target.step(expected[-1], return_logits=False).token_id))
        provider = Qwen35GGUFNextNDraftProvider.from_model(
            _DENSE_MODEL,
            max_positions=64,
            max_requests=1,
            runtime=target.runtime,
            require_cached_build=require_cached,
        )
        try:
            with Qwen35GGUFMTPDecodeSession(
                target,
                provider,
                candidate_budget=1,
                quant="gguf_q4_k_m",
                target_verify_mode="native",
            ) as decoder:
                actual = decoder.generate(
                    natural_prompt,
                    max_new_tokens=8,
                    use_bulk_prefill=False,
                )
        finally:
            provider.close()

    assert actual.token_ids == tuple(expected)
    assert actual.gpu_accept_match_cpu
    assert actual.accepted_draft_tokens >= 1
    assert any(record["draft_tail_advanced"] for record in actual.cycle_records)
    assert actual.graph_stats["entries"] == 1
    assert actual.graph_stats["hits"] >= 1
    assert all(record["quant"] == "gguf_q4_k_m" for record in actual.cycle_records)
    assert all(record["experts_per_token"] == 0 for record in actual.cycle_records)
    assert all(record["target_verify_mode"] == "native" for record in actual.cycle_records)
    assert all(record["span_role"] == "verify_chain" for record in actual.cycle_records)
    assert rowtile_rows == {2, 3, 4}


def test_target_commit_plan_fixture_keeps_shared_transaction_shape() -> None:
    plan = TargetCommitPlan(
        transaction_id=7,
        request_ids=(17,),
        accepted_counts=(2,),
        commit_rows=(2,),
        commit_tokens=(102,),
        commit_positions=(8,),
        candidate_counts=(3,),
        draft_depth=3,
        tree_shape=(0, 1, 2),
        mode="verify_chain",
    )
    transaction = KVTransaction(
        transaction_id=7,
        request_ids=(17,),
        draft_rows=3,
        role="verify_chain",
        candidate_counts=(3,),
    )

    assert plan.kv_accept_counts == (2,)
    assert transaction.role == plan.mode
    assert transaction.candidate_counts == plan.candidate_counts
