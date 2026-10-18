"""Dense INT8 MTP transaction gate against the same INT8 autoregressive model."""

import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.runtime import qwen35_gguf_runner as runner_module
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNDraftProvider,
    borrow_qwen35_gguf_nextn_fallback_weights,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.speculative import DraftBatch, TargetCommitPlan, TargetVerifyBatch


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


_MODEL = Path(os.environ.get("HIPENGINE_INT8_MTP_MODEL", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
pytestmark = pytest.mark.skipif(
    not _hip_available() or not _MODEL.is_file(), reason="HIP or dense GGUF model unavailable",
)


@pytest.fixture(autouse=True)
def restore_kernel_registrations(sessions):
    from hipengine.speculative.native_cycle_graph import register_native_spec_gguf_graphs

    load_backend_kernel_package(sessions[0].backend)
    register_native_spec_gguf_graphs()


@pytest.fixture(scope="module")
def sessions():
    # Exercise compact storage itself, not the legacy short-context BF16 mirror.
    # This diagnostic does not publish or forge artifact qualification.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner_module, "_GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS", 0)
        patch.setenv("HIPENGINE_GGUF_INT8_KV_BF16_PREFIX_FULL_LAYERS", "0")
        patch.setenv("HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG", "1")
        kwargs = {
            "max_sequence_length": 1536, "kv_scale_dtype": DType.FP32,
            "kv_policy": FixedPagedKVPolicy(block_size=256, storage_dtype=DType.INT8_PER_TOKEN_HEAD),
        }
        with Qwen35GGUFResidentSession(_MODEL, **kwargs) as target:
            target.select_prefill_quant("gguf_q4_k_m")
            with Qwen35GGUFResidentSession(_MODEL, shared_runner=target.runner, **kwargs) as reference:
                assert all(buf is None for buf in target.scratch.full_bf16_mirror_key_caches)
                yield target, reference


def _read(buffer, size=None):
    size = buffer.nbytes if size is None else size
    host = np.empty(size, dtype=np.uint8)
    copy_device_to_host(host_array_ptr(host), DeviceBuffer(buffer.ptr, size), size)
    return host


def _assert_live_state(target, reference):
    assert target.position == reference.position
    for name in ("layer_conv_states", "layer_recurrent_states", "full_key_caches", "full_value_caches"):
        for actual, expected in zip(getattr(target.scratch, name), getattr(reference.scratch, name), strict=True):
            if actual is None:
                assert expected is None
                continue
            size = actual.nbytes
            if name.startswith("full_"):
                size = target.position * (size // target.scratch.max_positions)
            np.testing.assert_array_equal(_read(actual, size), _read(expected, size), err_msg=name)
    for actual, expected in zip(
        target.scratch.full_kv_scale_metadata, reference.scratch.full_kv_scale_metadata, strict=True,
    ):
        if actual is None:
            assert expected is None
            continue
        for name in ("k_scale", "v_scale"):
            left, right = getattr(actual, name), getattr(expected, name)
            size = target.position * 4 * left.dtype.itemsize
            np.testing.assert_array_equal(
                _read(DeviceBuffer(left.ptr, size)), _read(DeviceBuffer(right.ptr, size)), err_msg=name,
            )


@pytest.mark.parametrize("accepted", [0, 1, 3])
@pytest.mark.parametrize("graph", [False, True])
def test_int8_mtp_commit_matches_int8_ar(sessions, accepted, graph):
    _run_commit_case(sessions, accepted, graph, (9707, 11, 220, 264))


@pytest.mark.parametrize("position,graph", [(254, False), (254, True), (1022, False), (1280, True)])
def test_int8_mtp_commit_at_page_and_split_transitions(sessions, position, graph):
    _run_commit_case(sessions, 1, graph, (9707,) * position)


def _run_commit_case(sessions, accepted, graph, prompt):
    target, reference = sessions
    bulk = len(prompt) > 4
    root = int(target.prefill(prompt, use_bulk=bulk, return_logits=False).token_id)
    assert root == int(reference.prefill(prompt, use_bulk=bulk, return_logits=False).token_id)
    oracle = []
    token = root
    for _ in range(3):
        token = int(reference.step(token, return_logits=False).token_id)
        oracle.append(token)
    candidates = list(oracle)
    if accepted < 3:
        candidates[accepted] = (candidates[accepted] + 1) % target.runner.vocab_size
    draft = DraftBatch(
        request_ids=(17,), candidate_tokens=tuple(candidates),
        parent_positions=tuple(len(prompt) + i for i in range(3)),
        draft_depths=(1, 2, 3), row_to_request=(17, 17, 17), mode="verify_chain",
    )
    batch = TargetVerifyBatch.from_draft(draft, root_tokens=(root,), root_positions=(len(prompt),))
    with Qwen35GGUFTransactionalVerifier(
        target, max_candidate_budget=3, quant="gguf_q4_k_m", target_verify_mode="native",
    ) as verifier:
        prepared = verifier.prepare(
            batch, transaction_id=1, graph_bucket=verifier.graph_bucket("int8", batch),
            remaining_decode=(4,), allow_graph=graph,
        )
        assert prepared.summary.accepted_counts == (accepted,)
        assert prepared.gpu_accept_match_cpu
        if graph:
            assert target.last_native_spec_target_submitted, target.last_native_spec_target_fallback_reason
        plan = TargetCommitPlan(
            transaction_id=1, request_ids=batch.request_ids,
            accepted_counts=prepared.summary.accepted_counts,
            commit_rows=prepared.summary.commit_rows, commit_tokens=prepared.summary.commit_tokens,
            commit_positions=prepared.summary.commit_positions, next_tokens=prepared.summary.next_tokens,
            candidate_counts=batch.candidate_counts, draft_depth=batch.draft_depth,
            tree_shape=batch.tree_shape, mode=batch.mode,
        )
        verifier.commit(prepared, plan)
        verifier.finish(prepared)
    reference.prefill(prompt, use_bulk=bulk, return_logits=False)
    reference.step(root, return_logits=False)
    for token in candidates[:accepted]:
        reference.step(token, return_logits=False)
    _assert_live_state(target, reference)
    correction = int(prepared.summary.next_tokens[0])
    actual = target.step(correction, return_logits=True)
    expected = reference.step(correction, return_logits=True)
    np.testing.assert_array_equal(actual.logits, expected.logits)


@pytest.mark.parametrize("graph", [False, True])
def test_int8_mtp_rollback_restores_live_payload_and_scales(sessions, graph):
    target, reference = sessions
    prompt = (9707, 11, 220, 264)
    root = int(target.prefill(prompt, use_bulk=False, return_logits=False).token_id)
    reference.prefill(prompt, use_bulk=False, return_logits=False)
    draft = DraftBatch(
        request_ids=(17,), candidate_tokens=(1, 2, 3),
        parent_positions=(4, 5, 6), draft_depths=(1, 2, 3),
        row_to_request=(17, 17, 17), mode="verify_chain",
    )
    batch = TargetVerifyBatch.from_draft(draft, root_tokens=(root,), root_positions=(4,))
    with Qwen35GGUFTransactionalVerifier(
        target, max_candidate_budget=3, quant="gguf_q4_k_m", target_verify_mode="native",
    ) as verifier:
        prepared = verifier.prepare(
            batch, transaction_id=2, graph_bucket=verifier.graph_bucket("rollback", batch),
            remaining_decode=(4,), allow_graph=graph,
        )
        verifier.rollback(prepared)
    _assert_live_state(target, reference)
    np.testing.assert_array_equal(
        target.step(root, return_logits=True).logits,
        reference.step(root, return_logits=True).logits,
    )


def _prompts():
    root = Path(__file__).resolve().parents[1] / "benchmarks/prompts"
    rows = []
    for filename in ("mtpbench-code-general-ja.jsonl", "gdn-prefill-category-heldouts.jsonl"):
        rows.extend(json.loads(line) for line in (root / filename).read_text().splitlines() if line.strip())
    return rows


@pytest.fixture(scope="module")
def provider(sessions):
    target, _ = sessions
    value = Qwen35GGUFNextNDraftProvider.from_model(
        _MODEL, max_positions=target.scratch.max_positions, max_requests=1,
        runtime=target.runtime,
        borrowed_fallback_weights=borrow_qwen35_gguf_nextn_fallback_weights(target),
    )
    try:
        yield value
    finally:
        value.close()


@pytest.mark.parametrize("prompt_row", _prompts(), ids=lambda row: row["id"])
def test_int8_real_nextn_matches_ar_category_suite(sessions, provider, prompt_row):
    from hipengine.loading import load_gguf_index
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    target, reference = sessions
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(_MODEL))
    text = "\n".join(message["content"] for message in prompt_row["messages"])
    prompt = tuple(tokenizer.encode(text))
    length = 24
    token = int(reference.prefill(prompt, use_bulk=True, return_logits=False).token_id)
    expected = [token]
    for _ in range(length - 1):
        token = int(reference.step(token, return_logits=False).token_id)
        expected.append(token)
    with Qwen35GGUFMTPDecodeSession(
        target, provider, candidate_budget=3, quant="gguf_q4_k_m", target_verify_mode="native",
    ) as decoder:
        result = decoder.generate(prompt, max_new_tokens=length, use_bulk_prefill=True, prefill_draft=True)
    assert result.token_ids == tuple(expected)
    assert result.cycles > 0
    assert result.gpu_accept_match_cpu
