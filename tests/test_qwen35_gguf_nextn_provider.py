"""GGUF trailing-NextN executors and candidate-only DraftModel provider gates."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free, malloc
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNDraftProvider,
    Qwen35GGUFNextNStepResult,
    Qwen35GGUFNextNExecutor,
)
from hipengine.speculative import MtpDraftProvider, MtpProposalContext

_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")
_ORACLE = Path(__file__).parent / "fixtures" / "gguf" / "q3km_nextn_one_step_oracle.json"
_DENSE_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
_DENSE_ORACLE = (
    Path(__file__).parent / "fixtures" / "gguf" / "qwen36_27b_q4km_nextn_one_step_oracle.json"
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


class _FakeExecutor:
    hidden_size = 8

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int]] = []

    def run_step(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        return_logits: bool = False,
    ) -> Qwen35GGUFNextNStepResult:
        self.calls.append((request_id, token_id, position, target_hidden.ptr))
        next_token = token_id + 1
        logits = np.asarray([[float(token_id), float(next_token)]], dtype=np.float32) if return_logits else None
        hidden = Tensor.from_handle(1000 + len(self.calls), (1, 8), DType.BF16, Device("hip", 0))
        return Qwen35GGUFNextNStepResult(
            request_id=request_id,
            input_token=token_id,
            position=position,
            token_id=next_token,
            logit=float(next_token),
            hidden=hidden,
            logits=logits,
        )

    def reset_request(self, request_id: int) -> None:
        del request_id

    def close(self) -> None:
        return None


def test_nextn_provider_emits_only_candidate_rows_under_locked_abi() -> None:
    executor = _FakeExecutor()
    provider = Qwen35GGUFNextNDraftProvider(executor)
    assert isinstance(provider, MtpDraftProvider)
    target_hidden = Tensor.from_handle(77, (2, 8), DType.BF16, Device("hip", 0))
    context = MtpProposalContext(
        request_ids=(41, 42),
        root_tokens=(9, 19),
        root_positions=(12, 30),
        target_hidden=target_hidden,
    )

    draft = provider.propose(context, candidate_budget=2)

    assert draft.request_ids == (41, 42)
    assert draft.candidate_tokens == (10, 11, 20, 21)
    assert draft.parent_positions == (12, 13, 30, 31)
    assert draft.draft_depths == (1, 2, 1, 2)
    assert draft.row_to_request == (41, 41, 42, 42)
    assert draft.mode == "verify_chain"
    assert executor.calls == [
        (41, 9, 12, 77),
        (41, 10, 13, 1001),
        (42, 19, 30, 93),
        (42, 20, 31, 1003),
    ]
    with pytest.raises(ValueError, match="one of 1, 2, 3, 5"):
        provider.propose(context, candidate_budget=4)
    assert len(executor.calls) == 4


def test_nextn_provider_advances_only_a_fully_accepted_tail() -> None:
    executor = _FakeExecutor()
    provider = Qwen35GGUFNextNDraftProvider(executor)
    context = MtpProposalContext(
        request_ids=(41,),
        root_tokens=(9,),
        root_positions=(12,),
        target_hidden=Tensor.from_handle(77, (1, 8), DType.BF16, Device("hip", 0)),
    )
    provider.propose(context, candidate_budget=2)

    assert provider.advance_full_accept_tail(41, accepted_count=1) is None
    assert len(executor.calls) == 2
    update = provider.advance_full_accept_tail(41, accepted_count=2)
    assert update is not None
    assert executor.calls[-1] == (41, 11, 14, 1002)
    assert update.token_id == 12
    with pytest.raises(ValueError, match="prior proposal"):
        provider.advance_full_accept_tail(42, accepted_count=0)
    with pytest.raises(ValueError, match="prior proposal budget"):
        provider.advance_full_accept_tail(41, accepted_count=3)


def _require_real_nextn(model: Path) -> None:
    if not model.exists():
        pytest.skip(f"local GGUF fixture not found: {model}")
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    free_bytes, _ = get_hip_runtime().mem_get_info()
    if free_bytes < 3 * 1024**3:
        pytest.skip(f"GGUF NextN one-step gate needs 3 GiB free VRAM; only {free_bytes / 1024**3:.2f} GiB")


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
def test_real_blk40_one_step_logits_match_direct_executor_and_provider() -> None:
    _require_real_nextn(_MODEL)
    runtime = get_hip_runtime()
    hidden_buf = malloc(2048 * DType.BF16.itemsize, runtime=runtime)
    runtime.memset(hidden_buf.ptr, 0, hidden_buf.nbytes)
    hidden = Tensor.from_handle(hidden_buf.ptr, (1, 2048), DType.BF16, Device("hip", 0))
    executor = Qwen35GGUFNextNExecutor(
        _MODEL,
        max_positions=256,
        max_requests=1,
        runtime=runtime,
        require_cached_build=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )
    try:
        direct = executor.run_step(7, 11, 0, hidden, return_logits=True)
        assert direct.logits is not None
        assert direct.logits.shape == (1, executor.vocab_size)
        assert np.all(np.isfinite(direct.logits))
        oracle = json.loads(_ORACLE.read_text())
        top_ids = np.argpartition(direct.logits[0], -10)[-10:]
        top_ids = top_ids[np.argsort(direct.logits[0, top_ids])[::-1]]
        expected_ids = np.asarray([row[0] for row in oracle["top10"]], dtype=np.int64)
        expected_values = np.asarray([row[1] for row in oracle["top10"]], dtype=np.float32)
        np.testing.assert_array_equal(top_ids, expected_ids)
        tolerance = oracle["tolerance"]
        np.testing.assert_allclose(
            direct.logits[0, top_ids],
            expected_values,
            atol=tolerance["top10_logits_atol"],
            rtol=tolerance["top10_logits_rtol"],
        )
        assert direct.token_id == oracle["token_id"]
        executor.reset_request(7)

        provider = Qwen35GGUFNextNDraftProvider(executor)
        draft = provider.propose(
            MtpProposalContext(
                request_ids=(7,),
                root_tokens=(11,),
                root_positions=(0,),
                target_hidden=hidden,
            ),
            candidate_budget=1,
            return_logits=True,
        )
        proposed = provider.last_results[7][-1]
        assert proposed.logits is not None
        np.testing.assert_array_equal(proposed.logits, direct.logits)
        assert proposed.token_id == direct.token_id
        assert proposed.logit == direct.logit
        assert draft.candidate_tokens == (direct.token_id,)
        assert draft.parent_positions == (0,)
        assert draft.draft_depths == (1,)
    finally:
        executor.close()
        free(hidden_buf, runtime=runtime)


@pytest.mark.skipif(not _DENSE_MODEL.exists(), reason=f"local GGUF fixture not found: {_DENSE_MODEL}")
def test_real_dense_blk64_one_step_logits_match_llamacpp_oracle() -> None:
    _require_real_nextn(_DENSE_MODEL)
    runtime = get_hip_runtime()
    hidden_buf = malloc(5120 * DType.BF16.itemsize, runtime=runtime)
    runtime.memset(hidden_buf.ptr, 0, hidden_buf.nbytes)
    hidden = Tensor.from_handle(hidden_buf.ptr, (1, 5120), DType.BF16, Device("hip", 0))
    executor = Qwen35GGUFNextNExecutor(
        _DENSE_MODEL,
        max_positions=256,
        max_requests=1,
        runtime=runtime,
        require_cached_build=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )
    try:
        assert executor.weights is not None
        assert executor.weights.block_id == 64
        assert not executor.weights.config.is_moe
        direct = executor.run_step(7, 9707, 0, hidden, return_logits=True)
        assert direct.logits is not None
        assert direct.logits.shape == (1, executor.vocab_size)
        assert np.all(np.isfinite(direct.logits))
        oracle = json.loads(_DENSE_ORACLE.read_text())
        top_ids = np.argpartition(direct.logits[0], -10)[-10:]
        top_ids = top_ids[np.argsort(direct.logits[0, top_ids])[::-1]]
        expected_ids = np.asarray([row[0] for row in oracle["top10"]], dtype=np.int64)
        expected_values = np.asarray([row[1] for row in oracle["top10"]], dtype=np.float32)
        np.testing.assert_array_equal(top_ids, expected_ids)
        tolerance = oracle["tolerance"]
        np.testing.assert_allclose(
            direct.logits[0, top_ids],
            expected_values,
            atol=tolerance["top10_logits_atol"],
            rtol=tolerance["top10_logits_rtol"],
        )
        assert direct.token_id == oracle["token_id"]
        executor.reset_request(7)

        provider = Qwen35GGUFNextNDraftProvider(executor)
        draft = provider.propose(
            MtpProposalContext(
                request_ids=(7,),
                root_tokens=(9707,),
                root_positions=(0,),
                target_hidden=hidden,
            ),
            candidate_budget=1,
            return_logits=True,
        )
        proposed = provider.last_results[7][-1]
        assert proposed.logits is not None
        np.testing.assert_array_equal(proposed.logits, direct.logits)
        assert proposed.token_id == direct.token_id
        assert proposed.logit == direct.logit
        assert draft.candidate_tokens == (direct.token_id,)
        assert draft.parent_positions == (0,)
        assert draft.draft_depths == (1,)
    finally:
        executor.close()
        free(hidden_buf, runtime=runtime)
