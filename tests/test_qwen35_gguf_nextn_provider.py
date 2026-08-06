"""GGUF trailing-NextN executors and candidate-only DraftModel provider gates."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.runtime import qwen35_gguf_nextn as nextn_mod
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNDraftProvider,
    Qwen35GGUFNextNExecutor,
    Qwen35GGUFNextNStateAdvance,
    Qwen35GGUFNextNStepResult,
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
        self.state_only_calls: list[tuple[int, int, int, int]] = []

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

    def advance_state_only(
        self,
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
    ) -> Qwen35GGUFNextNStateAdvance:
        self.state_only_calls.append((request_id, token_id, position, target_hidden.ptr))
        return Qwen35GGUFNextNStateAdvance(
            request_id=request_id,
            input_token=token_id,
            position=position,
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


def test_nextn_provider_prefers_an_executor_chain_without_changing_draft_abi() -> None:
    executor = _FakeExecutor()
    chain_calls: list[tuple[int, int, int, int, int, bool]] = []

    def run_chain(
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
        *,
        candidate_budget: int,
        return_logits: bool = False,
    ) -> tuple[Qwen35GGUFNextNStepResult, ...]:
        chain_calls.append(
            (
                request_id,
                token_id,
                position,
                target_hidden.ptr,
                candidate_budget,
                return_logits,
            )
        )
        rows = []
        hidden = target_hidden
        current = token_id
        for depth in range(candidate_budget):
            current += 1
            hidden = Tensor.from_handle(2000 + depth, (1, 8), DType.BF16, Device("hip", 0))
            rows.append(
                Qwen35GGUFNextNStepResult(
                    request_id=request_id,
                    input_token=current - 1,
                    position=position + depth,
                    token_id=current,
                    logit=float(current),
                    hidden=hidden,
                )
            )
        return tuple(rows)

    executor.run_chain = run_chain  # type: ignore[attr-defined]
    provider = Qwen35GGUFNextNDraftProvider(executor)
    target_hidden = Tensor.from_handle(77, (1, 8), DType.BF16, Device("hip", 0))

    draft = provider.propose(
        MtpProposalContext(
            request_ids=(41,),
            root_tokens=(9,),
            root_positions=(12,),
            target_hidden=target_hidden,
        ),
        candidate_budget=3,
    )

    assert chain_calls == [(41, 9, 12, 77, 3, False)]
    assert executor.calls == []
    assert draft.candidate_tokens == (10, 11, 12)
    assert draft.parent_positions == (12, 13, 14)
    assert draft.draft_depths == (1, 2, 3)
    assert tuple(row.input_token for row in provider.last_results[41]) == (9, 10, 11)


def test_nextn_executor_chain_falls_back_to_eager_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.closed = False
    executor.weights = object()
    executor.hidden_size = 8
    executor.vocab_size = 64
    executor.scratch = SimpleNamespace(max_positions=64)
    graph_calls: list[tuple[int, int, int, int, int]] = []
    step_calls: list[tuple[int, int, int, int, bool]] = []

    def graph_chain(request_id, token_id, position, target_hidden, *, candidate_budget):
        graph_calls.append((request_id, token_id, position, target_hidden.ptr, candidate_budget))
        return None

    def run_step(request_id, token_id, position, target_hidden, *, return_logits=False):
        step_calls.append((request_id, token_id, position, target_hidden.ptr, return_logits))
        hidden = Tensor.from_handle(3000 + len(step_calls), (1, 8), DType.BF16, Device("hip", 0))
        return Qwen35GGUFNextNStepResult(
            request_id=request_id,
            input_token=token_id,
            position=position,
            token_id=token_id + 1,
            logit=float(token_id + 1),
            hidden=hidden,
        )

    monkeypatch.setattr(executor, "_run_exact_graph_chain", graph_chain, raising=False)
    monkeypatch.setattr(executor, "run_step", run_step)
    hidden = Tensor.from_handle(77, (1, 8), DType.BF16, Device("hip", 0))

    rows = executor.run_chain(41, 9, 12, hidden, candidate_budget=3)

    assert graph_calls == [(41, 9, 12, 77, 3)]
    assert step_calls == [
        (41, 9, 12, 77, False),
        (41, 10, 13, 3001, False),
        (41, 11, 14, 3002, False),
    ]
    assert tuple(row.token_id for row in rows) == (10, 11, 12)


def test_nextn_proposal_graph_keeps_split_attention_context_on_eager_path() -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.scratch = SimpleNamespace(max_positions=2048)
    executor._proposal_target_hidden = object()
    executor._proposal_results = object()
    executor._proposal_results_host = object()
    executor._proposal_graph_last_status = "ready"
    hidden = Tensor.from_handle(77, (1, 8), DType.BF16, Device("hip", 0))

    rows = executor._run_exact_graph_chain(
        41,
        9,
        1021,
        hidden,
        candidate_budget=3,
    )

    assert rows is None
    assert executor._proposal_graph_last_status == "eager_long_context"


def test_nextn_proposal_graph_capture_failure_is_cached_as_eager_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor._proposal_graphs = {}
    executor._proposal_graph_unavailable = set()
    executor._proposal_graph_last_status = "ready"
    executor._proposal_graph_last_error = None
    executor._proposal_graph_captures = 0
    capture_calls: list[tuple[int, int, int]] = []

    def fail_capture(request_id: int, slot: int, budget: int):
        capture_calls.append((request_id, slot, budget))
        raise RuntimeError("unsupported graph")

    monkeypatch.setattr(executor, "_capture_exact_chain_graph", fail_capture)

    assert executor._proposal_graph(41, 0, 3) is None
    assert executor._proposal_graph(41, 0, 3) is None
    assert capture_calls == [(41, 0, 3)]
    assert executor._proposal_graph_unavailable == {(0, 3)}
    assert executor._proposal_graph_last_status == "capture_fallback"
    assert executor._proposal_graph_last_error == "RuntimeError: unsupported graph"


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
    assert isinstance(update, Qwen35GGUFNextNStateAdvance)
    assert len(executor.calls) == 2
    assert executor.state_only_calls == [(41, 11, 14, 1002)]
    assert update.input_token == 11
    with pytest.raises(ValueError, match="prior proposal"):
        provider.advance_full_accept_tail(42, accepted_count=0)
    with pytest.raises(ValueError, match="prior proposal budget"):
        provider.advance_full_accept_tail(41, accepted_count=3)


def test_nextn_executor_state_only_tail_skips_output_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    runtime_calls: list[str] = []
    block_calls: list[tuple[int, int, int, int]] = []
    executor.runtime = SimpleNamespace(
        device_synchronize=lambda: runtime_calls.append("synchronize")
    )
    hidden = Tensor.from_handle(0x7000, (1, 8), DType.BF16, Device("hip", 0))

    def fake_run_block(
        request_id: int,
        token_id: int,
        position: int,
        target_hidden: Tensor,
    ) -> tuple[int, int]:
        block_calls.append((request_id, token_id, position, target_hidden.ptr))
        return 0x8000, 0x9000

    monkeypatch.setattr(executor, "_run_block", fake_run_block, raising=False)
    monkeypatch.setattr(
        executor,
        "_sample_lm_head",
        lambda *args, **kwargs: pytest.fail("state-only tail must not sample lm-head"),
    )

    result = executor.advance_state_only(41, 11, 14, hidden)

    assert result == Qwen35GGUFNextNStateAdvance(
        request_id=41,
        input_token=11,
        position=14,
    )
    assert block_calls == [(41, 11, 14, 0x7000)]
    assert runtime_calls == ["synchronize"]


@pytest.mark.parametrize("quant_key", ["gguf_q6_k", "gguf_q6_k_t16_v1"])
def test_nextn_executor_prepares_exact_top1_through_quant_registry(
    monkeypatch: pytest.MonkeyPatch,
    quant_key: str,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    weight = SimpleNamespace(
        spec=SimpleNamespace(quant_key=quant_key),
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
        if name in {"raw", "tiles"}
        else (_ for _ in ()).throw(KeyError(name)),
    )
    executor.weights = SimpleNamespace(
        backend="hip_gfx1100",
        fallback=lambda slot: weight
        if slot == "lm_head"
        else (_ for _ in ()).throw(KeyError(slot)),
    )
    executor.hidden_size = 512
    executor.vocab_size = 1024
    executor.compiler_version = "compiler"
    executor.require_cached_build = True
    executor.runtime = object()
    executor._lm_head_top1_kernel = None
    executor._lm_head_top1_weight = None
    executor._lm_head_top1_block_values = None
    executor._lm_head_top1_block_indices = None
    executor._lm_head_top1_result = None
    executor._lm_head_top1_libraries = None
    kernel = object()
    pack8_library = object()
    t16_library = object()
    registered_keys: list[object] = []
    resolve_calls: list[dict[str, object]] = []
    malloc_calls: list[int] = []

    def fake_is_registered(key) -> bool:
        registered_keys.append(key)
        return True

    def fake_resolve(**kwargs):
        resolve_calls.append(kwargs)
        return kernel

    def fake_malloc(nbytes, *, runtime):
        assert runtime is executor.runtime
        malloc_calls.append(nbytes)
        return SimpleNamespace(ptr=0x3000 + len(malloc_calls) * 0x1000, nbytes=nbytes)

    monkeypatch.setattr(nextn_mod, "is_registered", fake_is_registered)
    monkeypatch.setattr(nextn_mod, "resolve", fake_resolve)
    monkeypatch.setattr(
        nextn_mod,
        "build_gguf_q6_k_pack8_gemv",
        lambda **kwargs: pack8_library,
    )
    monkeypatch.setattr(
        nextn_mod,
        "build_gguf_q6_k_t16_gemv",
        lambda **kwargs: t16_library,
        raising=False,
    )
    monkeypatch.setattr(nextn_mod, "malloc", fake_malloc)

    executor._prepare_exact_lm_head_top1()

    assert [(key.backend, key.layer, key.quant, key.variant) for key in registered_keys] == [
        (
            "hip_gfx1100",
            "linear+argmax",
            quant_key,
            "proposal_top1_exact_bf16",
        )
    ]
    assert resolve_calls == [
        {
            "backend": "hip_gfx1100",
            "layer": "linear+argmax",
            "quant": quant_key,
            "variant": "proposal_top1_exact_bf16",
        }
    ]
    assert malloc_calls == [512, 512, 8]
    assert executor._lm_head_top1_kernel is kernel
    assert executor._lm_head_top1_weight is weight
    assert executor._lm_head_top1_libraries == {
        "q6_pack8": pack8_library,
        "q6_t16": t16_library,
    }


def test_nextn_executor_exact_top1_reads_only_token_and_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    kernel_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sync_calls: list[str] = []
    weight = object()
    libraries = {"q6_pack8": object(), "q6_t16": object()}
    runtime = SimpleNamespace(device_synchronize=lambda: sync_calls.append("sync"))
    executor.runtime = runtime
    executor.hidden_size = 512
    executor.vocab_size = 1024
    executor._logits_buf = SimpleNamespace(ptr=0x6000)
    executor._lm_head_top1_kernel = lambda *args, **kwargs: kernel_calls.append((args, kwargs))
    executor._lm_head_top1_weight = weight
    executor._lm_head_top1_block_values = SimpleNamespace(ptr=0x3000)
    executor._lm_head_top1_block_indices = SimpleNamespace(ptr=0x4000)
    executor._lm_head_top1_result = SimpleNamespace(ptr=0x5000, nbytes=8)
    executor._lm_head_top1_libraries = libraries

    def fake_copy(host_ptr, _device, nbytes, *, runtime) -> None:
        assert nbytes == 8
        assert runtime is executor.runtime
        result = (ctypes.c_uint32 * 2).from_address(host_ptr)
        result[0] = 731
        result[1] = int(np.asarray([4.25], dtype=np.float32).view(np.uint32)[0])

    monkeypatch.setattr(nextn_mod, "copy_device_to_host", fake_copy)

    compact = executor._run_exact_lm_head_top1(0x1000, stream=7)

    assert compact == (731, 4.25)
    assert sync_calls == ["sync"]
    assert len(kernel_calls) == 1
    args, kwargs = kernel_calls[0]
    assert args == (
        weight,
        0x1000,
        0x6000,
        0x3000,
        0x4000,
        0x5000,
        0x5004,
        1,
        512,
        1024,
    )
    assert kwargs == {"stream": 7, "libraries": libraries, "runtime": runtime}


def test_nextn_executor_sample_prefers_compact_top1_without_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    compact_calls: list[tuple[int, int]] = []
    executor._run_exact_lm_head_top1 = lambda hidden_ptr, stream=0: (
        compact_calls.append((hidden_ptr, stream)) or (17, 3.5)
    )
    monkeypatch.setattr(
        nextn_mod,
        "launch_gguf_linear",
        lambda *args, **kwargs: pytest.fail("compact scoring must not launch full logits"),
    )

    token, logit, logits = executor._sample_lm_head(0x1234, return_logits=False, stream=9)

    assert (token, logit, logits) == (17, 3.5, None)
    assert compact_calls == [(0x1234, 9)]


def test_nextn_executor_logits_diagnostic_keeps_full_scoring_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object.__new__(Qwen35GGUFNextNExecutor)
    executor.hidden_size = 8
    executor.vocab_size = 4
    executor.weights = SimpleNamespace(fallback=lambda slot: f"weight:{slot}")
    executor._logits_buf = SimpleNamespace(ptr=0x6000, nbytes=16)
    executor._logits_host = np.empty((1, 4), dtype=np.float32)
    sync_calls: list[str] = []
    executor.runtime = SimpleNamespace(device_synchronize=lambda: sync_calls.append("sync"))
    executor._run_exact_lm_head_top1 = lambda *_args, **_kwargs: pytest.fail(
        "diagnostic logits must bypass compact scoring"
    )
    launch_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        nextn_mod,
        "launch_gguf_linear",
        lambda *args, **kwargs: launch_calls.append((args, kwargs)),
    )

    def fake_copy(host_ptr, _device, nbytes, *, runtime) -> None:
        assert nbytes == 16
        assert runtime is executor.runtime
        out = np.ctypeslib.as_array((ctypes.c_float * 4).from_address(host_ptr))
        out[:] = (1.0, 5.0, 3.0, 2.0)

    monkeypatch.setattr(nextn_mod, "copy_device_to_host", fake_copy)

    token, logit, logits = executor._sample_lm_head(0x7000, return_logits=True, stream=11)

    assert token == 1
    assert logit == 5.0
    np.testing.assert_array_equal(logits, np.asarray([[1.0, 5.0, 3.0, 2.0]], dtype=np.float32))
    assert sync_calls == ["sync"]
    assert len(launch_calls) == 1
    assert launch_calls[0][0][0:3] == ("weight:lm_head", 0x7000, 0x6000)
    assert launch_calls[0][1]["stream"] == 11


def _device_sha256(runtime, buffers: tuple[DeviceBuffer, ...]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for buffer in buffers:
        host = np.empty((buffer.nbytes,), dtype=np.uint8)
        copy_device_to_host(host_array_ptr(host), buffer, buffer.nbytes, runtime=runtime)
        digest.update(host)
    return digest.hexdigest()


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

        compact = executor.run_step(7, 11, 0, hidden, return_logits=False)
        assert compact.logits is None
        assert compact.token_id == direct.token_id
        assert compact.logit == direct.logit
        assert executor.last_lm_head_path == "exact_q6_top1"
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
        # Keep the allocation above the scalar-attention graph ceiling to prove
        # short live contexts still use the exact graph in a server-sized cache.
        max_positions=2048,
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

        compact = executor.run_step(7, 9707, 0, hidden, return_logits=False)
        assert compact.logits is None
        assert compact.token_id == direct.token_id
        assert compact.logit == direct.logit
        assert executor.last_lm_head_path == "exact_q6_top1"
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

        executor.reset_request(7)
        current_token = 9707
        current_hidden = hidden
        eager_rows = []
        for depth in range(3):
            row = executor.run_step(7, current_token, depth, current_hidden, return_logits=False)
            eager_rows.append(row)
            current_token = row.token_id
            current_hidden = row.hidden
        key_cache, value_cache = executor.scratch.full_cache(0)
        state_buffers = (
            DeviceBuffer(executor._final_hidden_buf.ptr, executor.hidden_size * DType.BF16.itemsize),
            key_cache,
            value_cache,
        )
        eager_state = _device_sha256(runtime, state_buffers)
        executor.reset_request(7)

        graph_rows = executor.run_chain(
            7,
            9707,
            0,
            hidden,
            candidate_budget=3,
            return_logits=False,
        )
        graph_state = _device_sha256(runtime, state_buffers)
        assert tuple(row.token_id for row in graph_rows) == tuple(row.token_id for row in eager_rows)
        assert tuple(row.logit for row in graph_rows) == tuple(row.logit for row in eager_rows)
        assert graph_state == eager_state
        assert executor.proposal_graph_contract()["replays"] >= 1
        assert executor.proposal_graph_contract()["last_status"] == "replay"
    finally:
        executor.close()
        free(hidden_buf, runtime=runtime)
