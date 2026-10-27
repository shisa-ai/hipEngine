"""The predictor input is [embedding, branch-hidden] for every row/branch."""
import numpy as np
import pytest

from hipengine.runtime.qwen4_exp_mtp_prefill import prompt_fusion_indices


@pytest.mark.parametrize("rows,branches", [(1, 1), (3, 4), (17, 2), (64, 4)])
def test_prompt_fusion_gather_keeps_row_and_branch_order(rows, branches):
    indices = prompt_fusion_indices(rows, branches)
    embedding = np.arange(rows * 5).reshape(rows, 5)
    hidden = 1000 + np.arange(rows * branches * 5).reshape(rows * branches, 5)
    source = np.concatenate((embedding, hidden), axis=0)
    got = source[indices].reshape(rows, branches, 10)
    for row in range(rows):
        for branch in range(branches):
            np.testing.assert_array_equal(got[row, branch, :5], embedding[row])
            np.testing.assert_array_equal(got[row, branch, 5:], hidden[row * branches + branch])
    assert indices.dtype == np.int64
    assert indices.min() >= 0 and indices.max() < 2 * rows * branches


@pytest.mark.parametrize("rows,branches", [(0, 4), (1, 0), (-1, 2)])
def test_prompt_fusion_gather_rejects_empty_geometry(rows, branches):
    with pytest.raises(ValueError, match="positive"):
        prompt_fusion_indices(rows, branches)


def test_target_consumer_borrows_each_chunk_outside_projection_context(monkeypatch):
    from tests.test_unit_qwen4exp_prefill_chunks import _target
    from hipengine.runtime.gguf_linear import raw_k_prefill_rowbatch

    runner = _target(monkeypatch)
    runner.config.residual_width = 2
    consumed = []

    def chunk(tokens, **kwargs):
        assert not kwargs["capture_hidden_seeds"]
        assert raw_k_prefill_rowbatch() == 32
        runner.position += len(tokens)
        return 100 + (len(tokens) - 1) * 4, None

    def consume(tokens, buffer):
        assert raw_k_prefill_rowbatch() == 0
        assert buffer.ptr == 100 and buffer.nbytes == len(tokens) * 4
        consumed.append(tuple(tokens))
        return np.array([tokens[-1], runner.position], dtype=np.float32)

    runner._prefill_chunk = chunk
    result = runner.prefill_append((1, 2, 3, 4, 5), capture_hidden_seeds=True,
                                  hidden_seed_consumer=consume)
    assert consumed == [(1, 2), (3, 4), (5,)]
    np.testing.assert_array_equal(result.hidden_seeds, [[5, 5]])


def test_bulk_stage_failure_restores_authoritative_cursors():
    from tests.test_unit_qwen4exp_prefill_chunks import _draft
    draft = _draft()
    positions = []
    draft.attention.set_position = positions.append

    def fail(*args):
        draft.index.count = 2
        raise RuntimeError("injected stage failure")

    draft._prefill_workspace.append = fail
    with pytest.raises(RuntimeError, match="injected"):
        draft.prime_prompt_chunk((1, 2, 3), np.zeros((3, 2)), previous_hidden=None)
    assert draft.position == draft.index.count == 0
    assert positions == [0]


def test_draft_closes_bulk_workspace_once():
    from tests.test_unit_qwen4exp_prefill_chunks import _draft
    draft = _draft()
    closed = []
    draft._prefill_workspace.close = lambda: closed.append("workspace")
    draft.index.close = lambda: closed.append("index")
    draft.attention.close = lambda: closed.append("attention")
    draft.close()
    draft.close()
    assert closed == ["workspace", "index", "attention"]
    assert draft._prefill_workspace is None


@pytest.mark.parametrize("abi,variant,batched", [
    ("pack8", "pack8_gemv_bf16_bf16_out", True),
    ("raw", "wmma_bf16_bf16_out", False),
    ("t16", "gemv_bf16_bf16_out", False),
])
def test_bulk_linear_preserves_parent_policy_and_trace(monkeypatch, abi, variant, batched):
    from hipengine.runtime import gguf_linear as linear
    calls, groups = [], []
    weight, fn = object(), object()
    key = dict(backend="test", layer="linear", quant="test", variant=variant)

    def launch(weight, source, destination, rows, inputs, outputs, **kw):
        calls.append((source, destination, rows))
        linear._record_gguf_linear_dispatch(**key, abi=abi, rows=rows,
            in_features=inputs, out_features=outputs,
            activation_dtype=kw["activation_dtype"], output_dtype=kw["output_dtype"])

    def batch(kernel, resident, source, destination, rows, inputs, outputs, kw):
        assert kernel is fn and resident is weight and kw["threads"] == 128
        groups.append((source, destination, rows))

    monkeypatch.setattr(linear, "launch_gguf_linear", launch)
    monkeypatch.setattr(linear, "resolve", lambda **kw: fn)
    monkeypatch.setattr(linear, "_pack8_gemv_decode_threads", lambda **kw: 128)
    monkeypatch.setitem(linear._LAUNCH_ABI, abi, batch)
    with linear.record_gguf_linear_dispatch() as records:
        linear.launch_gguf_linear_decode_rows(weight, 100, 200, 12, 3, 2,
            reference_rows=4, activation_dtype="bf16", output_dtype="bf16")
    assert calls == ([(100, 200, 4)] if batched else [(100, 200, 4), (124, 216, 4), (148, 232, 4)])
    assert groups == ([(124, 216, 8)] if batched else [])
    assert sum(record["rows"] for record in records) == 12
    assert all(record["key"] == key for record in records)


def test_workspace_allocation_failure_releases_owned_buffers(monkeypatch):
    from types import SimpleNamespace as NS
    from hipengine.core.memory import DeviceBuffer
    from hipengine.runtime import qwen4_exp_mtp_prefill as module
    allocated, freed, closed = [], [], []
    owner = NS(runtime=object(), max_sequence_length=128, config=NS(
        hidden_size=4, residual_branch_count=2, residual_width=8, residual_low_rank=2,
        attention_head_count=2, attention_kv_head_count=1, attention_key_length=2,
        indexer_head_count=1, indexer_key_length=2))

    def allocate(size, **kw):
        buffer = DeviceBuffer(100 + len(allocated), size)
        allocated.append(buffer)
        return buffer

    def fail(**kw):
        raise MemoryError("injected QSA allocation failure")

    monkeypatch.setattr(module, "malloc", allocate)
    monkeypatch.setattr(module, "free", lambda buffer, **kw: freed.append(buffer))
    monkeypatch.setattr(module.Qwen4ExpGRScratch, "allocate", lambda **kw: NS(close=lambda: closed.append("gr")))
    monkeypatch.setattr(module.Qwen4ExpQSAScratch, "allocate", fail)
    with pytest.raises(MemoryError, match="injected"):
        module.Qwen4ExpMTPPrefillWorkspace(owner)
    assert freed == list(reversed(allocated)) and len(freed) == 7
    assert closed == ["gr"]


def test_public_submission_consumes_device_rows_once(monkeypatch):
    from types import SimpleNamespace as NS
    from tests.test_unit_qwen4exp_mtp_submission import _driver, _request, _finish

    from tests._qwen4_exp_c1_control import select_request_owned_c1
    select_request_owned_c1(monkeypatch)
    driver, target, draft = _driver()
    draft._prefill_workspace = NS(append_calls=0)
    original = target.prefill_append

    def consume(tokens, buffer, **kw):
        draft.prime_prompt_chunk(tokens, None, **kw)
        draft._prefill_workspace.append_calls += 1
        return target._hidden(tokens[-1])

    def prefill(tokens, **kw):
        result = original(tokens, **kw)
        seed = kw["hidden_seed_consumer"](tokens, NS(ptr=100, nbytes=len(tokens) * 4))
        result.hidden_seeds, result.hidden_seed = seed[None, :], seed
        return result

    draft.prime_prompt_device_chunk = consume
    target.prefill_append = prefill
    try:
        submission = driver.submit_speculative_detailed(_request())
        output, _ = _finish(driver, submission)
        assert [tokens for tokens, _ in draft.chunks] == [(1, 2), (3, 4)]
        assert output.telemetry.diagnostics["mtp_prefill"] == {"device_hidden_rows": 4, "cache_tiles": 2}
    finally:
        driver.close()


@pytest.mark.parametrize("mode", ["serial", "batched", "k0", "head_only"])
def test_direct_device_prefill_does_not_leak_consumer_into_decode(mode):
    from tests.test_unit_qwen4exp_mtp_steps import (
        _TargetRunner, _BatchedTargetRunner, _DraftRunner, _provider, _request,
    )
    target = _BatchedTargetRunner() if mode == "batched" else _TargetRunner()
    draft = _DraftRunner()
    consumed = []
    original = target.prefill
    draft.reset = lambda: setattr(draft, "position", 0)

    def consume(tokens, buffer, **kw):
        kw["check_cancelled"]()
        consumed.append(tuple(tokens))
        draft.position += len(tokens)
        return target._hidden(tokens[-1])

    def prefill(tokens, *, hidden_seed_consumer, **kw):
        result = original(tokens, **kw)
        seed = hidden_seed_consumer(tokens, object())
        result.hidden_seeds = seed[None, :]
        return result

    target.prefill = prefill
    draft.prime_prompt_device_chunk = consume
    provider, _, _ = _provider(runner=target, draft=draft)
    count = 1 if mode == "head_only" else 5
    iterator = provider.iter_generate((1, 2), _request(max_tokens=count))
    next(iterator)
    while True:
        try:
            iterator.send(0 if mode == "k0" else None)
        except StopIteration as stopped:
            output = stopped.value
            break
    assert consumed == [(1, 2)] and draft.prime_calls == 0
    assert output.generated_token_ids == tuple(range(10, 10 + count))
    assert output.telemetry.diagnostics["phase_census"]["target_and_draft_prefill"]["calls"] == 1
