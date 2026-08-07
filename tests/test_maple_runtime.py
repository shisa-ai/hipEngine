"""Focused orchestration regressions for the Maple resident runner."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime import maple as maple_runtime
from hipengine.runtime.maple import MapleRunner


def test_reset_clears_captured_correctness_state() -> None:
    reset_calls = []
    runner = object.__new__(MapleRunner)
    runner.closed = False
    runner.position = 7
    runner.last_hidden_states = (object(),)
    runner.runtime = SimpleNamespace(device_synchronize=lambda: None)
    runner.buffers = SimpleNamespace(reset=lambda: reset_calls.append(True))

    runner.reset()

    assert reset_calls == [True]
    assert runner.position == 0
    assert runner.last_hidden_states == ()


def test_equal_capacity_swa_and_global_span_owners_are_both_published(monkeypatch) -> None:
    sliding_spans = object()
    global_spans = object()
    runner = object.__new__(MapleRunner)
    runner.buffers = SimpleNamespace(
        sliding_span_owner=SimpleNamespace(capacity=64, spans=sliding_spans),
        global_span_owner=SimpleNamespace(capacity=64, spans=global_spans),
    )
    runner.libraries = SimpleNamespace(attention="attention-library")
    runner.runtime = "runtime"
    calls = []

    def record(spans, *, position, library, runtime):
        calls.append((spans, position, library, runtime))

    monkeypatch.setattr(maple_runtime, "maple_kv_span_update", record)
    runner._publish_span_position(7)

    assert calls == [
        (sliding_spans, 7, "attention-library", "runtime"),
        (global_spans, 7, "attention-library", "runtime"),
    ]


def test_maple_batch_step_rejects_invalid_tokens_before_launch() -> None:
    from hipengine.runtime.maple import MapleBatchRunner

    runner = object.__new__(MapleBatchRunner)
    runner.closed = False
    runner.batch_size = 2
    runner.checkpoint = SimpleNamespace(spec=SimpleNamespace(vocab_size=151_936))

    with pytest.raises(ValueError, match="token_id"):
        runner.batch_step((1, -1))
    with pytest.raises(ValueError, match="token_id"):
        runner.batch_step((1, 151_936))


def test_maple_batch_span_reset_copies_to_request_offsets(monkeypatch) -> None:
    from hipengine.core.memory import DeviceBuffer
    from hipengine.runtime.maple import MapleBatchSpanOwner

    span_owner = object.__new__(MapleBatchSpanOwner)
    span_owner.batch_size = 2
    span_owner.per_request_capacity = 4
    span_owner.runtime = SimpleNamespace(device_synchronize=lambda: None)
    span_owner.token_positions = DeviceBuffer(ptr=100, nbytes=64)
    span_owner.evict_mask = DeviceBuffer(ptr=200, nbytes=8)
    span_owner.live_counts = DeviceBuffer(ptr=300, nbytes=16)
    span_owner.row_positions = DeviceBuffer(ptr=400, nbytes=16)
    span_owner.token_host = np.arange(8, dtype=np.int64)
    span_owner.evict_host = np.zeros(8, dtype=np.bool_)
    span_owner.live_host = np.ones(2, dtype=np.int64)
    span_owner.row_host = np.ones(2, dtype=np.int64)
    calls: list[tuple[int, int]] = []

    def fake_copy(buffer, host_ptr, nbytes=None, *, runtime=None):
        del host_ptr, runtime
        calls.append((buffer.ptr, buffer.nbytes if nbytes is None else nbytes))

    monkeypatch.setattr(maple_runtime, "copy_host_to_device", fake_copy)
    span_owner.reset_request(1)

    assert calls == [(132, 32), (204, 4), (308, 8), (408, 8)]
    assert np.all(span_owner.token_host[4:8] == -1)
    assert np.all(span_owner.evict_host[4:8])
    assert span_owner.live_host[1] == 0
    assert span_owner.row_host[1] == -1


def test_maple_prefill_buffers_exclude_all_row_sampling_scratch() -> None:
    from hipengine.core.memory import DeviceBuffer
    from hipengine.runtime.maple import _BatchBuffers, _PrefillBuffers

    class FakeOwner:
        def __init__(self) -> None:
            self.next_ptr = 1_000

        def allocate(self, nbytes: int) -> DeviceBuffer:
            buffer = DeviceBuffer(ptr=self.next_ptr, nbytes=nbytes)
            self.next_ptr += nbytes
            return buffer

    buffers = _PrefillBuffers(
        owner=FakeOwner(),
        spec=SimpleNamespace(
            hidden_size=4,
            q_size=4,
            kv_size=2,
            vocab_size=10,
            num_experts=3,
        ),
        top_k=2,
        intermediate=2,
        T=3,
    )

    assert not hasattr(buffers, "logits")
    assert not hasattr(buffers, "argmax_block_values")
    assert not hasattr(buffers, "argmax_block_indices")
    assert not hasattr(buffers, "argmax_index")
    assert not hasattr(buffers, "argmax_value")

    batch_buffers = _BatchBuffers(
        owner=FakeOwner(),
        spec=SimpleNamespace(
            hidden_size=4,
            q_size=4,
            kv_size=2,
            vocab_size=10,
            num_experts=3,
        ),
        top_k=2,
        intermediate=2,
        T=3,
    )
    assert batch_buffers.logits.nbytes == 3 * 10 * 4
    assert batch_buffers.argmax_index.nbytes == 3 * 4
    assert batch_buffers.argmax_value.nbytes == 3 * 4


def test_maple_prefill_native_samples_only_the_final_row(monkeypatch) -> None:
    from hipengine.core.memory import DeviceBuffer

    def buffer(ptr: int, nbytes: int = 4_096) -> DeviceBuffer:
        return DeviceBuffer(ptr=ptr, nbytes=nbytes)

    spec = SimpleNamespace(
        vocab_size=10,
        sliding_window=8,
        hidden_size=4,
        q_size=4,
        kv_size=2,
        num_experts_per_tok=2,
        moe_intermediate_size=2,
        num_experts=3,
        rms_norm_eps=1e-6,
    )
    pf = SimpleNamespace(
        token_ids=buffer(100),
        hidden=buffer(1_000),
        normalized=buffer(1_100),
        logits=buffer(1_200),
        argmax_block_values=buffer(1_300),
        argmax_block_indices=buffer(1_400),
        argmax_index=buffer(1_500),
        argmax_value=buffer(1_600),
    )
    buffers = SimpleNamespace(
        pf=pf,
        normalized=buffer(2_000),
        logits=buffer(3_000),
        argmax_block_values=buffer(4_000),
        argmax_block_indices=buffer(5_000),
        argmax_index=buffer(6_000),
        argmax_value=buffer(7_000),
        sliding_span_owner=SimpleNamespace(spans=object()),
        global_span_owner=SimpleNamespace(spans=object()),
        layers=(),
    )
    pointer = lambda value: SimpleNamespace(ptr=value)  # noqa: E731
    runner = object.__new__(MapleRunner)
    runner.closed = False
    runner.position = 0
    runner.max_context = 8
    runner.runtime = object()
    runner.checkpoint = SimpleNamespace(spec=spec)
    runner.buffers = buffers
    runner.weights = SimpleNamespace(
        embeddings=SimpleNamespace(
            weight=pointer(10), scales=pointer(11), biases=pointer(12)
        ),
        final_norm=pointer(13),
        lm_head=SimpleNamespace(
            weight=pointer(14), scales=pointer(15), biases=pointer(16)
        ),
        layers=(),
    )
    runner.libraries = SimpleNamespace(
        ternary=object(),
        attention=object(),
        moe=object(),
        norm=object(),
        lm_head=object(),
    )

    embed_rows: list[int] = []
    tail_norm_calls: list[tuple[int, int, int]] = []
    head_calls: list[tuple[int, int]] = []
    argmax_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        maple_runtime, "maple_kv_span_update_batched", lambda *a, **k: None
    )
    monkeypatch.setattr(maple_runtime, "copy_host_to_device", lambda *a, **k: None)
    monkeypatch.setattr(
        maple_runtime,
        "maple_affine4_embed_batched_bf16",
        lambda *args, **kwargs: embed_rows.append(int(args[5])),
    )
    monkeypatch.setattr(
        maple_runtime,
        "paro_rmsnorm_out_bf16",
        lambda *args, **kwargs: tail_norm_calls.append(
            (int(args[0]), int(args[2]), int(args[3]))
        ),
    )
    monkeypatch.setattr(
        maple_runtime,
        "maple_affine4_gemv_f32",
        lambda *args, **kwargs: head_calls.append((int(args[0]), int(args[4]))),
    )
    monkeypatch.setattr(
        maple_runtime,
        "argmax_f32",
        lambda *args, **kwargs: argmax_calls.append((int(args[0]), int(args[3]))),
    )
    monkeypatch.setattr(
        maple_runtime,
        "maple_affine4_gemv_batched_f32",
        lambda *a, **k: pytest.fail("prefill must not launch the all-row LM head"),
    )
    monkeypatch.setattr(
        maple_runtime,
        "argmax_f32_rows_i32",
        lambda *a, **k: pytest.fail("prefill must not launch all-row argmax"),
    )

    def fake_copy_device_to_host(host_ptr, source, nbytes=None, *, runtime=None):
        del nbytes, runtime
        if source.ptr == buffers.argmax_index.ptr:
            ctypes.c_int64.from_address(host_ptr).value = 7
        elif source.ptr == buffers.argmax_value.ptr:
            ctypes.c_float.from_address(host_ptr).value = 3.5
        else:
            raise AssertionError(f"unexpected D2H source {source.ptr}")

    monkeypatch.setattr(
        maple_runtime, "copy_device_to_host", fake_copy_device_to_host
    )

    result = runner.prefill_native((1, 2, 3, 4, 5), chunk_size=3)

    assert embed_rows == [3, 2]
    assert tail_norm_calls == [(1_008, buffers.normalized.ptr, 1)]
    assert head_calls == [(buffers.normalized.ptr, buffers.logits.ptr)]
    assert argmax_calls == [(buffers.logits.ptr, buffers.argmax_index.ptr)]
    assert result.position == 4
    assert result.token_id == 7
    assert result.top_logit == pytest.approx(3.5)


def test_maple_prefill_native_rejects_invalid_chunk_and_token_before_launch() -> None:
    runner = object.__new__(MapleRunner)
    runner.closed = False
    runner.position = 0
    runner.max_context = 4_096
    runner.checkpoint = SimpleNamespace(
        spec=SimpleNamespace(vocab_size=151_936, sliding_window=512)
    )

    with pytest.raises(ValueError, match="chunk_size"):
        runner.prefill_native((1,), chunk_size=257)
    with pytest.raises(ValueError, match="token_id"):
        runner.prefill_native((-1,), chunk_size=1)
    with pytest.raises(NotImplementedError, match="sliding-window"):
        runner.prefill_native(tuple(1 for _ in range(513)), chunk_size=1)


def test_maple_prefill_native_matches_serial_step_gate(hip_test_target_arch) -> None:
    """Bulk prefill must preserve the serial next-token and continuation contract.

    GPU + real checkpoint guarded: skipped when no supported HIP target or when
    the Maple checkpoint cannot be resolved locally.
    """
    del hip_test_target_arch
    try:
        from hipengine.loading.maple import load_maple_checkpoint
    except Exception as exc:  # noqa: BLE001 - import guard
        pytest.skip(f"maple loading unavailable: {exc}")

    model = "deepgrove/maple-preview-2bit-mlx"
    backend = "hip_gfx1151"
    prompt = (9707, 13, 358, 1093, 220, 3100, 1066, 13, 366, 264, 1156, 15)
    try:
        checkpoint = load_maple_checkpoint(model)
    except Exception as exc:  # noqa: BLE001 - checkpoint missing
        pytest.skip(f"maple checkpoint unavailable: {exc}")

    serial_token = None
    runner = MapleRunner.load(checkpoint, backend=backend, max_context=64)
    try:
        for tok in prompt:
            serial_token = runner.step(tok).token_id
        assert serial_token is not None
        serial_continuation = runner.step(serial_token)
    finally:
        runner.close()

    runner2 = MapleRunner.load(checkpoint, backend=backend, max_context=64)
    try:
        native = runner2.prefill_native(prompt)
        native_continuation = runner2.step(native.token_id)
    finally:
        runner2.close()

    assert native.position == len(prompt) - 1
    assert np.isfinite(native.top_logit)
    assert native.token_id == serial_token
    assert native_continuation.token_id == serial_continuation.token_id


def test_maple_prefill_native_natural_prompt_continuations(hip_test_target_arch) -> None:
    """Natural English/Japanese prompts preserve seed and decode continuation."""
    del hip_test_target_arch
    from hipengine.loading.maple import load_maple_checkpoint
    from hipengine.tokenization.maple import MapleTokenizer

    try:
        checkpoint = load_maple_checkpoint("deepgrove/maple-preview-2bit-mlx")
    except Exception as exc:  # noqa: BLE001 - checkpoint missing
        pytest.skip(f"maple checkpoint unavailable: {exc}")
    spec = checkpoint.spec
    tokenizer = MapleTokenizer.from_model_path(
        checkpoint.index.model_path,
        model_vocab_size=spec.vocab_size,
        eos_token_id=spec.eos_token_id,
        bos_token_id=spec.bos_token_id,
    )
    prompts = tuple(
        tokenizer.encode_chat(text)
        for text in (
            "Write one short sentence about maple trees.",
            "What is 2 + 2? Answer briefly.",
            "日本語で短く挨拶してください。",
        )
    )
    serial = MapleRunner.load(checkpoint, backend="hip_gfx1151", max_context=512)
    native = MapleRunner.load(checkpoint, backend="hip_gfx1151", max_context=512)
    try:
        for prompt in prompts:
            serial.reset()
            native.reset()
            serial_token = serial.prefill(prompt).token_id
            native_token = native.prefill_native(prompt).token_id
            assert native_token == serial_token
            for _ in range(2):
                serial_step = serial.step(serial_token)
                native_step = native.step(native_token)
                assert native_step.token_id == serial_step.token_id
                serial_token = serial_step.token_id
                native_token = native_step.token_id
    finally:
        native.close()
        serial.close()


def test_maple_prefill_native_multichunk_continuation_gate(hip_test_target_arch) -> None:
    """A prompt crossing the 256-row chunk boundary keeps decode state aligned."""
    del hip_test_target_arch
    from hipengine.loading.maple import load_maple_checkpoint

    try:
        checkpoint = load_maple_checkpoint("deepgrove/maple-preview-2bit-mlx")
    except Exception as exc:  # noqa: BLE001 - checkpoint missing
        pytest.skip(f"maple checkpoint unavailable: {exc}")
    prompt = tuple(9_000 + (index % 512) for index in range(260))
    backend = "hip_gfx1151"

    serial = MapleRunner.load(checkpoint, backend=backend, max_context=512)
    native = MapleRunner.load(checkpoint, backend=backend, max_context=512)
    try:
        serial_result = serial.prefill(prompt)
        native_result = native.prefill_native(prompt)
        assert native_result.position == len(prompt) - 1
        assert native_result.token_id == serial_result.token_id
        serial_token = serial_result.token_id
        native_token = native_result.token_id
        for _ in range(3):
            serial_step = serial.step(serial_token)
            native_step = native.step(native_token)
            assert native_step.token_id == serial_step.token_id
            serial_token = serial_step.token_id
            native_token = native_step.token_id
    finally:
        native.close()
        serial.close()


@pytest.mark.parametrize("c", [2, 4, 8])
def test_maple_batch_decode_matches_serial_steps(hip_test_target_arch, c) -> None:
    """MapleBatchRunner.batch_step (c>1) must equal c independent serial decodes.

    GPU + real checkpoint guarded: skipped when no supported HIP target or when
    the Maple checkpoint cannot be resolved locally. Each request gets a
    distinct prompt; batch request r must produce the same argmax tokens as the
    serial runner stepping the same prompt from position 0.
    """
    del hip_test_target_arch
    try:
        from hipengine.loading.maple import load_maple_checkpoint
    except Exception as exc:  # noqa: BLE001 - import guard
        pytest.skip(f"maple loading unavailable: {exc}")

    from hipengine.runtime.maple import MapleBatchRunner

    model = "deepgrove/maple-preview-2bit-mlx"
    backend = "hip_gfx1151"
    prompts = [
        tuple(9000 + r + i for i in range(4)) for r in range(c)
    ]
    steps = len(prompts[0])
    try:
        checkpoint = load_maple_checkpoint(model)
    except Exception as exc:  # noqa: BLE001 - checkpoint missing
        pytest.skip(f"maple checkpoint unavailable: {exc}")

    serial_tokens = []
    runner = MapleRunner.load(checkpoint, backend=backend, max_context=64)
    try:
        for prompt in prompts:
            runner.reset()
            serial_tokens.append([runner.step(token).token_id for token in prompt])
    finally:
        runner.close()

    batch_tokens = []
    batch = MapleBatchRunner.load(
        checkpoint, backend=backend, batch_size=c, per_capacity=64
    )
    try:
        for step in range(steps):
            batch_tokens.append(
                batch.batch_step([prompts[r][step] for r in range(c)])
            )
    finally:
        batch.close()

    # batch_tokens[step][r] must equal serial_tokens[r][step].
    for step in range(steps):
        for r in range(c):
            assert batch_tokens[step][r] == serial_tokens[r][step], (
                f"request {r} step {step}: batch={batch_tokens[step][r]} "
                f"serial={serial_tokens[r][step]}"
            )
    from hipengine.core.memory import memory_stats

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


def test_maple_batch_decode_preserves_swa_after_wrap(hip_test_target_arch) -> None:
    """c=2 batch decode remains serial-exact beyond the SWA-512 boundary."""
    del hip_test_target_arch
    from hipengine.loading.maple import load_maple_checkpoint
    from hipengine.runtime.maple import MapleBatchRunner

    try:
        checkpoint = load_maple_checkpoint("deepgrove/maple-preview-2bit-mlx")
    except Exception as exc:  # noqa: BLE001 - checkpoint missing
        pytest.skip(f"maple checkpoint unavailable: {exc}")
    steps = 514
    prompts = tuple(
        tuple(9_000 + ((request * 37 + step) % 512) for step in range(steps))
        for request in range(2)
    )
    serial_outputs: list[list[int]] = []
    serial = MapleRunner.load(
        checkpoint, backend="hip_gfx1151", max_context=steps + 2
    )
    try:
        for prompt in prompts:
            serial.reset()
            serial_outputs.append([serial.step(token).token_id for token in prompt])
    finally:
        serial.close()

    batch_outputs: list[list[int]] = [[], []]
    batch = MapleBatchRunner.load(
        checkpoint,
        backend="hip_gfx1151",
        batch_size=2,
        per_capacity=steps + 2,
    )
    try:
        for step in range(steps):
            output = batch.batch_step([prompts[0][step], prompts[1][step]])
            for request, token in enumerate(output):
                batch_outputs[request].append(token)
    finally:
        batch.close()

    assert batch_outputs == serial_outputs


def test_maple_continuous_batcher_validates_admission_and_steps_sparse_slots() -> None:
    from hipengine.runtime.maple_batch import MapleContinuousBatcher

    class FakeBatchRunner:
        batch_size = 2
        closed = False
        checkpoint = SimpleNamespace(spec=SimpleNamespace(vocab_size=100))

        def __init__(self) -> None:
            self.reset_requests: list[int] = []
            self.last_active_mask: list[bool] | None = None

        def reset_request(self, request: int) -> None:
            self.reset_requests.append(request)

        def batch_step(self, token_ids, *, active_mask=None):
            assert token_ids == [7, 0]
            self.last_active_mask = list(active_mask)
            return [11, 22]

    runner = FakeBatchRunner()
    batcher = MapleContinuousBatcher(runner)
    with pytest.raises(ValueError, match="max_new"):
        batcher.submit(7, max_new=0)
    with pytest.raises(ValueError, match="seed"):
        batcher.submit(100, max_new=1)

    assert batcher.submit(7, max_new=1) == 0
    assert batcher.step() == 1
    assert batcher.active() == 0
    assert batcher.completions == [[11]]
    assert runner.last_active_mask == [True, False]


def test_maple_continuous_batcher_matches_serial(hip_test_target_arch) -> None:
    """Continuous-batching owner loop matches serial autoregressive decode."""
    del hip_test_target_arch
    try:
        from hipengine.loading.maple import load_maple_checkpoint
    except Exception as exc:  # noqa: BLE001 - import guard
        pytest.skip(f"maple loading unavailable: {exc}")

    from hipengine.runtime.maple import MapleBatchRunner
    from hipengine.runtime.maple_batch import MapleContinuousBatcher

    model = "deepgrove/maple-preview-2bit-mlx"
    backend = "hip_gfx1151"
    c = 2
    seeds = [9000, 9001, 9002]
    lengths = [2, 5, 3]
    try:
        checkpoint = load_maple_checkpoint(model)
    except Exception as exc:  # noqa: BLE001 - checkpoint missing
        pytest.skip(f"maple checkpoint unavailable: {exc}")

    serial = []
    runner = MapleRunner.load(checkpoint, backend=backend, max_context=64)
    try:
        for seed, length in zip(seeds, lengths):
            runner.reset()
            out = [runner.step(seed).token_id]
            for _ in range(length - 1):
                out.append(runner.step(out[-1]).token_id)
            serial.append(out)
    finally:
        runner.close()

    batch = MapleBatchRunner.load(
        checkpoint, backend=backend, batch_size=c, per_capacity=64
    )
    batcher = MapleContinuousBatcher(batch)
    try:
        batcher.submit(seeds[0], max_new=lengths[0])
        batcher.submit(seeds[1], max_new=lengths[1])
        batcher.step()
        batcher.step()  # request 0 completes and slot 0 is reclaimed
        batcher.step()  # sparse round: request 1 advances while slot 0 is idle
        assert batcher.submit(seeds[2], max_new=lengths[2]) == 0
        while batcher.active():
            batcher.step()
    finally:
        batch.close()

    assert batcher.completions == serial
