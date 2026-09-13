"""Focused orchestration regressions for the Maple resident runner."""

from __future__ import annotations

import ast
import ctypes
import inspect
import textwrap
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


def test_maple_batch_span_exposes_request_local_prefill_views() -> None:
    from hipengine.core.memory import DeviceBuffer
    from hipengine.runtime.maple import MapleBatchSpanOwner

    class FakeOwner:
        runtime = object()

        def __init__(self) -> None:
            self.next_ptr = 1_000

        def put(self, array) -> DeviceBuffer:
            buffer = DeviceBuffer(ptr=self.next_ptr, nbytes=array.nbytes)
            self.next_ptr += array.nbytes
            return buffer

    owner = MapleBatchSpanOwner(
        FakeOwner(),
        batch_size=2,
        per_request_capacity=4,
        device=maple_runtime.Device("hip", 0),
    )

    assert len(owner.request_spans) == 2
    for request, spans in enumerate(owner.request_spans):
        assert spans.max_live_count == 4
        assert spans.base_offsets.ptr == owner._request_base_offsets.ptr
        assert spans.live_counts.ptr == owner.live_counts.ptr + request * 8
        assert spans.token_positions.ptr == owner.token_positions.ptr + request * 4 * 8
        assert spans.evict_mask.ptr == owner.evict_mask.ptr + request * 4
        assert spans.row_positions.ptr == owner.row_positions.ptr + request * 8


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
    assert buffers.expert_start.nbytes == 4 * 8
    assert buffers.active_experts.nbytes == 3 * 8
    assert buffers.active_count.nbytes == 8
    assert buffers.sorted_lanes.nbytes == 3 * 2 * 8
    assert buffers.sorted_experts.nbytes == 3 * 2 * 8
    assert buffers.sorted_weights.nbytes == 3 * 2 * 4

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


def test_maple_prefill_grouped_moe_is_default_with_explicit_gather_rollback(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_MAPLE_PREFILL_GROUPED_MOE", raising=False)
    assert maple_runtime._maple_prefill_grouped_moe() is True
    monkeypatch.setenv("HIPENGINE_MAPLE_PREFILL_GROUPED_MOE", "0")
    assert maple_runtime._maple_prefill_grouped_moe() is False


def test_maple_retained_paths_have_no_environment_rollback_seams() -> None:
    source = inspect.getsource(maple_runtime)
    for selector in (
        "HIPENGINE_MAPLE_PREFILL_GQA4",
        "HIPENGINE_MAPLE_ROUTER_SINGLE_DISPATCH",
        "HIPENGINE_MAPLE_AFFINE4_WAVE32_EXACT",
        "HIPENGINE_MAPLE_BATCH_AFFINE4_ROWREUSE_EXACT",
    ):
        assert selector not in source


def test_maple_batch_affine4_rowreuse_is_width_bounded() -> None:
    for rows in (2, 4, 8):
        assert (
            maple_runtime._maple_batch_affine4_head(rows)
            is maple_runtime.maple_affine4_gemv_batched_rowreuse_exact_f32
        )
    for rows in (1, 3, 16):
        assert (
            maple_runtime._maple_batch_affine4_head(rows)
            is maple_runtime.maple_affine4_gemv_batched_f32
        )


def test_maple_step_snapshots_decode_selectors_once() -> None:
    """Per-layer decode must not repeat invariant environment lookups."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(MapleRunner.step)))
    step = tree.body[0]
    assert isinstance(step, ast.FunctionDef)
    decode = next(
        node
        for node in step.body
        if isinstance(node, ast.FunctionDef) and node.name == "_decode_layers_and_tail"
    )
    for selector in ("_maple_fuse_qkattn", "_maple_fuse_moe"):
        all_calls = [
            node
            for node in ast.walk(step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == selector
        ]
        nested_calls = [
            node
            for node in ast.walk(decode)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == selector
        ]
        assert len(all_calls) == 1
        assert nested_calls == []


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
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_experts_per_tok=2,
        moe_intermediate_size=2,
        num_experts=3,
        rms_norm_eps=1e-6,
        layer_types=(),
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
        sliding_span_owner=SimpleNamespace(capacity=8, spans=object()),
        global_span_owner=SimpleNamespace(capacity=8, spans=object()),
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
    head_calls: list[tuple[str, int, int]] = []
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
        "maple_affine4_gemv_wave32_exact_f32",
        lambda *args, **kwargs: head_calls.append(
            ("wave32", int(args[0]), int(args[4]))
        ),
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
    assert head_calls == [("wave32", buffers.normalized.ptr, buffers.logits.ptr)]
    assert argmax_calls == [(buffers.logits.ptr, buffers.argmax_index.ptr)]
    assert result.position == 4
    assert result.token_id == 7
    assert result.top_logit == pytest.approx(3.5)


def test_maple_prefill_swa_segments_never_bulk_append_across_ring_wrap() -> None:
    segments = maple_runtime._maple_prefill_swa_segments

    assert segments(start=0, rows=256, capacity=512) == ((0, 256),)
    assert segments(start=256, rows=256, capacity=512) == ((0, 256),)
    assert segments(start=400, rows=200, capacity=512) == (
        (0, 112),
        *((offset, 1) for offset in range(112, 200)),
    )
    assert segments(start=512, rows=3, capacity=512) == ((0, 1), (1, 1), (2, 1))
    assert segments(start=700, rows=1, capacity=512) == ((0, 1),)


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


@pytest.mark.parametrize("prompt_length", [520, 770])
def test_maple_prefill_native_multichunk_continuation_gate(
    hip_test_target_arch, prompt_length
) -> None:
    """Post-wrap chunks keep physical state and continuation aligned."""
    del hip_test_target_arch
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        host_array_ptr,
    )
    from hipengine.loading.maple import load_maple_checkpoint

    try:
        checkpoint = load_maple_checkpoint("deepgrove/maple-preview-2bit-mlx")
    except Exception as exc:  # noqa: BLE001 - checkpoint missing
        pytest.skip(f"maple checkpoint unavailable: {exc}")
    prompt = tuple(9_000 + (index % 512) for index in range(prompt_length))
    backend = "hip_gfx1151"

    def device_bytes(runner, source, *, offset=0, nbytes=None):
        size = source.nbytes - offset if nbytes is None else nbytes
        host = np.empty(size, dtype=np.uint8)
        copy_device_to_host(
            host_array_ptr(host),
            DeviceBuffer(ptr=source.ptr + offset, nbytes=size),
            nbytes=size,
            runtime=runner.runtime,
        )
        return host

    max_context = prompt_length + 8
    serial = MapleRunner.load(checkpoint, backend=backend, max_context=max_context)
    native = MapleRunner.load(checkpoint, backend=backend, max_context=max_context)
    try:
        serial_result = serial.prefill(prompt)
        native_result = native.prefill_native(prompt)
        assert native_result.position == len(prompt) - 1
        assert native_result.token_id == serial_result.token_id
        assert (
            np.asarray(native_result.top_logit, dtype=np.float32).view(np.uint32)
            == np.asarray(serial_result.top_logit, dtype=np.float32).view(np.uint32)
        )

        spec = checkpoint.spec
        hidden_bytes = spec.hidden_size * np.dtype(np.uint16).itemsize
        final_row = (len(prompt) - 1) % maple_runtime.PREFILL_CHUNK
        assert np.array_equal(
            device_bytes(serial, serial.buffers.hidden, nbytes=hidden_bytes),
            device_bytes(
                native,
                native.buffers.pf.hidden,
                offset=final_row * hidden_bytes,
                nbytes=hidden_bytes,
            ),
        )
        assert np.array_equal(
            device_bytes(serial, serial.buffers.normalized, nbytes=hidden_bytes),
            device_bytes(native, native.buffers.normalized, nbytes=hidden_bytes),
        )
        for serial_layer, native_layer in zip(
            serial.buffers.layers, native.buffers.layers, strict=True
        ):
            live_bytes = (
                min(len(prompt), serial_layer.spans.max_live_count)
                * spec.kv_size
                * np.dtype(np.uint16).itemsize
            )
            for field in ("key_cache", "value_cache"):
                assert np.array_equal(
                    device_bytes(
                        serial, getattr(serial_layer, field), nbytes=live_bytes
                    ),
                    device_bytes(
                        native, getattr(native_layer, field), nbytes=live_bytes
                    ),
                )
        for owner_name in ("sliding_span_owner", "global_span_owner"):
            serial_owner = getattr(serial.buffers, owner_name)
            native_owner = getattr(native.buffers, owner_name)
            for field in (
                "base_offsets",
                "live_counts",
                "token_positions",
                "evict_mask",
                "row_positions",
            ):
                assert np.array_equal(
                    device_bytes(serial, getattr(serial_owner, field)),
                    device_bytes(native, getattr(native_owner, field)),
                )

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


def test_maple_batch_prefill_admission_matches_serial(hip_test_target_arch) -> None:
    """Request-local native prefill feeds exact shared-weight c2 decode."""
    del hip_test_target_arch
    from hipengine.core.memory import memory_stats
    from hipengine.loading.maple import load_maple_checkpoint
    from hipengine.runtime.maple import MapleBatchRunner

    try:
        checkpoint = load_maple_checkpoint("deepgrove/maple-preview-2bit-mlx")
    except Exception as exc:  # noqa: BLE001 - checkpoint missing
        pytest.skip(f"maple checkpoint unavailable: {exc}")
    prompts = (
        tuple(9_000 + index for index in range(12)),
        tuple(9_100 + index * 3 for index in range(17)),
    )
    serial = MapleRunner.load(
        checkpoint, backend="hip_gfx1151", max_context=64
    )
    expected: list[list[int]] = []
    expected_top_bits: list[int] = []
    try:
        for prompt in prompts:
            serial.reset()
            result = serial.prefill_native(prompt)
            trajectory = [result.token_id]
            expected_top_bits.append(
                int(np.asarray(result.top_logit, np.float32).view(np.uint32))
            )
            for _ in range(2):
                trajectory.append(serial.step(trajectory[-1]).token_id)
            expected.append(trajectory)

        batch = MapleBatchRunner.from_runner(
            serial, batch_size=2, per_capacity=64
        )
        try:
            admitted = [
                batch.prefill_request(request, prompt)
                for request, prompt in enumerate(prompts)
            ]
            assert batch._prefill_runners is not None
            for runner in batch._prefill_runners:
                assert runner.buffers.argmax_index.nbytes == 8
                assert runner.buffers.argmax_value.nbytes == 4
                assert runner.buffers.logits.nbytes == checkpoint.spec.vocab_size * 4
            assert [result.token_id for result in admitted] == [row[0] for row in expected]
            assert [
                int(np.asarray(result.top_logit, np.float32).view(np.uint32))
                for result in admitted
            ] == expected_top_bits
            current = [result.token_id for result in admitted]
            for step in range(1, 3):
                current = batch.batch_step(current)
                assert current == [row[step] for row in expected]

            batch.reset()
            single = batch.prefill_request(0, prompts[0])
            assert single.token_id == expected[0][0]
            for step in range(1, 3):
                single = batch.step_request(0, single.token_id)
                assert single.token_id == expected[0][step]
        finally:
            batch.close()
        assert not serial.closed
    finally:
        serial.close()

    assert memory_stats()["current_allocated_bytes"] == 0
    assert memory_stats()["active_allocations"] == 0


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


def test_maple_batch_benchmark_drives_sparse_rows_without_private_scheduler() -> None:
    from scripts.maple_batch_decode_bench import _run_batch

    class FakeBatchRunner:
        batch_size = 3

        def __init__(self) -> None:
            self.resets: list[int] = []
            self.calls: list[tuple[list[int], list[bool]]] = []

        def reset_request(self, request: int) -> None:
            self.resets.append(request)

        def batch_step(self, token_ids, *, active_mask=None):
            self.calls.append((list(token_ids), list(active_mask)))
            return [int(token) + 1 for token in token_ids]

    runner = FakeBatchRunner()
    outputs, elapsed = _run_batch(runner, [7, 11], steps=3)

    assert outputs == [[8, 9, 10], [12, 13, 14]]
    assert runner.resets == [0, 1]
    assert [mask for _, mask in runner.calls] == [[True, True, False]] * 3
    assert elapsed >= 0.0


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
