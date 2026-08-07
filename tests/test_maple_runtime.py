"""Focused orchestration regressions for the Maple resident runner."""

from __future__ import annotations

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


@pytest.mark.parametrize("c", [2, 4])
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
    runners = []
    try:
        for r in range(c):
            runner = MapleRunner.load(checkpoint, backend=backend, max_context=64)
            runners.append(runner)
            got = []
            for tok in prompts[r]:
                got.append(runner.step(tok).token_id)
            serial_tokens.append(got)
    finally:
        for runner in runners:
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
    seeds = [9000, 9001]
    n = 4
    try:
        checkpoint = load_maple_checkpoint(model)
    except Exception as exc:  # noqa: BLE001 - checkpoint missing
        pytest.skip(f"maple checkpoint unavailable: {exc}")

    serial = []
    runners = []
    try:
        for r in range(c):
            runner = MapleRunner.load(checkpoint, backend=backend, max_context=64)
            runners.append(runner)
            out = [runner.step(seeds[r]).token_id]
            for _ in range(n - 1):
                out.append(runner.step(out[-1]).token_id)
            serial.append(out)
    finally:
        for runner in runners:
            runner.close()

    batch = MapleBatchRunner.load(
        checkpoint, backend=backend, batch_size=c, per_capacity=64
    )
    batcher = MapleContinuousBatcher(batch)
    try:
        for r in range(c):
            batcher.submit(seeds[r], max_new=n)
        while batcher.active():
            batcher.step()
    finally:
        batch.close()

    assert len(batcher.completions) == c
    for r in range(c):
        assert batcher.completions[r] == serial[r], (
            f"request {r}: batch={batcher.completions[r]} serial={serial[r]}"
        )
