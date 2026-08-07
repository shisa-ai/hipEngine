"""Focused orchestration regressions for the Maple resident runner."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_maple_prefill_native_matches_serial_step_gate(hip_test_target_arch) -> None:
    """prefill_native (batched P4) must agree with the serial step-loop next token.

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
    finally:
        runner.close()

    runner2 = MapleRunner.load(checkpoint, backend=backend, max_context=64)
    try:
        native = runner2.prefill_native(prompt)
    finally:
        runner2.close()

    assert native.token_id == serial_token


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
