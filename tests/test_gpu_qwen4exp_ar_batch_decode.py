"""GPU: the serving generator decodes two AR rows with one shared batch step.

The synthetic model and the c1/batch owners come from the committed batch-runner
fixture. These tests add only the generator wiring: one c1 prefill per request,
one import per request, and one 2-row batch call per decode tick.

They are narrow plumbing checks, not the production numerical gate: the fixture
model is three layers with a 64-token vocabulary, and the envelope it can
establish is the fixture's own.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
from hipengine.generation.qwen4_exp_ar_batch import AR_BATCH_ENV
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.registry import GenerationRequest
from tests._qwen4exp_ar_batch_fixtures import FakeTokenizer
from tests.test_gpu_qwen4exp_batch_runner import _c1, model  # noqa: F401


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


class _Resident:
    """The fixture resident plus the ``close`` hook the generator owns."""

    def __init__(self, resident):
        self._resident = resident

    def __getattr__(self, name):
        return getattr(self._resident, name)

    def close(self):
        return None


def _c1_greedy(model, prompt, steps):
    """The committed single-request chain: prefill, then one token per step."""

    runner = _c1(model)
    try:
        result = runner.prefill_serial(list(prompt))
        ids = [int(result.token_id)]
        for _ in range(int(steps) - 1):
            result = runner.step(ids[-1])
            ids.append(int(result.token_id))
        return tuple(ids)
    finally:
        runner.close()


def _driver(model):
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="synthetic.gguf",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        tokenizer=FakeTokenizer(),
        runner=_c1(model),
        backend=model.backend,
    )
    generator._resident = _Resident(model.resident)
    driver = SubmitPollTextGenerator(
        generator,
        config=EngineLoopConfig(
            max_active_requests=2,
            max_prefill_chunk_tokens=4,
            # Complete both prefills before decoding, so the shared step's width
            # is the batch's and not the TTFT policy's.
            prefill_decode_policy="protect_ttft",
        ),
    )
    return generator, driver


def _run_to_completion(driver, native, submissions):
    """Poll until every submission completes, snapshotting the prefilled rows."""

    prefilled = None
    while not all(driver.generation_complete(item) for item in submissions):
        driver.poll(max_ticks=1)
        rows = native._rows
        if prefilled is None and rows and all(
            row.next_result is not None for row in rows.values()
        ):
            # A batch row returned its staging runner at import, so the slot
            # holds the prefilled position.
            prefilled = {
                request_id: (
                    len(row.prompt_ids),
                    int(native._ar_batch.runner.position(native._ar_batch.lease(request_id)[0])),
                    native._ar_batch.lease(request_id),
                )
                for request_id, row in rows.items()
            }
    return prefilled, [driver.take_result(item) for item in submissions]


def test_serving_generator_decodes_two_rows_in_one_batch_step(model, monkeypatch):
    """One 2-row step per tick, and each row samples from its own logits row.

    The two requests force different tokens with a logit bias, so a swapped
    result row would emit the neighbour's id. The c1 runners only prefill.
    """

    monkeypatch.setenv(AR_BATCH_ENV, "1")
    prompts = ((9, 4, 27), (15, 22))
    forced = (5, 17)
    steps = 3
    generator, driver = _driver(model)
    native = generator._resident_model_runner
    try:
        assert native is not None
        submissions = tuple(
            driver.submit_detailed(
                GenerationRequest(
                    prompts=(prompt,), max_tokens=steps, temperature=0.0,
                    top_p=1.0, ignore_eos=True, logit_bias={token: 100.0},
                )
            )
            for prompt, token in zip(prompts, forced)
        )
        prefilled, results = _run_to_completion(driver, native, submissions)
        outputs = [item[0] for item in results]
        telemetry = native.decode_telemetry()
        assert telemetry["decode_path"] == "qwen4exp_full_model_ar_batch"
        batch = telemetry["ar_batch"]
        assert batch["state_slots"] == 2
        assert batch["imports"] == 2
        # Two decode ticks run a step; the third samples the final token.
        assert batch["step_widths"] == {"2": 2}
        assert batch["failure"] is None
        assert batch["releases"] == 2
        assert batch["live_slots"] == 0
        # Row identity: each request keeps its own forced token.
        assert [output.generated_token_ids for output in outputs] == [
            (token,) * steps for token in forced
        ]
        # The c1 runners only prefilled: no c1 decode step ran for either row.
        assert prefilled is not None and len(prefilled) == 2
        for prompt_len, position, lease in prefilled.values():
            assert position == prompt_len
            assert lease is not None
        slots = sorted(lease[0] for *_, lease in prefilled.values())
        owners = {lease[1] for *_, lease in prefilled.values()}
        assert slots == [0, 1]
        assert len(owners) == 2
        # Both staging runners went back to the pool after the imports.
        assert len(native._available) == len(native._all_runners) == 2
        # Completion reclaimed both rows, so both pinned slots are free again.
        assert native._ar_batch.live_request_ids == ()
        assert native._ar_batch.runner.free_slots == (0, 1)
    finally:
        driver.close()


def test_serving_generator_greedy_batch_matches_the_c1_chain(model, monkeypatch):
    """The no-readback argmax path follows the committed c1 chain step for step.

    This fixture's greedy chain reaches a fixed point after the first token, so
    the comparison proves the imported state and the argmax step agree with c1;
    row identity is covered by the forced-token test above.
    """

    monkeypatch.setenv(AR_BATCH_ENV, "1")
    prompts = ((9, 4, 27), (15, 22))
    steps = 3
    expected = tuple(_c1_greedy(model, prompt, steps) for prompt in prompts)
    generator, driver = _driver(model)
    native = generator._resident_model_runner
    try:
        submission = driver.submit_detailed(
            GenerationRequest(
                prompts=prompts, max_tokens=steps, temperature=0.0, top_p=1.0,
                ignore_eos=True,
            )
        )
        _prefilled, results = _run_to_completion(driver, native, (submission,))
        outputs = results[0]
        batch = native.decode_telemetry()["ar_batch"]
        assert batch["step_widths"] == {"2": 2}
        assert tuple(output.generated_token_ids for output in outputs) == expected
        assert all(len(ids) == steps for ids in expected)
    finally:
        driver.close()


def test_a_generator_without_resident_weights_reports_the_c1_path(model, monkeypatch):
    """No model-owned weights means no batch; telemetry names the c1 route."""

    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="synthetic.gguf",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        tokenizer=FakeTokenizer(),
        runner=_c1(model),
        backend=model.backend,
    )
    try:
        native = generator.create_resident_model_runner(capacity=1)
        native.prepare()
        assert native.decode_telemetry()["decode_path"] == "qwen4exp_c1_request_owned_step"
        assert native._ar_batch is None
    finally:
        generator.close()
