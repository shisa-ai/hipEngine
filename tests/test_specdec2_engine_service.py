from __future__ import annotations

import threading

from hipengine.generation.engine_loop import SubmitPollTextGenerator
from hipengine.generation.engine_service import EngineService
from hipengine.generation.registry import GenerationOutput, GenerationRequest
from tests.test_specdec2_engine_frontier import _StagedCycleRunner


class _ServiceRunner(_StagedCycleRunner):
    def __init__(self) -> None:
        super().__init__()
        self.capacity = 4
        self._requests = {}
        self._outputs = {}
        self.target_entered = threading.Event()
        self.target_release = threading.Event()

    @property
    def active_request_ids(self):
        return tuple(self._requests)

    def prompt_tokens(self, prompt):
        return (10,)

    def scheduler_max_new_tokens(self, request):
        return int(request.max_tokens)

    def speculative_desired_candidate_count(self, request):
        return min(2, int(request.max_tokens))

    def register_batch(self, request_ids, request, *, prompt_rows):
        for request_id in request_ids:
            self._requests[int(request_id)] = request

    def reclaim(self, completed):
        self._outputs[int(completed.request_id)] = GenerationOutput(
            text=",".join(str(token) for token in completed.generated_tokens),
            generated_token_ids=completed.generated_tokens,
            finish_details=completed.finish_details,
        )

    def has_outputs(self, request_ids):
        return all(int(request_id) in self._outputs for request_id in request_ids)

    def missing_outputs(self, request_ids):
        return [
            int(request_id)
            for request_id in request_ids
            if int(request_id) not in self._outputs
        ]

    def take_outputs(self, request_ids):
        output = []
        for request_id in request_ids:
            rid = int(request_id)
            output.append(self._outputs.pop(rid))
            self._requests.pop(rid, None)
        return output

    def discard(self, request_ids):
        for request_id in request_ids:
            self._requests.pop(int(request_id), None)
            self._outputs.pop(int(request_id), None)

    def finalize_batch(self, request, request_ids, outputs):
        return None

    def execute_target_frontier(self, plan, frontier, complete_claims, *, commit):
        self.target_entered.set()
        assert self.target_release.wait(timeout=2.0)
        return super().execute_target_frontier(
            plan,
            frontier,
            complete_claims,
            commit=commit,
        )

    def close(self):
        self._requests.clear()
        self._outputs.clear()


class _Inner:
    supports_speculative_mtp = True

    def __init__(self, runner: _ServiceRunner) -> None:
        self.runner = runner
        self.legacy_calls = 0

    def create_resident_model_runner(self, *, capacity):
        self.runner.capacity = int(capacity or self.runner.capacity)
        return self.runner

    def generate_speculative_mtp_detailed(self, request):
        self.legacy_calls += 1
        raise AssertionError("staged EngineService route must not run legacy generation")

    def close(self):
        return None


def _request() -> GenerationRequest:
    return GenerationRequest(
        prompts=("staged",),
        max_tokens=3,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
    )


def test_engine_service_staged_submission_returns_before_generation_finishes() -> None:
    runner = _ServiceRunner()
    inner = _Inner(runner)
    adapter = SubmitPollTextGenerator(inner, capacity=1, prefill_chunk_size=8)
    service = EngineService(adapter, idle_wait_seconds=0.001)
    try:
        handle = service.submit_speculative_child(_request())
        assert runner.target_entered.wait(timeout=2.0)

        submission = adapter.last_speculative_submission
        assert submission is not None
        assert submission.execution_route == "engine_service_specdec2"
        assert submission.work_kind == "verify_chain"
        assert submission.work_item is None
        assert not handle.done
        assert inner.legacy_calls == 0
        assert adapter._speculative_outputs_by_request == {}

        runner.target_release.set()
        output = handle.result(timeout=2.0)

        assert output.generated_token_ids == (1, 2, 8000)
        assert output.text == "1,2,8000"
        assert runner.stage_order == [
            "claims",
            "reserve",
            "prepare",
            "propose",
            "target",
            "release",
        ]
        snapshot = service.live_loop_snapshot()["engine_service"]
        assert snapshot["active_children"] == 0
        assert snapshot["last_speculative_route"] == "engine_service_verify_chain"
        assert snapshot["speculative_routes"]["legacy_prelaunch_fallback"] == 0
    finally:
        runner.target_release.set()
        service.close()
