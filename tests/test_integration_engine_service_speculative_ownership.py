"""Owner-thread contracts for the generic speculative spelling of EngineService.

``generate_speculative_detailed`` and ``stream_speculative_detailed`` name the
explicit speculative-provider spelling, which is distinct from the model-attached
MTP spelling. Before this unit both names fell through
``EngineService.__getattr__`` to ``SubmitPollTextGenerator`` and then to the
target generator, so a frontend thread executed whole-request provider
generation. The service must instead admit the work as one speculative child on
its sole driver thread, or fail closed when the driver owns no such lifecycle.
"""

from __future__ import annotations

from dataclasses import replace
import threading

import pytest

from hipengine.generation import (
    EngineService,
    FinishDetails,
    GenerationCancellationToken,
    GenerationCancelled,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationSubmission,
    SubmitPollTextGenerator,
)


def _request(
    prompt: str,
    *,
    max_tokens: int = 2,
    cancellation_token: GenerationCancellationToken | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        prompts=(prompt,),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
        cancellation_token=cancellation_token,
    )


class _GenericProviderDriver:
    """Fake resident driver with one model-attached generic speculative provider.

    ``generate_speculative_detailed``/``stream_speculative_detailed`` model the
    whole-request generator spelling that a frontend thread used to reach
    directly. ``submit_speculative_detailed`` models the resident driver's
    speculative child admission, which executes the model-owned speculative
    spelling on the driver thread and returns one ``verify_chain`` submission.
    """

    supports_controlled_streaming = True
    supports_stream_many = True
    supports_speculative = True
    supports_speculative_mtp = True

    def __init__(
        self,
        *,
        completes: bool = True,
        admission_error: BaseException | None = None,
        completion_error: BaseException | None = None,
    ) -> None:
        self._next_request_id = 1
        self._active: dict[int, GenerationRequest] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self._completes = bool(completes)
        self.admission_error = admission_error
        self.completion_error = completion_error
        self.admitted = threading.Event()
        self.submit_thread_ids: list[int] = []
        self.model_thread_ids: list[int] = []
        self.poll_thread_ids: set[int] = set()
        self.whole_request_thread_ids: list[int] = []
        self.mtp_spelling_calls = 0
        self.abort_reasons: dict[int, str] = {}
        self.release_order: list[int] = []
        self.closed = False

    # --- whole-request generator spelling (must not run on a frontend thread) ---
    def generate_speculative_detailed(
        self,
        request: GenerationRequest,
    ) -> list[GenerationOutput]:
        self.whole_request_thread_ids.append(threading.get_ident())
        return [GenerationOutput(text="frontend-generic", generated_token_ids=(1,))]

    def stream_speculative_detailed(self, request: GenerationRequest):
        self.whole_request_thread_ids.append(threading.get_ident())
        yield GenerationStreamChunk(text="frontend-generic", generated_token_ids=(1,))

    def generate_speculative_mtp_detailed(
        self,
        request: GenerationRequest,
    ) -> list[GenerationOutput]:
        self.mtp_spelling_calls += 1
        self.model_thread_ids.append(threading.get_ident())
        if self.admission_error is not None:
            raise self.admission_error
        return [
            GenerationOutput(
                text=f"model-owned:{prompt}",
                generated_token_ids=(700 + index, 701 + index),
                finish_details=FinishDetails(
                    reason="length",
                    length_limit=int(request.max_tokens),
                ),
            )
            for index, prompt in enumerate(request.prompts)
        ]

    # --- resident driver contract ---
    def submit_speculative_detailed(
        self,
        request: GenerationRequest,
    ) -> GenerationSubmission:
        self.submit_thread_ids.append(threading.get_ident())
        outputs = self.generate_speculative_mtp_detailed(request)
        if len(outputs) != 1:
            raise RuntimeError("one speculative submission must produce one output")
        request_id = self._next_request_id
        self._next_request_id += 1
        self._active[request_id] = request
        if self._completes:
            self._outputs[request_id] = outputs[0]
        self.admitted.set()
        return GenerationSubmission(
            request_ids=(request_id,),
            request=request,
            max_ticks=1,
            work_kind="verify_chain",
            execution_route="engine_service_speculative",
        )

    def submit_detailed(self, request: GenerationRequest) -> GenerationSubmission:
        request_id = self._next_request_id
        self._next_request_id += 1
        self._active[request_id] = request
        self._outputs[request_id] = GenerationOutput(
            text="ar",
            generated_token_ids=(11,),
            finish_details=FinishDetails(reason="length", length_limit=1),
        )
        return GenerationSubmission(
            request_ids=(request_id,),
            request=request,
            max_ticks=1,
        )

    def poll(self, *, max_ticks: int = 1):
        assert max_ticks == 1
        self.poll_thread_ids.add(threading.get_ident())
        return ()

    def generation_complete(self, submission: GenerationSubmission) -> bool:
        return all(
            request_id in self._outputs for request_id in submission.request_ids
        )

    def take_result(self, submission: GenerationSubmission) -> list[GenerationOutput]:
        if self.completion_error is not None:
            raise self.completion_error
        outputs = [self._outputs.pop(request_id) for request_id in submission.request_ids]
        for request_id in submission.request_ids:
            self._active.pop(request_id, None)
            self.release_order.append(request_id)
        return outputs

    def abort_submission(
        self,
        submission: GenerationSubmission,
        *,
        reason: str = "cancel",
    ) -> None:
        for request_id in submission.request_ids:
            self._active.pop(request_id, None)
            self._outputs.pop(request_id, None)
            self.abort_reasons[request_id] = str(reason)
            self.release_order.append(request_id)

    def detokenize(self, token_ids) -> str:
        return ",".join(str(int(token)) for token in token_ids)

    def live_loop_snapshot(self) -> dict[str, object]:
        return {"active_request_ids": sorted(self._active)}

    def close(self) -> None:
        self.closed = True
        self._active.clear()
        self._outputs.clear()


def test_generic_speculative_blocking_runs_model_work_on_driver_thread() -> None:
    driver = _GenericProviderDriver()
    service = EngineService(driver, idle_wait_seconds=0.001)
    caller_thread_id = threading.get_ident()
    try:
        outputs = service.generate_speculative_detailed(_request("generic-blocking"))
        driver_thread_id = service.driver_thread_id
        advertises_route = service.supports_speculative
    finally:
        service.close()

    assert advertises_route is True
    assert driver_thread_id != caller_thread_id
    assert [output.text for output in outputs] == ["model-owned:generic-blocking"]
    assert outputs[0].generated_token_ids == (700, 701)
    assert outputs[0].finish_details == FinishDetails(reason="length", length_limit=2)
    assert driver.whole_request_thread_ids == []
    assert driver.submit_thread_ids == [driver_thread_id]
    assert driver.model_thread_ids == [driver_thread_id]
    assert driver.poll_thread_ids == {driver_thread_id}
    assert driver.closed is True


def test_generic_speculative_streaming_runs_model_work_on_driver_thread() -> None:
    driver = _GenericProviderDriver()
    service = EngineService(driver, idle_wait_seconds=0.001)
    caller_thread_id = threading.get_ident()
    try:
        chunks = tuple(service.stream_speculative_detailed(_request("generic-stream")))
        driver_thread_id = service.driver_thread_id
    finally:
        service.close()

    assert driver_thread_id != caller_thread_id
    assert driver.whole_request_thread_ids == []
    assert driver.submit_thread_ids == [driver_thread_id]
    assert driver.model_thread_ids == [driver_thread_id]
    assert len(chunks) == 1
    assert chunks[0].text == "model-owned:generic-stream"
    assert chunks[0].generated_token_ids == (700, 701)
    assert chunks[0].finish_details == FinishDetails(reason="length", length_limit=2)


def test_generic_speculative_blocking_and_streaming_share_one_committed_tail() -> None:
    blocking_driver = _GenericProviderDriver()
    blocking_service = EngineService(blocking_driver, idle_wait_seconds=0.001)
    streaming_driver = _GenericProviderDriver()
    streaming_service = EngineService(streaming_driver, idle_wait_seconds=0.001)
    try:
        blocking = blocking_service.generate_speculative_detailed(
            _request("generic-shared")
        )[0]
        chunks = tuple(
            streaming_service.stream_speculative_detailed(_request("generic-shared"))
        )
    finally:
        blocking_service.close()
        streaming_service.close()

    assert blocking.generated_token_ids == (700, 701)
    assert blocking.text == "model-owned:generic-shared"
    assert tuple(
        token for chunk in chunks for token in (chunk.generated_token_ids or ())
    ) == blocking.generated_token_ids
    assert chunks[-1].finish_details == blocking.finish_details
    assert chunks[-1].text == blocking.text


def test_generic_speculative_preserves_provider_error_from_driver_thread() -> None:
    driver = _GenericProviderDriver(
        admission_error=RuntimeError("provider exploded"),
    )
    service = EngineService(driver, idle_wait_seconds=0.001)
    try:
        with pytest.raises(RuntimeError, match="provider exploded"):
            service.generate_speculative_detailed(_request("generic-error"))

        assert driver.model_thread_ids == [service.driver_thread_id]
        assert driver.whole_request_thread_ids == []
        assert service.closed is False
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0

        survivor = service.generate_detailed(_request("after-error", max_tokens=1))
        assert survivor[0].text == "ar"
    finally:
        service.close()


def test_generic_speculative_completion_error_reclaims_the_child() -> None:
    driver = _GenericProviderDriver(
        completion_error=RuntimeError("readback failed"),
    )
    service = EngineService(driver, idle_wait_seconds=0.001)
    try:
        with pytest.raises(RuntimeError, match="readback failed"):
            service.generate_speculative_detailed(_request("generic-readback"))

        assert driver.whole_request_thread_ids == []
        assert driver.release_order == [1]
        assert service.closed is False
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0
    finally:
        service.close()


def test_generic_speculative_cancellation_reclaims_the_child() -> None:
    driver = _GenericProviderDriver(completes=False)
    service = EngineService(driver, idle_wait_seconds=0.01)
    token = GenerationCancellationToken()
    request = replace(_request("generic-cancel", max_tokens=8), cancellation_token=token)
    errors: list[BaseException] = []

    def run_blocking() -> None:
        try:
            service.generate_speculative_detailed(request)
        except BaseException as exc:  # pragma: no cover - captured for assertion
            errors.append(exc)

    worker = threading.Thread(target=run_blocking)
    worker.start()
    try:
        assert driver.admitted.wait(timeout=5.0)
        token.cancel(FinishDetails(reason="cancelled", cancelled=True))
        worker.join(timeout=5.0)

        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], GenerationCancelled)
        assert errors[0].finish_details.cancelled is True
        assert driver.abort_reasons == {1: "disconnect"}
        assert driver.whole_request_thread_ids == []
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0
    finally:
        service.close()
        worker.join(timeout=5.0)


def test_generic_speculative_multi_prompt_children_keep_public_order() -> None:
    driver = _GenericProviderDriver()
    service = EngineService(driver, idle_wait_seconds=0.001)
    request = replace(_request("generic-parent"), prompts=("first", "second"))
    try:
        outputs = service.generate_speculative_detailed(request)
        snapshot = service.live_loop_snapshot()["engine_service"]
    finally:
        service.close()

    assert [output.text for output in outputs] == [
        "model-owned:first",
        "model-owned:second",
    ]
    assert driver.submit_thread_ids == [service.driver_thread_id] * 2
    assert driver.whole_request_thread_ids == []
    assert driver.release_order == [1, 2]
    assert snapshot["active_children"] == 0
    assert snapshot["speculative_routes"] == {
        "engine_service_verify_chain": 1,
        "legacy_prelaunch_fallback": 0,
    }
    assert snapshot["generic_speculative_children"] == 2


def test_generic_speculative_declines_driver_without_speculative_child_admission() -> None:
    class _WholeRequestOnlyDriver(_GenericProviderDriver):
        """Laguna-shaped driver: generic provider, no model-attached MTP spelling."""

        supports_speculative_mtp = False

        def submit_speculative_detailed(self, request: GenerationRequest):
            raise NotImplementedError(
                "speculative MTP generation is not supported by the wrapped generator"
            )

    driver = _WholeRequestOnlyDriver()
    service = EngineService(driver, idle_wait_seconds=0.001)
    try:
        assert service.supports_speculative is False
        with pytest.raises(NotImplementedError, match="sole-driver child ownership"):
            service.generate_speculative_detailed(_request("whole-request"))

        with pytest.raises(NotImplementedError, match="sole-driver child ownership"):
            tuple(service.stream_speculative_detailed(_request("whole-request")))

        assert driver.whole_request_thread_ids == []
        assert driver.submit_thread_ids == []
        assert driver.mtp_spelling_calls == 0
        assert service.closed is False
        assert service.live_loop_snapshot()["engine_service"]["active_children"] == 0

        survivor = service.generate_detailed(_request("after-decline", max_tokens=1))
        assert survivor[0].text == "ar"
    finally:
        service.close()


def test_generic_speculative_declines_driver_without_generic_provider_spelling() -> None:
    class _MtpOnlyDriver:
        """MTP-only driver: no explicit speculative-provider spelling at all."""

        supports_controlled_streaming = True
        supports_speculative = False
        supports_speculative_mtp = True

        def __init__(self) -> None:
            self.submit_thread_ids: list[int] = []
            self.closed = False

        def submit_speculative_detailed(self, request: GenerationRequest):
            self.submit_thread_ids.append(threading.get_ident())
            raise AssertionError("an MTP-only driver must not admit a generic child")

        def poll(self, *, max_ticks: int = 1):
            return ()

        def live_loop_snapshot(self) -> dict[str, object]:
            return {}

        def close(self) -> None:
            self.closed = True

    driver = _MtpOnlyDriver()
    service = EngineService(driver, idle_wait_seconds=0.001)
    try:
        assert service.supports_speculative is False
        with pytest.raises(NotImplementedError, match="no generic speculative provider"):
            service.generate_speculative_detailed(_request("mtp-only"))
        assert driver.submit_thread_ids == []
        assert service.closed is False
    finally:
        service.close()


def test_generic_speculative_routes_through_real_submit_poll_adapter() -> None:
    """The production adapter owns the generic spelling through child admission."""

    class _Inner:
        supports_speculative = True
        supports_speculative_mtp = True

        def __init__(self) -> None:
            self.spellings: list[tuple[str, int]] = []
            self.closed = False

        def generate_speculative_detailed(self, request: GenerationRequest):
            self.spellings.append(("generic", threading.get_ident()))
            return [
                GenerationOutput(text="adapter-generic", generated_token_ids=(900, 901))
            ]

        def generate_speculative_mtp_detailed(self, request: GenerationRequest):
            self.spellings.append(("mtp", threading.get_ident()))
            return [
                GenerationOutput(text="adapter-generic", generated_token_ids=(900, 901))
            ]

        def close(self) -> None:
            self.closed = True

    inner = _Inner()
    adapter = SubmitPollTextGenerator(inner, capacity=1)
    service = EngineService(adapter)
    try:
        output = service.generate_speculative_detailed(_request("adapter-generic"))[0]
        driver_thread_id = service.driver_thread_id
        snapshot = service.live_loop_snapshot()["engine_service"]
    finally:
        service.close()

    assert output.generated_token_ids == (900, 901)
    assert output.text == "adapter-generic"
    # The adapter's speculative child admission executes the model-owned
    # speculative spelling; the whole-request generic method stays unreached.
    assert inner.spellings == [("mtp", driver_thread_id)]
    assert adapter._speculative_outputs_by_request == {}
    assert snapshot["active_children"] == 0
    assert snapshot["generic_speculative_children"] == 1
    assert snapshot["last_speculative_route"] == "engine_service_verify_chain"
    assert inner.closed is True


def test_generic_speculative_declines_real_adapter_over_whole_request_provider() -> None:
    class _WholeRequestInner:
        supports_speculative = True
        supports_speculative_mtp = False

        def __init__(self) -> None:
            self.spellings: list[str] = []

        def generate_speculative_detailed(self, request: GenerationRequest):
            self.spellings.append("generic")
            return [GenerationOutput(text="bypass", generated_token_ids=(1,))]

        def close(self) -> None:
            return None

    inner = _WholeRequestInner()
    adapter = SubmitPollTextGenerator(inner, capacity=1)
    service = EngineService(adapter)
    try:
        assert service.supports_speculative is False
        with pytest.raises(NotImplementedError, match="sole-driver child ownership"):
            service.generate_speculative_detailed(_request("whole-request-adapter"))

        assert inner.spellings == []
        assert adapter.last_speculative_submission is None
        assert service.closed is False
    finally:
        service.close()


def test_mtp_speculative_spellings_keep_the_child_lifecycle() -> None:
    driver = _GenericProviderDriver()
    service = EngineService(driver, idle_wait_seconds=0.001)
    try:
        blocking = service.generate_speculative_mtp_detailed(_request("mtp-blocking"))
        chunks = tuple(service.stream_speculative_mtp_detailed(_request("mtp-stream")))
        snapshot = service.live_loop_snapshot()["engine_service"]
    finally:
        service.close()

    assert [output.text for output in blocking] == ["model-owned:mtp-blocking"]
    assert chunks[-1].generated_token_ids == (700, 701)
    assert driver.submit_thread_ids == [service.driver_thread_id] * 2
    assert driver.whole_request_thread_ids == []
    assert snapshot["last_speculative_route"] == "engine_service_verify_chain"
    assert snapshot["generic_speculative_children"] == 0


def test_mtp_legacy_fallback_still_runs_on_the_driver_thread() -> None:
    class _LegacyMtpDriver:
        supports_controlled_streaming = True
        supports_speculative_mtp = True

        def __init__(self) -> None:
            self.legacy_thread_ids: list[int] = []
            self.closed = False

        def generate_speculative_mtp_detailed(self, request: GenerationRequest):
            self.legacy_thread_ids.append(threading.get_ident())
            return [GenerationOutput(text="legacy", generated_token_ids=(800,))]

        def poll(self, *, max_ticks: int = 1):
            return ()

        def live_loop_snapshot(self) -> dict[str, object]:
            return {}

        def close(self) -> None:
            self.closed = True

    driver = _LegacyMtpDriver()
    service = EngineService(driver, idle_wait_seconds=0.001)
    try:
        outputs = service.generate_speculative_mtp_detailed(_request("legacy"))
        snapshot = service.live_loop_snapshot()["engine_service"]
    finally:
        service.close()

    assert outputs[0].generated_token_ids == (800,)
    assert driver.legacy_thread_ids == [service.driver_thread_id]
    assert snapshot["last_speculative_route"] == "legacy_prelaunch_fallback"
