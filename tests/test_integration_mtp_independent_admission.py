"""Independent MTP clients must reach the resident scheduler before peers finish."""
import asyncio
import threading

from hipengine.llm import SamplingParams
from hipengine.server.api import _GenerationBatcher
from tests.test_integration_server_api import SpeculativeMTPFakeLLM


def test_mtp_clients_with_different_seeds_can_overlap_without_ar_submission():
    class ResidentEngine(SpeculativeMTPFakeLLM):
        supports_independent_generation = True
        server_mtp_batch_max_active_requests = 4

        def __init__(self):
            super().__init__()
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.release_first = threading.Event()

        def submit_independent_batches_detailed(self, batches):
            raise AssertionError("MTP clients must not enter the AR cohort API")

        def generate_speculative_mtp_detailed(self, prompts, sampling_params):
            if prompts == ("slow",):
                self.first_started.set()
                assert self.release_first.wait(5)
            else:
                self.second_started.set()
            return super().generate_speculative_mtp_detailed(prompts, sampling_params)

    async def run():
        engine = ResidentEngine()
        batcher = _GenerationBatcher(engine_factory=lambda: engine, batch_window_seconds=0, max_active_requests=4)
        slow = asyncio.create_task(batcher.submit(("slow",), SamplingParams(max_tokens=4, seed=1), route="speculative_mtp"))
        fast = None
        try:
            assert await asyncio.to_thread(engine.first_started.wait, 2)
            fast = asyncio.create_task(batcher.submit(("fast",), SamplingParams(max_tokens=4, seed=2), route="speculative_mtp"))
            assert await asyncio.to_thread(engine.second_started.wait, 1), "MTP admission waited for the first request to finish"
            assert await asyncio.wait_for(fast, 1) == ["mtp:fast"]
            assert not slow.done()
            assert engine.calls == []
        finally:
            engine.release_first.set()
            await asyncio.gather(*(task for task in (slow, fast) if task is not None), return_exceptions=True)
            await batcher.shutdown(grace_seconds=1)

    asyncio.run(run())
