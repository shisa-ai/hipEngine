"""Closing the resident driver releases the generator it wraps, and all it owns.

``LLM.close()`` reaches the model only through the resident driver. The driver
used to close just the resident runner, so a generator whose runner does not
close it (Gemma 4, PARO) kept its weights resident: a closed Gemma 4 ``LLM``
held about 30 GiB of device memory for the rest of the process.
"""

from __future__ import annotations

from hipengine.generation.engine_loop import SubmitPollTextGenerator


class _Runner:
    capacity = 2
    active_request_ids = ()

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def close(self) -> None:
        self._events.append("runner")


class _ResidentGenerator:
    def __init__(self) -> None:
        self.events: list[str] = []

    def create_resident_model_runner(self, *, capacity=None):
        return _Runner(self.events)

    def close(self) -> None:
        self.events.append("generator")


def test_close_releases_the_runner_then_the_generator() -> None:
    generator = _ResidentGenerator()
    SubmitPollTextGenerator(generator).close()
    assert generator.events == ["runner", "generator"]


def test_a_failing_runner_close_still_releases_the_generator() -> None:
    generator = _ResidentGenerator()
    driver = SubmitPollTextGenerator(generator)

    def fail() -> None:
        generator.events.append("runner")
        raise RuntimeError("runner close failed")

    driver._runner.close = fail
    try:
        driver.close()
    except RuntimeError:
        pass
    else:  # pragma: no cover - the runner error must surface
        raise AssertionError("runner close error was swallowed")
    assert generator.events == ["runner", "generator"]


def test_gemma4_generator_close_releases_its_speculative_provider() -> None:
    from hipengine.generation.gemma4_gguf import Gemma4GGUFGenerator

    class _Provider:
        closed = 0

        def capabilities(self):
            return {"max_verifier_rows": 1}

        def generate_detailed(self, request):  # pragma: no cover - not called
            raise AssertionError

        def stream_detailed(self, request):  # pragma: no cover - not called
            raise AssertionError

        def close(self) -> None:
            self.closed += 1

    generator = Gemma4GGUFGenerator(
        model_path="/nonexistent/gemma4.gguf",
        weight_index=None,
        model_plugin=None,
        backend="hip_gfx1100",
    )
    provider = _Provider()
    generator.attach_speculative_provider(provider)
    generator.close()
    generator.close()
    assert provider.closed == 1
