"""The Gemma 4 generator must not block its own staged speculative route.

``engine_loop`` accepts two speculative protocols. The *legacy* one is a method
on the wrapped generator. The *staged* one lives on the runner
(``speculative_capability`` plus ``execute_target_frontier`` or
``execute_speculative_cycle``). Before it consults the staged hooks it reads the
generator's ``supports_speculative_mtp`` attribute and returns ``False`` if that
attribute is present and falsy::

    supports = getattr(self._inner, "supports_speculative_mtp", None)
    if supports is not None and not bool(supports):
        return False

``Gemma4GGUFGenerator`` does not implement the legacy method, and it used to say
so with a ``supports_speculative_mtp = False`` class attribute. That is accurate
about the legacy method but it is not a statement about the artifact, and it
stopped the runner from ever opting in -- the staged hooks would have been dead
code no matter what they returned.

These tests pin the routing rather than the drafter: that the generator does not
declare the miss, that a missing attribute still means no legacy protocol, and
that staged hooks on the runner are reachable once they exist.
"""

from __future__ import annotations

from hipengine.generation.engine_loop import SubmitPollTextGenerator
from hipengine.generation.gemma4_gguf import Gemma4GGUFGenerator


def test_the_generator_does_not_declare_a_speculative_capability_miss() -> None:
    """A class-level ``False`` would short-circuit the runner's staged hooks."""

    assert "supports_speculative_mtp" not in vars(Gemma4GGUFGenerator), (
        "Gemma4GGUFGenerator declares supports_speculative_mtp again. If it is "
        "set to a falsy value, engine_loop returns False before consulting the "
        "runner's staged hooks, so the staged speculative route cannot be "
        "reached however the runner is implemented. Declare the legacy miss by "
        "leaving the attribute off instead."
    )


class _StubInner:
    """Minimal TextGenerator stand-in: no legacy protocol, optional runner.

    ``engine_loop`` decides whether a resident runner exists by looking for a
    callable ``create_resident_model_runner`` on the inner generator, so that is
    the hook these tests provide or omit.
    """

    def __init__(self, staged: bool) -> None:
        if staged:
            self.create_resident_model_runner = _StagedRunner

    def generate(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("these tests only exercise capability routing")


class _StagedRunner:
    """A runner exposing the staged hooks the engine looks for."""

    def __init__(self, *, capacity: int | None) -> None:
        # The engine passes whatever capacity it resolved, which is None when the
        # caller did not ask for one.
        self.capacity = 32 if capacity is None else int(capacity)

    def speculative_capability(self, request_semantics):
        return None

    def execute_target_frontier(self, plan, *, commit: bool):
        raise AssertionError("not executed by these tests")


def test_a_missing_attribute_means_no_legacy_protocol() -> None:
    """Absence is not a claim; it must not be read as one in either direction."""

    loop = SubmitPollTextGenerator(_StubInner(staged=False))
    assert loop.supports_speculative_mtp is False


def test_staged_hooks_on_the_runner_are_reachable() -> None:
    """With no class-level miss, the staged route is what decides.

    This is the case the removed class attribute was blocking: the generator
    declares no legacy protocol, but the runner supplies the staged hooks, so
    the engine's own routing reaches them.
    """

    loop = SubmitPollTextGenerator(_StubInner(staged=True))
    assert loop.supports_speculative_mtp is True
