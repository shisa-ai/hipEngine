"""Unit tests for the tier-1 GGUF capacity probe's architecture routing.

The probe is the cheap half of the capacity protocol: a short prompt at a
target ``max_sequence_length`` proves the same memory envelope a full-length
point would, in ~3 minutes instead of the hours the full point costs.  That
only works if the probe can actually reach the architecture in question, and
before this routing existed it could not reach gemma4 at all - it drove the
qwen35 resident session unconditionally and died inside the loader with
``expected GGUF architecture 'qwen35' or 'qwen35moe', got 'gemma4'``.

The contract these tests pin down:

* the branch is chosen from the artifact's declared ``general.architecture``,
  not from a filename, a path, or a substring of the model name;
* gemma4 reaches the gemma4 branch;
* an architecture with no branch fails *by name*, before any allocation,
  rather than falling into a branch that was never going to serve it;
* the declared context and the probe's prompt width stay independent, which
  is the whole reason tier 1 is cheap.

No HIP, no weights, no device: the reader and both branches are stubbed.
"""

from __future__ import annotations

import sys

import pytest

from scripts import gguf_capacity_probe as probe


class _StubReader:
    def __init__(self, architecture: str) -> None:
        self._architecture = architecture

    @property
    def info(self) -> object:
        return type("_Info", (), {"architecture": self._architecture})()


def _drive(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    *extra_argv: str,
) -> dict[str, object]:
    """Run ``main`` with every heavy dependency stubbed; return the branch taken."""

    taken: dict[str, object] = {}

    def _record(branch: str):
        def _branch(args, result, prompt_ids, *, np) -> None:
            taken["branch"] = branch
            taken["prompt_length"] = len(prompt_ids)
            taken["max_sequence_length"] = int(args.max_sequence_length)
            taken["result"] = result
        return _branch

    monkeypatch.setattr(
        "hipengine.loading.gguf.GGUFReader",
        lambda _path: _StubReader(architecture),
    )
    monkeypatch.setattr(probe, "_probe_gemma4", _record("gemma4"))
    monkeypatch.setattr(probe, "_probe_qwen35", _record("qwen35"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_capacity_probe.py",
            "--model", "/nonexistent/model.gguf",
            "--max-sequence-length", "262144",
            *extra_argv,
        ],
    )
    # ``main`` returns 0 only when a branch set ``status`` to pass, and the
    # stubs do not. The routing is what is under test, so the return value is
    # not asserted here.
    probe.main()
    return taken


def test_gemma4_routes_to_the_gemma4_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _drive(monkeypatch, "gemma4")["branch"] == "gemma4"


@pytest.mark.parametrize("architecture", ["qwen35", "qwen35moe"])
def test_the_qwen35_family_routes_to_the_qwen35_branch(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    assert _drive(monkeypatch, architecture)["branch"] == "qwen35"


def test_an_unknown_architecture_fails_by_name_before_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new family must not be silently handed to the qwen35 session.

    Falling through would raise a loader's "expected architecture" error
    describing a contract the caller never chose, which is exactly the
    confusion that made gemma4 look like it had no tier-1 answer.
    """

    with pytest.raises(SystemExit) as excinfo:
        _drive(monkeypatch, "laguna")

    message = str(excinfo.value)
    assert "laguna" in message
    assert "gemma4" in message and "qwen35" in message


def test_the_gemma4_branch_records_that_kv_storage_does_not_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The artifact must not claim a KV policy the probe did not select.

    The qwen35 branch takes ``--kv-storage``; the gemma4 branch certifies
    whatever route ``LLM`` builds, so echoing the flag's default into the
    artifact would misdescribe the envelope that was actually measured.
    """

    result = _drive(monkeypatch, "gemma4")["result"]
    assert isinstance(result, dict)
    assert result["kv_storage"] is None
    assert "does not apply" in str(result["kv_storage_note"])
    assert result["architecture"] == "gemma4"


def test_the_declared_context_does_not_size_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier 1's premise as a property of the arguments.

    The envelope is sized by ``max_sequence_length`` at session init, so the
    prompt width has to be able to stay small while the declared context is
    large.  If the probe ever coupled them, every point would silently become
    the tier-2 point it exists to avoid.
    """

    taken = _drive(monkeypatch, "gemma4")
    assert taken["max_sequence_length"] == 262144
    assert taken["prompt_length"] == 2048
    assert taken["prompt_length"] < taken["max_sequence_length"] // 100

    overridden = _drive(monkeypatch, "gemma4", "--prompt-length", "16")
    assert overridden["prompt_length"] == 16
    assert overridden["max_sequence_length"] == 262144
