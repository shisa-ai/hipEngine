"""The forced-token gate must be able to fail, not just to pass.

The gate's job is to catch a speculative route that ignores the forced queue, one
that consumes it more than once, and one that lets it leak into another request.
Each check is therefore driven against a synthetic arm pair that violates exactly
one thing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    import sys

    spec = importlib.util.spec_from_file_location(
        "mtp_forced_token_gate", ROOT / "scripts" / "mtp_forced_token_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    # The gate declares dataclasses, which resolve their module through
    # ``sys.modules``; a loader that skips registration breaks them.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_module()

MTP_PATH = gate.MTP_EXECUTION_PATH
AR_PATH = "gguf_packed_ar_host_sampler_decode"

# The forced prefixes and the continuation the fake arms publish after them. The
# values are arbitrary ids: the gate compares positions, not vocabularies.
FORCED = {
    "forced_two_tokens": (101, 102),
    "forced_three_tokens_greedy": (101, 102, 103),
    "forced_tokens_sampled": (101, 102),
}
SEQUENCE = (7, 103, 104)
CONTINUATION = (201, 202, 203, 204)
SEED_IDS = (301, 302, 303)
AFTER_IDS = SEED_IDS
PROBE_IDS = (401, 402, 403)


def _fields(name: str) -> dict:
    if name in FORCED:
        return {
            "forced_tokens_pending": list(FORCED[name]),
            "forced_token_reason": "gate",
        }
    return {
        "force_sequence_completion_token_sequences": [list(SEQUENCE)],
        "force_sequence_completion_reason": "gate",
    }


def _request(ids: tuple[int, ...], *, path: str) -> dict:
    return {
        "text": "".join(f" t{token}" for token in ids),
        "ids": list(ids),
        "telemetry": {"execution_path": path},
    }


DECLARED_CONTEXT = 4096


def _arm(
    *,
    arm: str,
    case_ids: dict[str, tuple[int, ...]] | None = None,
    seed_ids: tuple[int, ...] = SEED_IDS,
    after_ids: tuple[int, ...] = AFTER_IDS,
    probe_ids: tuple[int, ...] = PROBE_IDS,
    seed_path: str | None = None,
    case_path: str | None = None,
    probe_by_arm: tuple[int, ...] | None = None,
    resident_context: int | None = None,
) -> dict:
    """Build the observations one arm returns, with per-case id overrides."""

    path = MTP_PATH if arm == "mtp" else AR_PATH
    overrides = dict(case_ids or {})
    cases: dict[str, dict] = {}
    for case in gate.CASES:
        fields = _fields(case.name)
        default = (
            tuple(fields.get("forced_tokens_pending") or SEQUENCE) + CONTINUATION
        )
        cases[case.name] = {
            "fields": fields,
            "probe": (
                None
                if case.name != "force_sequence_completion"
                else _request(probe_by_arm or probe_ids, path=path)
            ),
            "request": _request(overrides.get(case.name, default), path=case_path or path),
        }
    return {
        "seed": _request(seed_ids, path=seed_path or path),
        "cases": cases,
        "after": _request(after_ids, path=path),
        "tokenizer": {"seed_token_text": " river"},
        "declared_context_tokens": DECLARED_CONTEXT,
        "resident_context_tokens": resident_context or DECLARED_CONTEXT,
    }


def _run_pair(mtp_overrides: dict | None = None, ar_overrides: dict | None = None) -> dict:
    mtp = gate._check(_arm(arm="mtp", **(mtp_overrides or {})), arm="mtp")
    ar = gate._check(_arm(arm="ar", **(ar_overrides or {})), arm="ar")
    return gate._compare({"mtp": mtp, "ar": ar})


def test_a_clean_pair_passes_every_check() -> None:
    comparison = _run_pair()
    assert set(comparison["acceptance"]) == {
        "queue_isolated",
        "force_sequence_completion",
        "forced_prefix_published",
        "continuation_parity",
        "route_ran",
    }
    rows = {row["case"]: row for row in comparison["rows"]}
    assert rows["forced_two_tokens"]["forced_tokens"] == 2
    assert rows["force_sequence_completion"]["force_sequence_tokens"] == len(SEQUENCE)
    assert all(row["mtp_execution_path"] == MTP_PATH for row in comparison["rows"])
    assert all(row["ar_execution_path"] == AR_PATH for row in comparison["rows"])


def test_a_route_that_ignores_the_queue_is_caught() -> None:
    with pytest.raises(AssertionError, match="forced_prefix_not_published"):
        _run_pair(
            {
                "case_ids": {
                    "forced_two_tokens": CONTINUATION,
                    "forced_three_tokens_greedy": CONTINUATION,
                    "forced_tokens_sampled": CONTINUATION,
                }
            }
        )


def test_a_route_that_consumes_the_queue_twice_is_caught() -> None:
    """The forced prefix published twice is a continuation mismatch."""

    doubled = FORCED["forced_two_tokens"] + FORCED["forced_two_tokens"] + CONTINUATION
    with pytest.raises(AssertionError, match="published_ids_differ_from_the_autoregressive_arm"):
        _run_pair({"case_ids": {"forced_two_tokens": doubled}})


def test_a_route_that_drops_the_forced_token_mid_chain_is_caught() -> None:
    dropped = (FORCED["forced_two_tokens"][0],) + CONTINUATION
    with pytest.raises(AssertionError, match="forced_prefix_not_published"):
        _run_pair({"case_ids": {"forced_two_tokens": dropped}})


def test_a_route_that_never_completes_the_sequence_is_caught() -> None:
    """Both arms agree on ids that are not the queued sequence."""

    with pytest.raises(AssertionError, match="force_sequence_not_completed"):
        _run_pair(
            {"case_ids": {"force_sequence_completion": CONTINUATION}},
            {"case_ids": {"force_sequence_completion": CONTINUATION}},
        )


def test_a_speculative_case_that_did_not_speculate_is_caught() -> None:
    with pytest.raises(AssertionError, match="speculative_case_did_not_speculate"):
        _run_pair({"case_path": AR_PATH})


def test_a_speculative_seed_that_did_not_speculate_is_caught() -> None:
    with pytest.raises(AssertionError, match="unforced_seed_did_not_speculate"):
        _run_pair({"seed_path": AR_PATH})


def test_an_autoregressive_arm_that_speculated_is_caught() -> None:
    with pytest.raises(AssertionError, match="autoregressive_arm_speculated"):
        gate._check(_arm(arm="ar", seed_path=MTP_PATH), arm="ar")


def test_a_leaked_queue_is_caught_by_the_next_request_comparison() -> None:
    """A leak moves the speculative arm's next request and nothing else."""

    with pytest.raises(
        AssertionError,
        match="unforced_request_after_a_forced_one_differs_between_arms",
    ):
        _run_pair({"after_ids": FORCED["forced_two_tokens"] + AFTER_IDS})


def test_a_next_request_that_differs_from_the_seed_is_caught() -> None:
    """Both arms agree, but the unforced request no longer matches the seed."""

    shifted = (999, 998, 997)
    with pytest.raises(AssertionError, match="queue_leaked_into_the_next_request"):
        _run_pair({"after_ids": shifted}, {"after_ids": shifted})


def test_a_probe_that_differs_between_arms_is_caught() -> None:
    with pytest.raises(AssertionError, match="unforced_probe_differs_between_arms"):
        _run_pair({"probe_by_arm": (999, 998)})


def test_a_case_that_published_nothing_is_caught() -> None:
    """An unforced case that published nothing has no shared prefix to compare."""

    with pytest.raises(AssertionError, match="case_published_nothing"):
        _run_pair(
            {"case_ids": {"force_sequence_completion": ()}},
            {"case_ids": {"force_sequence_completion": ()}},
        )


def test_a_seed_that_differs_between_arms_is_caught() -> None:
    """The unforced seed is the reference, so a disagreement is a failure."""

    mtp = gate._check(_arm(arm="mtp", seed_ids=(1, 2, 3)), arm="mtp")
    ar = gate._check(_arm(arm="ar", seed_ids=(4, 5, 6), after_ids=(4, 5, 6)), arm="ar")
    with pytest.raises(AssertionError, match="unforced_seed_differs_between_arms"):
        gate._compare({"mtp": mtp, "ar": ar})


class _FakeLLM:
    """The public LLM surface the gate uses, with a fixed greedy trajectory."""

    def __init__(self, model: str, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs
        self.prepared: list[int | None] = []
        self.requests: list[tuple[str, dict]] = []
        self.closed = False

    def prepare(self, *, max_sequence_length=None) -> int:
        self.prepared.append(max_sequence_length)
        return int(max_sequence_length)

    def tokenize(self, text: str) -> tuple[int, ...]:
        # One id per word, so a forced phrase is a multi-token queue.
        return tuple(600 + index for index, _word in enumerate(str(text).split()))

    def detokenize(self, token_ids) -> str:
        return " ".join(f"t{int(token)}" for token in token_ids)

    def generate_speculative_mtp_detailed(self, prompts, sampling_params):
        return self._generate(prompts, sampling_params, mtp=True)

    def generate_detailed(self, prompts, sampling_params):
        return self._generate(prompts, sampling_params, mtp=False)

    def _generate(self, prompts, sampling_params, *, mtp: bool):
        from hipengine.generation.registry import GenerationOutput

        prompt = str(prompts[0])
        forced = tuple(int(token) for token in sampling_params.forced_tokens_pending)
        sequences = tuple(
            tuple(int(token) for token in sequence)
            for sequence in sampling_params.force_sequence_completion_token_sequences
        )
        if sequences and not forced:
            ids = sequences[0] + (701, 702)
        elif forced:
            ids = forced + (703, 704)
        else:
            # An unforced request is deterministic, so the seed run and a run
            # after a forced request must publish the same ids.
            ids = (701, 702)
        self.requests.append((prompt, {"forced": forced}))
        return [
            GenerationOutput(
                text=self.detokenize(ids),
                generated_token_ids=ids,
                telemetry={
                    "decode_state": {
                        "execution_path": (
                            MTP_PATH if mtp else AR_PATH
                        )
                    }
                },
            )
        ]

    _text_generator = SimpleNamespace(
        # The sizing decision this engine would make: the declared context.
        resident_context_tokens=4096,
    )

    def close(self) -> None:
        self.closed = True


def test_the_arm_runner_pins_the_context_and_produces_a_checkable_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live path must pin the context and return observations the checks accept."""

    import hipengine.llm

    monkeypatch.setattr(hipengine.llm, "LLM", _FakeLLM)
    monkeypatch.setattr(gate, "_preflight_memory", lambda: 100.0)
    kwargs = {
        "backend": "hip_gfx1151",
        "kv_storage": "bf16",
        "prefix_cache": "off",
        "max_sequence_length": 4096,
    }
    mtp_raw = gate._run_arm("fake-model", serving="auto", mtp=True, kwargs=kwargs)
    mtp = gate._check(mtp_raw, arm="mtp")
    ar_raw = gate._run_arm("fake-model", serving="off", mtp=False, kwargs=kwargs)
    ar = gate._check(ar_raw, arm="ar")
    comparison = gate._compare({"mtp": mtp, "ar": ar})
    assert comparison["rows"]
    assert mtp_raw["tokenizer"]["seed_token_text"]
    # Every case that needs a probe got one, and the force-sequence case is the
    # only one that does.
    probed = [
        name for name, entry in mtp_raw["cases"].items() if entry["probe"] is not None
    ]
    assert probed == ["force_sequence_completion"]
    # The queue is published on the speculative arm and the request's fields
    # reach the sampler.
    forced_case = mtp_raw["cases"]["forced_two_tokens"]
    assert forced_case["request"]["ids"][:2] == forced_case["fields"][
        "forced_tokens_pending"
    ]


def test_a_sampled_case_may_diverge_after_its_forced_prefix() -> None:
    """The accept walk draws per verified row, so a sampled case is a new draw."""

    forced = FORCED["forced_tokens_sampled"]
    comparison = _run_pair(
        {"case_ids": {"forced_tokens_sampled": forced + (901, 902)}},
        {"case_ids": {"forced_tokens_sampled": forced + (903, 904)}},
    )
    row = next(
        item for item in comparison["rows"] if item["case"] == "forced_tokens_sampled"
    )
    assert row["identical_output"] is False
    # The forced prefix is shared; the draws after it are not.
    assert row["shared_prefix"] == len(forced)


def test_a_sampled_case_that_ignores_its_own_queue_is_caught() -> None:
    forced = FORCED["forced_tokens_sampled"]
    with pytest.raises(AssertionError, match="forced_prefix_not_published"):
        _run_pair(
            {"case_ids": {"forced_tokens_sampled": (901, 902) + forced}},
            {"case_ids": {"forced_tokens_sampled": forced + (903, 904)}},
        )


def test_a_greedy_case_that_diverges_after_its_forced_prefix_is_caught() -> None:
    """A greedy case is deterministic, so any divergence is a defect."""

    forced = FORCED["forced_two_tokens"]
    with pytest.raises(
        AssertionError, match="published_ids_differ_from_the_autoregressive_arm"
    ):
        _run_pair({"case_ids": {"forced_two_tokens": forced + (901, 902)}})


def test_a_session_sized_to_something_else_than_declared_is_caught() -> None:
    """The incident: declared 4096, resident sized to the model's 262144."""

    with pytest.raises(
        AssertionError, match="the_resident_session_was_not_sized_to_the_declared_context"
    ):
        gate._check(_arm(arm="mtp", resident_context=262144), arm="mtp")
