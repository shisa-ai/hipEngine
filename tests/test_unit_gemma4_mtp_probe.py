"""CPU checks for the public Gemma 4 MTP probe's verdict and accounting."""
from types import SimpleNamespace

import pytest

from scripts import gemma4_mtp_probe as probe


def output(ids, text="same", reason="length"):
    return SimpleNamespace(generated_token_ids=ids, text=text,
                           finish_details=SimpleNamespace(reason=reason))


@pytest.mark.parametrize("plain,spec", [
    (output((1, 2)), output((1, 3))),
    (output(None), output((1, 2))),
    (output((1, 2)), output(None)),
    (output((1, 2)), output((1, 2), text="different")),
    (output((1, 2)), output((1,), reason="length")),
    (output((1, 2, 3)), output((1, 2, 3))),
])
def test_probe_rejects_invalid_comparisons(plain, spec):
    row = probe.compare_outputs(plain, spec, requested_tokens=2, plain_s=2, spec_s=1)
    assert not row["passed"]
    assert row["speedup"] is None


def test_probe_accounts_for_actual_tokens_and_whole_call_scope():
    row = probe.compare_outputs(output((1, 2)), output((1, 2)),
                                requested_tokens=2, plain_s=2, spec_s=1)
    assert row["passed"]
    assert row["plain_tok_s"] == 1
    assert row["spec_tok_s"] == 2
    assert row["speedup"] == 2
    assert row["timing_scope"] == "whole_public_call_including_prefill"


@pytest.mark.parametrize("reason", ["eos", "stop"])
def test_probe_accepts_matching_eos_without_inventing_tokens(reason):
    row = probe.compare_outputs(output((1,), reason=reason), output((1,), reason=reason),
                                requested_tokens=64, plain_s=2, spec_s=1)
    assert row["passed"]
    assert row["plain_tok_s"] == 0.5
    assert row["spec_tok_s"] == 1
    assert not row["fixed_length_complete"]


def test_probe_does_not_accept_unexplained_short_output():
    row = probe.compare_outputs(output((1,)), output((1,)),
                                requested_tokens=64, plain_s=2, spec_s=1)
    assert not row["passed"]


@pytest.mark.parametrize("wall", [0, -1, float("inf"), float("nan")])
def test_probe_rejects_invalid_times(wall):
    row = probe.compare_outputs(output((1, 2)), output((1, 2)),
                                requested_tokens=2, plain_s=wall, spec_s=1)
    assert not row["passed"]
    assert row["speedup"] is None


def test_probe_loads_full_category_and_heldout_suites():
    from pathlib import Path
    cases = probe.load_cases([
        Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"),
        Path("benchmarks/prompts/gdn-prefill-category-heldouts.jsonl"),
    ], "unused")
    assert len(cases) == 18
    assert {case["category"] for case in cases} == {
        "code", "general_en", "general_ja", "mixed_ja_en",
    }


def test_probe_main_fails_on_equal_text_with_different_ids(monkeypatch):
    class LLM:
        speculative_mtp_serving = True
        def __init__(self, **kwargs): pass
        def _get_text_generator(self): return SimpleNamespace(supports_speculative_mtp=True)
        def generate(self, *args): return ["same"]
        def generate_detailed(self, *args): return [output((1, 2))]
        def generate_speculative_mtp_detailed(self, *args): return [output((1, 3))]
    monkeypatch.setattr(probe.hipengine, "LLM", LLM)
    monkeypatch.setattr(probe.os.path, "exists", lambda _: True)
    monkeypatch.setattr("sys.argv", ["probe", "--tokens", "2"])
    assert probe.main() == 1
