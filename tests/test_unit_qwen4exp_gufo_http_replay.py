"""Contracts for the ordered, source-pinned Gufo HTTP replay adapter."""
from types import SimpleNamespace

import pytest

from scripts import qwen4exp_gufo_http_replay as replay


def observation(**changes):
    values = dict(prompt_tokens=6144, cached_prompt_tokens=4096, prefill_tokens=2048,
                  completion_tokens=128, prefill_ms=1000.0, decode_ms=2000.0,
                  draft_tokens=32, draft_accepted_tokens=16, wall_ms=3100.0,
                  completion_sha256="a" * 64, execution_plan="serial-c1")
    values.update(changes)
    return SimpleNamespace(**values, public=lambda: dict(values))


def test_native_denominators_keep_first_output_and_cached_prefix_separate():
    row = replay.checked_sample(observation(), depth=4096, outputs=128, speculative=True)
    assert row["rates"]["prefill_new_tok_s"] == 2048
    assert row["rates"]["decode_output_tok_s"] == 64
    assert row["rates"]["decode_transition_tok_s"] == 63.5
    assert row["output_identity"] == "visible_text_sha256_not_native_ids"


@pytest.mark.parametrize("change", [
    {"cached_prompt_tokens": 0}, {"prefill_tokens": 4096},
    {"prompt_tokens": 1}, {"completion_tokens": 127},
    {"decode_ms": 0}, {"prefill_ms": float("nan")},
    {"prefill_tokens": True}, {"draft_tokens": 0, "draft_accepted_tokens": 0},
    {"draft_tokens": 16, "draft_accepted_tokens": 17},
])
def test_invalid_counts_cache_timing_and_mtp_fallback_cannot_be_speed_rows(change):
    with pytest.raises(ValueError):
        replay.checked_sample(observation(**change), depth=4096, outputs=128, speculative=True)


def test_ar_rejects_any_draft_work():
    with pytest.raises(ValueError, match="AR"):
        replay.checked_sample(observation(), depth=4096, outputs=128, speculative=False)


def test_ordered_replay_carries_recalibration_to_later_depths():
    events = []
    session = SimpleNamespace(scope=None)
    tokenizer = SimpleNamespace(ratio=1.2, overhead=10, words_for=lambda n: n)
    session.request = lambda *a, **kw: events.append(("warmup", a[2]))
    def measure(session, url, tokenizer, **kw):
        events.append((kw["depth"], kw["repetition"], tokenizer.ratio))
        tokenizer.ratio += 0.1
        return observation(prompt_tokens=kw["depth"] + 2048,
                           cached_prompt_tokens=kw["depth"], draft_tokens=0,
                           draft_accepted_tokens=0)
    upstream = SimpleNamespace(_measure_depth=measure, synthetic_text=lambda seed, words: str(seed))
    rows = replay.ordered_depths(upstream, session, "url", tokenizer,
                                 depths=[0, 4096, 8192], repetitions=2, task="prose", speculative=False)
    assert [(e[0], e[1]) for e in events] == [("warmup", 16), (0, 0), (0, 1),
        (4096, 0), (4096, 1), (8192, 0), (8192, 1)]
    assert events[-1][2] == pytest.approx(1.7)
    assert len(rows) == 6


def test_reference_checkout_must_not_receive_output_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="outside"):
        replay.validate_paths(source, source / "artifacts")
    replay.validate_paths(source, tmp_path / "outputs")


def test_quality_binding_is_explicitly_incomplete_until_characterized():
    quality = replay.quality_columns()
    assert set(quality) == {"G0", "G1", "G2", "G3", "G4"}
    assert quality["G2"]["status"] == "not_applicable"
    assert quality["G1"]["status"] == "unavailable"
    assert quality["G4"]["status"] == "pending"
    assert quality["G0"]["status"] != "passed"
