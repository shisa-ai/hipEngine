"""Host contracts for Laguna's generalized runtime-mode comparison harness."""

from types import SimpleNamespace

import pytest

from scripts import laguna_gate_doublebuf_ab as comparison
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession


@pytest.mark.parametrize("values,message", [
    (["missing_separator"], "LABEL=RUNTIME_MODE"),
    (["=mode"], "LABEL=RUNTIME_MODE"),
    (["label="], "LABEL=RUNTIME_MODE"),
    (["a=x", "a=y"], "duplicate"),
    (["a=x"], "at least two"),
])
def test_mode_parser_rejects_ambiguous_comparison_arms(values, message):
    with pytest.raises(ValueError, match=message):
        comparison._modes_from_args(values)


def test_mode_parser_preserves_generalized_arms_and_owns_default_copy():
    defaults = comparison._modes_from_args(None)
    assert defaults == comparison._MODES
    defaults.clear()
    assert comparison._MODES
    assert comparison._modes_from_args(["gate=custom_gate", "down=selected_down:custom_down"]) == {
        "gate": "custom_gate", "down": "selected_down:custom_down",
    }


@pytest.mark.parametrize("mode,expected", [
    ("mmq128x32_d8_f32_wavecols_direct_doublebuf", [("gate", "mmq128x32_d8_f32_wavecols_direct_doublebuf")]),
    ("selected_down:custom_down", [("down", "custom_down")]),
])
def test_comparison_applies_only_requested_runtime_controls(mode, expected):
    calls = []
    session = SimpleNamespace(**{
        method: lambda value, label=label: calls.append((label, value))
        for method, label in (
            ("set_selected_gate_up_mode", "gate"), ("set_selected_down_mode", "down"),
        )
    })
    comparison._apply_mode(session, mode)
    assert calls == expected


@pytest.mark.parametrize("failure", [False, True])
def test_runtime_comparison_captures_complete_state_and_closes_child(monkeypatch, failure):
    calls = []

    def prefill(tokens):
        calls.append(("prefill", tuple(tokens)))
        if failure:
            raise RuntimeError("synthetic prefill failure")
        return SimpleNamespace(next_token_id=9, next_token_logit=0.5, logits="logits", final_hidden="final", post_layer_hidden="post")

    child = SimpleNamespace(
        set_selected_gate_up_mode=lambda mode: calls.append(("mode", mode)),
        prefill=prefill, position=3, close=lambda: calls.append(("close",)),
        runtime=SimpleNamespace(device_synchronize=lambda: calls.append(("sync",))),
    )

    def child_session(owner, *, context_length):
        calls.append(("child", context_length))
        return child

    monkeypatch.setattr(comparison, "_child_session", child_session)
    monkeypatch.setattr(comparison, "_device_digest", lambda runtime, value, dtype: "digest:" + value)
    monkeypatch.setattr(comparison, "_kv_digest", lambda session: "digest:kv")
    clock = iter((1.0, 2.0))
    monkeypatch.setattr(comparison.time, "perf_counter", lambda: next(clock))
    if failure:
        with pytest.raises(RuntimeError, match="synthetic prefill failure"):
            comparison._run(object(), "custom_gate", [1, 2, 3])
    else:
        assert comparison._run(object(), "custom_gate", [1, 2, 3]) == {
            "tok_s": 3.0, "length": 3, "next_token": 9, "next_token_logit_hex": float(0.5).hex(),
            "logits_sha256": "digest:logits", "final_hidden_sha256": "digest:final",
            "post_layer_hidden_sha256": "digest:post", "kv_sha256": "digest:kv", "final_position": 3,
        }
        assert ("sync",) in calls
    assert calls[:2] == [("child", 3), ("mode", "custom_gate")]
    assert calls[-1] == ("close",)


@pytest.mark.parametrize("mode,expected", [
    ("q6_wmma_current", (False, False, False)),
    ("q6_wmma_prefetch", (True, False, False)),
    ("q6_wmma_prefetch_activation", (True, True, False)),
    ("q6_wmma_precomputed_sums", (True, True, True)),
])
@pytest.mark.parametrize("initial", [(True, True, True), (False, False, False)])
def test_q6_comparison_scope_keeps_sum_producer_and_consumer_consistent(mode, expected, initial):
    for name in ("set_q6_wmma_prefetch_weight", "set_q6_wmma_prefetch_activation", "set_q6_precomputed_activation_sums"):
        assert not hasattr(LagunaGGUFResidentSession, name)
    session = object.__new__(LagunaGGUFResidentSession)
    session.q6_qmicro_planar = True
    (session.q6_wmma_prefetch_weight, session.q6_wmma_prefetch_activation,
     session.q6_precomputed_activation_sums) = initial
    with comparison._mode_scope(session, mode):
        assert (
            session.q6_wmma_prefetch_weight, session.q6_wmma_prefetch_activation,
            session.q6_precomputed_activation_sums,
        ) == expected
        with pytest.raises(RuntimeError, match="synthetic"):
            with session.q6_prefetch_diagnostic(weight=False, activation=False, sums=False):
                raise RuntimeError("synthetic")
        assert (
            session.q6_wmma_prefetch_weight, session.q6_wmma_prefetch_activation,
            session.q6_precomputed_activation_sums,
        ) == expected
    assert (
        session.q6_wmma_prefetch_weight, session.q6_wmma_prefetch_activation,
        session.q6_precomputed_activation_sums,
    ) == initial


@pytest.mark.parametrize("planar,selected", [
    (False, (True, False, False)),
    (True, (False, True, False)),
    (True, (True, False, True)),
])
def test_q6_diagnostic_rejects_invalid_bundle_before_mutation(planar, selected):
    session = object.__new__(LagunaGGUFResidentSession)
    session.q6_qmicro_planar = planar
    session.q6_wmma_prefetch_weight = False
    session.q6_wmma_prefetch_activation = False
    session.q6_precomputed_activation_sums = False
    with pytest.raises(ValueError):
        with session.q6_prefetch_diagnostic(
            weight=selected[0], activation=selected[1], sums=selected[2],
        ):
            pytest.fail("invalid bundle entered")
    assert not session.q6_wmma_prefetch_weight
    assert not session.q6_wmma_prefetch_activation
    assert not session.q6_precomputed_activation_sums
