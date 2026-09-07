"""CPU guards for actual-route conditional-logit diagnostics (not certification)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.qwen38_packed_c1_logits import compare_logits, validate_capture, replay_strict


@pytest.mark.parametrize("batched_logits", [False, True])
def test_strict_replay_prefills_only_prompt_then_steps_generated_prefix(batched_logits):
    from types import SimpleNamespace
    calls = []
    session = SimpleNamespace(position=0)

    def reset():
        session.position = 0

    def prefill(tokens, *, return_logits):
        calls.append(("prefill", tuple(tokens), return_logits))
        session.position += len(tokens)

    def step(token, *, return_logits):
        calls.append(("step", token, return_logits))
        session.position += 1
        logits = np.arange(16, dtype=np.float32)
        return SimpleNamespace(logits=logits[None, :] if batched_logits else logits)

    session.reset, session.prefill, session.step = reset, prefill, step
    result = replay_strict(session, dict(prefix=[1, 2, 3], prompt_length=2,
                                         position=3, tokens=[4, 5]))
    assert result.shape == (2, 16)
    assert calls == [("prefill", (1, 2), False), ("step", 3, False),
                     ("step", 4, True), ("step", 5, True)]
    assert session.position == 5


@pytest.mark.parametrize("failure", [None, "stale", "direct_top1", "root", "prefix"])
def test_actual_route_capture_requires_fresh_head_and_exact_context(monkeypatch, tmp_path, failure):
    from types import SimpleNamespace as NS
    import json
    from scripts import qwen38_packed_c1_logits as module
    from scripts import gguf_mtp_c1c8_server_bench as bench
    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter as Adapter
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession as Session

    prompt = dict(id="test", category="code", rendered_prompt="test prompt")
    monkeypatch.setattr(bench, "load_prompt_suite", lambda path: [prompt])
    target = NS(position=2, runtime=None, runner=NS(vocab_size=16),
                _verify_logits_buf=NS(ptr=1))
    job = dict(session=target, input_token_ids=(0 if failure == "root" else 12, 13))

    def head(owner, *args, **kwargs):
        owner._last_packed_lm_head_decode_path = (
            "direct_top1_rows" if failure == "direct_top1" else "row_linear_f32_logits")

    def verify(owner, jobs, **kwargs):
        if failure != "stale":
            Session._enqueue_target_block_rows_from_hidden(owner, 1, 2)
        return [object()]

    monkeypatch.setattr(Session, "_enqueue_target_block_rows_from_hidden", head)
    monkeypatch.setattr(Session, "verify_target_blocks_batch", verify)
    monkeypatch.setattr(module, "_read_device", lambda ptr, shape, dtype, runtime: np.zeros(shape, dtype=dtype))
    row = NS(prompt_ids=(10,) if failure == "prefix" else (10, 11),
             slot=NS(generated_ids=(12,)))
    adapter = NS(owner=NS(capacity=1, _row=lambda rid: row),
                 _physical_c1_request=lambda rid: True,
                 generator=NS(execution_profile="production", execution_profile_manifest_sha256="a"*64,
                              _kv_model_artifact_identity=lambda: NS(content_verified=True, sha256="b"*64)))
    plan = NS(speculative_request_ids=(7,), request_ids=(7,), candidate_counts=(1,),
              resident_slots=(0,), cycle_id=1)
    monkeypatch.setattr(Adapter, "_execute_target_frontier_batch",
                        lambda *args, **kwargs: Session.verify_target_blocks_batch(target, [job], device_result=True))
    monkeypatch.setattr(bench, "_run_arm", lambda *args, **kwargs: Adapter._execute_target_frontier_batch(adapter, plan))
    recorder = module.PackedC1Capture(tmp_path / "capture")
    original = Session.verify_target_blocks_batch
    try:
        recorder.install()
        if failure:
            with pytest.raises(ValueError):
                bench._run_arm(prompt="test prompt", width=1)
            assert recorder.records == []
        else:
            bench._run_arm(prompt="test prompt", width=1)
            assert recorder.records[0]["prefix"] == (10, 11)
            assert recorder.records[0]["tokens"] == [12, 13]
    finally:
        recorder.close(success=failure is None)
    assert Session.verify_target_blocks_batch is original
    payload = json.loads((tmp_path / "capture/capture.json").read_text())
    assert payload["complete"] is (failure is None)


def _capture():
    return dict(prefix=(10, 11), tokens=(12, 13, 0, 0), position=2,
                logical_rows=2, logits=np.zeros((4, 16), dtype=np.float32))


def test_capture_slices_only_valid_logical_rows():
    logits = validate_capture(**_capture())
    assert logits.shape == (2, 16)


@pytest.mark.parametrize("field,value", [
    ("position", 1), ("logical_rows", 0), ("logical_rows", 5),
    ("tokens", (12,)), ("tokens", (16, 13, 0, 0)),
    ("prefix", (10, -1)), ("logits", np.zeros((4, 0))),
    ("logits", np.full((4, 16), np.nan)),
])
def test_capture_rejects_misaligned_or_nonfinite_rows(field, value):
    row = _capture()
    row[field] = value
    with pytest.raises(ValueError):
        validate_capture(**row)


def test_identical_logits_pass_but_not_full_campaign_certification():
    logits = np.array([[0., 1., 2., 3., 4.]], dtype=np.float32)
    result = compare_logits(logits, logits.copy())
    assert result["numerical_envelope_passed"]
    assert result["mean_kl"] == 0
    assert result["top1"] == 1
    assert result["full_profile_qualification"] is False


def test_numerics_cannot_hide_a_tail_or_top1_failure():
    reference = np.tile(np.array([5., 0., 0., 0., 0.]), (100, 1))
    candidate = reference.copy()
    candidate[-1] = [0., 5., 0., 0., 0.]
    result = compare_logits(reference, candidate)
    assert not result["numerical_envelope_passed"]
    assert result["max_kl"] > .05
    assert result["review_rows"] == [99]


@pytest.mark.parametrize("candidate", [np.zeros((2, 5)), np.full((1, 5), np.inf)])
def test_comparison_rejects_shape_and_finite_failures(candidate):
    with pytest.raises(ValueError):
        compare_logits(np.zeros((1, 5)), candidate)
