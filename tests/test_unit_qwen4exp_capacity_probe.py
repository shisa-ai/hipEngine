"""Flash-Next tier-1 probing must use the public allocation and request route."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from scripts import gguf_capacity_probe as probe


def args(**changes):
    values = dict(model=Path("model.gguf"), max_sequence_length=133760,
                  max_batch_size=2, prefill_chunk_size=0, prompt_length=997,
                  decode_tokens=4, kv_storage="bf16", execution_profile="production",
                  sidecar=None, candidate_budget=3, prefix_cache="radix")
    return SimpleNamespace(**{**values, **changes})


def fake_llm(monkeypatch, *, error=None, nonfinite=False, short=False, speculative=False,
             prepared_capacity=133760, leak=False, generation_error=None):
    events = []
    class Runner:
        max_sequence_length = 133760
        prefill_chunk_size = 4096
        config = SimpleNamespace(vocab_size=4)
        def reset(self):
            events.append("reset")
        def prefill(self, ids, **kw):
            from inspect import signature
            from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
            signature(Qwen4ExpGGUFResidentModelRunner.prefill).bind(self, ids, **kw)
            events.append(("logit_prefill", len(ids), kw))
            return SimpleNamespace(token_id=1, logits=np.array([1, np.nan if nonfinite else 2, 3, 4]))
        def step(self, token, **kw):
            from inspect import signature
            from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
            signature(Qwen4ExpGGUFResidentModelRunner.step).bind(self, token, **kw)
            events.append(("logit_step", kw))
            return SimpleNamespace(token_id=1, logits=np.ones(4))
    owner = SimpleNamespace(runner=Runner(), context_admission={"mode": "explicit"},
                            decode_telemetry=lambda: {"decode_path": "batch", "request_capacity": 2})
    class LLM:
        def __init__(self, **kw):
            events.append(("construct", kw))
        def prepare(self, **kw):
            events.append(("prepare", kw))
            if error:
                raise error
            return prepared_capacity
        def prepare_request_scratch(self, **kw):
            events.append(("scratch", kw))
            return {"execution_path": "qwen4exp_request_owned_runner_pool"}
        def _get_text_generator(self):
            return SimpleNamespace(_inner=owner)
        def generate_detailed(self, prompts, params):
            import threading
            assert len(prompts) == 1, "probe must submit independent requests, not a multi-prompt group"
            events.append(("generate", prompts, params, threading.get_ident()))
            if generation_error is not None:
                raise generation_error
            return [SimpleNamespace(generated_token_ids=(1,) * (3 if short else 4),
                                    prompt_tokens=None, telemetry=SimpleNamespace(
                                        decode_state=SimpleNamespace(prompt_tokens=len(prompt)),
                                        diagnostics={"speculative_provider": "qwen4_exp_mtp" if speculative else None,
                                                     "proposed_draft_tokens": 3 if speculative else 0})) for prompt in prompts]
        def close(self):
            events.append("close")
    monkeypatch.setattr("hipengine.LLM", LLM)
    monkeypatch.setattr("hipengine.core.memory.memory_stats", lambda: {
        "current_allocated_bytes": 16 if leak and "close" in events else 0,
        "peak_allocated_bytes": 1024, "active_allocations": 0})
    monkeypatch.setattr("hipengine.core.memory.reset_memory_stats", lambda: None)
    return events


def test_routes_by_gguf_architecture_without_allocating(monkeypatch):
    monkeypatch.setattr("hipengine.loading.gguf.GGUFReader", lambda _: SimpleNamespace(
        info=SimpleNamespace(architecture="qwen4exp")))
    seen = []
    monkeypatch.setattr(probe, "_probe_qwen4exp", lambda a, r, ids, **kw: seen.append((a, r, ids)))
    monkeypatch.setattr(sys, "argv", ["probe", "--max-sequence-length", "133760", "--prompt-length", "997"])
    probe.main()
    assert seen[0][0].kv_storage == "bf16"
    assert len(seen[0][2]) == 997


def test_model_directory_keeps_complete_shard_set(monkeypatch, tmp_path):
    first = tmp_path / "model-00001-of-00002.gguf"
    first.touch()
    (tmp_path / "model-00002-of-00002.gguf").touch()
    opened = []
    def reader(path):
        opened.append(path)
        return SimpleNamespace(info=SimpleNamespace(architecture="qwen4exp"))
    monkeypatch.setattr("hipengine.loading.gguf.GGUFReader", reader)
    seen = []
    monkeypatch.setattr(probe, "_probe_qwen4exp", lambda a, r, ids, **kw: seen.append(a.model))
    monkeypatch.setattr(sys, "argv", ["probe", "--model", str(tmp_path), "--max-sequence-length", "4096"])
    probe.main()
    assert opened == [first]
    assert seen == [tmp_path]


def test_public_capacity_and_cohort_then_full_vocab_finite_check(monkeypatch):
    events = fake_llm(monkeypatch)
    result = {}
    probe._probe_qwen4exp(args(), result, [42] * 997, np=np)
    assert result["status"] == "pass" and result["finite_logits"] is True
    assert events[0][1]["max_sequence_length"] == 133760
    assert events[0][1]["max_active_requests"] == 2
    assert events[1][0] == "prepare" and events[2][0] == "scratch"
    generated = [e for e in events if isinstance(e, tuple) and e[0] == "generate"]
    assert len(generated) == 2
    assert all(len(e[1]) == 1 and len(e[1][0]) == 997 for e in generated)
    assert len({e[3] for e in generated}) == 2
    assert generated[0][2].ignore_eos and generated[0][2].temperature == 0
    assert result["runner_capacity"] == 133760 and result["prefill_chunk_size"] == 4096
    assert events[-1] == "close"


@pytest.mark.parametrize("error", [MemoryError("out of memory"), RuntimeError("allocation failed")])
def test_preparation_failure_is_recorded_and_owner_closed(monkeypatch, error):
    events = fake_llm(monkeypatch, error=error)
    result = {}
    probe._probe_qwen4exp(args(), result, [42] * 997, np=np)
    assert result["status"] != "pass" and str(error) in result["error"]
    assert events[-1] == "close"
    assert not any(isinstance(e, tuple) and e[0] == "generate" for e in events)


@pytest.mark.parametrize("options", [{"nonfinite": True}, {"short": True}])
def test_bad_output_does_not_certify_capacity(monkeypatch, options):
    events = fake_llm(monkeypatch, **options)
    result = {}
    probe._probe_qwen4exp(args(), result, [42] * 997, np=np)
    assert result["status"] != "pass" and events[-1] == "close"


def test_request_failure_drains_threads_before_owner_close(monkeypatch):
    events = fake_llm(monkeypatch, generation_error=RuntimeError("request failed"))
    result = {}
    probe._probe_qwen4exp(args(), result, [42] * 997, np=np)
    assert result["status"] == "error" and "request failed" in result["error"]
    assert len([e for e in events if isinstance(e, tuple) and e[0] == "generate"]) == 2
    assert not any(isinstance(e, tuple) and e[0] == "logit_prefill" for e in events)
    assert events[-1] == "close"


def test_requested_chunk_cannot_be_silently_changed(monkeypatch):
    events = fake_llm(monkeypatch)
    result = {}
    probe._probe_qwen4exp(args(prefill_chunk_size=1024), result, [42] * 997, np=np)
    assert result["status"] != "pass" and "chunk" in result["error"]
    assert events[-1] == "close"


def test_requested_sidecar_is_loaded_and_mtp_must_run(monkeypatch):
    events = fake_llm(monkeypatch, speculative=False)
    result = {}
    probe._probe_qwen4exp(args(sidecar=Path("draft.gguf")), result, [42] * 997, np=np)
    assert events[0][1]["draft_model"] == "draft.gguf"
    assert events[0][1]["speculative_provider"] == "qwen4_exp_mtp"
    assert result["status"] != "pass" and "MTP" in result["error"]
    assert events[-1] == "close"


def test_successful_mtp_retains_actual_provider_accounting(monkeypatch):
    fake_llm(monkeypatch, speculative=True)
    result = {}
    probe._probe_qwen4exp(args(sidecar=Path("draft.gguf")), result, [42] * 997, np=np)
    assert result["status"] == "pass"
    assert all(row["speculative_used"] for row in result["responses"])


@pytest.mark.parametrize("options", [{"prepared_capacity": 4096}, {"leak": True}])
def test_smaller_capacity_or_leaked_owner_cannot_pass(monkeypatch, options):
    fake_llm(monkeypatch, **options)
    result = {}
    probe._probe_qwen4exp(args(), result, [42] * 997, np=np)
    assert result["status"] != "pass"


def test_explicit_incompatible_kv_is_rejected_before_probe(monkeypatch):
    monkeypatch.setattr("hipengine.loading.gguf.GGUFReader", lambda _: SimpleNamespace(
        info=SimpleNamespace(architecture="qwen4exp")))
    seen = []
    monkeypatch.setattr(probe, "_probe_qwen4exp", lambda *a, **kw: seen.append(True))
    monkeypatch.setattr(sys, "argv", ["probe", "--max-sequence-length", "133760", "--kv-storage", "int8_per_token_head"])
    with pytest.raises(SystemExit) as exc:
        probe.main()
    assert exc.value.code == 2 and seen == []
