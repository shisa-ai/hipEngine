from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.generation.registry import GenerationRequest
from hipengine.speculative.registry import SpeculativeProviderConfig


class _Tokenizer:
    eos_token_id = 2
    eot_token_id = 24
    byte_decoder = {}

    def decode(self, token_ids, *, skip_special: bool = False) -> str:
        pieces = {10: "A", 11: "B", 12: "C", 13: "D", 24: "</assistant>"}
        hidden = {2, 24} if skip_special else set()
        return "".join(pieces.get(int(token), f"T{int(token)}") for token in token_ids if int(token) not in hidden)


class _Runtime:
    def device_synchronize(self) -> None:
        return None


class _TargetSession:
    events: list[str] = []

    def __init__(self) -> None:
        self.runtime = _Runtime()
        self.position = -1
        self.closed = False

    def reset_state(self) -> None:
        self.events.append("target_reset")
        self.position = -1

    def close(self) -> None:
        self.events.append("target_close")
        self.closed = True


class _TargetGenerator:
    def __init__(self, model: Path, cache: Path) -> None:
        self.model_path = model
        self.backend = "hip_gfx1151"
        self.context_length = 4096
        self.tokenizer = _Tokenizer()
        self.repacked_cache_path = cache
        self._lock = threading.RLock()
        self._closed = False
        self.prepared = 0
        self.bound_sha256 = None
        self.session = _TargetSession()
        self.last_generation_outputs = ()
        self.last_batch_generation = None
        self.resident_weights = None

    def bind_repacked_cache_source_sha256(self, sha256: str) -> None:
        self.bound_sha256 = str(sha256)

    def _prepare_locked(self) -> None:
        self.prepared += 1

    def _open_session_locked(self) -> _TargetSession:
        return self.session

    def _prepare_request(self, request: GenerationRequest) -> tuple[int, ...]:
        if len(request.prompts) != 1:
            raise ValueError("exactly one prompt")
        prompt = request.prompts[0]
        if isinstance(prompt, str):
            raise TypeError("test target accepts token IDs")
        return tuple(int(token) for token in prompt)


class _Drafter:
    events: list[str] = []

    def __init__(self, target_session, drafter_model, *, candidate_budget, **kwargs) -> None:
        del drafter_model, kwargs
        self.target = target_session
        self.candidate_budget = int(candidate_budget)
        self._closed = False
        self.reset_calls = 0

    def reset_state(self) -> None:
        self.events.append("drafter_reset")
        self.reset_calls += 1

    def close(self) -> None:
        self.events.append("drafter_close")
        self._closed = True


class _Cycle:
    events: list[str] = []
    sequence = (10, 11, 24, 13)

    def __init__(self, target, drafter) -> None:
        self.target = target
        self.drafter = drafter
        self.closed = False
        self.index = 0

    def prefill(self, token_ids):
        self.events.append("prefill")
        self.target.position = len(tuple(token_ids)) - 1
        self.index = 1
        return SimpleNamespace(next_token_id=self.sequence[0], next_token_logit=1.0)

    def run_cycle(self, root_token_id, *, remaining_decode, stop_token_ids=()):
        assert int(root_token_id) == int(self.sequence[self.index - 1])
        stops = set(int(token) for token in stop_token_ids)
        visible = []
        while self.index < len(self.sequence) and len(visible) < min(2, remaining_decode):
            token = int(self.sequence[self.index])
            visible.append(token)
            self.index += 1
            if token in stops:
                break
        next_token = None if not visible or visible[-1] in stops or self.index >= len(self.sequence) else visible[-1]
        self.target.position += len(visible)
        proposal = SimpleNamespace(
            candidate_token_ids=(11, 24, 13, 14),
            topk_values=((1.0,),) * 4,
        )
        target_result = SimpleNamespace(
            accepted_draft_count=max(0, len(visible) - 1),
            target_top1_values=(1.0,) * (len(visible) + 1),
            next_token_id=next_token,
        )
        return SimpleNamespace(
            visible_output_ids=tuple(visible),
            proposal=proposal,
            target_batch=SimpleNamespace(tokens=(root_token_id, *visible)),
            target_result=target_result,
            proposal_seconds=0.001,
            target_verify_seconds=0.002,
            draft_commit_enqueue_seconds=0.0001,
            cycle_host_seconds=0.004,
            verifier_addresses_stable=True,
        )

    def close(self) -> None:
        self.events.append("cycle_close")
        self.closed = True


def _request(**overrides) -> GenerationRequest:
    values = {
        "prompts": ((7, 8),),
        "max_tokens": 3,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
    }
    values.update(overrides)
    return GenerationRequest(**values)


def _identity_tree(tmp_path: Path):
    from hipengine.generation.laguna_dflash import (
        LAGUNA_DFLASH_DRAFTER_REVISION,
        LAGUNA_DFLASH_DRAFTER_SHA256,
        LAGUNA_DFLASH_TARGET_SHA256,
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "laguna.gguf"
    model.touch()
    cache = tmp_path / "laguna.hipengine-repacked-v1"
    cache.mkdir()
    (cache / "manifest.json").write_text(
        json.dumps({"source": {"sha256": LAGUNA_DFLASH_TARGET_SHA256}}),
        encoding="utf-8",
    )
    hub = tmp_path / "hub"
    blob = hub / "blobs" / LAGUNA_DFLASH_DRAFTER_SHA256
    blob.parent.mkdir(parents=True)
    blob.touch()
    snapshot = hub / "snapshots" / LAGUNA_DFLASH_DRAFTER_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to(blob)
    return model, cache, snapshot


def test_laguna_dflash_provider_binds_identities_and_reports_rejected_economics(
    tmp_path,
) -> None:
    from hipengine.generation.laguna_dflash import (
        LAGUNA_DFLASH_DRAFTER_REVISION,
        LAGUNA_DFLASH_DRAFTER_SHA256,
        LAGUNA_DFLASH_TARGET_SHA256,
        LagunaDFlashTextProvider,
    )

    model, cache, snapshot = _identity_tree(tmp_path)
    target = _TargetGenerator(model, cache)
    provider = LagunaDFlashTextProvider(
        target,
        SpeculativeProviderConfig("dflash", snapshot, 4),
    )

    assert target.bound_sha256 == LAGUNA_DFLASH_TARGET_SHA256
    assert provider.capabilities() == {
        "provider": "dflash",
        "policy": "explicit_only",
        "default_enabled": False,
        "streaming_compatible": True,
        "candidate_budget": 4,
        "exactness_mode": "target_corrected_greedy",
        "processed_target_verification": False,
        "target": {
            "model": "poolside/Laguna-S-2.1-GGUF",
            "sha256": LAGUNA_DFLASH_TARGET_SHA256,
            "quant": "Q4_K_M",
        },
        "drafter": {
            "model": "poolside/Laguna-S-2.1-DFlash",
            "revision": LAGUNA_DFLASH_DRAFTER_REVISION,
            "sha256": LAGUNA_DFLASH_DRAFTER_SHA256,
            "dtype": "bf16",
        },
        "fallback_reason": "d4_full_suite_speedup_0p9469x_below_1p10",
        "performance_claim": False,
        "economics_evidence": "benchmarks/results/2026-07-23-gfx1151-laguna-dflash-category-economics-post-prefill.json",
    }
    assert target.prepared == 0


def test_laguna_dflash_provider_blocking_streaming_stop_and_close_order(
    tmp_path,
    monkeypatch,
) -> None:
    import hipengine.generation.laguna_dflash as module

    model, cache, snapshot = _identity_tree(tmp_path)
    target = _TargetGenerator(model, cache)
    events: list[str] = []
    _TargetSession.events = events
    _Drafter.events = events
    _Cycle.events = events
    monkeypatch.setattr(module, "LagunaDFlashResidentDrafter", _Drafter)
    monkeypatch.setattr(module, "LagunaDFlashResidentCycle", _Cycle)
    provider = module.LagunaDFlashTextProvider(
        target,
        SpeculativeProviderConfig("dflash", snapshot, 4),
    )

    blocked = provider.generate_detailed(_request())[0]
    chunks = list(provider.stream_detailed(_request()))

    assert blocked.text == "AB"
    assert blocked.generated_token_ids == (10, 11, 24)
    assert blocked.finish_details is not None
    assert blocked.finish_details.reason == "stop"
    assert "".join(chunk.text for chunk in chunks) == "AB"
    assert chunks[-1].generated_token_ids == (10, 11, 24)
    assert chunks[-1].finish_details is not None
    assert chunks[-1].finish_details.reason == "stop"
    assert target.prepared == 1
    assert events.count("target_reset") == 4
    assert events.count("drafter_reset") == 4
    assert blocked.telemetry is not None
    diagnostics = blocked.telemetry.to_json_dict()["diagnostics"]
    assert diagnostics["provider"] == "dflash"
    assert diagnostics["candidate_budget"] == 4
    assert diagnostics["performance_claim"] is False
    assert target.last_generation_outputs[0].generated_token_ids == (10, 11, 24)
    assert target.last_batch_generation["path"] == "laguna_dflash_b4_c1"
    assert target.last_batch_generation["speculative"] == {
        "provider": "dflash",
        "candidate_budget": 4,
        "cycles": 1,
        "accepted_draft_tokens": 1,
        "draft_tokens_proposed": 4,
        "target_verify_rows": 3,
        "exactness_mode": "target_corrected_greedy",
        "performance_claim": False,
    }

    provider.close()
    assert events[-3:] == ["cycle_close", "drafter_close", "target_close"]


def test_laguna_dflash_provider_rejects_identity_budget_and_sampling_before_load(
    tmp_path,
) -> None:
    from hipengine.generation.laguna_dflash import LagunaDFlashTextProvider

    model, cache, snapshot = _identity_tree(tmp_path)
    target = _TargetGenerator(model, cache)
    with pytest.raises(ValueError, match="B4"):
        LagunaDFlashTextProvider(
            target,
            SpeculativeProviderConfig("dflash", snapshot, 2),
        )

    (cache / "manifest.json").write_text(
        json.dumps({"source": {"sha256": "0" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target SHA-256"):
        LagunaDFlashTextProvider(
            target,
            SpeculativeProviderConfig("dflash", snapshot, 4),
        )

    _model, cache, snapshot = _identity_tree(tmp_path / "second")
    target = _TargetGenerator(model, cache)
    provider = LagunaDFlashTextProvider(
        target,
        SpeculativeProviderConfig("dflash", snapshot, 4),
    )
    with pytest.raises(NotImplementedError, match="raw greedy"):
        provider.generate_detailed(_request(temperature=0.2))
    with pytest.raises(NotImplementedError, match="raw greedy"):
        list(provider.stream_detailed(_request(eos_token_id=2)))
    assert target.prepared == 0
