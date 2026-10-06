"""The sampled gate must be able to fail, not just to pass.

A live gate that only ever passes is indistinguishable from no gate, so each
check is driven against a fake server that violates exactly one thing: a
speculative arm that did not speculate, a baseline whose usage disagrees, and a
second run that does not reproduce the first.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "int8_mtp_sampled_gate", ROOT / "scripts" / "int8_mtp_sampled_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_module()


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeServer:
    """A server whose two arms answer with whatever the test configured."""

    def __init__(
        self,
        *,
        storage: str = "int8_per_token_head",
        used: bool = True,
        cycles: int = 5,
        ar_completion_tokens: int | None = None,
        second_run_cycles: int | None = None,
    ) -> None:
        self.storage = storage
        self.used = used
        self.cycles = cycles
        self.ar_completion_tokens = ar_completion_tokens
        self.second_run_cycles = second_run_cycles
        self.requests: list[dict] = []

    def ready(self) -> dict:
        return {
            "status": "ready",
            "model": {
                "id": "fake-model",
                "kv_capability": {
                    "effective_kv_storage": self.storage,
                    "capability_id": "fake",
                    "status": "rejected",
                    "runtime_action": "diagnostic_override",
                    "diagnostic_override": True,
                    "persistent_bf16_mirror": False,
                },
            },
        }

    def completion(self, payload: dict) -> dict:
        self.requests.append(payload)
        speculative = bool(payload.get("speculative_mtp"))
        cycles = self.cycles
        if speculative and self.second_run_cycles is not None:
            # The first pass through the config/prompt matrix is one run; every
            # repeat afterwards reports the drifted cycle count.
            seen = sum(1 for request in self.requests if request.get("speculative_mtp"))
            if seen > len(gate.SAMPLER_CONFIGS) * len(gate.PROMPTS):
                cycles = self.second_run_cycles
        completion_tokens = 24
        if not speculative and self.ar_completion_tokens is not None:
            completion_tokens = self.ar_completion_tokens
        ids = [7, 8, 9] if speculative or self.used else [7, 8, 9]
        return {
            "choices": [
                {
                    "finish_reason": "length",
                    "hipengine": {
                        "generated_token_ids": ids,
                        "decode_state": {
                            "sampler_mode": "gpu_sample",
                            "sampler_fast_path_blockers": ["temperature"],
                        },
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": completion_tokens,
                "total_tokens": 5 + completion_tokens,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 3 if speculative else 0,
                    "rejected_prediction_tokens": 2 if speculative else 0,
                },
            },
            "hipengine": {
                "speculative_mtp": {
                    "used": self.used if speculative else False,
                    "effective_route": "speculative_mtp" if speculative else "default",
                    "draft_cycles": cycles if speculative else 0,
                    "draft_tokens": 5 if speculative else 0,
                    "accepted_draft_tokens": 3 if speculative else 0,
                }
            },
        }


class _FakeClient:
    def __init__(self, server: _FakeServer) -> None:
        self._server = server

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def get(self, path: str) -> _Response:
        assert path == "/ready"
        return _Response(self._server.ready())

    def post(self, path: str, json: dict) -> _Response:
        assert path == "/v1/completions"
        return _Response(self._server.completion(json))


def _run_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, server: _FakeServer, runs: int = 2):
    monkeypatch.setattr(gate.httpx, "Client", lambda **kwargs: _FakeClient(server))
    output = tmp_path / "sampled.json"
    status = gate.main([
        "--model", "fake-model", "--json", str(output), "--runs", str(runs),
        "--allow-kv-diagnostic-override",
    ])
    return status, json.loads(output.read_text())


def test_the_gate_passes_a_server_that_speculates_and_repeats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(monkeypatch, tmp_path, _FakeServer())

    assert status == 0
    assert report["passed"] is True
    assert report["summary"] == {
        "rows": len(gate.SAMPLER_CONFIGS) * len(gate.PROMPTS) * 2,
        "configs": len(gate.SAMPLER_CONFIGS),
        "prompts": len(gate.PROMPTS),
        "runs": 2,
        "repeatable": True,
    }


def test_the_gate_refuses_a_server_that_is_not_the_int8_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(storage="bf16"), runs=1
    )

    assert status == 2
    assert "bf16" in report["error"]
def test_the_gate_refuses_a_server_running_the_diagnostic_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _FakeServer()
    monkeypatch.setattr(gate.httpx, "Client", lambda **kwargs: _FakeClient(server))
    output = tmp_path / "sampled.json"

    status = gate.main(["--model", "fake-model", "--json", str(output), "--runs", "1"])

    assert status == 2
    assert "diagnostic override" in json.loads(output.read_text())["error"]


def test_the_gate_records_the_override_when_it_is_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(monkeypatch, tmp_path, _FakeServer(), runs=1)

    assert status == 0
    assert report["server"]["diagnostic_override"] is True


def test_the_gate_fails_when_the_speculative_arm_did_not_speculate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(monkeypatch, tmp_path, _FakeServer(used=False), runs=1)

    assert status == 1
    assert "sampled_arm_did_not_speculate" in report["error"]


def test_the_gate_fails_when_the_arms_disagree_about_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(ar_completion_tokens=23), runs=1
    )

    assert status == 1
    assert "usage_mismatch" in report["error"]


def test_the_gate_fails_when_a_second_run_does_not_reproduce_the_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(second_run_cycles=99), runs=2
    )

    assert status == 1
    assert "sampled_cycles_not_repeatable" in report["error"]


def test_sampler_params_carries_every_field_but_the_label() -> None:
    config = {"name": "penalties", "temperature": 0.7, "presence_penalty": 0.5}

    assert gate.sampler_params(config) == {"temperature": 0.7, "presence_penalty": 0.5}


def test_every_config_is_a_sampler_request_the_declaration_covers() -> None:
    """The gate's configs must exercise the four families the task names."""

    names = [config["name"] for config in gate.SAMPLER_CONFIGS]

    assert names == ["temperature_top_p", "penalties", "logit_bias", "suppress_token_ids"]
    for config in gate.SAMPLER_CONFIGS:
        assert config.get("temperature", 0.0) > 0.0, "a greedy config would not test sampling"
