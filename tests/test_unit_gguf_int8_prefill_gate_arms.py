"""Exercise evaluator arm binding and artifact manifests without GPU work."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from scripts import execution_profile_gguf_int8_direct_prefill_gate as gate


@pytest.mark.parametrize("inherited", [None, "0", "1"])
@pytest.mark.parametrize("candidate", ["slot_local_aotriton", "int8_direct_prefill"])
def test_capture_and_artifact_bind_distinct_declared_arms(monkeypatch, tmp_path, inherited, candidate):
    from hipengine.loading import gguf
    from hipengine.runtime import qwen35_gguf_runner as runner
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    key = gate.SLOT_LOCAL_AOTRITON_ENV
    monkeypatch.delenv(key, raising=False)
    if inherited is not None:
        monkeypatch.setenv(key, inherited)
    model = tmp_path / "model.gguf"
    model.touch()
    args = gate.build_parser().parse_args([
        "--model", str(model), "--candidate", candidate,
        "--prefill-entry", "packed_slot_local", "--decode-steps", "1",
        "--prompts", str(tmp_path / "suite.jsonl"),
    ])
    observed = []

    class Session:
        def __init__(self, *_args, **_kwargs):
            self.runner = SimpleNamespace(backend="hip_gfx1100", target_arch="gfx1100")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def _release_bulk_prefill_workspace(self):
            pass

    def trajectory(_session, **_kwargs):
        observed.append((
            runner._gguf_slot_local_prefill_allow_aotriton(transient_direct_oracle=True),
            runner._gguf_int8_prefill_direct_enabled(),
        ))
        return [{"token_id": 1, "logits": [0.0, 1.0]}]

    monkeypatch.setattr(gguf, "scan_gguf", lambda _path: object())
    monkeypatch.setattr(Qwen35GGUFTokenizer, "from_gguf_info", lambda _info: object())
    monkeypatch.setattr(runner, "Qwen35GGUFResidentSession", Session)
    monkeypatch.setattr(gate, "build_chat_prompt", lambda *_args: [7] * 613)
    monkeypatch.setattr(gate, "_packed_trajectory", trajectory)
    monkeypatch.setattr(gate, "_state_summary", lambda *_args: {"finite": True, "state_sha256": "same"})
    monkeypatch.setattr(gate, "validate_strict_baseline", lambda **_kwargs: None)
    monkeypatch.setattr(gate, "_configure_gate_environment", lambda **_kwargs: None)
    monkeypatch.setattr(gate, "_load_suites", lambda _paths: [
        {"id": "example", "category": "general_en", "suite": "test", "prompt": "example"},
    ])
    monkeypatch.setattr(gate, "build_candidate_quality", lambda *_args, **_kwargs: {
        "repeat_determinism": {"passed": True}, "quality": {"hard_gates_passed": True},
    })
    monkeypatch.setattr(gate, "collect_artifact_provenance", lambda **kwargs: {
        "dirty": False, "environment": kwargs["environment"],
    })

    result = gate.run(args, command=["synthetic gate"])

    expected = (False, False) if candidate == "slot_local_aotriton" else (True, False)
    selected = (True, False) if candidate == "slot_local_aotriton" else (True, True)
    assert observed == [expected, selected, selected, selected]
    strict_manifest = result["candidate"]["strict_environment"]
    assert strict_manifest[key] == ("0" if candidate == "slot_local_aotriton" else "1")
    assert result["provenance"]["environment"]["strict_route_environment"] == strict_manifest
    assert result["candidate"]["candidate_environment"][key] == "1"
    assert os.environ.get(key) == inherited
