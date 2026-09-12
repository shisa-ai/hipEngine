from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import scripts.qwen35_kv_e2e_fixture_gate as gate


def _args(tmp_path: Path, fixture: Path, *, kv_storage: str = "int8_per_token_head") -> SimpleNamespace:
    return SimpleNamespace(
        model="/tmp/model",
        fixture=fixture,
        max_layers=4,
        max_new_tokens=1,
        compiler_version_file=None,
        require_cached_build=False,
        kl_threshold=0.05,
        top1_threshold=0.90,
        attn_aotriton_min_tokens=0,
        prefill_linear_chunk_size=0,
        prefill_moe_chunk_size=0,
        prefill_full_attn_query_chunk_size=0,
        prefill_full_attn_post_chunk_size=0,
        prefill_full_attn_rope_chunk_size=0,
        prefill_chunk_autotune=True,
        prefill_chunk_memory_budget_gib=0.0,
        kv_storage=kv_storage,
        kv_scale_dtype="fp16",
        kv_scale_granularity="per_token_head",
        json=tmp_path / "out.json",
    )


def test_compare_logits_reports_kl_and_top1_agreement() -> None:
    reference = [np.asarray([0.0, 4.0, 1.0], dtype=np.float32), np.asarray([0.0, 1.0, 5.0], dtype=np.float32)]
    candidate = [np.asarray([0.0, 3.9, 1.0], dtype=np.float32), np.asarray([0.0, 1.0, 4.8], dtype=np.float32)]

    comparison = gate._compare_logits(reference, candidate)

    assert comparison["positions"] == 2
    assert comparison["max_kl"] < 0.05
    assert comparison["reference_top1"] == [1, 2]
    assert comparison["candidate_top1"] == [1, 2]
    assert comparison["top1_agreement"] == 1.0


def test_kv_e2e_fixture_gate_payload_records_policy_and_logit_gate(monkeypatch, tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        '{"prompt_ids":[10,11],"expected_generated_token_ids":[2],"decode_len":1}',
        encoding="utf-8",
    )

    class FakeResult:
        def __init__(self, token_id: int) -> None:
            self.token_id = token_id
            self.token_text = f"tok{token_id}"
            self.logit = float(token_id)

    class FakeSession:
        def __init__(self, runner, **kwargs) -> None:
            self.storage = kwargs["kv_policy"].storage_dtype.value
            self.scale_dtype = kwargs["kv_scale_dtype"]
            self.scale_granularity = kwargs["kv_scale_granularity"]
            self.prefill_config = kwargs["prefill_config"]
            self.prefill_chunk_tuning = None
            self.last_prefill_execution = {"kv_storage_dtype": self.storage}
            self.allocations = []
            self.buffers = []
            self.states = []
            self.last_token_id = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill_native(self, prompt_tokens, *, sample: bool = True):
            self.last_token_id = 1
            return FakeResult(1) if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            self.last_token_id = 2
            return FakeResult(2) if sample else None

        def owned_buffer_summary(self):
            if self.storage == "int8_per_token_head":
                layer = {
                    "layer_id": 0,
                    "storage_dtype": "int8_per_token_head",
                    "payload_dtype": "int8",
                    "scale_metadata": {"scale_bytes": 32, "scale_dtype": self.scale_dtype.value, "granularity": self.scale_granularity},
                }
                return {
                    "kv_storage_dtype": self.storage,
                    "kv_scale_dtype": self.scale_dtype.value,
                    "kv_scale_granularity": self.scale_granularity,
                    "full_attention_layers": [layer],
                    "full_attention_kv_payload_bytes": 64,
                    "full_attention_kv_scale_bytes": 32,
                }
            return {"kv_storage_dtype": "bf16", "full_attention_layers": []}

    def fake_read_logits(session: FakeSession) -> np.ndarray:
        logits = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        logits[int(session.last_token_id)] = 5.0
        if session.storage == "int8_per_token_head":
            logits[int(session.last_token_id)] -= np.float32(0.05)
        return logits

    monkeypatch.setattr(gate, "Qwen35ParoNextTokenRunner", lambda model: object())
    monkeypatch.setattr(gate, "Qwen35ParoResidentSession", FakeSession)
    monkeypatch.setattr(gate, "_read_logits", fake_read_logits)

    payload = gate.run(_args(tmp_path, fixture))

    assert payload["passed"] is True
    assert payload["fixture"] == fixture.as_posix()
    assert payload["kv_storage_dtype"] == "int8_per_token_head"
    assert payload["reference_kv_storage_dtype"] == "bf16"
    assert payload["candidate_kv_policy"]["scale_metadata_format"] == {
        "present": True,
        "scale_dtype": "fp16",
        "granularity": "per_token_head",
        "k_scale": "per_token_head",
        "v_scale": "per_token_head",
    }
    assert payload["candidate_kv_policy"]["int8_explicit"] is True
    assert payload["candidate_kv_policy"]["int8_admission_gated"] is False
    assert payload["thresholds"] == {"kl_max": 0.05, "top1_agreement_min": 0.90}
    assert payload["reference_generated_token_ids"] == [2]
    assert payload["candidate_generated_token_ids"] == [2]
    assert payload["expected_generated_token_ids"] == [2]
    assert payload["logit_position_labels"] == ["prefill_seed", "decode_0"]
    assert payload["logit_gate"]["top1_agreement"] == 1.0
    assert payload["logit_gate"]["max_kl"] <= 0.05
    assert payload["candidate_kv_memory_audit_after_decode"]["passed"] is True


def test_kv_e2e_fixture_gate_fails_on_generated_mismatch(monkeypatch, tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        '{"prompt_ids":[10,11],"expected_generated_token_ids":[2],"decode_len":1}',
        encoding="utf-8",
    )

    class FakeResult:
        def __init__(self, token_id: int) -> None:
            self.token_id = token_id
            self.token_text = f"tok{token_id}"
            self.logit = float(token_id)

    class FakeSession:
        def __init__(self, runner, **kwargs) -> None:
            self.storage = kwargs["kv_policy"].storage_dtype.value
            self.prefill_config = kwargs["prefill_config"]
            self.prefill_chunk_tuning = None
            self.last_prefill_execution = {}
            self.allocations = []
            self.buffers = []
            self.states = []
            self.last_token_id = 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill_native(self, prompt_tokens, *, sample: bool = True):
            self.last_token_id = 1
            return FakeResult(1)

        def step(self, token_id: int, *, position: int, sample: bool = True):
            self.last_token_id = 3 if self.storage == "int8_per_token_head" else 2
            return FakeResult(self.last_token_id)

        def owned_buffer_summary(self):
            if self.storage == "int8_per_token_head":
                return {
                    "kv_storage_dtype": self.storage,
                    "full_attention_layers": [
                        {
                            "layer_id": 0,
                            "storage_dtype": "int8_per_token_head",
                            "payload_dtype": "int8",
                            "scale_metadata": {"scale_bytes": 1},
                        }
                    ],
                    "full_attention_kv_payload_bytes": 1,
                    "full_attention_kv_scale_bytes": 1,
                }
            return {"kv_storage_dtype": "bf16", "full_attention_layers": []}

    monkeypatch.setattr(gate, "Qwen35ParoNextTokenRunner", lambda model: object())
    monkeypatch.setattr(gate, "Qwen35ParoResidentSession", FakeSession)
    monkeypatch.setattr(gate, "_read_logits", lambda session: np.eye(4, dtype=np.float32)[session.last_token_id] * 5.0)

    payload = gate.run(_args(tmp_path, fixture))

    assert payload["passed"] is False
    assert payload["generated_match"] is False
    assert payload["generated_first_mismatch"] == {"index": 0, "left": 2, "right": 3}
    assert payload["expected_match"] is False
