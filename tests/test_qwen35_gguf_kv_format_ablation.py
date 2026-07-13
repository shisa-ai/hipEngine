from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import qwen35_gguf_kv_format_ablation as ablation


class _FakeResult:
    def __init__(self, token_id: int, logits: list[float]) -> None:
        self.token_id = token_id
        self.logits = np.asarray(logits, dtype=np.float32)


class _FakeScratch:
    full_key_caches = (None, object(), None, object())
    full_value_caches = (None, object(), None, object())

    def full_cache(self, layer_id: int):
        return self.full_key_caches[layer_id], self.full_value_caches[layer_id]


class _FakeSession:
    def __init__(self) -> None:
        config = SimpleNamespace(head_count_kv=2, key_length=8)
        self.runner = SimpleNamespace(weights=SimpleNamespace(config=config))
        self.scratch = _FakeScratch()
        self.max_sequence_length = 32
        self.reset_count = 0
        self.prefills: list[list[int]] = []
        self.steps: list[tuple[int, int | None]] = []

    def reset(self) -> None:
        self.reset_count += 1

    def prefill(self, prompt_tokens, *, use_bulk: bool, bulk_attention_mode: str, return_logits: bool):
        assert use_bulk
        assert bulk_attention_mode == "bulk"
        assert return_logits
        self.prefills.append(list(prompt_tokens))
        return _FakeResult(7, [[0.0, 2.0, 1.0]])

    def step(self, token_id: int, position: int | None = None, *, return_logits: bool):
        assert return_logits
        self.steps.append((token_id, position))
        return _FakeResult(token_id + 10, [float(token_id), 0.0, 1.0])


def test_cache_layout_uses_only_full_attention_buffers() -> None:
    layout = ablation._cache_layout(_FakeSession())

    assert layout.full_layer_ids == (1, 3)
    assert layout.num_kv_heads == 2
    assert layout.head_dim == 8


def test_run_loaded_session_reuses_gguf_session_and_teacher_history(monkeypatch) -> None:
    session = _FakeSession()
    roundtrips: list[tuple[int, int, str]] = []
    capture = (
        [np.zeros((3, 2, 8), dtype=np.float32)] * 2,
        [np.ones((3, 2, 8), dtype=np.float32)] * 2,
    )
    monkeypatch.setattr(ablation, "_capture_cache_sample", lambda _session, *, tokens: capture)
    monkeypatch.setattr(
        ablation,
        "_roundtrip_session_cache",
        lambda _session, _spec, *, start, rows, scale_dtype: roundtrips.append((start, rows, scale_dtype)),
    )

    result, captured, full_layers = ablation._run_loaded_session(
        session,
        prompt_tokens=[1, 2, 3],
        prompt_length=3,
        decode_steps=2,
        forced_input_ids=[7, 8],
        scale_dtype="fp16",
        emulated_spec=ablation.FormatSpec("group32", k_group_size=4, v_group_size=4),
        sample_tokens=3,
    )

    assert session.reset_count == 1
    assert session.prefills == [[1, 2, 3]]
    assert session.steps == [(7, 3), (8, 4)]
    assert roundtrips == [(0, 3, "fp16"), (3, 1, "fp16"), (4, 1, "fp16")]
    assert result["seed_token_id"] == 7
    assert result["generated_token_ids"] == [17, 18]
    assert result["finite_logits"]
    assert captured is capture
    assert full_layers == 2


def test_fixed_mixed_prompt_matches_paro_screen_fixture() -> None:
    prompt = ablation._fixed_mixed_prompt_tokens(512)

    assert len(prompt) == 512
    assert len(set(prompt)) == 36
    assert ablation._prompt_sha256(prompt) == "933b5f11bdfb5766ab729e06c6fe024f5e9041fb287aee9472589560d350a5f8"
