from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.generation.qwen35_gguf as qwen35_gguf
from hipengine.dispatch import WorkItem, WorkKind
from hipengine.generation import (
    EngineLoopConfig,
    GenerationCancellationToken,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationRequest,
    GenerationStreamChunk,
    SubmitPollTextGenerator,
    TokenLogprob,
)
from hipengine.kvcache import DeviceChunkedKVPool


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, prompt: str) -> list[int]:
        return {
            "first": [10, 11],
            "second": [20],
            "long": [10, 11, 12, 13],
            "long2": [20, 21, 22, 23],
            "{": [5],
            "}": [4],
        }[prompt]

    def decode(self, ids) -> str:
        table = {1: "B", 2: "C", 3: "D", 4: "}", 5: "{", 6: "X", 16: "Q", 99: "<eos>", 114: "T114"}
        return "".join(table[int(token)] for token in ids)


def _generator() -> qwen35_gguf.Qwen35GGUFBringupGenerator:
    generator = qwen35_gguf.Qwen35GGUFBringupGenerator.__new__(
        qwen35_gguf.Qwen35GGUFBringupGenerator
    )
    generator.model_path = "/tmp/fake.gguf"
    generator.weight_index = SimpleNamespace()
    generator.model_plugin = SimpleNamespace()
    generator.tokenizer = _FakeTokenizer()
    generator._mtp_serving_assets = None
    generator._mtp_serving_lock = threading.Lock()
    return generator


def _request(**overrides) -> GenerationRequest:
    values = {
        "prompts": ("first",),
        "max_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": False,
    }
    values.update(overrides)
    return GenerationRequest(**values)


def _decode_state(output):
    assert output.telemetry is not None
    return output.telemetry.to_json_dict()["decode_state"]


def _mtp_capable_weight_index():
    return SimpleNamespace(
        tensors=[
            SimpleNamespace(name=name)
            for name in qwen35_gguf._GGUF_MTP_REQUIRED_TENSORS
        ]
    )


def test_gguf_mtp_server_defer_verify_scatter_default_on_with_opt_out(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER", raising=False)
    assert qwen35_gguf._gguf_mtp_server_defer_verify_scatter_enabled() is True

    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER", "0")
    assert qwen35_gguf._gguf_mtp_server_defer_verify_scatter_enabled() is False


def test_gguf_decode_graph_default_on_with_opt_out(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_GRAPH", raising=False)
    assert qwen35_gguf._gguf_decode_graph_enabled() is True

    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_GRAPH", "0")
    assert qwen35_gguf._gguf_decode_graph_enabled() is False


def test_gguf_speculative_mtp_hook_runs_llama_compat_direct_commit(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        def memcpy(self, dst, src, nbytes, kind):  # pragma: no cover - short prompt skips context copy
            calls.append(("memcpy", int(nbytes)))

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            calls.append(("session_init", str(model_path), dict(kwargs)))
            self.runtime = FakeRuntime()
            self.position = 0
            self.runner = SimpleNamespace(
                weights=SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=4))
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("session_close",))

        def prefill(self, token_ids, **kwargs):
            calls.append(("prefill", tuple(token_ids), dict(kwargs)))
            self.position = len(token_ids)
            return SimpleNamespace(token_id=1)

        def mtp_draft_seed(self, *, token_id: int, position: int):
            return SimpleNamespace(
                token_id=int(token_id),
                position=int(position),
                hidden_ptr=0xABC0,
                hidden_contract=SimpleNamespace(
                    ready_for_mtp=True,
                    rows=1,
                    hidden_size=2,
                ),
            )

        def verify_target_block(self, input_token_ids, **kwargs):
            calls.append(("verify_block", tuple(input_token_ids), dict(kwargs)))
            return SimpleNamespace(
                token_ids=[2, 3],
                hidden_seeds=np.ones((2, 2), dtype=np.float32),
                linear_state_rows_captured=True,
            )

        def _commit_verify_linear_state_row(self, row_index: int, *, position: int):
            calls.append(("commit_row", int(row_index), int(position)))
            self.position = int(position)

        def fp32_verify_hidden_seed_ptr(self, row_index: int = 0) -> int:
            return 0xD000 + int(row_index) * 8

    class FakeDraft:
        def __init__(self):
            calls.append(
                (
                    "draft_init_env",
                    os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"),
                    os.environ.get("HIPENGINE_GGUF_SELECTED_X8_REPACK"),
                )
            )

        def propose_chain_from_device_seed(self, hidden_seed_ptr, **kwargs):
            calls.append(("draft", int(hidden_seed_ptr), dict(kwargs)))
            return [2], [[2]], int(kwargs["dense_cache_len"]) + 2

        def write_kv_rows(self, hidden_rows, token_ids, **kwargs):  # pragma: no cover - short prompt skips context
            calls.append(("write_context_kv", tuple(int(token) for token in token_ids)))
            return len(token_ids)

        def write_kv_rows_from_device_seed_base(self, hidden_seed_ptr, token_ids, **kwargs):
            calls.append(
                (
                    "write_commit_kv",
                    int(hidden_seed_ptr),
                    tuple(int(token) for token in token_ids.tolist()),
                    tuple(int(pos) for pos in kwargs["positions"].tolist()),
                    int(kwargs["dense_cache_len"]),
                )
            )
            return int(kwargs["dense_cache_len"]) + len(token_ids)

        def close(self):
            calls.append(("draft_close",))

    assets = qwen35_gguf._GGUFMTPServingAssets(
        weights={
            "blk.40.attn_q_norm.weight": (np.zeros((2,), dtype=np.float32), 0, (2,)),
            "output.weight": (np.zeros((8, 1), dtype=np.uint8), 0, (8, 1)),
        },
        token_embd_f32=np.zeros((8, 2), dtype=np.float32),
        rope_cos=np.ones((16, 2), dtype=np.float32),
        rope_sin=np.zeros((16, 2), dtype=np.float32),
    )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setattr(
        qwen35_gguf.Qwen35GGUFBringupGenerator,
        "_load_mtp_serving_assets",
        lambda self: assets,
    )
    monkeypatch.setattr(qwen35_gguf, "_new_mtp_draft_runner", lambda assets, *, runtime: FakeDraft())
    monkeypatch.setattr(
        qwen35_gguf,
        "_allocate_mtp_dense_kv",
        lambda **kwargs: (SimpleNamespace(ptr=0x1000), SimpleNamespace(ptr=0x2000), []),
    )
    monkeypatch.setattr(qwen35_gguf, "_free_mtp_buffers", lambda buffers, *, runtime: calls.append(("free_kv", len(buffers))))
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_SELECTED_X8_REPACK", "preexisting")

    generator = _generator()
    generator.weight_index = _mtp_capable_weight_index()
    outputs = generator.generate_speculative_mtp_detailed(_request(prompts=("long",), max_tokens=3))

    assert outputs[0].text == "BCD"
    assert ("draft_init_env", "1", "q6") in calls
    assert ("verify_block", (1, 2), {
        "bulk_attention_mode": "bulk",
        "use_wmma_prefill": False,
        "capture_linear_state_rows": True,
        "defer_linear_state_commit": True,
    }) in calls
    assert ("commit_row", 1, 6) in calls
    assert ("write_context_kv", (10, 11, 12, 13)) in calls
    assert ("write_commit_kv", 0xD000, (2,), (5,), 5) in calls
    assert ("draft_close",) in calls
    assert ("free_kv", 0) in calls
    assert os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN") is None
    assert os.environ["HIPENGINE_GGUF_SELECTED_X8_REPACK"] == "preexisting"
    assert generator.last_batch_generation is not None
    assert generator.last_batch_generation["path"] == "gguf_llama_compat_mtp_server"
    assert generator.last_batch_generation["speculative_mtp"]["total_draft_tokens"] == 1
    assert generator.last_batch_generation["speculative_mtp"]["total_accepted_draft_tokens"] == 1
    assert generator.last_batch_generation["speculative_mtp"]["accepted_draft_tokens_histogram"] == {"1": 1}
    assert generator.last_batch_generation["speculative_mtp"]["cycle_shape_histogram"] == {"draft1_accept1": 1}
    assert generator.last_batch_generation["speculative_mtp"]["full_accept_cycles"] == 1
    assert generator.last_batch_generation["speculative_mtp"]["linear_state_extra_rows"] == 1
    timing = outputs[0].telemetry.to_json_dict()["timing"]
    assert timing["mtp_cycles_count"] == 1.0
    assert timing["mtp_generated_draft_tokens"] == 1.0
    assert timing["mtp_accepted_draft_tokens"] == 1.0
    assert timing["mtp_visible_output_tokens"] == 2.0
    assert timing["mtp_target_verify_rows"] == 2.0
    assert timing["mtp_accept_per_draft"] == 1.0
    assert timing["mtp_full_accept_cycles"] == 1.0
    assert timing["mtp_linear_state_captured_rows"] == 2.0
    assert timing["mtp_linear_state_commit_rows"] == 1.0
    assert timing["mtp_hidden_seed_extra_rows"] == 0.0


def test_gguf_speculative_mtp_c2_uses_resident_slots(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        def memcpy(self, dst, src, nbytes, kind):
            calls.append(("memcpy", int(nbytes)))

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.model_path = str(model_path)
            self.runtime = FakeRuntime()
            self.weights = SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=4))
            calls.append(("shared_runner_init", self.model_path))

        def __enter__(self):
            calls.append(("shared_runner_enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def close(self):
            calls.append(("shared_runner_close",))

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.slot_id = len([call for call in calls if call and call[0] == "session_init"])
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            calls.append(("session_init", self.slot_id, str(model_path), kwargs["shared_runner"]))

        def prefill(self, token_ids, **kwargs):
            calls.append(("prefill", self.slot_id, tuple(token_ids), dict(kwargs)))
            self.position = len(token_ids)
            return SimpleNamespace(token_id=1)

        def mtp_draft_seed(self, *, token_id: int, position: int):
            return SimpleNamespace(
                token_id=int(token_id),
                position=int(position),
                hidden_ptr=0xABC0 + self.slot_id * 0x100,
                hidden_contract=SimpleNamespace(
                    ready_for_mtp=True,
                    rows=1,
                    hidden_size=2,
                ),
            )

        def verify_target_block(self, input_token_ids, **kwargs):
            calls.append(("verify_block", self.slot_id, tuple(input_token_ids), dict(kwargs)))
            return SimpleNamespace(
                token_ids=[2, 3],
                hidden_seeds=np.ones((2, 2), dtype=np.float32),
                linear_state_rows_captured=True,
            )

        def _commit_verify_linear_state_row(self, row_index: int, *, position: int):
            calls.append(("commit_row", self.slot_id, int(row_index), int(position)))
            self.position = int(position)

        def fp32_verify_hidden_seed_ptr(self, row_index: int = 0) -> int:
            return 0xD000 + self.slot_id * 0x100 + int(row_index) * 8

        def close(self):
            calls.append(("session_close", self.slot_id))

    class FakeDraft:
        def __init__(self):
            self.draft_id = len([call for call in calls if call and call[0] == "draft_init"])
            calls.append(("draft_init", self.draft_id))

        def propose_chain_from_device_seed(self, hidden_seed_ptr, **kwargs):
            calls.append(("draft", self.draft_id, int(hidden_seed_ptr), dict(kwargs)))
            return [2], [[2]], int(kwargs["dense_cache_len"]) + 2

        def write_kv_rows(self, hidden_rows, token_ids, **kwargs):
            calls.append(("write_context_kv", self.draft_id, tuple(int(token) for token in token_ids)))
            return len(token_ids)

        def write_kv_rows_from_device_seed_base(self, hidden_seed_ptr, token_ids, **kwargs):
            calls.append(
                (
                    "write_commit_kv",
                    self.draft_id,
                    int(hidden_seed_ptr),
                    tuple(int(token) for token in token_ids.tolist()),
                    tuple(int(pos) for pos in kwargs["positions"].tolist()),
                    int(kwargs["dense_cache_len"]),
                )
            )
            return int(kwargs["dense_cache_len"]) + len(token_ids)

        def close(self):
            calls.append(("draft_close", self.draft_id))

    assets = qwen35_gguf._GGUFMTPServingAssets(
        weights={
            "blk.40.attn_q_norm.weight": (np.zeros((2,), dtype=np.float32), 0, (2,)),
            "output.weight": (np.zeros((8, 1), dtype=np.uint8), 0, (8, 1)),
        },
        token_embd_f32=np.zeros((8, 2), dtype=np.float32),
        rope_cos=np.ones((16, 2), dtype=np.float32),
        rope_sin=np.zeros((16, 2), dtype=np.float32),
    )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setattr(
        qwen35_gguf.Qwen35GGUFBringupGenerator,
        "_load_mtp_serving_assets",
        lambda self: assets,
    )
    monkeypatch.setattr(qwen35_gguf, "_new_mtp_draft_runner", lambda assets, *, runtime: FakeDraft())
    monkeypatch.setattr(
        qwen35_gguf,
        "_allocate_mtp_dense_kv",
        lambda **kwargs: (SimpleNamespace(ptr=0x1000), SimpleNamespace(ptr=0x2000), []),
    )
    monkeypatch.setattr(qwen35_gguf, "_free_mtp_buffers", lambda buffers, *, runtime: calls.append(("free_kv", len(buffers))))

    generator = _generator()
    generator.weight_index = _mtp_capable_weight_index()
    generator.prepare()
    outputs = generator.generate_speculative_mtp_detailed(_request(prompts=("long", "long2"), max_tokens=3))

    assert [output.text for output in outputs] == ["BCD", "BCD"]
    assert [call[0] for call in calls].count("shared_runner_init") == 1
    session_inits = [call for call in calls if call and call[0] == "session_init"]
    assert len(session_inits) == 2
    assert session_inits[0][3] is session_inits[1][3]
    assert [call for call in calls if call and call[0] == "verify_block"] == [
        ("verify_block", 0, (1, 2), {
            "bulk_attention_mode": "bulk",
            "use_wmma_prefill": False,
            "capture_linear_state_rows": True,
            "defer_linear_state_commit": True,
        }),
        ("verify_block", 1, (1, 2), {
            "bulk_attention_mode": "bulk",
            "use_wmma_prefill": False,
            "capture_linear_state_rows": True,
            "defer_linear_state_commit": True,
        }),
    ]
    assert generator.last_batch_generation is not None
    mtp = generator.last_batch_generation["speculative_mtp"]
    assert generator.last_batch_generation["batch_size"] == 2
    assert generator.last_batch_generation["serial_decode_fallback"] is False
    assert mtp["resident_slot_count"] == 2
    assert mtp["scheduler"] == "resident_slots_phase_serial"
    assert mtp["target_verify_batching"] == "per_slot_serial"
    assert mtp["total_draft_tokens"] == 2
    assert mtp["total_accepted_draft_tokens"] == 2
    assert mtp["accepted_draft_tokens_histogram"] == {"1": 2}
    assert mtp["cycle_shape_histogram"] == {"draft1_accept1": 2}
    assert mtp["full_accept_cycles"] == 2
    assert mtp["linear_state_captured_rows"] == 4
    assert mtp["linear_state_commit_rows"] == 2
    assert mtp["linear_state_extra_rows"] == 2
    assert sorted(mtp["cycles_by_request"]) == ["0", "1"]
    timing = outputs[0].telemetry.to_json_dict()["timing"]
    assert "slots_draft_phase_ms" in timing
    assert "slots_verify_phase_ms" in timing
    assert "slots_commit_phase_ms" in timing
    assert timing["mtp_cycles_count"] == 1.0
    assert timing["mtp_generated_draft_tokens"] == 1.0
    assert timing["mtp_accepted_draft_tokens"] == 1.0
    assert timing["mtp_target_verify_rows"] == 2.0
    assert timing["mtp_full_accept_cycles"] == 1.0
    assert timing["mtp_linear_state_extra_rows"] == 1.0


def test_gguf_speculative_mtp_c2_uses_packed_prefill_by_default(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        pass

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.model_path = str(model_path)
            self.runtime = FakeRuntime()
            self.weights = SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=4))
            calls.append(("shared_runner_init", self.model_path))

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.slot_id = len([call for call in calls if call and call[0] == "session_init"])
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            calls.append(("session_init", self.slot_id))

        def reset(self):
            self.position = 0

        def close(self):
            calls.append(("session_close", self.slot_id))

        def prefill(self, token_ids, **kwargs):  # pragma: no cover - must not be used
            raise AssertionError("packed MTP prefill should bypass per-slot prefill")

        def prefill_batch_native(self, prompt_token_ids, *, sessions, return_logits=False, return_hidden_seeds=False):
            calls.append(
                (
                    "prefill_batch",
                    self.slot_id,
                    tuple(tuple(int(token) for token in prompt) for prompt in prompt_token_ids),
                    tuple(session.slot_id for session in sessions),
                    os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"),
                    return_logits,
                    return_hidden_seeds,
                )
            )
            results = []
            for session, prompt in zip(sessions, prompt_token_ids, strict=True):
                session.position = len(prompt)
                hidden = np.full((len(prompt), 2), float(session.slot_id + 1), dtype=np.float32)
                results.append(SimpleNamespace(token_id=1, hidden_seeds=hidden))
            return results

        def mtp_draft_seed(self, *, token_id: int, position: int):
            calls.append(("mtp_seed", self.slot_id, int(token_id), int(position)))
            return SimpleNamespace(
                token_id=int(token_id),
                position=int(position),
                hidden_ptr=0xABC0 + self.slot_id * 0x100,
                hidden_contract=SimpleNamespace(
                    ready_for_mtp=True,
                    rows=1,
                    hidden_size=2,
                ),
            )

        def verify_target_block(self, input_token_ids, **kwargs):
            calls.append(("verify_block", self.slot_id, tuple(input_token_ids), dict(kwargs)))
            return SimpleNamespace(
                token_ids=[2, 3],
                hidden_seeds=np.ones((2, 2), dtype=np.float32),
                linear_state_rows_captured=True,
            )

        def _commit_verify_linear_state_row(self, row_index: int, *, position: int):
            calls.append(("commit_row", self.slot_id, int(row_index), int(position)))
            self.position = int(position)

        def fp32_verify_hidden_seed_ptr(self, row_index: int = 0) -> int:
            return 0xD000 + self.slot_id * 0x100 + int(row_index) * 8

    class FakeDraft:
        def __init__(self):
            self.draft_id = len([call for call in calls if call and call[0] == "draft_init"])
            calls.append(("draft_init", self.draft_id))

        def propose_chain_from_device_seed(self, hidden_seed_ptr, **kwargs):
            calls.append(("draft", self.draft_id, int(hidden_seed_ptr), dict(kwargs)))
            return [2], [[2]], int(kwargs["dense_cache_len"]) + 2

        def write_kv_rows(self, hidden_rows, token_ids, **kwargs):
            calls.append(
                (
                    "write_context_kv",
                    self.draft_id,
                    tuple(int(token) for token in token_ids),
                    tuple(hidden_rows.shape),
                    float(hidden_rows[1, 0]),
                )
            )
            return len(token_ids)

        def write_kv_rows_from_device_seed_base(self, hidden_seed_ptr, token_ids, **kwargs):
            calls.append(
                (
                    "write_commit_kv",
                    self.draft_id,
                    int(hidden_seed_ptr),
                    tuple(int(token) for token in token_ids.tolist()),
                )
            )
            return int(kwargs["dense_cache_len"]) + len(token_ids)

        def close(self):
            calls.append(("draft_close", self.draft_id))

    assets = qwen35_gguf._GGUFMTPServingAssets(
        weights={
            "blk.40.attn_q_norm.weight": (np.zeros((2,), dtype=np.float32), 0, (2,)),
            "output.weight": (np.zeros((8, 1), dtype=np.uint8), 0, (8, 1)),
        },
        token_embd_f32=np.zeros((8, 2), dtype=np.float32),
        rope_cos=np.ones((16, 2), dtype=np.float32),
        rope_sin=np.zeros((16, 2), dtype=np.float32),
    )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setattr(
        qwen35_gguf.Qwen35GGUFBringupGenerator,
        "_load_mtp_serving_assets",
        lambda self: assets,
    )
    monkeypatch.setattr(qwen35_gguf, "_new_mtp_draft_runner", lambda assets, *, runtime: FakeDraft())
    monkeypatch.setattr(
        qwen35_gguf,
        "_allocate_mtp_dense_kv",
        lambda **kwargs: (SimpleNamespace(ptr=0x1000), SimpleNamespace(ptr=0x2000), []),
    )
    monkeypatch.setattr(qwen35_gguf, "_free_mtp_buffers", lambda buffers, *, runtime: calls.append(("free_kv", len(buffers))))
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL", raising=False)

    generator = _generator()
    generator.weight_index = _mtp_capable_weight_index()
    generator.prepare()
    outputs = generator.generate_speculative_mtp_detailed(_request(prompts=("long", "long2"), max_tokens=3))

    assert [output.text for output in outputs] == ["BCD", "BCD"]
    assert [call for call in calls if call[0] == "prefill_batch"] == [
        (
            "prefill_batch",
            0,
            ((10, 11, 12, 13), (20, 21, 22, 23)),
            (0, 1),
            "1",
            False,
            True,
        )
    ]
    assert [call for call in calls if call[0] == "mtp_seed"] == [
        ("mtp_seed", 0, 1, 3),
        ("mtp_seed", 1, 1, 3),
    ]
    assert [call for call in calls if call[0] == "write_context_kv"] == [
        ("write_context_kv", 0, (10, 11, 12, 13), (4, 2), 1.0),
        ("write_context_kv", 1, (20, 21, 22, 23), (4, 2), 2.0),
    ]
    assert os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN") is None
    timing = outputs[0].telemetry.to_json_dict()["timing"]
    assert "prefill_batch_ms" in timing


def test_gguf_speculative_mtp_rolls_slots_above_four(monkeypatch) -> None:
    events: list[tuple] = []
    cycle_counts: dict[int, int] = {}

    def fake_open_slots(shared_runner, assets, encoded_prompts, request, *, pool_sessions):
        del shared_runner, assets, pool_sessions
        events.append(("open", tuple(request.prompts)))
        slots = []
        for local_index, _prompt in enumerate(request.prompts):
            slots.append(
                qwen35_gguf._GGUFMTPServingSlot(
                    request_id=local_index,
                    prompt_ids=list(encoded_prompts[local_index]),
                    session=SimpleNamespace(runtime=SimpleNamespace()),
                    resident_draft=SimpleNamespace(),
                    resident_context=SimpleNamespace(),
                    mtp_key_cache=SimpleNamespace(ptr=0x1000 + local_index),
                    mtp_value_cache=SimpleNamespace(ptr=0x2000 + local_index),
                    mtp_buffers=[],
                    hidden_size=2,
                    prev_token=1,
                    seq_position=4,
                    generated_ids=[1],
                )
            )
        return slots

    def fake_run_cycle(slots, assets, request, *, base_env, verify_owner_session=None):
        del assets, request, base_env
        assert verify_owner_session is not None
        live_ids = tuple(slot.request_id for slot in slots if not slot.done)
        events.append(("cycle", live_ids))
        assert len(live_ids) <= qwen35_gguf._MTP_SERVING_TARGET_BATCH_MAX_SLOTS
        for slot in slots:
            if slot.done:
                continue
            request_id = int(slot.request_id)
            cycle_counts[request_id] = cycle_counts.get(request_id, 0) + 1
            slot.timing["target_verify_batch_ms"] = 1.0
            slot.generated_ids.append(2)
            slot.cycles.append(
                {
                    "mode": "llama_compat_direct_commit",
                    "generated_draft_tokens": 1,
                    "accepted_draft_tokens": 1,
                    "visible_output_tokens": 2,
                }
            )
            slot.done = (request_id in {0, 1} and cycle_counts[request_id] == 1) or cycle_counts[request_id] >= 2

    def fake_close(slots, *, reuse=True):
        events.append(("close", tuple(slot.request_id for slot in slots), bool(reuse)))

    def fake_output(prompt_ids, generated_ids, request, *, row_index, resident_slot_count, timing=None):
        del prompt_ids, generated_ids, request, timing
        return SimpleNamespace(text=f"out{row_index}", resident_slot_count=int(resident_slot_count))

    generator = _generator()
    monkeypatch.setattr(generator, "_open_mtp_serving_slots", fake_open_slots)
    monkeypatch.setattr(generator, "_run_mtp_serving_slots_cycle", fake_run_cycle)
    monkeypatch.setattr(generator, "_close_mtp_serving_slots", fake_close)
    monkeypatch.setattr(generator, "_mtp_generation_output", fake_output)

    request = _request(prompts=("first", "second", "long", "long2", "{", "}"), max_tokens=3)
    outputs: list[object] = []
    resident_slots, verify_batching = generator._generate_rolling_mtp_serving_slots(
        SimpleNamespace(),
        qwen35_gguf._GGUFMTPServingAssets(
            weights={},
            token_embd_f32=np.zeros((8, 2), dtype=np.float32),
            rope_cos=np.ones((16, 2), dtype=np.float32),
            rope_sin=np.zeros((16, 2), dtype=np.float32),
        ),
        {
            0: [10, 11],
            1: [20],
            2: [10, 11, 12, 13],
            3: [20, 21, 22, 23],
            4: [5],
            5: [4],
        },
        request,
        base_env={},
        prompt_rows_by_request={},
        generated_ids_by_request={},
        mtp_cycles_by_request={},
        tokenize_ms_by_request={},
        assets_load_ms=0.0,
        pool_sessions=True,
        outputs=outputs,
    )

    assert resident_slots == 4
    assert verify_batching == "packed_slot_batch"
    assert [output.text for output in outputs] == ["out0", "out1", "out2", "out3", "out4", "out5"]
    assert events[:5] == [
        ("open", ("first", "second", "long", "long2")),
        ("cycle", (0, 1, 2, 3)),
        ("close", (1,), True),
        ("open", ("{", "}")),
        ("cycle", (2, 3, 4, 5)),
    ]
    assert events[-1] == ("close", (0,), True)


def test_gguf_speculative_mtp_c2_uses_batch_verifier_when_available() -> None:
    calls: list[tuple] = []

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.position = 4
            self.runtime = SimpleNamespace()

        def verify_target_block(self, input_token_ids, **kwargs):  # pragma: no cover - must not be used
            calls.append(("verify_block", self.slot_id, tuple(input_token_ids), dict(kwargs)))
            raise AssertionError("per-slot verifier should not be used when batch verifier exists")

        def verify_target_blocks_batch(self, jobs):
            self.last_packed_verify_stage_timings_ms = {"packed_verify_total": 1.25}
            calls.append(
                (
                    "verify_batch",
                    tuple((job["session"].slot_id, tuple(job["input_token_ids"])) for job in jobs),
                    tuple(
                        (
                            job["bulk_attention_mode"],
                            job["use_wmma_prefill"],
                            job["capture_linear_state_rows"],
                            job["defer_linear_state_commit"],
                            job["defer_state_scatter"],
                        )
                        for job in jobs
                    ),
                )
            )
            return [
                SimpleNamespace(
                    token_ids=[2, 3],
                    hidden_seeds=np.ones((2, 2), dtype=np.float32),
                    linear_state_rows_captured=True,
                )
                for _job in jobs
            ]

        def _commit_verify_linear_state_row(self, row_index: int, *, position: int):
            calls.append(("commit_row", self.slot_id, int(row_index), int(position)))
            self.position = int(position)

        def fp32_verify_hidden_seed_ptr(self, row_index: int = 0) -> int:
            return 0xD000 + self.slot_id * 0x100 + int(row_index) * 8

    class FakeDraft:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)

        def propose_chain_from_device_seed(self, hidden_seed_ptr, **kwargs):
            calls.append(("draft", self.slot_id, int(hidden_seed_ptr), dict(kwargs)))
            return [2], [[2]], int(kwargs["dense_cache_len"]) + 2

        def write_kv_rows_from_device_seed_base(self, hidden_seed_ptr, token_ids, **kwargs):
            calls.append(
                (
                    "write_commit_kv",
                    self.slot_id,
                    int(hidden_seed_ptr),
                    tuple(int(token) for token in token_ids.tolist()),
                    tuple(int(pos) for pos in kwargs["positions"].tolist()),
                    int(kwargs["dense_cache_len"]),
                )
            )
            return int(kwargs["dense_cache_len"]) + len(token_ids)

    class FakeContext:
        def __init__(self, slot_id: int):
            self.pending_seed = SimpleNamespace(hidden_ptr=0xABC0 + slot_id * 0x100)
            self.accepted: list[int] = []

        def record_verify_seeds(self, rows):
            calls.append(("record_verify_seeds", tuple(row.token_id for row in rows)))

        def accept(self, accepted_draft_tokens: int):
            self.accepted.append(int(accepted_draft_tokens))
            calls.append(("accept", int(accepted_draft_tokens)))

    assets = qwen35_gguf._GGUFMTPServingAssets(
        weights={},
        token_embd_f32=np.zeros((8, 2), dtype=np.float32),
        rope_cos=np.ones((16, 2), dtype=np.float32),
        rope_sin=np.zeros((16, 2), dtype=np.float32),
    )
    slots = [
        qwen35_gguf._GGUFMTPServingSlot(
            request_id=slot_id,
            prompt_ids=[10, 11, 12, 13],
            session=FakeSession(slot_id),
            resident_draft=FakeDraft(slot_id),
            resident_context=FakeContext(slot_id),
            mtp_key_cache=SimpleNamespace(ptr=0x1000 + slot_id),
            mtp_value_cache=SimpleNamespace(ptr=0x2000 + slot_id),
            mtp_buffers=[],
            hidden_size=2,
            prev_token=1,
            seq_position=4,
            generated_ids=[1],
            mtp_device_kv_len=4,
        )
        for slot_id in range(2)
    ]

    generator = _generator()
    generator._run_mtp_serving_slots(
        slots,
        assets,
        _request(prompts=("long", "long2"), max_tokens=3),
        base_env={},
    )

    assert [slot.generated_ids for slot in slots] == [[1, 2, 3], [1, 2, 3]]
    assert [call for call in calls if call[0] == "verify_batch"] == [
        (
            "verify_batch",
            ((0, (1, 2)), (1, (1, 2))),
            (("bulk", False, True, True, True), ("bulk", False, True, True, True)),
        )
    ]
    assert not [call for call in calls if call[0] == "verify_block"]
    assert [call for call in calls if call[0] == "commit_row"] == [
        ("commit_row", 0, 1, 6),
        ("commit_row", 1, 1, 6),
    ]
    assert all("target_verify_batch_ms" in slot.timing for slot in slots)
    assert all(slot.timing["target_packed_verify_total_ms"] == pytest.approx(1.25) for slot in slots)
    assert all("slots_verify_phase_ms" in slot.timing for slot in slots)


def test_gguf_speculative_mtp_final_state_fastpath_uses_batch_final_state(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.position = 4
            self.runtime = SimpleNamespace()

        def _linear_state_snapshot(self):
            calls.append(("snapshot", self.slot_id))
            return ("snapshot", self.slot_id)

        def _free_linear_state_snapshot(self, snapshot):
            calls.append(("free_snapshot", self.slot_id, snapshot))

        def verify_target_block(self, input_token_ids, **kwargs):  # pragma: no cover - must not be used
            calls.append(("verify_block", self.slot_id, tuple(input_token_ids), dict(kwargs)))
            raise AssertionError("full-accept final-state fastpath should not replay")

        def verify_target_blocks_batch(self, jobs):
            calls.append(
                (
                    "verify_batch",
                    tuple((job["session"].slot_id, tuple(job["input_token_ids"])) for job in jobs),
                    tuple(
                        (
                            job["capture_linear_state_rows"],
                            job["defer_linear_state_commit"],
                        )
                        for job in jobs
                    ),
                )
            )
            return [
                SimpleNamespace(
                    token_ids=[2, 3],
                    hidden_seeds=np.ones((2, 2), dtype=np.float32),
                    linear_state_rows_captured=False,
                    final_linear_state_committed=True,
                )
                for _job in jobs
            ]

        def _commit_verify_linear_state_row(self, row_index: int, *, position: int):
            calls.append(("commit_row", self.slot_id, int(row_index), int(position)))

        def fp32_verify_hidden_seed_ptr(self, row_index: int = 0) -> int:
            return 0xD000 + self.slot_id * 0x100 + int(row_index) * 8

    class FakeDraft:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)

        def propose_chain_from_device_seed(self, hidden_seed_ptr, **kwargs):
            return [2], [[2]], int(kwargs["dense_cache_len"]) + 2

        def write_kv_rows_from_device_seed_base(self, hidden_seed_ptr, token_ids, **kwargs):
            calls.append(("write_commit_kv", self.slot_id, tuple(int(token) for token in token_ids.tolist())))
            return int(kwargs["dense_cache_len"]) + len(token_ids)

    class FakeContext:
        def __init__(self, slot_id: int):
            self.pending_seed = SimpleNamespace(hidden_ptr=0xABC0 + slot_id * 0x100)

        def record_verify_seeds(self, rows):
            calls.append(("record_verify_seeds", tuple(row.token_id for row in rows)))

        def accept(self, accepted_draft_tokens: int):
            calls.append(("accept", int(accepted_draft_tokens)))

    assets = qwen35_gguf._GGUFMTPServingAssets(
        weights={},
        token_embd_f32=np.zeros((8, 2), dtype=np.float32),
        rope_cos=np.ones((16, 2), dtype=np.float32),
        rope_sin=np.zeros((16, 2), dtype=np.float32),
    )
    slots = [
        qwen35_gguf._GGUFMTPServingSlot(
            request_id=slot_id,
            prompt_ids=[10, 11, 12, 13],
            session=FakeSession(slot_id),
            resident_draft=FakeDraft(slot_id),
            resident_context=FakeContext(slot_id),
            mtp_key_cache=SimpleNamespace(ptr=0x1000 + slot_id),
            mtp_value_cache=SimpleNamespace(ptr=0x2000 + slot_id),
            mtp_buffers=[],
            hidden_size=2,
            prev_token=1,
            seq_position=4,
            generated_ids=[1],
            mtp_device_kv_len=4,
        )
        for slot_id in range(2)
    ]

    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_VERIFY_FINAL_STATE_FASTPATH", "1")

    generator = _generator()
    generator._run_mtp_serving_slots(
        slots,
        assets,
        _request(prompts=("long", "long2"), max_tokens=3),
        base_env={},
    )

    assert [call for call in calls if call[0] == "verify_batch"] == [
        ("verify_batch", ((0, (1, 2)), (1, (1, 2))), ((False, False), (False, False)))
    ]
    assert [call for call in calls if call[0] == "snapshot"] == [("snapshot", 0), ("snapshot", 1)]
    assert not [call for call in calls if call[0] == "commit_row"]
    assert not [call for call in calls if call[0] == "verify_block"]
    assert [call for call in calls if call[0] == "accept"] == [("accept", 1), ("accept", 1)]


def test_gguf_speculative_mtp_deferred_verify_scatter_commits_from_owner(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.position = 4
            self.runtime = SimpleNamespace()

        def verify_target_blocks_batch(self, jobs):
            calls.append(
                (
                    "verify_batch",
                    tuple((job["session"].slot_id, tuple(job["input_token_ids"])) for job in jobs),
                    tuple(
                        (
                            job["capture_linear_state_rows"],
                            job["defer_linear_state_commit"],
                            job["defer_state_scatter"],
                        )
                        for job in jobs
                    ),
                )
            )
            return [
                SimpleNamespace(
                    token_ids=[2, 3],
                    hidden_seeds=np.ones((2, 2), dtype=np.float32),
                    linear_state_rows_captured=True,
                    final_linear_state_committed=False,
                    deferred_packed_state=SimpleNamespace(owner=self, slot_index=index),
                )
                for index, _job in enumerate(jobs)
            ]

        def _commit_deferred_packed_verify_state(
            self,
            deferred_state,
            destination_session,
            *,
            commit_row_index: int,
            position: int,
            hidden_rows: int,
        ):
            calls.append(
                (
                    "commit_deferred",
                    self.slot_id,
                    int(deferred_state.slot_index),
                    destination_session.slot_id,
                    int(commit_row_index),
                    int(position),
                    int(hidden_rows),
                )
            )
            destination_session.position = int(position)

        def verify_target_block(self, input_token_ids, **kwargs):  # pragma: no cover - must not be used
            calls.append(("verify_block", self.slot_id, tuple(input_token_ids), dict(kwargs)))
            raise AssertionError("per-slot verifier should not be used when batch verifier exists")

        def _commit_verify_linear_state_row(self, row_index: int, *, position: int):
            calls.append(("commit_row", self.slot_id, int(row_index), int(position)))

        def fp32_verify_hidden_seed_ptr(self, row_index: int = 0) -> int:
            return 0xD000 + self.slot_id * 0x100 + int(row_index) * 8

    class FakeDraft:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)

        def propose_chain_from_device_seed(self, hidden_seed_ptr, **kwargs):
            return [2], [[2]], int(kwargs["dense_cache_len"]) + 2

        def write_kv_rows_from_device_seed_base(self, hidden_seed_ptr, token_ids, **kwargs):
            calls.append(("write_commit_kv", self.slot_id, tuple(int(token) for token in token_ids.tolist())))
            return int(kwargs["dense_cache_len"]) + len(token_ids)

    class FakeContext:
        def __init__(self, slot_id: int):
            self.pending_seed = SimpleNamespace(hidden_ptr=0xABC0 + slot_id * 0x100)

        def record_verify_seeds(self, rows):
            calls.append(("record_verify_seeds", tuple(row.token_id for row in rows)))

        def accept(self, accepted_draft_tokens: int):
            calls.append(("accept", int(accepted_draft_tokens)))

    assets = qwen35_gguf._GGUFMTPServingAssets(
        weights={},
        token_embd_f32=np.zeros((8, 2), dtype=np.float32),
        rope_cos=np.ones((16, 2), dtype=np.float32),
        rope_sin=np.zeros((16, 2), dtype=np.float32),
    )
    slots = [
        qwen35_gguf._GGUFMTPServingSlot(
            request_id=slot_id,
            prompt_ids=[10, 11, 12, 13],
            session=FakeSession(slot_id),
            resident_draft=FakeDraft(slot_id),
            resident_context=FakeContext(slot_id),
            mtp_key_cache=SimpleNamespace(ptr=0x1000 + slot_id),
            mtp_value_cache=SimpleNamespace(ptr=0x2000 + slot_id),
            mtp_buffers=[],
            hidden_size=2,
            prev_token=1,
            seq_position=4,
            generated_ids=[1],
            mtp_device_kv_len=4,
        )
        for slot_id in range(2)
    ]

    monkeypatch.delenv("HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER", raising=False)

    generator = _generator()
    generator._run_mtp_serving_slots(
        slots,
        assets,
        _request(prompts=("long", "long2"), max_tokens=3),
        base_env={},
    )

    assert [call for call in calls if call[0] == "verify_batch"] == [
        (
            "verify_batch",
            ((0, (1, 2)), (1, (1, 2))),
            ((True, True, True), (True, True, True)),
        )
    ]
    assert [call for call in calls if call[0] == "commit_deferred"] == [
        ("commit_deferred", 0, 0, 0, 1, 6, 2),
        ("commit_deferred", 0, 1, 1, 1, 6, 2),
    ]
    assert not [call for call in calls if call[0] == "commit_row"]
    assert not [call for call in calls if call[0] == "verify_block"]


def test_gguf_speculative_mtp_final_state_fastpath_partial_replays_prefix() -> None:
    calls: list[tuple] = []

    class FakeSession:
        def __init__(self):
            self.position = 6

        def _restore_linear_state_snapshot(self, snapshot, *, position: int):
            calls.append(("restore", snapshot, int(position)))
            self.position = int(position)

        def verify_target_block(self, input_token_ids, **kwargs):
            calls.append(("verify_replay", tuple(input_token_ids), dict(kwargs)))
            self.position += len(input_token_ids)
            return SimpleNamespace(token_ids=list(input_token_ids))

        def _free_linear_state_snapshot(self, snapshot):
            calls.append(("free_snapshot", snapshot))

        def _commit_verify_linear_state_row(self, row_index: int, *, position: int):
            calls.append(("commit_row", int(row_index), int(position)))

        def fp32_verify_hidden_seed_ptr(self, row_index: int = 0) -> int:
            return 0xD000 + int(row_index) * 8

    class FakeContext:
        def record_verify_seeds(self, rows):
            calls.append(("record_verify_seeds", tuple(row.token_id for row in rows)))

        def accept(self, accepted_draft_tokens: int):
            calls.append(("accept", int(accepted_draft_tokens)))

    session = FakeSession()
    slot = qwen35_gguf._GGUFMTPServingSlot(
        request_id=0,
        prompt_ids=[10, 11, 12, 13],
        session=session,
        resident_draft=SimpleNamespace(),
        resident_context=FakeContext(),
        mtp_key_cache=SimpleNamespace(ptr=0x1000),
        mtp_value_cache=SimpleNamespace(ptr=0x2000),
        mtp_buffers=[],
        hidden_size=2,
        prev_token=1,
        seq_position=4,
        generated_ids=[1],
        mtp_device_kv_len=4,
    )
    drafted = qwen35_gguf._GGUFMTPDraftedCycle(
        slot=slot,
        advance_start=0.0,
        cycle_mtp_kv_base_len=4,
        draft_tokens=[2],
        block_inputs=[1, 2],
        block_start=4,
        direct_commit_exact=True,
        snapshot="snap",
    )
    verified = qwen35_gguf._GGUFMTPVerifiedCycle(
        drafted=drafted,
        block_result=SimpleNamespace(
            token_ids=[8, 3],
            linear_state_rows_captured=False,
            final_linear_state_committed=True,
        ),
        block_target_tokens=[8, 3],
        acceptance={"accepted_draft_tokens": 0, "output_tokens": [8]},
    )
    assets = qwen35_gguf._GGUFMTPServingAssets(
        weights={},
        token_embd_f32=np.zeros((8, 2), dtype=np.float32),
        rope_cos=np.ones((16, 2), dtype=np.float32),
        rope_sin=np.zeros((16, 2), dtype=np.float32),
    )

    generator = _generator()
    generator._commit_mtp_serving_cycle(
        verified,
        assets,
        _request(prompts=("long",), max_tokens=2),
    )

    assert ("restore", "snap", 4) in calls
    assert [call for call in calls if call[0] == "verify_replay"] == [
        (
            "verify_replay",
            (1,),
            {
                "bulk_attention_mode": "bulk",
                "use_wmma_prefill": False,
                "advance_state_only": True,
            },
        )
    ]
    assert not [call for call in calls if call[0] == "commit_row"]
    assert ("free_snapshot", "snap") in calls
    assert ("record_verify_seeds", (8,)) in calls
    assert [slot.generated_ids, slot.prev_token, slot.seq_position] == [[1, 8], 8, 5]


def test_gguf_speculative_mtp_stream_draft_uses_slot_streams(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        next_stream = 700

        def stream_create(self, *, nonblocking=True):
            stream = FakeRuntime.next_stream
            FakeRuntime.next_stream += 1
            calls.append(("stream_create", stream, bool(nonblocking)))
            return stream

        def stream_synchronize(self, stream):
            calls.append(("stream_sync", int(stream)))

        def stream_destroy(self, stream):
            calls.append(("stream_destroy", int(stream)))

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.runtime = FakeRuntime()

        def close(self):
            calls.append(("session_close", self.slot_id))

    class FakeDraft:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)

        def propose_chain_from_device_seed(self, hidden_seed_ptr, **kwargs):
            calls.append(
                (
                    "draft",
                    self.slot_id,
                    int(hidden_seed_ptr),
                    int(kwargs["stream"]),
                    int(kwargs["dense_cache_len"]),
                )
            )
            return [2], [[2]], int(kwargs["dense_cache_len"]) + 2

        def close(self):
            calls.append(("draft_close", self.slot_id))

    class FakeContext:
        def __init__(self, slot_id: int):
            self.pending_seed = SimpleNamespace(hidden_ptr=0xABC0 + slot_id * 0x100)

    assets = qwen35_gguf._GGUFMTPServingAssets(
        weights={},
        token_embd_f32=np.zeros((8, 2), dtype=np.float32),
        rope_cos=np.ones((16, 2), dtype=np.float32),
        rope_sin=np.zeros((16, 2), dtype=np.float32),
    )
    slots = [
        qwen35_gguf._GGUFMTPServingSlot(
            request_id=slot_id,
            prompt_ids=[10, 11, 12, 13],
            session=FakeSession(slot_id),
            resident_draft=FakeDraft(slot_id),
            resident_context=FakeContext(slot_id),
            mtp_key_cache=SimpleNamespace(ptr=0x1000 + slot_id),
            mtp_value_cache=SimpleNamespace(ptr=0x2000 + slot_id),
            mtp_buffers=[],
            hidden_size=2,
            prev_token=1,
            seq_position=4,
            generated_ids=[1],
            mtp_device_kv_len=4,
        )
        for slot_id in range(2)
    ]

    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT", "1")
    monkeypatch.setattr(qwen35_gguf, "_free_mtp_buffers", lambda buffers, *, runtime: None)
    generator = _generator()
    drafted = generator._try_draft_mtp_serving_slots_streams(
        slots,
        assets,
        _request(prompts=("long", "long2"), max_tokens=3),
        base_env={},
    )

    assert drafted is not None
    assert [cycle.block_inputs for cycle in drafted] == [[1, 2], [1, 2]]
    assert [slot.draft_stream for slot in slots] == [700, 701]
    assert sorted(call for call in calls if call[0] == "draft") == [
        ("draft", 0, 0xABC0, 700, 4),
        ("draft", 1, 0xACC0, 701, 4),
    ]
    assert all("draft_stream_batch_ms" in slot.timing for slot in slots)

    generator._close_mtp_serving_slots(slots, reuse=False)

    assert [call for call in calls if call[0] == "stream_destroy"] == [
        ("stream_destroy", 701),
        ("stream_destroy", 700),
    ]


def test_gguf_speculative_mtp_batch_verifier_notimplemented_falls_back() -> None:
    calls: list[tuple] = []

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.position = 4

        def verify_target_blocks_batch(self, jobs):
            calls.append(("verify_batch", tuple((job["session"].slot_id, tuple(job["input_token_ids"])) for job in jobs)))
            raise NotImplementedError("unsupported packed shape")

    slots = [
        qwen35_gguf._GGUFMTPServingSlot(
            request_id=slot_id,
            prompt_ids=[10, 11, 12, 13],
            session=FakeSession(slot_id),
            resident_draft=SimpleNamespace(),
            resident_context=SimpleNamespace(),
            mtp_key_cache=SimpleNamespace(ptr=0x1000 + slot_id),
            mtp_value_cache=SimpleNamespace(ptr=0x2000 + slot_id),
            mtp_buffers=[],
            hidden_size=2,
            prev_token=1,
            seq_position=4,
            generated_ids=[1],
            mtp_device_kv_len=4,
        )
        for slot_id in range(2)
    ]
    drafted_cycles = [
        qwen35_gguf._GGUFMTPDraftedCycle(
            slot=slot,
            advance_start=0.0,
            cycle_mtp_kv_base_len=4,
            draft_tokens=[2],
            block_inputs=[1, 2],
            block_start=4,
            direct_commit_exact=True,
        )
        for slot in slots
    ]

    generator = _generator()

    assert generator._try_verify_mtp_serving_cycles_batch(drafted_cycles) is None
    assert calls == [("verify_batch", ((0, (1, 2)), (1, (1, 2))))]


def test_gguf_speculative_mtp_batch_verifier_chunks_above_four_slots() -> None:
    calls: list[tuple] = []

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.position = 4

        def verify_target_blocks_batch(self, jobs):
            calls.append(("verify_batch", self.slot_id, tuple(job["session"].slot_id for job in jobs)))
            return [
                SimpleNamespace(
                    token_ids=[2, 3],
                    hidden_seeds=np.ones((2, 2), dtype=np.float32),
                    linear_state_rows_captured=True,
                )
                for _job in jobs
            ]

    slots = [
        qwen35_gguf._GGUFMTPServingSlot(
            request_id=slot_id,
            prompt_ids=[10, 11, 12, 13],
            session=FakeSession(slot_id),
            resident_draft=SimpleNamespace(),
            resident_context=SimpleNamespace(),
            mtp_key_cache=SimpleNamespace(ptr=0x1000 + slot_id),
            mtp_value_cache=SimpleNamespace(ptr=0x2000 + slot_id),
            mtp_buffers=[],
            hidden_size=2,
            prev_token=1,
            seq_position=4,
            generated_ids=[1],
            mtp_device_kv_len=4,
        )
        for slot_id in range(8)
    ]
    drafted_cycles = [
        qwen35_gguf._GGUFMTPDraftedCycle(
            slot=slot,
            advance_start=0.0,
            cycle_mtp_kv_base_len=4,
            draft_tokens=[2],
            block_inputs=[1, 2],
            block_start=4,
            direct_commit_exact=True,
        )
        for slot in slots
    ]

    generator = _generator()
    results = generator._try_verify_mtp_serving_cycles_batch(drafted_cycles)

    assert len(results or []) == 8
    assert calls == [
        ("verify_batch", 0, (0, 1, 2, 3)),
        ("verify_batch", 4, (4, 5, 6, 7)),
    ]
    assert all("target_verify_batch_ms" in slot.timing for slot in slots)


def test_gguf_speculative_mtp_batch_verifier_streams_chunks_above_four_slots(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        next_stream = 900

        def stream_create(self, *, nonblocking=True):
            stream = FakeRuntime.next_stream
            FakeRuntime.next_stream += 1
            calls.append(("stream_create", stream, bool(nonblocking)))
            return stream

        def stream_synchronize(self, stream):
            calls.append(("stream_sync", int(stream)))

        def stream_destroy(self, stream):
            calls.append(("stream_destroy", int(stream)))

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.position = 4
            self.runtime = FakeRuntime()

        def verify_target_blocks_batch(self, jobs, *, stream: int = 0):
            calls.append(("verify_batch", self.slot_id, int(stream), tuple(job["session"].slot_id for job in jobs)))
            return [
                SimpleNamespace(
                    token_ids=[2, 3],
                    hidden_seeds=np.ones((2, 2), dtype=np.float32),
                    linear_state_rows_captured=True,
                )
                for _job in jobs
            ]

        def reset(self):
            calls.append(("reset", self.slot_id))

        def close(self):
            calls.append(("session_close", self.slot_id))

    class FakeDraft:
        def close(self):
            calls.append(("draft_close",))

    slots = [
        qwen35_gguf._GGUFMTPServingSlot(
            request_id=slot_id,
            prompt_ids=[10, 11, 12, 13],
            session=FakeSession(slot_id),
            resident_draft=FakeDraft(),
            resident_context=SimpleNamespace(),
            mtp_key_cache=SimpleNamespace(ptr=0x1000 + slot_id),
            mtp_value_cache=SimpleNamespace(ptr=0x2000 + slot_id),
            mtp_buffers=[],
            hidden_size=2,
            prev_token=1,
            seq_position=4,
            generated_ids=[1],
            mtp_device_kv_len=4,
        )
        for slot_id in range(8)
    ]
    drafted_cycles = [
        qwen35_gguf._GGUFMTPDraftedCycle(
            slot=slot,
            advance_start=0.0,
            cycle_mtp_kv_base_len=4,
            draft_tokens=[2],
            block_inputs=[1, 2],
            block_start=4,
            direct_commit_exact=True,
        )
        for slot in slots
    ]

    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY", "1")
    monkeypatch.setattr(qwen35_gguf, "_free_mtp_buffers", lambda buffers, *, runtime: None)

    generator = _generator()
    results = generator._try_verify_mtp_serving_cycles_batch(drafted_cycles)

    assert len(results or []) == 8
    assert [call for call in calls if call[0] == "stream_create"] == [
        ("stream_create", 900, True),
        ("stream_create", 901, True),
    ]
    assert sorted(call for call in calls if call[0] == "verify_batch") == [
        ("verify_batch", 0, 900, (0, 1, 2, 3)),
        ("verify_batch", 4, 901, (4, 5, 6, 7)),
    ]
    assert all("target_verify_batch_ms" in slot.timing for slot in slots)
    assert all("target_verify_stream_chunks_ms" in slot.timing for slot in slots)

    generator._close_mtp_serving_slots(slots, reuse=False)

    assert [call for call in calls if call[0] == "stream_destroy"] == [
        ("stream_destroy", 901),
        ("stream_destroy", 900),
    ]


def test_gguf_mtp_metadata_reports_packed_slot_batch() -> None:
    payload = qwen35_gguf._gguf_mtp_last_batch_generation(
        _FakeTokenizer(),
        _request(prompts=("long", "long2"), max_tokens=3),
        SimpleNamespace(active_processors=(), fast_path_blockers=(), fallback_reason=None, mode=SimpleNamespace(value="greedy_fast")),
        {0: [10, 11, 12, 13], 1: [20, 21, 22, 23]},
        {0: [1, 2, 3], 1: [1, 2, 3]},
        {},
        outputs=(
            SimpleNamespace(),
            SimpleNamespace(),
        ),
        cycles_by_request={
            0: [
                {
                    "mode": "llama_compat_direct_commit",
                    "generated_draft_tokens": 2,
                    "accepted_draft_tokens": 2,
                    "visible_output_tokens": 3,
                },
                {
                    "mode": "llama_compat_direct_commit",
                    "generated_draft_tokens": 2,
                    "accepted_draft_tokens": 1,
                    "visible_output_tokens": 2,
                },
            ],
            1: [
                {
                    "mode": "llama_compat_direct_commit",
                    "generated_draft_tokens": 2,
                    "accepted_draft_tokens": 0,
                    "visible_output_tokens": 1,
                }
            ],
        },
        resident_slot_count=2,
        target_verify_batching="packed_slot_batch",
    )

    mtp = payload["speculative_mtp"]
    assert mtp["target_verify_batching"] == "packed_slot_batch"
    assert mtp["target_verify_rows"] == 9
    assert mtp["accepted_draft_tokens_histogram"] == {"0": 1, "1": 1, "2": 1}
    assert mtp["cycle_shape_histogram"] == {
        "draft2_accept0": 1,
        "draft2_accept1": 1,
        "draft2_accept2": 1,
    }
    assert mtp["full_accept_cycles"] == 1
    assert mtp["partial_accept_cycles"] == 1
    assert mtp["reject_cycles"] == 1
    assert mtp["full_accept_rate"] == pytest.approx(1.0 / 3.0)
    assert mtp["linear_state_captured_rows"] == 9
    assert mtp["linear_state_commit_rows"] == 3
    assert mtp["linear_state_extra_rows"] == 6
    assert mtp["hidden_seed_captured_rows"] == 9
    assert mtp["hidden_seed_needed_rows"] == 6
    assert mtp["hidden_seed_extra_rows"] == 3


def test_gguf_submit_poll_runner_owns_and_reuses_resident_sessions(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.model_path = str(model_path)
            self.runtime = SimpleNamespace()
            self.backend = kwargs.get("backend", "hip_gfx1100")
            self.target_arch = "gfx1100"
            self.vocab_size = 128
            self.weights = SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=4))
            calls.append(("runner_init", self.model_path))

    class FakeSession:
        next_slot = 0

        def __init__(self, model_path, **kwargs):
            self.slot_id = FakeSession.next_slot
            FakeSession.next_slot += 1
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            self._packed_decode_state_dirty = False
            self._packed_decode_sessions = ()
            calls.append(("session_init", self.slot_id, kwargs.get("max_sequence_length")))

        def reset(self):
            self.position = 0
            calls.append(("reset", self.slot_id))

        def prefill(self, token_ids, **kwargs):
            self.position = len(token_ids)
            calls.append(("prefill", self.slot_id, tuple(token_ids), dict(kwargs)))
            logits = None
            if kwargs.get("return_logits"):
                logits = np.full((1, 128), -100.0, dtype=np.float32)
                logits[0, 1] = 100.0
            return SimpleNamespace(token_id=1, logits=logits)

        def step(self, token_id, **kwargs):
            self.position += 1
            calls.append(("step", self.slot_id, int(token_id), dict(kwargs)))
            next_token = int(token_id) + 1
            logits = None
            if kwargs.get("return_logits"):
                logits = np.full((1, 128), -100.0, dtype=np.float32)
                logits[0, next_token] = 100.0
            return SimpleNamespace(token_id=next_token, logits=logits)

        def step_batch_native(self, token_ids, *, sessions, positions, **kwargs):
            self.last_packed_execution_manifest = {
                "schema": 1,
                "kind": "gguf_packed_ar_execution_manifest",
                "rows": len(token_ids),
                "model_step": {"complete_c1_session_replays": 0},
            }
            calls.append(
                (
                    "step_batch_native",
                    self.slot_id,
                    tuple(int(token) for token in token_ids),
                    tuple(session.slot_id for session in sessions),
                    tuple(int(position) for position in positions),
                    dict(kwargs),
                )
            )
            for session in sessions:
                session.position += 1
            self._packed_decode_state_dirty = True
            self._packed_decode_sessions = tuple(sessions)
            return [SimpleNamespace(token_id=int(token) + 1) for token in token_ids]

        def flush_packed_decode_state(self):
            calls.append(("flush", self.slot_id))
            self._packed_decode_state_dirty = False
            return True

        def close(self):
            calls.append(("close", self.slot_id))

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()
    generator.backend = "hip_gfx1100"
    generator._shared_runner = None
    generator._shared_runner_lock = threading.Lock()
    generator._prepared_max_sequence_length = 64
    generator._shared_session_pool = {}
    generator._shared_session_pool_lock = threading.Lock()
    generator._shared_mtp_draft_pool = {}
    generator._shared_mtp_draft_pool_lock = threading.Lock()

    adapter = SubmitPollTextGenerator(generator, capacity=2)
    runner = adapter._runner
    assert isinstance(runner, qwen35_gguf.Qwen35GGUFResidentModelRunner)
    assert [call[0] for call in calls].count("runner_init") == 1
    assert [call[0] for call in calls].count("session_init") == 2

    adapter.prepare(max_sequence_length=128)
    assert [call[0] for call in calls].count("runner_init") == 1
    assert [call[0] for call in calls].count("session_init") == 4
    assert [call[2] for call in calls if call[0] == "session_init"][-2:] == [128, 128]

    first = adapter.generate_detailed(_request(prompts=("first", "second"), max_tokens=3))
    second = adapter.generate_detailed(_request(prompts=("first",), max_tokens=2))
    greedy_last = dict(generator.last_batch_generation or {})
    sampled = adapter.generate_detailed(
        _request(prompts=("first",), max_tokens=2, temperature=0.7, seed=17)
    )
    zero = adapter.generate_detailed(_request(prompts=("first",), max_tokens=0))

    assert [output.text for output in first] == ["BCD", "BCD"]
    assert [output.generated_token_ids for output in first] == [(1, 2, 3), (1, 2, 3)]
    assert [output.text for output in second] == ["BC"]
    assert [output.generated_token_ids for output in sampled] == [(1, 2)]
    assert [output.generated_token_ids for output in zero] == [()]
    assert [call[0] for call in calls].count("runner_init") == 1
    assert [call[0] for call in calls].count("session_init") == 4
    assert [call[2] for call in calls if call[0] == "step_batch_native"] == [(1, 1), (2, 2)]
    assert [call for call in calls if call[0] == "step"][-1][2] == 1
    assert runner.active_request_ids == ()
    assert runner.available_session_count == 2
    assert greedy_last["path"] == "gguf_packed_ar_server_decode"
    assert generator.last_batch_generation is not None
    assert generator.last_batch_generation["path"] == "gguf_resident_model_loop"
    assert generator.last_batch_generation["serial_decode_fallback"] is True

    observability = adapter.live_loop_snapshot()["runner"]
    assert observability["model_runner"] == {
        "capacity": 2,
        "active_request_ids": [],
        "active_requests": 0,
        "available_sessions": 2,
    }
    assert observability["routes"]["counts"] == {
        "native_full_prefill_rows": 3,
        "native_incremental_prefill_chunks": 0,
        "native_packed_decode_steps": 2,
        "native_c1_decode_steps": 1,
        "serial_decode_fallback_steps": 0,
        "resident_fallback_requests": 2,
    }
    assert observability["routes"]["last_execution_manifest"] == {
        "schema": 1,
        "kind": "gguf_packed_ar_execution_manifest",
        "rows": 2,
        "model_step": {"complete_c1_session_replays": 0},
    }
    assert observability["routes"]["fallback_reasons"]
    assert len(observability["routes"]["recent_completed"]) == 5
    assert observability["graph_buckets"]["captures_total"] == 0
    assert observability["graph_buckets"]["buckets"] == {}


def test_gguf_resident_runner_device_kv_admission_is_atomic_at_high_water() -> None:
    class FakeSession:
        def __init__(self, slot_id: int) -> None:
            self.slot_id = int(slot_id)
            self.scratch = SimpleNamespace(max_positions=513)
            self.allocation = None
            self.pool = None

        def create_device_kv_pool(self, **config):
            return DeviceChunkedKVPool(
                page_bytes=4096,
                initial_pages=int(config["initial_pages"]),
                low_water_pages=int(config["low_water_pages"]),
                high_water_pages=(
                    None
                    if config["high_water_pages"] is None
                    else int(config["high_water_pages"])
                ),
                chunk_pages=int(config["chunk_pages"]),
                idle_grace_seconds=float(config["idle_grace_seconds"]),
                allocate_chunk=lambda start, pages: {
                    "ptr": 0xA0000000 + int(start) * 4096,
                    "pages": int(pages),
                },
                free_chunk=lambda backing: None,
                page_pointer=lambda backing, local_page: int(backing["ptr"]) + int(local_page) * 4096,
            )

        def bind_device_kv_allocation(self, pool, allocation) -> None:
            assert self.allocation is None
            self.pool = pool
            self.allocation = allocation

        def invalidate_device_kv_graphs(self) -> int:
            return 0

        def unbind_device_kv_allocation(self):
            allocation = self.allocation
            assert allocation is not None
            self.pool = None
            self.allocation = None
            return allocation

        def reset(self) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeOwner:
        backend = "hip_gfx1100"
        target_arch = "gfx1100"
        tokenizer = _FakeTokenizer()
        _prepared_max_sequence_length = 256

        def __init__(self) -> None:
            self.sessions = [FakeSession(index) for index in range(3)]

        def _get_shared_runner(self):
            return SimpleNamespace(runtime=SimpleNamespace(mem_get_info=lambda: (100, 200)))

        def _acquire_shared_session(self, shared_runner, **kwargs):
            del shared_runner, kwargs
            session = self.sessions.pop(0)
            return session, ("continuous_ar_dynamic_kv", True, True, 256), False

        def _release_shared_session(self, key, session) -> None:
            del key
            self.sessions.append(session)

    owner = FakeOwner()
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner(owner, capacity=3)
    runner.configure_engine_loop(
        EngineLoopConfig(
            max_active_requests=3,
            kv_pool_initial_pages=1,
            kv_pool_low_water_pages=1,
            kv_pool_high_water_pages=2,
            kv_pool_chunk_pages=1,
            kv_pool_idle_grace_seconds=0.0,
        )
    )
    request = _request(prompts=("first", "first", "first"), max_tokens=2, ignore_eos=True)
    runner.register_batch((1, 2, 3), request, prompt_rows=((10, 11), (10, 11), (10, 11)))

    runner.reserve_admission(SimpleNamespace(request_id=1))
    runner.reserve_admission(SimpleNamespace(request_id=2))
    before = runner.kv_pool_stats
    assert before is not None

    with pytest.raises(MemoryError, match="high-water"):
        runner.reserve_admission(SimpleNamespace(request_id=3))

    after = runner.kv_pool_stats
    assert after is not None
    assert after.current_pages == before.current_pages == 2
    assert after.refcounted_pages == before.refcounted_pages == 2
    assert after.grow_failures == before.grow_failures + 1
    assert runner._rows[3].lease is None
    assert runner._rows[3].kv_allocation is None

    runner.rollback_admission(SimpleNamespace(request_id=1))
    runner.reserve_admission(SimpleNamespace(request_id=3))
    assert runner._rows[3].lease is not None
    assert runner._rows[3].kv_allocation is not None
    assert runner.kv_pool_memory_snapshot()["dynamic_pool"]["refcounted_pages"] == 2
    observability = runner.observability_snapshot()
    assert observability["kv_pool"] == runner.kv_pool_stats.to_json_dict()
    assert observability["kv_pool"]["current_pages"] == 2
    assert observability["kv_pool"]["high_water_observed_pages"] == 2
    assert observability["kv_pool"]["refcounted_pages"] == 2
    assert observability["kv_pool"]["pinned_pages"] == 0
    assert observability["kv_pool"]["grow_events"] == 1
    assert observability["kv_pool"]["grow_failures"] == 1
    assert observability["kv_pool"]["shrink_events"] == 0

    runner.rollback_admission(SimpleNamespace(request_id=2))
    runner.rollback_admission(SimpleNamespace(request_id=3))
    runner.close()

    ceiling_runner = qwen35_gguf.Qwen35GGUFResidentModelRunner(owner, capacity=3)
    ceiling_runner.configure_engine_loop(EngineLoopConfig(max_active_requests=3))
    assert ceiling_runner.kv_pool is not None
    assert ceiling_runner.kv_pool.high_water_pages is None
    assert ceiling_runner.kv_pool.current_pages == 9
    assert ceiling_runner.kv_pool.low_water_pages == 9
    assert ceiling_runner.kv_pool.chunk_pages == 9
    ceiling_runner.close()


def test_gguf_resident_runner_graph_observability_is_bucketed_and_cumulative() -> None:
    class FakeGraphKey:
        key_sha256 = "bucket-c2"

        def as_dict(self):
            return {
                "key_sha256": self.key_sha256,
                "physical_rows": 2,
                "active_rows": 2,
                "active_mask": [True, True],
            }

    class FakeGraph:
        def __init__(self) -> None:
            self.bucket_key = FakeGraphKey()
            # Match the real Qwen35GGUFDecodeGraph replay ABI.
            self.replayed_steps = 0
            self.steps_per_replay = 2
            self.closed = False

    class FakeSession:
        def __init__(self) -> None:
            self._decode_graphs = []
            self._device_kv_graph_handles = {}

        def invalidate_device_kv_graphs(self) -> int:
            invalidated = 0
            for handle in self._decode_graphs:
                if not handle.closed:
                    handle.closed = True
                    invalidated += 1
            return invalidated

        def reset(self) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeOwner:
        backend = "hip_gfx1100"
        target_arch = "gfx1100"
        tokenizer = _FakeTokenizer()
        _prepared_max_sequence_length = 64

        def __init__(self) -> None:
            self.session = FakeSession()

        def _get_shared_runner(self):
            return SimpleNamespace(runtime=SimpleNamespace(mem_get_info=lambda: (100, 200)))

        def _acquire_shared_session(self, shared_runner, **kwargs):
            del shared_runner, kwargs
            return self.session, ("continuous_ar_dynamic_kv", True, True, 64), False

        def _release_shared_session(self, key, session) -> None:
            del key, session

    owner = FakeOwner()
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner(owner, capacity=1)
    handle = FakeGraph()
    owner.session._decode_graphs.append(handle)

    captured = runner.observability_snapshot()["graph_buckets"]
    assert captured["entries"] == 1
    assert captured["captures_total"] == 1
    assert captured["hits_total"] == 0
    assert captured["replays_total"] == 0
    assert captured["invalidations_total"] == 0
    assert captured["buckets"]["bucket-c2"] == {
        "bucket_key": {
            "key_sha256": "bucket-c2",
            "physical_rows": 2,
            "active_rows": 2,
            "active_mask": [True, True],
        },
        "entries": 1,
        "captures": 1,
        "hits": 0,
        "replays": 0,
        "invalidations": 0,
    }

    handle.replayed_steps = 4
    replayed = runner.observability_snapshot()["graph_buckets"]
    assert replayed["hits_total"] == 2
    assert replayed["replays_total"] == 2
    assert replayed["buckets"]["bucket-c2"]["hits"] == 2
    assert replayed["buckets"]["bucket-c2"]["replays"] == 2

    request = _request(prompts=("first",), max_tokens=1, ignore_eos=True)
    runner.register_batch((1,), request, prompt_rows=((10, 11),))
    runner._rows[1].lease = runner._available.pop()
    runner.rollback_admission(SimpleNamespace(request_id=1))

    invalidated = runner.observability_snapshot()["graph_buckets"]
    assert invalidated["entries"] == 0
    assert invalidated["captures_total"] == 1
    assert invalidated["replays_total"] == 2
    assert invalidated["invalidations_total"] == 1
    assert invalidated["buckets"]["bucket-c2"]["invalidations"] == 1
    runner.close()


def test_gguf_resident_runner_commits_incremental_prefill_chunks() -> None:
    calls: list[tuple] = []

    class FakeSession:
        def __init__(self) -> None:
            self.position = 0

        def reset(self) -> None:
            self.position = 0
            calls.append(("reset",))

        def prefill(self, token_ids, **kwargs):
            raise AssertionError(f"incremental prefill must not replay the full prompt: {token_ids!r} {kwargs!r}")

        def prefill_batch_native(self, prompt_token_ids, *, sessions, **kwargs):
            assert sessions == [self]
            chunk = tuple(int(token) for token in prompt_token_ids[0])
            start = self.position
            self.position += len(chunk)
            calls.append(("prefill_batch_native", chunk, start, self.position, dict(kwargs)))
            return [SimpleNamespace(token_id=100 + chunk[-1])]

        def close(self) -> None:
            calls.append(("close",))

    class FakeOwner:
        backend = "hip_gfx1100"
        target_arch = "gfx1100"
        tokenizer = _FakeTokenizer()
        _prepared_max_sequence_length = 64

        def __init__(self) -> None:
            self.session = FakeSession()

        def _get_shared_runner(self):
            return SimpleNamespace()

        def _acquire_shared_session(self, shared_runner, **kwargs):
            del shared_runner, kwargs
            return self.session, ("continuous_ar", True, True, 64), False

        def _release_shared_session(self, key, session) -> None:
            del key, session

    owner = FakeOwner()
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner(owner, capacity=1)
    request = _request(prompts=((10, 11, 12, 13, 14),), max_tokens=2, ignore_eos=True)
    runner.register_batch((7,), request, prompt_rows=((10, 11, 12, 13, 14),))

    for chunk in ((10, 11), (12, 13), (14,)):
        runner.prefill_batch(
            WorkItem(
                kind=WorkKind.PREFILL,
                request_ids=(7,),
                row_to_request=(7,),
                token_rows=(chunk,),
            ),
            commit=True,
        )

    assert [call[1] for call in calls if call[0] == "prefill_batch_native"] == [
        (10, 11),
        (12, 13),
        (14,),
    ]
    assert [call[2:4] for call in calls if call[0] == "prefill_batch_native"] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert [call[4]["full_prompt_lengths"] for call in calls if call[0] == "prefill_batch_native"] == [
        [5],
        [5],
        [5],
    ]
    row = runner._rows[7]
    assert row.slot is not None
    assert row.slot.generated_ids == [114]
    assert row.slot.seq_position == 5
    generated = runner.decode_batch(
        WorkItem(kind=WorkKind.DECODE, request_ids=(7,), row_to_request=(7,)),
        commit=True,
    )
    assert [(token.request_id, token.token_id, token.finished) for token in generated] == [(7, 114, False)]
    assert generated[0].stream_chunk is not None
    assert generated[0].stream_chunk.text == "T114"
    assert generated[0].stream_chunk.telemetry is not None
    assert generated[0].stream_chunk.telemetry.decode_state.request_id == "7"


def test_gguf_prepare_reuses_shared_runner_for_ar(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        pass

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.model_path = str(model_path)
            self.runtime = FakeRuntime()
            self.weights = SimpleNamespace(config=SimpleNamespace())
            calls.append(("runner_init", self.model_path))

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            calls.append(
                (
                    "session_init",
                    str(model_path),
                    kwargs["shared_runner"],
                    kwargs.get("max_sequence_length"),
                )
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("session_close",))

        def prefill(self, token_ids, *, return_logits=False):
            calls.append(("prefill", tuple(token_ids), return_logits))
            return SimpleNamespace(token_id=1)

        def step(self, token_id: int, *, return_logits=False):
            calls.append(("step", int(token_id), return_logits))
            return SimpleNamespace(token_id=2)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    assert generator.prepare(max_sequence_length=1024) == 1024
    outputs = generator.generate_detailed(_request(max_tokens=2))

    assert outputs[0].text == "BC"
    runner_inits = [call for call in calls if call[0] == "runner_init"]
    assert len(runner_inits) == 1
    session_inits = [call for call in calls if call[0] == "session_init"]
    assert len(session_inits) == 1
    assert session_inits[0][2] is generator._shared_runner
    assert session_inits[0][3] == 1024


def test_gguf_generate_preserves_exact_token_prompt(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=False):
            calls.append(("prefill", tuple(token_ids), return_logits))
            return SimpleNamespace(token_id=1)

        def step(self, token_id: int, *, return_logits=False):
            calls.append(("step", int(token_id), return_logits))
            return SimpleNamespace(token_id=2)

    generator = _generator()
    generator.tokenizer.encode = lambda prompt: (_ for _ in ()).throw(
        AssertionError("raw token prompt was retokenized")
    )
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    outputs = generator.generate_detailed(
        _request(prompts=((30, 31, 32, 33),), max_tokens=2, ignore_eos=True)
    )

    assert outputs[0].generated_token_ids == (1, 2)
    assert calls[0] == ("prefill", (30, 31, 32, 33), False)
    assert _decode_state(outputs[0])["prompt_tokens"] == 4


def test_gguf_ar_c2_uses_packed_decode_when_prepared(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        pass

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.model_path = str(model_path)
            self.runtime = FakeRuntime()
            self.weights = SimpleNamespace(config=SimpleNamespace())
            calls.append(("runner_init", self.model_path))

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.slot_id = len([call for call in calls if call and call[0] == "session_init"])
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            calls.append(("session_init", self.slot_id, str(model_path), kwargs["shared_runner"]))

        def reset(self):
            calls.append(("reset", self.slot_id))
            self.position = 0

        def close(self):
            calls.append(("close", self.slot_id))

        def prefill(self, token_ids, *, return_logits=False):
            calls.append(("prefill", self.slot_id, tuple(token_ids), return_logits))
            self.position = len(token_ids)
            return SimpleNamespace(token_id=1)

        def step(self, token_id: int, *, return_logits=False):  # pragma: no cover - must not be used
            calls.append(("step", self.slot_id, int(token_id), return_logits))
            raise AssertionError("scalar step should not be used when packed AR decode works")

        def step_batch_native(self, token_ids, *, sessions, positions, return_logits=False, scatter_state=True):
            calls.append(
                (
                    "step_batch",
                    self.slot_id,
                    tuple((session.slot_id, int(token), int(position)) for session, token, position in zip(sessions, token_ids, positions, strict=True)),
                    os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"),
                    return_logits,
                    scatter_state,
                )
            )
            for session in sessions:
                session.position += 1
            return [SimpleNamespace(token_id=int(token) + 1) for token in token_ids]

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")

    generator = _generator()
    assert generator.prepare() is None
    outputs = generator.generate_detailed(_request(prompts=("long", "long2"), max_tokens=3))

    assert [output.text for output in outputs] == ["BCD", "BCD"]
    assert [call for call in calls if call[0] == "step_batch"] == [
        ("step_batch", 0, ((0, 1, 4), (1, 1, 4)), "1", False, False),
        ("step_batch", 0, ((0, 2, 5), (1, 2, 5)), "1", False, False),
    ]
    assert not [call for call in calls if call[0] == "step"]
    assert os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN") is None
    assert generator.last_batch_generation is not None
    assert generator.last_batch_generation["path"] == "gguf_packed_ar_server_decode"
    assert generator.last_batch_generation["native_caware_decode"] is True
    assert generator.last_batch_generation["serial_decode_fallback"] is False
    assert generator.last_batch_generation["native_decode_steps"] == 2
    assert all(_decode_state(output)["native_caware_decode"] is True for output in outputs)
    telemetry = [output.telemetry for output in outputs]
    assert all(item is not None for item in telemetry)
    assert [item.timing_scope for item in telemetry if item is not None] == ["batch", "batch"]
    assert len({item.batch_id for item in telemetry if item is not None}) == 1
    assert [item.group_rows for item in telemetry if item is not None] == [2, 2]
    assert [item.timing_owner for item in telemetry if item is not None] == [True, False]
    owned_timing = [item for item in telemetry if item is not None and item.timing_owner]
    assert len(owned_timing) == 1
    assert sum(float(item.timing["decode_batch_ms"]) for item in owned_timing if item.timing) == float(
        telemetry[0].timing["decode_batch_ms"]
    )
    assert generator.last_batch_generation["batch_id"] == telemetry[0].batch_id


def test_gguf_ar_packed_decode_can_be_disabled(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        pass

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.runtime = FakeRuntime()

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            calls.append(("session_init", str(model_path)))

        def reset(self):
            self.position = 0

        def close(self):
            calls.append(("close",))

        def prefill(self, token_ids, *, return_logits=False):
            calls.append(("prefill", tuple(token_ids), return_logits))
            self.position = len(token_ids)
            return SimpleNamespace(token_id=1)

        def step(self, token_id: int, *, return_logits=False):
            calls.append(("step", int(token_id), return_logits))
            self.position += 1
            return SimpleNamespace(token_id=int(token_id) + 1)

        def verify_target_blocks_batch(self, jobs):  # pragma: no cover - must not be used by default
            raise AssertionError("packed AR decode must be opt-in")

        def step_batch_native(self, token_ids, **kwargs):  # pragma: no cover - must not be used by default
            raise AssertionError("packed AR decode must be opt-in")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "0")
    monkeypatch.setenv("HIPENGINE_GGUF_AR_STREAM_DECODE", "0")

    generator = _generator()
    generator.prepare()
    outputs = generator.generate_detailed(_request(prompts=("long", "long2"), max_tokens=3))

    assert [output.text for output in outputs] == ["BCD", "BCD"]
    assert [call[0] for call in calls].count("session_init") == 1
    assert [call for call in calls if call[0] == "step"] == [
        ("step", 1, False),
        ("step", 2, False),
        ("step", 1, False),
        ("step", 2, False),
    ]
    assert generator.last_batch_generation is not None
    assert generator.last_batch_generation["path"] == "gguf_serial_greedy_decode"
    assert generator.last_batch_generation["native_caware_decode"] is False
    assert generator.last_batch_generation["serial_decode_fallback"] is True


def test_gguf_ar_stream_decode_uses_async_slot_streams(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        next_stream = 100

        def stream_create(self, *, nonblocking=True):
            stream = FakeRuntime.next_stream
            FakeRuntime.next_stream += 1
            calls.append(("stream_create", stream, nonblocking))
            return stream

        def stream_synchronize(self, stream):
            calls.append(("stream_sync", int(stream)))

        def stream_destroy(self, stream):
            calls.append(("stream_destroy", int(stream)))

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.runtime = FakeRuntime()

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.slot_id = len([call for call in calls if call and call[0] == "session_init"])
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            self._pending_token = 0
            calls.append(("session_init", self.slot_id))

        def reset(self):
            self.position = 0

        def close(self):
            calls.append(("close", self.slot_id))

        def prefill(self, token_ids, *, return_logits=False):
            self.position = len(token_ids)
            return SimpleNamespace(token_id=1)

        def step_async_top1(self, token_id: int, *, position: int, stream: int):
            calls.append(("step_async", self.slot_id, int(token_id), int(position), int(stream)))
            self.position += 1
            self._pending_token = int(token_id) + 1

        def read_top1_sample(self):
            calls.append(("read_sample", self.slot_id))
            return SimpleNamespace(token_id=self._pending_token)

        def step(self, token_id: int, *, return_logits=False):  # pragma: no cover - must not be used
            raise AssertionError("stream decode should not use scalar step")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setenv("HIPENGINE_GGUF_AR_STREAM_DECODE", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "0")

    generator = _generator()
    generator.prepare()
    outputs = generator.generate_detailed(_request(prompts=("long", "long2"), max_tokens=3))

    assert [output.text for output in outputs] == ["BCD", "BCD"]
    assert [call for call in calls if call[0] == "step_async"] == [
        ("step_async", 0, 1, 4, 100),
        ("step_async", 1, 1, 4, 101),
        ("step_async", 0, 2, 5, 100),
        ("step_async", 1, 2, 5, 101),
    ]
    assert [call for call in calls if call[0] == "stream_sync"] == [
        ("stream_sync", 100),
        ("stream_sync", 101),
        ("stream_sync", 100),
        ("stream_sync", 101),
    ]
    assert [call for call in calls if call[0] == "stream_destroy"] == [
        ("stream_destroy", 101),
        ("stream_destroy", 100),
    ]
    assert generator.last_batch_generation is not None
    assert generator.last_batch_generation["native_caware_decode"] is True
    assert generator.last_batch_generation["serial_decode_fallback"] is False


def test_gguf_ar_stream_prefill_reuses_decode_streams(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        next_stream = 300

        def stream_create(self, *, nonblocking=True):
            stream = FakeRuntime.next_stream
            FakeRuntime.next_stream += 1
            calls.append(("stream_create", stream, nonblocking))
            return stream

        def stream_synchronize(self, stream):
            calls.append(("stream_sync", int(stream)))

        def stream_destroy(self, stream):
            calls.append(("stream_destroy", int(stream)))

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.runtime = FakeRuntime()

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.slot_id = len([call for call in calls if call and call[0] == "session_init"])
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            self._pending_token = 0
            calls.append(("session_init", self.slot_id))

        def reset(self):
            self.position = 0

        def close(self):
            calls.append(("close", self.slot_id))

        def prefill(self, token_ids, *, return_logits=False):  # pragma: no cover - must not be used
            raise AssertionError("stream prefill should not use synchronous prefill")

        def prefill_async_top1(self, token_ids, *, stream: int, **kwargs):
            calls.append(("prefill_async", self.slot_id, tuple(token_ids), int(stream)))
            self.position = len(token_ids)
            self._pending_token = 1

        def step_async_top1(self, token_id: int, *, position: int, stream: int):
            calls.append(("step_async", self.slot_id, int(token_id), int(position), int(stream)))
            self.position += 1
            self._pending_token = int(token_id) + 1

        def read_top1_sample(self):
            calls.append(("read_sample", self.slot_id))
            return SimpleNamespace(token_id=self._pending_token)

        def step(self, token_id: int, *, return_logits=False):  # pragma: no cover - must not be used
            raise AssertionError("stream decode should not use scalar step")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setenv("HIPENGINE_GGUF_AR_STREAM_DECODE", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_AR_STREAM_PREFILL", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "0")

    generator = _generator()
    generator.prepare()
    outputs = generator.generate_detailed(_request(prompts=("long", "long2"), max_tokens=3))

    assert [output.text for output in outputs] == ["BCD", "BCD"]
    assert [call for call in calls if call[0] == "stream_create"] == [
        ("stream_create", 300, True),
        ("stream_create", 301, True),
    ]
    assert [call for call in calls if call[0] == "prefill_async"] == [
        ("prefill_async", 0, (10, 11, 12, 13), 300),
        ("prefill_async", 1, (20, 21, 22, 23), 301),
    ]
    assert [call for call in calls if call[0] == "step_async"] == [
        ("step_async", 0, 1, 4, 300),
        ("step_async", 1, 1, 4, 301),
        ("step_async", 0, 2, 5, 300),
        ("step_async", 1, 2, 5, 301),
    ]
    assert [call for call in calls if call[0] == "stream_destroy"] == [
        ("stream_destroy", 301),
        ("stream_destroy", 300),
    ]


def test_gguf_ar_packed_prefill_uses_batch_prompt_path(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        pass

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.runtime = FakeRuntime()

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.slot_id = len([call for call in calls if call and call[0] == "session_init"])
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            calls.append(("session_init", self.slot_id))

        def reset(self):
            self.position = 0

        def close(self):
            calls.append(("close", self.slot_id))

        def prefill(self, token_ids, *, return_logits=False):  # pragma: no cover - must not be used
            raise AssertionError("packed prompt prefill should bypass synchronous prefill")

        def prefill_async_top1(self, token_ids, *, stream: int, **kwargs):  # pragma: no cover - must not be used
            raise AssertionError("packed prompt prefill should bypass stream prefill")

        def prefill_batch_native(self, prompt_token_ids, *, sessions, return_logits=False):
            calls.append(
                (
                    "prefill_batch",
                    self.slot_id,
                    tuple(tuple(int(token) for token in prompt) for prompt in prompt_token_ids),
                    tuple(session.slot_id for session in sessions),
                    os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"),
                    return_logits,
                )
            )
            for session, prompt in zip(sessions, prompt_token_ids, strict=True):
                session.position = len(prompt)
            return [SimpleNamespace(token_id=1) for _session in sessions]

        def step_batch_native(self, token_ids, *, sessions, positions, return_logits=False, scatter_state=True):
            calls.append(
                (
                    "step_batch",
                    self.slot_id,
                    tuple((session.slot_id, int(token), int(position)) for session, token, position in zip(sessions, token_ids, positions, strict=True)),
                    os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"),
                    return_logits,
                    scatter_state,
                )
            )
            for session in sessions:
                session.position += 1
            return [SimpleNamespace(token_id=int(token) + 1) for token in token_ids]

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_PREFILL", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_AR_STREAM_PREFILL", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")

    generator = _generator()
    generator.prepare()
    outputs = generator.generate_detailed(_request(prompts=("long", "long2"), max_tokens=2))

    assert [output.text for output in outputs] == ["BC", "BC"]
    assert [call for call in calls if call[0] == "prefill_batch"] == [
        (
            "prefill_batch",
            0,
            ((10, 11, 12, 13), (20, 21, 22, 23)),
            (0, 1),
            "1",
            False,
        )
    ]
    assert [call for call in calls if call[0] == "step_batch"] == [
        ("step_batch", 0, ((0, 1, 4), (1, 1, 4)), "1", False, False)
    ]
    assert os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN") is None
    assert generator.last_batch_generation is not None
    assert generator.last_batch_generation["native_compact_prefill"] is True
    assert generator.last_batch_generation["native_caware_decode"] is True
    assert all(_decode_state(output)["native_compact_prefill"] is True for output in outputs)


def test_gguf_ar_packed_prefill_runs_one_native_c8_slab(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        pass

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.runtime = FakeRuntime()

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.slot_id = len([call for call in calls if call and call[0] == "session_init"])
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            calls.append(("session_init", self.slot_id))

        def reset(self):
            self.position = 0

        def close(self):
            calls.append(("close", self.slot_id))

        def prefill(self, token_ids, *, return_logits=False):  # pragma: no cover - must not be used
            raise AssertionError("chunked packed prompt prefill should bypass scalar prefill")

        def prefill_batch_native(self, prompt_token_ids, *, sessions, return_logits=False):
            calls.append(
                (
                    "prefill_batch",
                    self.slot_id,
                    tuple(tuple(int(token) for token in prompt) for prompt in prompt_token_ids),
                    tuple(session.slot_id for session in sessions),
                    os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"),
                    return_logits,
                )
            )
            for session, prompt in zip(sessions, prompt_token_ids, strict=True):
                session.position = len(prompt)
            return [SimpleNamespace(token_id=1) for _session in sessions]

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_PREFILL", "1")

    generator = _generator()
    generator.prepare()
    outputs = generator.generate_detailed(
        _request(prompts=("long", "long2", "long", "long2", "long", "long2", "long", "long2"), max_tokens=1)
    )

    assert [output.text for output in outputs] == ["B"] * 8
    assert [call for call in calls if call[0] == "prefill_batch"] == [
        (
            "prefill_batch",
            0,
            (
                (10, 11, 12, 13),
                (20, 21, 22, 23),
                (10, 11, 12, 13),
                (20, 21, 22, 23),
                (10, 11, 12, 13),
                (20, 21, 22, 23),
                (10, 11, 12, 13),
                (20, 21, 22, 23),
            ),
            (0, 1, 2, 3, 4, 5, 6, 7),
            "1",
            False,
        ),
    ]
    assert all("prefill_batch_ms" in output.telemetry.to_json_dict()["timing"] for output in outputs)
    assert all("prefill_batch_chunk_ms" not in output.telemetry.to_json_dict()["timing"] for output in outputs)
    assert os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN") is None


def test_gguf_ar_packed_prefill_notimplemented_falls_back(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        pass

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.runtime = FakeRuntime()

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.slot_id = len([call for call in calls if call and call[0] == "session_init"])
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            calls.append(("session_init", self.slot_id))

        def reset(self):
            self.position = 0

        def close(self):
            pass

        def prefill_batch_native(self, prompt_token_ids, *, sessions, return_logits=False):
            calls.append(("prefill_batch", self.slot_id))
            raise NotImplementedError("shape unsupported")

        def prefill(self, token_ids, *, return_logits=False):
            calls.append(("prefill", self.slot_id, tuple(token_ids), return_logits))
            self.position = len(token_ids)
            return SimpleNamespace(token_id=1)

        def step_batch_native(self, token_ids, *, sessions, positions, return_logits=False, scatter_state=True):
            calls.append(("step_batch", tuple(session.slot_id for session in sessions)))
            for session in sessions:
                session.position += 1
            return [SimpleNamespace(token_id=int(token) + 1) for token in token_ids]

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_PREFILL", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")

    generator = _generator()
    generator.prepare()
    outputs = generator.generate_detailed(_request(prompts=("long", "long2"), max_tokens=2))

    assert [output.text for output in outputs] == ["BC", "BC"]
    assert [call for call in calls if call[0] == "prefill_batch"] == [("prefill_batch", 0)]
    assert [call for call in calls if call[0] == "prefill"] == [
        ("prefill", 0, (10, 11, 12, 13), False),
        ("prefill", 1, (20, 21, 22, 23), False),
    ]


def test_gguf_prepare_request_scratch_warms_ar_packed_prefill_widths(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        pass

    class FakeFullStackRunner:
        vocab_size = 128

        def __init__(self, model_path, **kwargs):
            self.runtime = FakeRuntime()

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.slot_id = len([call for call in calls if call and call[0] == "session_init"])
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            calls.append(("session_init", self.slot_id))

        def reset(self):
            self.position = 0
            calls.append(("reset", self.slot_id))

        def close(self):  # pragma: no cover - warmup should keep reusable pooled sessions
            calls.append(("close", self.slot_id))

        def prefill_batch_native(self, prompt_token_ids, *, sessions, return_logits=False):
            calls.append(
                (
                    "prefill_batch",
                    self.slot_id,
                    len(prompt_token_ids),
                    tuple(tuple(int(token) for token in prompt) for prompt in prompt_token_ids),
                    tuple(session.slot_id for session in sessions),
                    os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"),
                    return_logits,
                )
            )
            for session, prompt in zip(sessions, prompt_token_ids, strict=True):
                session.position = len(prompt)
            return [SimpleNamespace(token_id=1) for _session in sessions]

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_PREFILL", "1")
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", raising=False)

    generator = _generator()
    result = generator.prepare_request_scratch(max_prompt_tokens=4, max_batch_size=4)

    assert result["packed_ar_prefill_widths"] == [2, 4]
    assert result["packed_ar_prefill_skipped"] is False
    assert [call for call in calls if call[0] == "prefill_batch"] == [
        (
            "prefill_batch",
            0,
            2,
            ((1, 2, 3, 4), (2, 3, 4, 5)),
            (0, 1),
            "1",
            False,
        ),
        (
            "prefill_batch",
            0,
            4,
            ((1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6), (4, 5, 6, 7)),
            (0, 1, 2, 3),
            "1",
            False,
        ),
    ]
    assert os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN") is None


def test_gguf_prepare_request_scratch_warms_mtp_hidden_seed_prefill_when_enabled(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        pass

    class FakeFullStackRunner:
        vocab_size = 128

        def __init__(self, model_path, **kwargs):
            self.runtime = FakeRuntime()

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.slot_id = len([call for call in calls if call and call[0] == "session_init"])
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            calls.append(("session_init", self.slot_id))

        def reset(self):
            self.position = 0
            calls.append(("reset", self.slot_id))

        def close(self):  # pragma: no cover - warmup should release reusable sessions to the pool
            calls.append(("close", self.slot_id))

        def prefill_batch_native(self, prompt_token_ids, *, sessions, return_logits=False, return_hidden_seeds=False):
            calls.append(
                (
                    "prefill_batch",
                    self.slot_id,
                    len(prompt_token_ids),
                    tuple(tuple(int(token) for token in prompt) for prompt in prompt_token_ids),
                    tuple(session.slot_id for session in sessions),
                    os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"),
                    return_logits,
                    return_hidden_seeds,
                )
            )
            results = []
            for session, prompt in zip(sessions, prompt_token_ids, strict=True):
                session.position = len(prompt)
                results.append(
                    SimpleNamespace(
                        token_id=1,
                        hidden_seeds=np.full((len(prompt), 2), float(session.slot_id + 1), dtype=np.float32),
                    )
            )
            return results

        def verify_target_blocks_batch(self, jobs):
            calls.append(
                (
                    "verify_batch",
                    self.slot_id,
                    tuple((job["session"].slot_id, tuple(job["input_token_ids"])) for job in jobs),
                    tuple(
                        (
                            job["bulk_attention_mode"],
                            job["use_wmma_prefill"],
                            job["capture_linear_state_rows"],
                            job["defer_linear_state_commit"],
                            job["defer_state_scatter"],
                        )
                        for job in jobs
                    ),
                    os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"),
                )
            )
            return [
                SimpleNamespace(
                    token_ids=list(job["input_token_ids"]),
                    hidden_seeds=np.ones((len(job["input_token_ids"]), 2), dtype=np.float32),
                    linear_state_rows_captured=True,
                )
                for job in jobs
            ]

    class FakeDraft:
        def __init__(self):
            self.draft_id = len([call for call in calls if call and call[0] == "draft_init"])
            calls.append(("draft_init", self.draft_id))

        def close(self):  # pragma: no cover - warmup should release reusable drafts to the pool
            calls.append(("draft_close", self.draft_id))

    assets = qwen35_gguf._GGUFMTPServingAssets(
        weights={
            "blk.40.attn_q_norm.weight": (np.zeros((2,), dtype=np.float32), 0, (2,)),
            "output.weight": (np.zeros((8, 1), dtype=np.uint8), 0, (8, 1)),
        },
        token_embd_f32=np.zeros((8, 2), dtype=np.float32),
        rope_cos=np.ones((16, 2), dtype=np.float32),
        rope_sin=np.zeros((16, 2), dtype=np.float32),
    )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setattr(
        qwen35_gguf.Qwen35GGUFBringupGenerator,
        "_load_mtp_serving_assets",
        lambda self: assets,
    )
    monkeypatch.setattr(qwen35_gguf, "_new_mtp_draft_runner", lambda assets, *, runtime: FakeDraft())
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_PREFILL", "0")
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP", "1")
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", raising=False)

    generator = _generator()
    generator.weight_index = _mtp_capable_weight_index()
    result = generator.prepare_request_scratch(max_prompt_tokens=4, max_batch_size=8)

    assert result["packed_ar_prefill_skipped"] is True
    assert result["packed_mtp_prefill_skipped"] is False
    assert result["packed_mtp_prefill_widths"] == [2, 4, 8]
    assert result["packed_mtp_prefill_prompt_lengths"] == [4]
    assert result["packed_mtp_verify_skipped"] is False
    assert result["packed_mtp_verify_widths"] == [2, 4, 8]
    assert result["packed_mtp_verify_prompt_lengths"] == [4]
    assert [call for call in calls if call[0] == "draft_init"] == [
        ("draft_init", 0),
        ("draft_init", 1),
        ("draft_init", 2),
        ("draft_init", 3),
        ("draft_init", 4),
        ("draft_init", 5),
        ("draft_init", 6),
        ("draft_init", 7),
    ]
    assert [call for call in calls if call[0] == "prefill_batch"] == [
        (
            "prefill_batch",
            0,
            2,
            ((1, 2, 3, 4), (2, 3, 4, 5)),
            (0, 1),
            "1",
            False,
            True,
        ),
        (
            "prefill_batch",
            0,
            4,
            ((1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6), (4, 5, 6, 7)),
            (0, 1, 2, 3),
            "1",
            False,
            True,
        ),
        (
            "prefill_batch",
            0,
            4,
            ((1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6), (4, 5, 6, 7)),
            (0, 1, 2, 3),
            "1",
            False,
            True,
        ),
        (
            "prefill_batch",
            4,
            4,
            ((5, 6, 7, 8), (6, 7, 8, 9), (7, 8, 9, 10), (8, 9, 10, 11)),
            (4, 5, 6, 7),
            "1",
            False,
            True,
        ),
    ]
    assert [call for call in calls if call[0] == "verify_batch"] == [
        (
            "verify_batch",
            0,
            ((0, (1, 2)), (1, (2, 3))),
            (("bulk", False, True, True, True), ("bulk", False, True, True, True)),
            "1",
        ),
        (
            "verify_batch",
            0,
            ((0, (1, 2)), (1, (2, 3)), (2, (3, 4)), (3, (4, 5))),
            (
                ("bulk", False, True, True, True),
                ("bulk", False, True, True, True),
                ("bulk", False, True, True, True),
                ("bulk", False, True, True, True),
            ),
            "1",
        ),
        (
            "verify_batch",
            0,
            ((0, (1, 2)), (1, (2, 3)), (2, (3, 4)), (3, (4, 5))),
            (
                ("bulk", False, True, True, True),
                ("bulk", False, True, True, True),
                ("bulk", False, True, True, True),
                ("bulk", False, True, True, True),
            ),
            "1",
        ),
        (
            "verify_batch",
            4,
            ((4, (5, 6)), (5, (6, 7)), (6, (7, 8)), (7, (8, 9))),
            (
                ("bulk", False, True, True, True),
                ("bulk", False, True, True, True),
                ("bulk", False, True, True, True),
                ("bulk", False, True, True, True),
            ),
            "1",
        ),
    ]
    assert os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN") is None


def test_gguf_ar_batch_decode_notimplemented_falls_back_to_step(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        pass

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.runtime = FakeRuntime()

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.slot_id = len([call for call in calls if call and call[0] == "session_init"])
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0
            calls.append(("session_init", self.slot_id))

        def reset(self):
            self.position = 0

        def close(self):
            calls.append(("close", self.slot_id))

        def prefill(self, token_ids, *, return_logits=False):
            self.position = len(token_ids)
            return SimpleNamespace(token_id=1)

        def step_batch_native(self, token_ids, *, sessions, positions, return_logits=False, scatter_state=True):
            calls.append(("step_batch", tuple(session.slot_id for session in sessions)))
            raise NotImplementedError("packed shape unavailable")

        def step(self, token_id: int, *, return_logits=False):
            calls.append(("step", self.slot_id, int(token_id), return_logits))
            self.position += 1
            return SimpleNamespace(token_id=int(token_id) + 1)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")

    generator = _generator()
    generator.prepare()
    outputs = generator.generate_detailed(_request(prompts=("long", "long2"), max_tokens=3))

    assert [output.text for output in outputs] == ["BCD", "BCD"]
    assert [call for call in calls if call[0] == "step_batch"] == [
        ("step_batch", (0, 1)),
        ("step_batch", (0, 1)),
    ]
    assert [call for call in calls if call[0] == "step"] == [
        ("step", 0, 1, False),
        ("step", 1, 1, False),
        ("step", 0, 2, False),
        ("step", 1, 2, False),
    ]
    assert generator.last_batch_generation is not None
    assert generator.last_batch_generation["path"] == "gguf_packed_ar_server_decode"
    assert generator.last_batch_generation["native_caware_decode"] is False
    assert generator.last_batch_generation["serial_decode_fallback"] is True


def test_gguf_ar_batch_decode_fallback_advances_each_slot_once_per_cycle(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.position = 4

        def step_batch_native(self, token_ids, **kwargs):
            calls.append(("step_batch", self.slot_id, tuple(int(token) for token in token_ids)))
            raise NotImplementedError("packed shape unavailable")

        def step(self, token_id: int, *, return_logits=False):
            calls.append(("step", self.slot_id, int(token_id), return_logits))
            self.position += 1
            return SimpleNamespace(token_id=int(token_id) + 1)

    slots = [
        qwen35_gguf._GGUFARServingSlot(
            request_id=slot_id,
            prompt_ids=[10, 11, 12, 13],
            session=FakeSession(slot_id),
            prev_token=1,
            seq_position=4,
            generated_ids=[1],
        )
        for slot_id in range(2)
    ]

    generator = _generator()
    monkeypatch.setenv("HIPENGINE_GGUF_AR_STREAM_DECODE", "0")

    assert generator._try_step_ar_serving_slots_batch(
        slots,
        _request(prompts=("long", "long2"), max_tokens=4),
    ) is True

    assert calls == [
        ("step_batch", 0, (1, 1)),
        ("step", 0, 1, False),
        ("step", 1, 1, False),
    ]
    assert [slot.generated_ids for slot in slots] == [[1, 2], [1, 2]]
    assert [slot.serial_decode_steps for slot in slots] == [1, 1]


def test_gguf_ar_batch_decode_runs_one_native_c8_group(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.position = 4

        def step_batch_native(self, token_ids, *, sessions, positions, return_logits=False, scatter_state=True):
            calls.append(("step_batch", self.slot_id, tuple(session.slot_id for session in sessions), scatter_state))
            for session in sessions:
                session.position += 1
            return [SimpleNamespace(token_id=2) for _token in token_ids]

        def step(self, token_id: int, *, return_logits=False):  # pragma: no cover - must not be used
            raise AssertionError("chunked packed AR should not use scalar step")

    slots = [
        qwen35_gguf._GGUFARServingSlot(
            request_id=slot_id,
            prompt_ids=[10, 11, 12, 13],
            session=FakeSession(slot_id),
            prev_token=1,
            seq_position=4,
            generated_ids=[1],
        )
        for slot_id in range(8)
    ]

    generator = _generator()
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_AR_STREAM_DECODE", "0")
    generator._run_ar_serving_slots(slots, _request(prompts=("long",) * 8, max_tokens=2))

    assert [slot.generated_ids for slot in slots] == [[1, 2]] * 8
    assert [call for call in calls if call[0] == "step_batch"] == [
        ("step_batch", 0, (0, 1, 2, 3, 4, 5, 6, 7), False),
    ]
    assert all(slot.native_decode_steps == 1 for slot in slots)
    assert all("decode_batch_ms" in slot.timing for slot in slots)


def test_gguf_ar_batch_decode_streams_chunks_above_native_c8(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeRuntime:
        next_stream = 500

        def stream_create(self, *, nonblocking=True):
            stream = FakeRuntime.next_stream
            FakeRuntime.next_stream += 1
            calls.append(("stream_create", stream, bool(nonblocking)))
            return stream

        def stream_synchronize(self, stream):  # pragma: no cover - fake batch owns sync contract
            calls.append(("stream_sync", int(stream)))

        def stream_destroy(self, stream):
            calls.append(("stream_destroy", int(stream)))

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.position = 4
            self.runtime = FakeRuntime()

        def step_batch_native(self, token_ids, *, sessions, positions, return_logits=False, scatter_state=True, stream=0):
            calls.append(
                (
                    "step_batch",
                    self.slot_id,
                    int(stream),
                    tuple(session.slot_id for session in sessions),
                    tuple(int(position) for position in positions),
                    scatter_state,
                    os.environ.get("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"),
                )
            )
            for session in sessions:
                session.position += 1
            return [SimpleNamespace(token_id=2) for _token in token_ids]

        def step(self, token_id: int, *, return_logits=False):  # pragma: no cover - must not be used
            raise AssertionError("streamed packed AR should not use scalar step")

        def close(self):
            calls.append(("close", self.slot_id))

    slots = [
        qwen35_gguf._GGUFARServingSlot(
            request_id=slot_id,
            prompt_ids=[10, 11, 12, 13],
            session=FakeSession(slot_id),
            prev_token=1,
            seq_position=4,
            generated_ids=[1],
        )
        for slot_id in range(10)
    ]

    generator = _generator()
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_AR_STREAM_DECODE", "1")
    generator._run_ar_serving_slots(slots, _request(prompts=("long",) * 10, max_tokens=2))

    assert [slot.generated_ids for slot in slots] == [[1, 2]] * 10
    assert [call for call in calls if call[0] == "stream_create"] == [
        ("stream_create", 500, True),
        ("stream_create", 501, True),
    ]
    assert sorted(call for call in calls if call[0] == "step_batch") == [
        ("step_batch", 0, 500, (0, 1, 2, 3, 4, 5, 6, 7), (4, 4, 4, 4, 4, 4, 4, 4), False, "1"),
        ("step_batch", 8, 501, (8, 9), (4, 4), False, "1"),
    ]
    assert all(slot.native_decode_steps == 1 for slot in slots)
    assert all("decode_batch_ms" in slot.timing for slot in slots)
    assert all("decode_stream_chunks_ms" in slot.timing for slot in slots)

    generator._close_ar_serving_slots(slots, reuse=False)

    assert [call for call in calls if call[0] == "stream_destroy"] == [
        ("stream_destroy", 501),
        ("stream_destroy", 500),
    ]


def test_gguf_ar_batch_decode_flushes_before_singleton_fallback(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.position = 4

        def step_batch_native(self, token_ids, *, sessions, positions, return_logits=False, scatter_state=True):
            calls.append(("step_batch", self.slot_id, tuple(session.slot_id for session in sessions), scatter_state))
            for session in sessions:
                session.position += 1
            return [
                SimpleNamespace(token_id=99 if session.slot_id == 0 else 2)
                for session in sessions
            ]

        def flush_packed_decode_state(self):
            calls.append(("flush", self.slot_id))
            return True

        def step(self, token_id: int, *, return_logits=False):
            calls.append(("step", self.slot_id, int(token_id), return_logits))
            self.position += 1
            return SimpleNamespace(token_id=int(token_id) + 1)

    slots = [
        qwen35_gguf._GGUFARServingSlot(
            request_id=slot_id,
            prompt_ids=[10, 11, 12, 13],
            session=FakeSession(slot_id),
            prev_token=1,
            seq_position=4,
            generated_ids=[1],
        )
        for slot_id in range(2)
    ]

    generator = _generator()
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")
    monkeypatch.delenv("HIPENGINE_GGUF_AR_STREAM_DECODE", raising=False)
    generator._run_ar_serving_slots(slots, _request(prompts=("long", "long2"), max_tokens=3))

    assert [slot.generated_ids for slot in slots] == [[1, 99], [1, 2, 3]]
    assert calls == [
        ("step_batch", 0, (0, 1), False),
        ("flush", 0),
        ("step", 1, 2, False),
    ]


def test_gguf_sampled_thinking_budget_suppresses_tokenizer_eos(monkeypatch) -> None:
    logits = np.full((1, 100), -10.0, dtype=np.float32)
    logits[0, 2] = 1.0
    logits[0, 99] = 5.0

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(token_id=99, logits=logits)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()

    outputs = generator.generate_detailed(
        _request(
            max_tokens=1,
            thinking_close_token_ids=(2,),
            thinking_hard_token_cap=5,
        )
    )

    assert outputs[0].text == "C"
    assert outputs[0].finish_details is not None
    assert outputs[0].finish_details.reason == "length"


def test_gguf_sampled_request_forced_token_overrides_logits(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(token_id=1, logits=np.array([[0.0, 10.0, 1.0]], dtype=np.float32))

        def step(self, token_id: int, *, return_logits=True):  # pragma: no cover - max_tokens=1
            raise AssertionError("forced-token fixture should finish after prefill")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()

    outputs = generator.generate_detailed(
        _request(
            max_tokens=1,
            forced_tokens_pending=(2,),
            forced_token_reason="tool_choice_required",
        )
    )

    assert outputs[0].text == "C"
    assert outputs[0].finish_details is not None
    assert outputs[0].finish_details.to_json_dict()["sampler_mode"] == "processed_argmax"
    decode_state = _decode_state(outputs[0])
    assert decode_state["active_processors"] == ["forced_tokens_pending"]
    assert decode_state["forced_token_id"] == 2
    assert decode_state["forced_token_reason"] == "tool_choice_required"
    assert decode_state["forced_tokens_remaining"] == 0


def test_gguf_json_object_close_forcing_goes_through_decode(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 5] = 10.0
            return SimpleNamespace(token_id=5, logits=logits)

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 6] = 10.0
            logits[0, 4] = 1.0
            return SimpleNamespace(token_id=6, logits=logits)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()

    outputs = generator.generate_detailed(_request(max_tokens=2, json_object_close_forcing=True))

    assert outputs[0].text == "{}"
    assert ("step", 5, True) in calls
    decode_state = _decode_state(outputs[0])
    assert decode_state["forced_token_id"] == 4
    assert decode_state["forced_token_reason"] == "json_object_close_forcing"
    assert decode_state["forced_tokens_remaining"] == 0
    assert "json_object_close_forcing" in decode_state["active_processors"]
    assert "json_object_close_forcing" in decode_state["sampler_fast_path_blockers"]


def test_gguf_sampled_post_thinking_forced_tokens_queue_after_close(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 2] = 5.0
            return SimpleNamespace(token_id=2, logits=logits)

        def step(self, token_id: int, *, return_logits=True):
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 1] = 10.0
            return SimpleNamespace(token_id=1, logits=logits)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()

    outputs = generator.generate_detailed(
        _request(
            max_tokens=3,
            thinking_close_token_ids=(2,),
            thinking_hard_token_cap=8,
            post_thinking_forced_tokens_pending=(3, 16),
            post_thinking_forced_token_reason="tool_choice_required",
        )
    )

    assert outputs[0].text == "CDQ"
    assert outputs[0].finish_details is not None
    assert outputs[0].finish_details.to_json_dict()["phase"] == "answer"
    assert _decode_state(outputs[0])["active_processors"] == [
        "thinking_budget",
        "post_thinking_forced_tokens_pending",
    ]


def test_gguf_sampled_force_sequence_completion_repairs_tool_close(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 3] = 5.0
            return SimpleNamespace(token_id=3, logits=logits)

        def step(self, token_id: int, *, return_logits=True):
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 1] = 10.0
            return SimpleNamespace(token_id=1, logits=logits)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()

    outputs = generator.generate_detailed(
        _request(
            max_tokens=2,
            force_sequence_completion_token_sequences=((3, 16),),
            force_sequence_completion_reason="tool_call_close_repair",
        )
    )

    assert outputs[0].text == "DQ"
    assert outputs[0].finish_details is not None
    assert outputs[0].finish_details.to_json_dict()["sampler_mode"] == "processed_argmax"
    decode_state = _decode_state(outputs[0])
    assert decode_state["active_processors"] == ["force_sequence_completion_token_sequences"]
    assert decode_state["force_sequence_completion_token_sequences"] == [[3, 16]]
    assert decode_state["force_sequence_completion_reason"] == "tool_call_close_repair"


def test_gguf_telemetry_reports_post_thinking_forced_queue_before_close() -> None:
    request = _request(
        thinking_close_token_ids=(2,),
        thinking_hard_token_cap=8,
        post_thinking_forced_tokens_pending=(3, 16),
        post_thinking_forced_token_reason="tool_choice_required",
    )
    state = qwen35_gguf._gguf_row_sampling_state(request, [10, 11], row_index=0)
    state.observe(1)

    telemetry = qwen35_gguf._gguf_telemetry(
        [10, 11],
        [1],
        request,
        row_index=0,
        sampling_state=state,
    )

    decode_state = telemetry.to_json_dict()["decode_state"]
    assert decode_state["phase"] == "think"
    assert decode_state["reasoning_tokens"] == 1
    assert decode_state["post_thinking_forced_tokens_pending"] == [3, 16]
    assert decode_state["post_thinking_forced_token_reason"] == "tool_choice_required"


def test_gguf_greedy_equivalent_request_uses_eager_step(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            calls.append(("init", str(model_path), dict(kwargs)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=1,
                logits=np.array([[0.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(token_id=16, logits=np.array([[0.0, 1.0]], dtype=np.float32))

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    out = generator.generate(_request(top_p=0.5, top_k=2, min_p=0.5))

    assert out == ["BQ"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "length",
        "length_limit": 2,
        "sampler_mode": "greedy_fast",
    }
    assert _decode_state(generator.last_generation_outputs[0]) == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "greedy_fast",
    }
    assert calls[0] == (
        "init",
        "/tmp/fake.gguf",
        {
            "backend": "hip_gfx1100",
            "use_wmma_prefill": True,
            "use_gemv_decode": True,
        },
    )
    assert ("prefill", (10, 11), False) in calls
    assert ("step", 1, False) in calls
    assert not any(call[0] == "capture_decode_graph" for call in calls)


def test_gguf_long_greedy_request_uses_backend_graph_capability(monkeypatch) -> None:
    calls = []

    class FakeGraph:
        def __init__(self):
            self.token = 1

        def replay(self, steps):
            calls.append(("graph_replay", int(steps)))
            self.token += int(steps)

        def read_sample(self, *, return_logits=True):
            calls.append(("graph_read", bool(return_logits)))
            return SimpleNamespace(token_id=self.token)

        def close(self):
            calls.append(("graph_close",))

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.position = 2

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def decode_graph_min_replay_steps(self):
            return 2

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(token_id=1)

        def capture_decode_graph(self, **kwargs):
            calls.append(("capture_decode_graph", dict(kwargs)))
            return FakeGraph()

        def step(self, token_id, *, return_logits=True):  # pragma: no cover - graph must own the long route
            raise AssertionError("long greedy route fell back to eager")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_GRAPH", raising=False)

    generator = _generator()
    out = generator.generate(_request(max_tokens=4))

    assert out == ["BCD}"]
    assert calls[0] == ("prefill", (10, 11), False)
    assert calls[1] == (
        "capture_decode_graph",
        {
            "position": 2,
            "steps_per_replay": 1,
            "max_replay_steps": 3,
            "attention_max_context_len": 5,
        },
    )
    assert calls.count(("graph_replay", 1)) == 3
    assert calls.count(("graph_read", False)) == 3
    assert calls[-1] == ("graph_close",)


@pytest.mark.parametrize(
    ("native_requested", "expected_fallback"),
    [
        (False, "host_sampling_required"),
        (True, "native_gpu_unsupported_request"),
    ],
)
def test_gguf_non_greedy_request_uses_host_logits_sampler(
    monkeypatch,
    native_requested: bool,
    expected_fallback: str,
) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            calls.append(("init", str(model_path)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 1.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    if native_requested:
        monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    else:
        monkeypatch.delenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", raising=False)

    generator = _generator()
    out = generator.generate(_request(temperature=0.7, top_k=1, seed=5))

    assert out == ["BC"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "length",
        "length_limit": 2,
        "sampler_mode": "host_logits_sample",
    }
    assert _decode_state(generator.last_generation_outputs[0]) == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_fast_path_blockers": ["temperature"],
        "sampler_fallback_reason": expected_fallback,
        "sampler_mode": "host_logits_sample",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 12,
    }
    assert ("prefill", (10, 11), True) in calls
    assert ("step", 1, True) in calls
    assert not any(call[0] == "capture_decode_graph" for call in calls)


def test_gguf_generate_detailed_records_scheduler_token_chunks_for_serial_rows(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 1.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.delenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", raising=False)

    generator = _generator()
    outputs = generator.generate_detailed(
        _request(
            prompts=("first", "second"),
            temperature=0.7,
            top_k=1,
            logprobs=True,
            top_logprobs=1,
            seed=5,
        )
    )

    assert [output.text for output in outputs] == ["BC", "BC"]
    batch = generator.last_batch_generation
    assert batch is not None
    assert {key: value for key, value in batch.items() if key != "scheduler_token_chunks"} == {
        "path": "gguf_serial_host_sampler_decode",
        "batch_size": 2,
        "request_ids": [0, 1],
        "prompt_lengths": [2, 1],
        "decode_steps": 2,
        "native_decode_steps": 0,
        "serial_decode_fallback": True,
        "native_compact_prefill": False,
        "native_caware_decode": False,
        "native_sampler_rows": False,
        "throughput_claim_eligible": False,
        "sampler_plan_metadata": [
            {
                "active_processors": [],
                "sampler_fast_path_blockers": ["temperature", "logprobs", "top_logprobs"],
                "native_gpu_available": False,
                "sampler_fallback_reason": "host_sampling_required",
                "sampler_mode": "host_logits_sample",
            },
            {
                "active_processors": [],
                "sampler_fast_path_blockers": ["temperature", "logprobs", "top_logprobs"],
                "native_gpu_available": False,
                "sampler_fallback_reason": "host_sampling_required",
                "sampler_mode": "host_logits_sample",
            },
        ],
    }
    chunks = batch["scheduler_token_chunks"]
    assert [
        (chunk["request_id"], chunk["token_index"], chunk["token_id"], chunk["chunk"]["text"])
        for chunk in chunks
    ] == [
        (0, 0, 1, "B"),
        (0, 1, 2, "C"),
        (1, 0, 1, "B"),
        (1, 1, 2, "C"),
    ]
    assert [chunk["finished"] for chunk in chunks] == [False, True, False, True]
    assert chunks[0]["chunk"]["token_logprobs"] == [
        {
            "token_id": 1,
            "token_text": "B",
            "logprob": 0.0,
            "top_logprobs": [{"token_id": 1, "token_text": "B", "logprob": 0.0}],
        }
    ]
    assert chunks[1]["chunk"]["finish_details"] == {
        "reason": "length",
        "length_limit": 2,
        "sampler_mode": "host_logits_sample",
    }
    assert chunks[2]["chunk"]["telemetry"]["decode_state"] == {
        "request_id": "1",
        "row_index": 1,
        "step_index": 1,
        "prompt_tokens": 1,
        "generated_tokens": 1,
        "phase": "answer",
        "continuation_eligible": False,
        "sampler_fast_path_blockers": ["temperature", "logprobs", "top_logprobs"],
        "sampler_fallback_reason": "host_sampling_required",
        "sampler_mode": "host_logits_sample",
        "execution_path": "gguf_serial_host_sampler_decode",
        "native_compact_prefill": False,
        "native_caware_decode": False,
        "serial_decode_fallback": True,
        "native_sampler_rows": False,
    }
    assert calls == [
        ("enter",),
        ("prefill", (10, 11), True),
        ("step", 1, True),
        ("prefill", (20,), True),
        ("step", 1, True),
        ("exit", True),
    ]


def test_gguf_stream_detailed_emits_live_greedy_telemetry(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            calls.append(("init", str(model_path), dict(kwargs)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(token_id=1)

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(token_id=2)

        def capture_decode_graph(self, **kwargs):  # pragma: no cover - streaming should stay live
            raise AssertionError("streaming should emit live one-token steps")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    chunks = list(generator.stream_detailed(_request(max_tokens=2)))

    assert [chunk.text for chunk in chunks] == ["B", "C"]
    assert all(isinstance(chunk, GenerationStreamChunk) for chunk in chunks)
    assert [_decode_state(chunk) for chunk in chunks] == [
        {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 2,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_mode": "greedy_fast",
        },
        {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 2,
            "generated_tokens": 2,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_mode": "greedy_fast",
        },
    ]
    assert [None if chunk.finish_details is None else chunk.finish_details.to_json_dict() for chunk in chunks] == [
        None,
        {"reason": "length", "length_limit": 2, "sampler_mode": "greedy_fast"},
    ]
    assert calls == [
        (
            "init",
            "/tmp/fake.gguf",
            {
                "backend": "hip_gfx1100",
                "use_wmma_prefill": True,
                "use_gemv_decode": True,
            },
        ),
        ("enter",),
        ("prefill", (10, 11), False),
        ("step", 1, False),
        ("exit", True),
    ]


def test_gguf_stream_text_wrapper_preserves_plain_chunks(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(token_id=1)

        def step(self, token_id: int, *, return_logits=True):
            return SimpleNamespace(token_id=2)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()

    assert list(generator.stream(_request(max_tokens=2))) == ["B", "C"]


@pytest.mark.parametrize(
    ("native_requested", "expected_fallback"),
    [
        (False, "host_sampling_required"),
        (True, "native_gpu_unsupported_request"),
    ],
)
def test_gguf_stream_detailed_emits_live_sampled_telemetry(
    monkeypatch,
    native_requested: bool,
    expected_fallback: str,
) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 1.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    if native_requested:
        monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    else:
        monkeypatch.delenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", raising=False)

    generator = _generator()
    chunks = list(generator.stream_detailed(_request(temperature=0.7, top_k=1, seed=5)))

    assert [chunk.text for chunk in chunks] == ["B", "C"]
    assert [_decode_state(chunk) for chunk in chunks] == [
        {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 2,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": expected_fallback,
            "sampler_mode": "host_logits_sample",
            "full_vocab_logits_d2h": True,
            "logits_d2h_bytes": 12,
        },
        {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 2,
            "generated_tokens": 2,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": expected_fallback,
            "sampler_mode": "host_logits_sample",
            "full_vocab_logits_d2h": True,
            "logits_d2h_bytes": 12,
        },
    ]
    assert [None if chunk.finish_details is None else chunk.finish_details.to_json_dict() for chunk in chunks] == [
        None,
        {"reason": "length", "length_limit": 2, "sampler_mode": "host_logits_sample"},
    ]
    assert calls == [
        ("enter",),
        ("prefill", (10, 11), True),
        ("step", 1, True),
        ("exit", True),
    ]


def test_gguf_stream_detailed_emits_live_sampled_logprobs(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 1.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    chunks = list(
        generator.stream_detailed(
            _request(temperature=0.7, top_k=1, logprobs=True, top_logprobs=1, seed=5)
        )
    )

    assert [chunk.text for chunk in chunks] == ["B", "C"]
    assert chunks[0].token_logprobs == (
        TokenLogprob(
            token_id=1,
            token_text="B",
            logprob=0.0,
            top_logprobs=((1, "B", 0.0),),
        ),
    )
    assert chunks[1].token_logprobs == (
        TokenLogprob(
            token_id=2,
            token_text="C",
            logprob=0.0,
            top_logprobs=((2, "C", 0.0),),
        ),
    )
    assert [None if chunk.finish_details is None else chunk.finish_details.to_json_dict() for chunk in chunks] == [
        None,
        {"reason": "length", "length_limit": 2, "sampler_mode": "host_logits_sample"},
    ]


def test_gguf_stream_detailed_reports_thinking_budget_pressure(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):  # pragma: no cover - max_tokens=1
            raise AssertionError("hard-close stream fixture should finish after prefill")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    chunks = list(
        generator.stream_detailed(
            _request(
                max_tokens=1,
                thinking_close_token_ids=(2,),
                thinking_hard_token_cap=0,
            )
        )
    )

    assert [chunk.text for chunk in chunks] == ["C"]
    assert _decode_state(chunks[0]) == {
        "row_index": 0,
        "step_index": 1,
        "prompt_tokens": 2,
        "generated_tokens": 1,
        "phase": "answer",
        "continuation_eligible": False,
        "reasoning_tokens": 1,
        "active_processors": ["thinking_budget"],
        "sampler_fast_path_blockers": ["thinking_budget"],
        "sampler_fallback_reason": "processed_logits_required",
        "forced_token_id": 2,
        "forced_token_reason": "thinking_hard_close",
        "forced_tokens_remaining": 0,
        "budget_pressure": "hard_close",
        "sampler_mode": "processed_argmax",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 12,
    }


def test_gguf_greedy_host_decode_checks_deadline_after_step(monkeypatch) -> None:
    calls = []

    def check_deadline(value) -> None:
        calls.append(("deadline", None if value is None else getattr(value, "deadline_at", value)))
        if ("step", 1, False) in calls:
            raise GenerationDeadlineExceeded(deadline_at=getattr(value, "deadline_at", value))

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            calls.append(("init", str(model_path)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(token_id=1, logits=np.array([[0.0, 1.0]], dtype=np.float32))

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(token_id=2, logits=np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

        def capture_decode_graph(self, **kwargs):  # pragma: no cover - host decode forced
            raise AssertionError("host-routed decode should not capture graph")

    monkeypatch.setattr(qwen35_gguf, "raise_if_generation_deadline_expired", check_deadline)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    with pytest.raises(GenerationDeadlineExceeded):
        generator.generate(_request(max_tokens=2, deadline_at=123.0))

    assert ("prefill", (10, 11), False) in calls
    assert ("step", 1, False) in calls
    assert ("exit", False) in calls


def test_gguf_greedy_host_decode_checks_cancellation_after_step(monkeypatch) -> None:
    calls = []
    token = GenerationCancellationToken()

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            calls.append(("init", str(model_path)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(token_id=1, logits=np.array([[0.0, 1.0]], dtype=np.float32))

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            token.cancel()
            return SimpleNamespace(token_id=2, logits=np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

        def capture_decode_graph(self, **kwargs):  # pragma: no cover - host decode forced
            raise AssertionError("host-routed decode should not capture graph")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    with pytest.raises(GenerationCancelled) as raised:
        generator.generate(_request(max_tokens=2, cancellation_token=token))

    assert raised.value.finish_details.to_json_dict() == {"reason": "cancelled", "cancelled": True}
    assert ("prefill", (10, 11), False) in calls
    assert ("step", 1, False) in calls
    assert ("exit", False) in calls


def test_gguf_finish_details_report_forced_thinking_close(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):  # pragma: no cover - max_tokens=1
            raise AssertionError("hard-close fixture should finish after prefill")

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    out = generator.generate(
        _request(
            max_tokens=1,
            thinking_close_token_ids=(2,),
            thinking_hard_token_cap=0,
        )
    )

    assert out == ["C"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "thinking_budget_exhausted",
        "length_limit": 1,
        "forced_close": True,
        "reasoning_tokens": 1,
        "budget_pressure": "hard_close",
        "sampler_mode": "processed_argmax",
        "phase": "answer",
    }
    decode_state = _decode_state(generator.last_generation_outputs[0])
    assert decode_state["phase"] == "answer"
    assert decode_state["reasoning_tokens"] == 1
    assert decode_state["forced_token_id"] == 2
    assert decode_state["forced_token_reason"] == "thinking_hard_close"
    assert decode_state["forced_tokens_remaining"] == 0
    assert decode_state["budget_pressure"] == "hard_close"


def test_gguf_host_sampler_stops_on_stop_token_id(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(
                token_id=2,
                logits=np.array([[0.0, 0.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    out = generator.generate(_request(temperature=0.7, top_k=1, stop_token_ids=(1,)))

    assert out == ["B"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "stop",
        "stop_sequence": [1],
        "sampler_mode": "host_logits_sample",
    }
    assert not any(call[0] == "step" for call in calls)


def test_gguf_host_sampler_stops_on_multi_token_stop_sequence(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 5.0, 1.0]], dtype=np.float32),
            )

        def step(self, token_id: int, *, return_logits=True):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(
                token_id=0,
                logits=np.array([[0.0, 0.0, 5.0]], dtype=np.float32),
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    generator = _generator()
    out = generator.generate(
        _request(temperature=0.7, top_k=1, max_tokens=3, stop_token_sequences=((1, 2),))
    )

    assert out == ["BC"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "stop",
        "stop_sequence": [1, 2],
        "sampler_mode": "host_logits_sample",
    }
    assert _decode_state(generator.last_generation_outputs[0]) == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "stop_suffix_state": {"matched_sequence": [1, 2]},
        "active_processors": ["stop_token_sequences"],
        "sampler_fast_path_blockers": ["temperature", "stop_token_sequences"],
        "sampler_fallback_reason": "host_sampling_required",
        "sampler_mode": "host_logits_sample",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 12,
    }
    assert len([call for call in calls if call[0] == "step"]) == 1
