from __future__ import annotations

import os
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.generation.qwen35_gguf as qwen35_gguf
from hipengine.dispatch import SlotMove, WorkItem, WorkKind
from hipengine.dispatch.d2_resolver import CostTable, PhysicalWidthCost, d2_partition
from hipengine.generation import (
    EngineLoopConfig,
    GenerationAdmissionRejected,
    GenerationCancellationToken,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationRequest,
    GenerationStreamChunk,
    PreparedPromptInput,
    SubmitPollTextGenerator,
    TokenLogprob,
)
from hipengine.generation.batch_scheduler import GeneratedTokenEvent
from hipengine.generation.sampling import SampleResult, SamplingMode, ToolCallConstraintSpec
from hipengine.kvcache import DeviceChunkedKVPool
from hipengine.models.qwen35 import Qwen35GGUFModel


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, prompt: str) -> list[int]:
        return {
            "first": [10, 11],
            "second": [20],
            "third": [30],
            "fourth": [40],
            "long": [10, 11, 12, 13],
            "long2": [20, 21, 22, 23],
            "{": [5],
            "}": [4],
        }[prompt]

    def decode(self, ids, *, skip_special: bool = False) -> str:
        table = {1: "B", 2: "C", 3: "D", 4: "}", 5: "{", 6: "X", 16: "Q", 99: "<eos>", 114: "T114"}
        return "".join(
            table[int(token)]
            for token in ids
            if not (skip_special and int(token) == self.eos_token_id)
        )


def _generator() -> qwen35_gguf.Qwen35GGUFBringupGenerator:
    generator = qwen35_gguf.Qwen35GGUFBringupGenerator.__new__(
        qwen35_gguf.Qwen35GGUFBringupGenerator
    )
    generator.model_path = "/tmp/fake.gguf"
    generator.weight_index = SimpleNamespace()
    generator.model_plugin = SimpleNamespace()
    generator.backend = "hip_gfx1100"
    generator.native_batch_decode = False
    generator.tokenizer = _FakeTokenizer()
    generator._mtp_serving_assets = None
    generator._mtp_serving_lock = threading.Lock()
    return generator


def test_qwen_gguf_generator_detokenizes_through_model_tokenizer() -> None:
    generator = _generator()

    assert generator.detokenize((1, 2, 99), skip_special=False) == "BC<eos>"
    assert generator.detokenize((1, 2, 99), skip_special=True) == "BC"


def test_speculative_stream_event_decorator_attaches_tokenizer_text() -> None:
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = SimpleNamespace(tokenizer=_FakeTokenizer())
    events = (
        GeneratedTokenEvent(
            request_id=0,
            token_id=1,
            finished=False,
            stream_chunk=GenerationStreamChunk(text=""),
        ),
        GeneratedTokenEvent(
            request_id=0,
            token_id=2,
            finished=False,
            stream_chunk=GenerationStreamChunk(text=""),
        ),
        GeneratedTokenEvent(
            request_id=0,
            token_id=99,
            finished=False,
            stream_chunk=GenerationStreamChunk(text=""),
        ),
        GeneratedTokenEvent(
            request_id=0,
            token_id=3,
            finished=True,
            stream_chunk=GenerationStreamChunk(
                text="",
                generated_token_ids=(1, 2, 99, 3),
            ),
        ),
    )

    decorated = runner.decorate_speculative_stream_events(events)

    assert [event.stream_chunk.text for event in decorated] == ["B", "C", "", ""]
    assert decorated[3].stream_chunk.generated_token_ids == (1, 2, 99, 3)


def test_submit_poll_adapter_explicitly_delegates_model_owned_mtp_route() -> None:
    sentinel = [SimpleNamespace(text="mtp")]
    inner = SimpleNamespace(
        supports_speculative_mtp=True,
        generate_speculative_mtp_detailed=lambda request: sentinel,
    )
    adapter = SubmitPollTextGenerator(inner)

    assert "supports_speculative_mtp" in SubmitPollTextGenerator.__dict__
    assert "generate_speculative_mtp_detailed" in SubmitPollTextGenerator.__dict__
    assert adapter.supports_speculative_mtp is True
    assert adapter.generate_speculative_mtp_detailed(_request()) == sentinel


def test_qwen36_dense_inventory_advertises_only_present_public_mtp() -> None:
    model = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
    if not model.exists():
        pytest.skip(f"local GGUF fixture not found: {model}")
    info = qwen35_gguf.GGUFReader(model).info
    config = qwen35_gguf.qwen35_gguf_config_from_metadata(info)

    assert config.architecture == "qwen35"
    if not config.ignored_block_ids:
        nextn_prefix = f"blk.{config.declared_block_count}.nextn."
        assert not any(tensor.name.startswith(nextn_prefix) for tensor in info.tensors)
        assert qwen35_gguf._gguf_info_has_mtp_tensors(info) is False
        return

    resolved_config, block_id, required = qwen35_gguf._gguf_mtp_required_tensor_names(info)
    assert resolved_config == config
    assert config.ignored_block_ids == (block_id,)
    assert f"blk.{block_id}.nextn.eh_proj.weight" in required
    assert f"blk.{block_id}.ffn_gate.weight" in required
    assert qwen35_gguf._gguf_info_has_mtp_tensors(info) is True


def test_dense_public_mtp_route_uses_transactional_provider_and_recycles_owner(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_MTP_VERIFY_MODE", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET", raising=False)
    calls: list[tuple] = []
    config = SimpleNamespace(
        ignored_block_ids=(64,),
        is_moe=False,
        architecture="qwen35",
    )

    class FakeDecoder:
        def __init__(self, target, provider, **kwargs):
            calls.append(("decoder_init", target, provider, kwargs))

        def generate(self, prompt_ids, **kwargs):
            calls.append(("generate", tuple(prompt_ids), kwargs))
            return SimpleNamespace(
                token_ids=(1, 2, 3),
                cycle_records=(
                    {"draft_tokens": [2, 4], "accepted": 1},
                ),
                prefill_seconds=0.001,
                decode_seconds=0.002,
                proposal_seconds=0.0005,
                verify_seconds=0.00075,
            )

        def close(self):
            calls.append(("decoder_close",))

    provider = SimpleNamespace(
        release_request=lambda request_id: calls.append(("provider_release", int(request_id))),
        close=lambda: calls.append(("provider_close",)),
    )
    @contextmanager
    def session_scope(**kwargs):
        yield SimpleNamespace(
            require_cached_build=True,
            target_layout=SimpleNamespace(max_sequence_length=256),
        ), False

    generator = _generator()
    generator._prepared_max_sequence_length = 64
    generator._shared_runner = SimpleNamespace()
    generator._resident_session_scope = session_scope
    def acquire_provider(*args, **kwargs):
        calls.append(("provider_acquire", kwargs))
        return provider, (7, "dense_nextn", int(kwargs["max_positions"])), False

    generator._acquire_dense_mtp_draft_provider = acquire_provider
    generator._release_mtp_draft_runner = lambda key, draft: calls.append(
        ("release", key, draft)
    )
    monkeypatch.setattr(
        "hipengine.runtime.qwen35_gguf_mtp.Qwen35GGUFMTPDecodeSession",
        FakeDecoder,
    )

    outputs = generator._generate_dense_speculative_mtp_detailed(
        _request(prompts=("long",), max_tokens=3),
        config=config,
    )

    assert outputs[0].generated_token_ids == (1, 2, 3)
    generate_call = next(call for call in calls if call[0] == "generate")
    assert {key: generate_call[2][key] for key in (
        "max_new_tokens",
        "request_id",
        "eos_token_id",
        "stop_token_ids",
    )} == {
        "max_new_tokens": 3,
        "request_id": 0,
        "eos_token_id": 99,
        "stop_token_ids": (),
    }
    assert callable(generate_call[2]["checkpoint"])
    assert ("provider_acquire", {"max_positions": 256, "pool_enabled": True}) in calls
    assert ("provider_release", 0) in calls
    assert calls[-1] == ("release", (7, "dense_nextn", 256), provider)
    assert generator.last_batch_generation["speculative_mtp"]["nextn_block_id"] == 64
    decoder_init = next(call for call in calls if call[0] == "decoder_init")
    assert decoder_init[3]["candidate_budget"] == 3
    assert decoder_init[3]["target_verify_mode"] == "native"
    mtp = generator.last_batch_generation["speculative_mtp"]
    assert mtp["draft_n_max"] == 3
    assert mtp["target_verify"] == "transactional_native"
    assert mtp["target_verify_batching"] == "single_slot_transactional_native"


def test_gguf_generator_close_releases_pooled_children_before_shared_weights() -> None:
    events: list[str] = []

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(self.name)

    generator = _generator()
    generator.backend = "hip_gfx1100"
    generator._closed = False
    generator._shared_runner = Resource("shared_runner")
    generator._shared_runner_lock = threading.Lock()
    generator._shared_session_pool = {("pool",): [Resource("session0"), Resource("session1")]}
    generator._shared_session_pool_lock = threading.Lock()
    generator._shared_mtp_draft_pool = {7: [Resource("draft0"), Resource("draft1")]}
    generator._shared_mtp_draft_pool_lock = threading.Lock()
    generator._mtp_serving_assets = SimpleNamespace()

    generator.close()
    generator.close()

    assert events == ["session1", "session0", "draft1", "draft0", "shared_runner"]
    assert generator._shared_session_pool == {}
    assert generator._shared_mtp_draft_pool == {}
    assert generator._mtp_serving_assets is None
    assert generator._shared_runner is None


def test_gfx1100_generator_factory_registers_plain_ar_width(monkeypatch) -> None:
    monkeypatch.setattr(
        qwen35_gguf,
        "Qwen35GGUFBringupGenerator",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    generator = qwen35_gguf.make_qwen35_gguf_bringup_generator(
        model_path="/tmp/fake.gguf",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )

    assert generator.backend == "hip_gfx1100"
    assert generator.server_plain_ar_max_active_requests == 4
    assert generator.server_plain_ar_max_active_requests_by_max_sequence_length == {
        768: 13,
    }


def test_gguf_ar_physical_widths_default_capability_and_override(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", raising=False
    )
    # Promoted default (2026-08-20): direct c3/c5/c6/c7 are advertised.
    assert qwen35_gguf._gguf_ar_physical_widths("hip_gfx1100") == (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    )
    assert qwen35_gguf._gguf_ar_physical_widths(
        "hip_gfx1100", use_capability=True
    ) == (1, 2, 3, 4, 5, 6, 7, 8)
    # The env override can still narrow/widen for diagnostics.
    monkeypatch.setenv(
        "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", "1,2,4,8"
    )
    assert qwen35_gguf._gguf_ar_physical_widths(
        "hip_gfx1100", use_capability=True
    ) == (1, 2, 4, 8)
    # Invalid overrides fail closed.
    monkeypatch.setenv("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", "1 3 2")
    with pytest.raises(RuntimeError, match="sorted registered AR widths"):
        qwen35_gguf._gguf_ar_physical_widths("hip_gfx1100")
    monkeypatch.setenv("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", "1 9")
    with pytest.raises(RuntimeError, match="sorted registered AR widths"):
        qwen35_gguf._gguf_ar_physical_widths("hip_gfx1100")


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
    required = (
        "token_embd.weight",
        "output.weight",
        "blk.40.nextn.eh_proj.weight",
        "blk.40.nextn.hnorm.weight",
        "blk.40.nextn.enorm.weight",
        "blk.40.nextn.shared_head_norm.weight",
        "blk.40.attn_norm.weight",
        "blk.40.attn_q.weight",
        "blk.40.attn_k.weight",
        "blk.40.attn_v.weight",
        "blk.40.attn_output.weight",
        "blk.40.attn_q_norm.weight",
        "blk.40.attn_k_norm.weight",
        "blk.40.post_attention_norm.weight",
        "blk.40.ffn_gate_inp.weight",
        "blk.40.ffn_gate_exps.weight",
        "blk.40.ffn_up_exps.weight",
        "blk.40.ffn_down_exps.weight",
        "blk.40.ffn_gate_inp_shexp.weight",
        "blk.40.ffn_gate_shexp.weight",
        "blk.40.ffn_up_shexp.weight",
        "blk.40.ffn_down_shexp.weight",
    )
    metadata = {
        "general.architecture": "qwen35moe",
        "qwen35moe.block_count": 41,
        "qwen35moe.embedding_length": 8,
        "qwen35moe.context_length": 128,
        "qwen35moe.attention.head_count": 2,
        "qwen35moe.attention.head_count_kv": 1,
        "qwen35moe.attention.key_length": 4,
        "qwen35moe.attention.value_length": 4,
        "qwen35moe.rope.dimension_count": 4,
        "qwen35moe.ssm.inner_size": 16,
        "qwen35moe.ssm.group_count": 2,
        "qwen35moe.ssm.state_size": 3,
        "qwen35moe.ssm.conv_kernel": 4,
        "qwen35moe.ssm.time_step_rank": 2,
    }
    tensors = [
        SimpleNamespace(name=name, shape=(11, 8) if name == "token_embd.weight" else (8, 8))
        for name in required
    ]
    by_name = {tensor.name: tensor for tensor in tensors}
    return SimpleNamespace(
        metadata=metadata,
        tensors=tensors,
        tensor=lambda name: by_name[name],
    )


def test_gguf_generator_prepares_explicit_int8_session_policy_and_rejects_switch(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED", "1")
    generator = _generator()
    int8_request = _request(kv_storage="int8_per_token_head", kv_scale_dtype="fp16")

    generator._prepare_kv_policy(int8_request)

    kwargs = generator._prepared_session_kv_kwargs()
    assert kwargs["kv_policy"].storage_dtype.value == "int8_per_token_head"
    assert kwargs["kv_policy"].storage_layout == "uniform"
    assert kwargs["kv_policy"].scale_granularity == "per_token_head"
    assert kwargs["kv_scale_dtype"] == "fp16"
    assert kwargs["kv_scale_granularity"] == "per_token_head"
    assert kwargs["kv_capability"]["runtime_action"] == "diagnostic_override"
    assert kwargs["kv_capability"]["promotion_eligible"] is False
    assert generator._prepared_kv_signature == (
        "int8_per_token_head",
        "uniform",
        "fp16",
        "per_token_head",
    )
    assert generator.kv_capability_provenance["status"] == "unknown"
    assert generator.kv_capability_provenance["runtime_action"] == "diagnostic_override"
    assert generator.kv_capability_provenance["promotion_eligible"] is False

    with pytest.raises(ValueError, match="cannot change after preparation"):
        generator._prepare_kv_policy(_request(kv_storage="bf16"))


def test_gguf_unknown_int8_artifact_fails_closed_to_bf16(monkeypatch) -> None:
    for name in qwen35_gguf._GGUF_INT8_KV_DIAGNOSTIC_OVERRIDE_ENVS:
        monkeypatch.delenv(name, raising=False)
    generator = _generator()

    generator._prepare_kv_policy(
        _request(kv_storage="int8_per_token_head", kv_scale_dtype="fp32")
    )

    assert generator._prepared_kv_signature == (
        "bf16",
        "uniform",
        "fp16",
        "per_token_head",
    )
    assert generator._prepared_session_kv_kwargs() == {}
    assert generator.kv_capability_provenance["status"] == "unknown"
    assert generator.kv_capability_provenance["runtime_action"] == "fallback_bf16"
    assert generator.kv_capability_provenance["effective_kv_storage"] == "bf16"


def test_gguf_exact_gfx1100_qwen38_artifact_admits_fp32_scale_int8(monkeypatch) -> None:
    for name in qwen35_gguf._GGUF_INT8_KV_DIAGNOSTIC_OVERRIDE_ENVS:
        monkeypatch.delenv(name, raising=False)
    generator = _generator()
    generator.model_plugin = Qwen35GGUFModel()
    generator.weight_index = SimpleNamespace(file_type_name="MOSTLY_Q4_K_M")
    generator._kv_model_artifact_identity = lambda: qwen35_gguf.ModelArtifactIdentity(
        path="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
        size_bytes=17_106_773_984,
        sha256="7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b",
        content_verified=True,
    )

    generator._prepare_kv_policy(
        _request(kv_storage="int8_per_token_head", kv_scale_dtype="fp32")
    )

    assert generator._prepared_kv_signature == (
        "int8_per_token_head",
        "uniform",
        "fp32",
        "per_token_head",
    )
    assert generator.kv_capability_provenance["status"] == "qualified"
    assert generator.kv_capability_provenance["runtime_action"] == "admit"
    assert generator.kv_capability_provenance["promotion_eligible"] is True
    kwargs = generator._prepared_session_kv_kwargs()
    assert kwargs["kv_capability"]["evidence"]["persistent_bf16_mirror"] is False
    assert kwargs["kv_capability"]["evidence"]["max_serial_resident_rows"] == 4


def test_gguf_exact_gfx1151_qwen38_rejected_artifact_falls_back_to_bf16(
    monkeypatch,
) -> None:
    for name in qwen35_gguf._GGUF_INT8_KV_DIAGNOSTIC_OVERRIDE_ENVS:
        monkeypatch.delenv(name, raising=False)
    generator = _generator()
    generator.backend = "hip_gfx1151"
    generator.model_plugin = Qwen35GGUFModel()
    generator.weight_index = SimpleNamespace(file_type_name="MOSTLY_Q4_K_M")
    generator._kv_model_artifact_identity = lambda: qwen35_gguf.ModelArtifactIdentity(
        path="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
        size_bytes=17_106_775_008,
        sha256="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
        content_verified=True,
    )

    generator._prepare_kv_policy(
        _request(kv_storage="int8_per_token_head", kv_scale_dtype="fp32")
    )

    assert generator._prepared_kv_signature == (
        "bf16",
        "uniform",
        "fp16",
        "per_token_head",
    )
    assert generator.kv_capability_provenance["status"] == "rejected"
    assert generator.kv_capability_provenance["runtime_action"] == "fallback_bf16"
    assert generator.kv_capability_provenance["promotion_eligible"] is False


def test_gguf_mtp_hot_vocab_defaults_only_in_production_with_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_MTP_HOT_VOCAB", raising=False)
    assert qwen35_gguf._gguf_mtp_hot_vocab_setting("production") == "auto"
    assert qwen35_gguf._gguf_mtp_hot_vocab_setting("strict") is None

    monkeypatch.setenv("HIPENGINE_GGUF_MTP_HOT_VOCAB", "0")
    assert qwen35_gguf._gguf_mtp_hot_vocab_setting("production") is None
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_HOT_VOCAB", "/tmp/custom.json")
    assert qwen35_gguf._gguf_mtp_hot_vocab_setting("strict") == "/tmp/custom.json"


def test_gguf_mtp_server_defer_verify_scatter_default_on_with_opt_out(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER", raising=False)
    assert qwen35_gguf._gguf_mtp_server_defer_verify_scatter_enabled() is True

    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER", "0")
    assert qwen35_gguf._gguf_mtp_server_defer_verify_scatter_enabled() is False


def test_gguf_mtp_server_target_verify_mode_defaults_to_native(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_MTP_VERIFY_MODE", raising=False)
    assert qwen35_gguf._gguf_mtp_server_target_verify_mode() == "native"

    monkeypatch.setenv("HIPENGINE_GGUF_MTP_VERIFY_MODE", "serial-exact")
    assert qwen35_gguf._gguf_mtp_server_target_verify_mode() == "serial_exact"

    monkeypatch.setenv("HIPENGINE_GGUF_MTP_VERIFY_MODE", "serial_exact")
    assert qwen35_gguf._gguf_mtp_server_target_verify_mode() == "serial_exact"

    # Unknown modes fall back to the native default.
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_VERIFY_MODE", "bogus")
    assert qwen35_gguf._gguf_mtp_server_target_verify_mode() == "native"


def test_gguf_mtp_server_candidate_budget_defaults_to_three(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET", raising=False)
    assert qwen35_gguf._gguf_mtp_server_candidate_budget() == 3

    monkeypatch.setenv("HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET", "2")
    assert qwen35_gguf._gguf_mtp_server_candidate_budget() == 2

    # Out-of-range / non-numeric values fall back to the default.
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET", "9")
    assert qwen35_gguf._gguf_mtp_server_candidate_budget() == 3
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET", "abc")
    assert qwen35_gguf._gguf_mtp_server_candidate_budget() == 3



def test_gguf_decode_graph_default_on_with_opt_out(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_GRAPH", raising=False)
    assert qwen35_gguf._gguf_decode_graph_enabled() is True

    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_GRAPH", "0")
    assert qwen35_gguf._gguf_decode_graph_enabled() is False


def test_native_complete_cycle_contributes_public_mtp_ownership_metrics() -> None:
    timing: dict[str, float] = {}
    qwen35_gguf._add_mtp_cycle_timing_metrics(
        timing,
        [
            {
                "mode": "llama_compat_native_complete_cycle",
                "generated_draft_tokens": 2,
                "accepted_draft_tokens": 1,
                "visible_output_tokens": 2,
            }
        ],
    )

    assert timing["mtp_target_verify_rows"] == 3.0
    assert timing["mtp_direct_cycles_count"] == 1.0
    assert timing["mtp_partial_accept_cycles"] == 1.0
    assert timing["mtp_linear_state_captured_rows"] == 3.0
    assert timing["mtp_linear_state_commit_rows"] == 1.0
    assert timing["mtp_hidden_seed_needed_rows"] == 2.0


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


def test_gguf_speculative_mtp_cancellation_releases_request_buffers_and_poisoned_draft(
    monkeypatch,
) -> None:
    calls: list[tuple] = []
    token = GenerationCancellationToken()

    class FakeRuntime:
        def memcpy(self, dst, src, nbytes, kind):
            calls.append(("memcpy", int(nbytes)))

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            self.runtime = kwargs["runtime"]
            self.runner = kwargs["shared_runner"]
            self.position = 0

        def reset(self):
            self.position = 0

        def prefill(self, token_ids, **kwargs):
            self.position = len(token_ids)
            return SimpleNamespace(token_id=1)

        def fp32_verify_hidden_seed_ptr(self, row_index: int = 0) -> int:
            return 0xD000 + int(row_index) * 8

        def mtp_draft_seed(self, *, token_id: int, position: int):
            return SimpleNamespace(
                token_id=int(token_id),
                position=int(position),
                hidden_ptr=0xABC0,
                hidden_contract=SimpleNamespace(ready_for_mtp=True, rows=1, hidden_size=2),
            )

        def close(self):
            calls.append(("session_close",))

    class FakeDraft:
        def propose_chain_from_device_seed(self, hidden_seed_ptr, **kwargs):
            token.cancel()
            calls.append(("draft_cancel", int(hidden_seed_ptr)))
            return [2], [[2]], int(kwargs["dense_cache_len"]) + 2

        def write_kv_rows(self, hidden_rows, token_ids, **kwargs):
            return len(token_ids)

        def close(self):
            calls.append(("draft_close",))

    runner = SimpleNamespace(
        runtime=FakeRuntime(),
        weights=SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=4)),
        close=lambda: calls.append(("runner_close",)),
    )
    draft = FakeDraft()
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
    monkeypatch.setattr(qwen35_gguf, "_new_mtp_draft_runner", lambda assets, *, runtime: draft)
    monkeypatch.setattr(
        qwen35_gguf,
        "_allocate_mtp_dense_kv",
        lambda **kwargs: (SimpleNamespace(ptr=0x1000), SimpleNamespace(ptr=0x2000), ["k", "v"]),
    )
    monkeypatch.setattr(
        qwen35_gguf,
        "_free_mtp_buffers",
        lambda buffers, *, runtime: calls.append(("free_kv", tuple(buffers))),
    )

    generator = _generator()
    generator.backend = "hip_gfx1100"
    generator.weight_index = _mtp_capable_weight_index()
    generator._shared_runner = runner
    generator._shared_runner_lock = threading.Lock()
    generator._shared_session_pool = {}
    generator._shared_session_pool_lock = threading.Lock()
    generator._shared_mtp_draft_pool = {}
    generator._shared_mtp_draft_pool_lock = threading.Lock()
    generator._prepared_max_sequence_length = 64
    generator._mtp_serving_assets = assets

    with pytest.raises(GenerationCancelled):
        generator.generate_speculative_mtp_detailed(
            _request(prompts=("long",), max_tokens=4, cancellation_token=token)
        )

    assert ("free_kv", ("k", "v")) in calls
    assert ("draft_close",) in calls
    assert generator._shared_mtp_draft_pool == {}
    assert generator._shared_session_pool == {}
    assert ("session_close",) in calls
    generator.close()
    assert calls[-1] == ("runner_close",)


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


def test_gguf_speculative_mtp_has_no_rejected_rolling_slot_route() -> None:
    assert not hasattr(
        qwen35_gguf.Qwen35GGUFBringupGenerator,
        "_generate_rolling_mtp_serving_slots",
    )
    source = Path(qwen35_gguf.__file__).read_text()
    assert "HIPENGINE_GGUF_MTP_SERVER_ROLLING_SLOTS" not in source


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


def test_gguf_speculative_mtp_has_no_rejected_final_state_fastpath() -> None:
    source = Path(qwen35_gguf.__file__).read_text()
    assert "HIPENGINE_GGUF_MTP_SERVER_VERIFY_FINAL_STATE_FASTPATH" not in source
    assert "final_state_fastpath" not in source


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


def test_gguf_speculative_mtp_partial_commit_uses_captured_row() -> None:
    calls: list[tuple] = []

    class FakeSession:
        position = 6

        def _commit_verify_linear_state_row(self, row_index: int, *, position: int):
            calls.append(("commit_row", int(row_index), int(position)))

        def fp32_verify_hidden_seed_ptr(self, row_index: int = 0) -> int:
            return 0xD000 + int(row_index) * 8

    class FakeContext:
        def record_verify_seeds(self, rows):
            calls.append(("record_verify_seeds", tuple(row.token_id for row in rows)))

        def accept(self, accepted_draft_tokens: int):
            calls.append(("accept", int(accepted_draft_tokens)))

    slot = qwen35_gguf._GGUFMTPServingSlot(
        request_id=0,
        prompt_ids=[10, 11, 12, 13],
        session=FakeSession(),
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
    )
    verified = qwen35_gguf._GGUFMTPVerifiedCycle(
        drafted=drafted,
        block_result=SimpleNamespace(
            token_ids=[8, 3],
            linear_state_rows_captured=True,
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

    _generator()._commit_mtp_serving_cycle(
        verified,
        assets,
        _request(prompts=("long",), max_tokens=2),
    )

    assert ("commit_row", 0, 5) in calls
    assert ("record_verify_seeds", (8, 3)) in calls
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


def test_gguf_submit_poll_sampled_rows_use_packed_model_ticks(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.model_path = str(model_path)
            self.runtime = SimpleNamespace()
            self.backend = kwargs.get("backend", "hip_gfx1151")
            self.target_arch = "gfx1151"
            self.vocab_size = 128
            self.weights = SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=4))

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

        @staticmethod
        def _logits(primary: int, secondary: int) -> np.ndarray:
            logits = np.full((1, 128), -100.0, dtype=np.float32)
            logits[0, int(primary)] = 1.0
            logits[0, int(secondary)] = 0.75
            return logits

        def reset(self):
            self.position = 0

        def prefill(self, token_ids, *, return_logits=False):
            self.position = len(token_ids)
            primary = int(token_ids[-1]) % 3 + 1
            calls.append(("prefill", self.slot_id, tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_id=primary,
                logits=(self._logits(primary, primary + 1) if return_logits else None),
            )

        def step(self, token_id, *, return_logits=False):  # pragma: no cover - packed c2 must own decode
            raise AssertionError(
                f"sampled row {self.slot_id} fell back to scalar model step for token {token_id}"
            )

        def step_batch_native(
            self,
            token_ids,
            *,
            sessions,
            positions,
            return_logits=False,
            scatter_state=True,
            **kwargs,
        ):
            calls.append(
                (
                    "step_batch_native",
                    tuple(int(token) for token in token_ids),
                    tuple(session.slot_id for session in sessions),
                    tuple(int(position) for position in positions),
                    bool(return_logits),
                    bool(scatter_state),
                    dict(kwargs),
                )
            )
            results = []
            for token, session in zip(token_ids, sessions, strict=True):
                session.position += 1
                primary = int(token) % 3 + 1
                results.append(
                    SimpleNamespace(
                        token_id=primary,
                        logits=(self._logits(primary, primary + 1) if return_logits else None),
                    )
                )
            self._packed_decode_state_dirty = True
            self._packed_decode_sessions = tuple(sessions)
            return results

        def discard_packed_decode_state(self):
            was_dirty = self._packed_decode_state_dirty
            self._packed_decode_sessions = ()
            self._packed_decode_state_dirty = False
            return was_dirty

        def flush_packed_decode_state(self):
            self._packed_decode_state_dirty = False
            return True

        def close(self):
            pass

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()
    generator.backend = "hip_gfx1151"
    generator._shared_runner = None
    generator._shared_runner_lock = threading.Lock()
    generator._prepared_max_sequence_length = 64
    generator._shared_session_pool = {}
    generator._shared_session_pool_lock = threading.Lock()
    generator._shared_mtp_draft_pool = {}
    generator._shared_mtp_draft_pool_lock = threading.Lock()

    adapter = SubmitPollTextGenerator(generator, capacity=2)
    request = _request(
        prompts=("first", "second"),
        max_tokens=3,
        temperature=0.8,
        top_k=2,
        top_p=1.0,
        row_seeds=(17, 29),
        logprobs=True,
        top_logprobs=2,
    )
    first = adapter.generate_detailed(request)
    second = adapter.generate_detailed(request)

    assert [output.generated_token_ids for output in first] == [
        output.generated_token_ids for output in second
    ]
    assert all(len(output.generated_token_ids or ()) == 3 for output in first)
    packed_calls = [call for call in calls if call[0] == "step_batch_native"]
    assert len(packed_calls) == 4
    assert all(call[4] is True for call in packed_calls)
    assert all(call[5] is False for call in packed_calls)
    assert all(call[6]["physical_rows"] == 2 for call in packed_calls)
    assert all(call[6]["active_slot_indices"] == (0, 1) for call in packed_calls)
    assert not any(call[0] == "step" for call in calls)

    for output in second:
        decode_state = _decode_state(output)
        assert decode_state["execution_path"] == "gguf_packed_ar_host_sampler_decode"
        assert decode_state["sampler_fallback_reason"] == "host_sampling_required"
        assert decode_state["native_caware_decode"] is True
        assert decode_state["serial_decode_fallback"] is False
        assert decode_state["native_sampler_rows"] is False
        assert len(output.token_logprobs) == 3

    assert generator.last_batch_generation is not None
    assert generator.last_batch_generation["path"] == "gguf_packed_ar_host_sampler_decode"
    assert generator.last_batch_generation["native_decode_steps"] == 2
    assert generator.last_batch_generation["serial_decode_fallback"] is False
    observability = adapter.live_loop_snapshot()["runner"]["routes"]
    assert observability["counts"]["native_packed_decode_steps"] == 4
    assert observability["counts"]["host_sampler_requests"] == 4
    assert observability["counts"]["serial_decode_fallback_steps"] == 0
    assert observability["counts"]["resident_fallback_requests"] == 0
    assert observability["fallback_reasons"] == {"host_sampling_required": 4}


def test_gguf_resident_sampler_plan_admits_only_supported_stochastic_rows(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    strict = _request(
        temperature=0.0,
        forced_tokens_pending=(7,),
        forced_token_reason="tool_choice_required",
        force_sequence_completion_token_sequences=((7, 8),),
        force_sequence_completion_reason="tool_call_sequence_completion",
    )

    direct = qwen35_gguf._gguf_sampler_plan(strict)
    resident = qwen35_gguf._gguf_sampler_plan(
        strict,
        native_gpu_available=True,
    )

    assert direct.mode is SamplingMode.PROCESSED_ARGMAX
    assert direct.native_gpu_available is False
    assert direct.fallback_reason == "processed_logits_required"
    assert resident.mode is SamplingMode.PROCESSED_ARGMAX
    assert resident.native_gpu_available is True
    assert resident.fallback_reason == "processed_logits_required"
    assert qwen35_gguf._gguf_native_sampler_plan_enabled(strict, resident) is False

    dynamic = _request(temperature=0.8, json_object_close_forcing=True)
    dynamic_plan = qwen35_gguf._gguf_sampler_plan(
        dynamic,
        native_gpu_available=True,
    )
    assert dynamic_plan.mode is SamplingMode.HOST_LOGITS_SAMPLE
    assert dynamic_plan.fallback_reason == "native_gpu_unsupported_request"
    assert qwen35_gguf._gguf_native_sampler_plan_enabled(dynamic, dynamic_plan) is False

    sampled = _request(temperature=0.8, top_k=8, seed=3)
    sampled_plan = qwen35_gguf._gguf_sampler_plan(
        sampled,
        native_gpu_available=True,
    )
    assert sampled_plan.mode is SamplingMode.GPU_SAMPLE
    assert qwen35_gguf._gguf_native_sampler_plan_enabled(sampled, sampled_plan) is True

    for unsupported in (
        _request(temperature=0.8, top_k=65, seed=3),
        _request(temperature=0.8, top_k=2, top_logprobs=3, seed=3),
        _request(temperature=0.8, top_k=8, seed=3, forced_tokens_pending=(7,)),
    ):
        unsupported_plan = qwen35_gguf._gguf_sampler_plan(
            unsupported,
            native_gpu_available=True,
        )
        assert unsupported_plan.mode is SamplingMode.HOST_LOGITS_SAMPLE
        assert unsupported_plan.fallback_reason == "native_gpu_unsupported_request"
        assert (
            qwen35_gguf._gguf_native_sampler_plan_enabled(
                unsupported,
                unsupported_plan,
            )
            is False
        )


def test_gguf_resident_mixed_native_and_host_sampler_rows_fail_closed_to_serial(
    monkeypatch,
) -> None:
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner._last_execution_manifest = {}
    runner._last_physical_group_plan = {}
    serial_calls: list[tuple[tuple[int, ...], str]] = []
    runner._step_native_chunk = lambda *args, **kwargs: pytest.fail(
        "mixed native/host rows must not share a full-logits packed result"
    )
    runner._step_native_serial = lambda rows, *, fallback_reason: serial_calls.append(
        (
            tuple(int(row.request_id) for row in rows),
            str(fallback_reason),
        )
    )
    rows = (
        SimpleNamespace(
            request_id=11,
            native_sampled=True,
            native_sampler=True,
        ),
        SimpleNamespace(
            request_id=12,
            native_sampled=True,
            native_sampler=False,
        ),
    )
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")

    runner._step_native_rows(rows)

    assert serial_calls == [((11, 12), "mixed_sampler_routes")]
    assert runner._last_physical_group_plan["groups"][0]["execution_path"] == (
        "serial_fallback"
    )


def test_gguf_submit_poll_stochastic_rows_use_batched_native_sampler_without_logits_d2h(
    monkeypatch,
) -> None:
    calls: list[tuple] = []

    class FakeFullStackRunner:
        def __init__(self, model_path, **kwargs):
            self.model_path = str(model_path)
            self.runtime = SimpleNamespace()
            self.backend = kwargs.get("backend", "hip_gfx1100")
            self.target_arch = "gfx1100"
            self.vocab_size = 128
            self.weights = SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=4))

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

        def reset(self):
            self.position = 0

        def prefill(self, token_ids, *, return_logits=False):
            self.position = len(token_ids)
            calls.append(("prefill", self.slot_id, tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(token_id=1, logits=None)

        def step(self, token_id, *, return_logits=False):  # pragma: no cover
            raise AssertionError("native sampled c2 must stay on packed model steps")

        def step_batch_native(
            self,
            token_ids,
            *,
            sessions,
            positions,
            return_logits=False,
            scatter_state=True,
            **kwargs,
        ):
            calls.append(
                (
                    "step_batch_native",
                    tuple(int(token) for token in token_ids),
                    tuple(session.slot_id for session in sessions),
                    bool(return_logits),
                    bool(scatter_state),
                    dict(kwargs),
                )
            )
            for session in sessions:
                session.position += 1
            self._packed_decode_state_dirty = True
            self._packed_decode_sessions = tuple(sessions)
            return [SimpleNamespace(token_id=1, logits=None) for _ in sessions]

        @staticmethod
        def _sample(params, state):
            token_id = (5, 4, 2)[state.step_index]
            state.observe(token_id)
            return SampleResult(
                token_id=token_id,
                logit=1.0,
                logprob=-0.25,
                mode=SamplingMode.GPU_SAMPLE,
                candidate_count=int(params.top_k),
            )

        def sample_native_from_last_logits(self, params, state):
            calls.append(("sample_native_last", self.slot_id, state.step_index))
            return self._sample(params, state)

        def sample_native_from_packed_logits_rows(
            self,
            physical_rows,
            params_rows,
            states,
        ):
            calls.append(
                (
                    "sample_native_packed_rows",
                    tuple(int(row) for row in physical_rows),
                    tuple(state.step_index for state in states),
                )
            )
            return tuple(
                self._sample(params, state)
                for params, state in zip(params_rows, states, strict=True)
            )

        def sample_native_from_packed_logits(
            self,
            physical_row,
            params,
            state,
            *,
            output_session,
        ):  # pragma: no cover - compatible c2 rows must use one batch launch
            raise AssertionError(
                f"native row {physical_row} for slot {output_session.slot_id} "
                "did not use the batch sampler"
            )

        def discard_packed_decode_state(self):
            was_dirty = self._packed_decode_state_dirty
            self._packed_decode_sessions = ()
            self._packed_decode_state_dirty = False
            return was_dirty

        def flush_packed_decode_state(self):
            self._packed_decode_state_dirty = False
            return True

        def close(self):
            pass

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
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
    request = _request(
        prompts=("first", "second"),
        max_tokens=3,
        temperature=0.8,
        top_k=8,
        seed=17,
    )
    outputs = adapter.generate_detailed(request)

    assert [output.generated_token_ids for output in outputs] == [
        (5, 4, 2),
        (5, 4, 2),
    ]
    assert all(call[3] is False for call in calls if call[0] == "prefill")
    packed_calls = [call for call in calls if call[0] == "step_batch_native"]
    assert len(packed_calls) == 2
    assert all(call[3] is False and call[4] is False for call in packed_calls)
    assert len([call for call in calls if call[0] == "sample_native_last"]) == 2
    packed_sample_calls = [
        call for call in calls if call[0] == "sample_native_packed_rows"
    ]
    assert packed_sample_calls == [
        ("sample_native_packed_rows", (0, 1), (1, 1)),
        ("sample_native_packed_rows", (0, 1), (2, 2)),
    ]

    for output in outputs:
        decode_state = _decode_state(output)
        assert decode_state["execution_path"] == "gguf_packed_ar_native_sampler_decode"
        assert decode_state["sampler_mode"] == "gpu_sample"
        assert "sampler_fallback_reason" not in decode_state
        assert decode_state["full_vocab_logits_d2h"] is False
        assert decode_state["logits_d2h_bytes"] == 0
        assert decode_state["native_sampler_rows"] is True
        assert decode_state["native_caware_decode"] is True
        assert decode_state["serial_decode_fallback"] is False
        assert output.finish_details is not None
        assert output.finish_details.to_json_dict()["sampler_mode"] == "gpu_sample"
        assert len(output.token_logprobs) == 3
        assert all(token.logprob == pytest.approx(-0.25) for token in output.token_logprobs)

    assert generator.last_batch_generation is not None
    assert generator.last_batch_generation["path"] == (
        "gguf_packed_ar_native_sampler_decode"
    )
    assert generator.last_batch_generation["native_sampler_rows"] is True
    observability = adapter.live_loop_snapshot()["runner"]["routes"]
    assert observability["counts"]["native_sampler_requests"] == 2
    assert observability["counts"]["native_sampler_batch_launches"] == 2
    assert observability["counts"]["native_sampler_row_launches"] == 2
    assert observability["counts"]["host_sampler_requests"] == 0
    assert observability["fallback_reasons"] == {}


def test_gguf_sampled_packed_unavailable_reports_model_serial_fallback(monkeypatch) -> None:
    calls: list[tuple] = []
    tokenizer = _FakeTokenizer()

    class FakeSession:
        def __init__(self, slot_id: int):
            self.slot_id = int(slot_id)
            self.position = 2

        def step_batch_native(self, token_ids, **kwargs):
            calls.append(("step_batch_native", tuple(token_ids), dict(kwargs)))
            raise NotImplementedError("sampled packed logits unavailable")

        def step(self, token_id, *, return_logits=False):
            calls.append(("step", self.slot_id, int(token_id), bool(return_logits)))
            self.position += 1
            logits = np.full((1, 128), -100.0, dtype=np.float32)
            logits[0, 2] = 10.0
            return SimpleNamespace(token_id=2, logits=logits if return_logits else None)

    generator = SimpleNamespace(
        tokenizer=tokenizer,
        _flush_ar_packed_decode_owners_if_chunk_changed=lambda slots: None,
        _flush_ar_packed_decode_owners=lambda slots: None,
    )
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = generator
    runner._route_counts = Counter()
    runner._fallback_reasons = Counter()
    runner._last_execution_manifest = {}
    runner._last_physical_group_plan = {}

    request = _request(
        prompts=("first", "second"),
        max_tokens=3,
        temperature=0.7,
        top_k=1,
        row_seeds=(17, 29),
    )
    sampling_request = qwen35_gguf._request_with_tokenizer_eos(request, tokenizer)
    rows = []
    for row_index, (request_id, prompt_ids) in enumerate(((10, (10, 11)), (11, (20,)))):
        state = qwen35_gguf._gguf_row_sampling_state(
            sampling_request,
            list(prompt_ids),
            row_index=row_index,
        )
        logits = np.full((1, 128), -100.0, dtype=np.float32)
        logits[0, 1] = 10.0
        sample = qwen35_gguf._select_from_gguf_logits(
            SimpleNamespace(logits=logits),
            sampling_request,
            state,
        )
        session = FakeSession(row_index)
        row = qwen35_gguf._GGUFResidentLoopRow(
            request_id=request_id,
            batch_id=0,
            row_index=row_index,
            request=request,
            prompt_ids=prompt_ids,
            native_greedy=False,
            native_sampled=True,
            submitted_at=0.0,
            sampling_request=sampling_request,
            sampling_state=state,
            samples=[sample],
            first_token_emitted=True,
            slot=qwen35_gguf._GGUFARServingSlot(
                request_id=request_id,
                prompt_ids=list(prompt_ids),
                session=session,
                prev_token=1,
                seq_position=2,
                generated_ids=[1],
            ),
        )
        rows.append(row)

    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")
    runner._step_native_rows(rows)

    assert [call[0] for call in calls] == ["step_batch_native", "step", "step"]
    assert all(call[-1] is True for call in calls if call[0] == "step")
    assert [row.slot.generated_ids for row in rows] == [[1, 2], [1, 2]]
    assert [len(row.samples) for row in rows] == [2, 2]
    assert [row.slot.serial_decode_steps for row in rows] == [1, 1]
    assert runner._route_counts["native_packed_decode_steps"] == 0
    assert runner._route_counts["serial_decode_fallback_steps"] == 1
    assert runner._fallback_reasons == {"packed_decode_unavailable": 1}
    for row in rows:
        decode_state = _decode_state(runner._native_stream_chunk(row))
        assert decode_state["sampler_fallback_reason"] == "host_sampling_required"
        assert decode_state["serial_decode_fallback"] is True


def test_gguf_resident_stream_reuses_registered_sampler_plan(monkeypatch) -> None:
    request = _request(max_tokens=2, ignore_eos=True)
    plan = qwen35_gguf._gguf_sampler_plan(request)
    slot = qwen35_gguf._GGUFARServingSlot(
        request_id=1,
        prompt_ids=[10, 11],
        session=SimpleNamespace(),
        prev_token=16,
        seq_position=3,
        generated_ids=[16],
        timing={"prefill_ms": 7.5},
        native_decode_steps=1,
    )
    row = qwen35_gguf._GGUFResidentLoopRow(
        request_id=1,
        batch_id=0,
        row_index=0,
        request=request,
        prompt_ids=(10, 11),
        native_greedy=True,
        native_sampled=False,
        submitted_at=0.0,
        slot=slot,
        first_token_emitted=True,
        sampler_plan=plan,
    )
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = SimpleNamespace(tokenizer=_FakeTokenizer())

    monkeypatch.setattr(
        qwen35_gguf,
        "_gguf_sampler_plan",
        lambda request: (_ for _ in ()).throw(
            AssertionError(f"registered sampler plan was recomputed: {request!r}")
        ),
    )

    chunk = runner._native_stream_chunk(row)

    assert _decode_state(chunk)["sampler_mode"] == "greedy_fast"
    assert chunk.telemetry is not None
    assert chunk.telemetry.timing == {"prefill_ms": 7.5}
    assert chunk.telemetry.timing_scope == "choice"
    assert chunk.telemetry.diagnostics == {
        "prefix_cache": {
            "mode": "off",
            "block_size_tokens": 256,
            "eligible": False,
            "lookup": False,
            "hit": False,
            "source": None,
            "matched_tokens": 0,
            "reused_tokens": 0,
            "avoided_prefill_tokens": 0,
            "executed_prefill_tokens": 2,
            "reused_pages": 0,
            "reused_page_bytes": 0,
            "state_clone_bytes": 0,
            "snapshot_hit": False,
            "admission_fallback": False,
            "fallback_reason": "cache_off",
            "cache_resident_entries": 0,
            "cache_resident_pages": 0,
            "cache_resident_bytes": 0,
        }
    }
    assert chunk.generated_token_ids is None


def test_gguf_empty_stop_suffix_skips_history_scan(monkeypatch) -> None:
    monkeypatch.setattr(
        qwen35_gguf,
        "token_sequence_state_for_tokens",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(f"empty stop policy must not scan token history: {args!r} {kwargs!r}")
        ),
    )

    assert qwen35_gguf._gguf_stop_suffix_state((1, 2, 3), ()) is None


def test_gguf_resident_prepare_is_idempotent_at_full_occupancy() -> None:
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = SimpleNamespace(_prepared_max_sequence_length=562)
    runner._max_sequence_length = 562
    runner._available = []
    runner._rows = {request_id: object() for request_id in range(8)}

    runner.prepare(max_sequence_length=562)

    assert len(runner._rows) == 8


@pytest.mark.parametrize(
    ("attention_source", "block_ids"),
    ((None, (9,)), ("int8_direct", (8,))),
    ids=("shifted-allocation", "direct-int8-base-zero"),
)
def test_gguf_resident_full_prefill_uses_block_table_path_when_required(
    attention_source: str | None,
    block_ids: tuple[int, ...],
) -> None:
    calls: list[tuple] = []

    class FakeSession:
        position = 0
        kv_attention_source = attention_source
        _device_kv_allocation = SimpleNamespace(
            block_ids=block_ids,
            chunk_start_block_id=8,
        )

        def prefill(self, token_ids, **kwargs):
            raise AssertionError(
                f"block-table-aware KV must not use raw-cache full prefill: {token_ids!r} {kwargs!r}"
            )

        def prefill_batch_native(self, token_rows, *, sessions, **kwargs):
            calls.append(
                (
                    tuple(tuple(int(token) for token in row) for row in token_rows),
                    tuple(sessions),
                    dict(kwargs),
                )
            )
            self.position = len(token_rows[0])
            return [SimpleNamespace(token_id=7)]

    session = FakeSession()
    row = qwen35_gguf._GGUFResidentLoopRow(
        request_id=1,
        batch_id=0,
        row_index=0,
        request=_request(prompts=("first",), max_tokens=2, ignore_eos=True),
        prompt_ids=(10, 11),
        native_greedy=True,
        native_sampled=False,
        submitted_at=0.0,
        lease=qwen35_gguf._GGUFResidentSessionLease(
            session=session,
            pool_key=("continuous_ar_dynamic_kv", True, True, 256),
        ),
    )
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = SimpleNamespace(tokenizer=_FakeTokenizer())
    runner._route_counts = Counter()
    runner._refresh_prefix_cache = lambda candidate: None

    runner._prefill_native_row(row)

    assert len(calls) == 1
    token_rows, sessions, kwargs = calls[0]
    assert token_rows == ((10, 11),)
    assert sessions == (session,)
    assert kwargs == {
        "full_prompt_lengths": [2],
        "return_logits": False,
        "return_hidden_seeds": False,
    }
    assert row.slot is not None
    assert row.slot.generated_ids == [7]
    assert row.slot.native_compact_prefill is True
    assert runner._route_counts["native_full_prefill_rows"] == 1


@pytest.mark.parametrize(
    ("attention_source", "block_ids"),
    ((None, (9,)), ("int8_direct", (8,))),
    ids=("shifted-allocation", "direct-int8-base-zero"),
)
def test_gguf_resident_sampled_prefill_uses_block_table_path_when_required(
    attention_source: str | None,
    block_ids: tuple[int, ...],
) -> None:
    calls: list[tuple] = []

    class FakeSession:
        position = 0
        kv_attention_source = attention_source
        _device_kv_allocation = SimpleNamespace(
            block_ids=block_ids,
            chunk_start_block_id=8,
        )

        def prefill(self, token_ids, **kwargs):
            raise AssertionError(
                f"block-table-aware sampled KV must not use raw-cache full prefill: {token_ids!r} {kwargs!r}"
            )

        def prefill_batch_native(self, token_rows, *, sessions, **kwargs):
            calls.append(
                (
                    tuple(tuple(int(token) for token in row) for row in token_rows),
                    tuple(sessions),
                    dict(kwargs),
                )
            )
            self.position = len(token_rows[0])
            logits = np.full((1, 128), -100.0, dtype=np.float32)
            logits[0, 1] = 10.0
            return [SimpleNamespace(token_id=1, logits=logits)]

    session = FakeSession()
    row = qwen35_gguf._GGUFResidentLoopRow(
        request_id=1,
        batch_id=0,
        row_index=0,
        request=_request(prompts=("first",), max_tokens=2, temperature=0.7, top_k=1),
        prompt_ids=(10, 11),
        native_greedy=False,
        native_sampled=True,
        submitted_at=0.0,
        lease=qwen35_gguf._GGUFResidentSessionLease(
            session=session,
            pool_key=("continuous_ar_dynamic_kv", True, True, 256),
        ),
    )
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = SimpleNamespace(tokenizer=_FakeTokenizer())
    runner._route_counts = Counter()
    runner._fallback_reasons = Counter()
    runner._refresh_prefix_cache = lambda candidate: None

    runner._prefill_sampled_row(row)

    assert len(calls) == 1
    token_rows, sessions, kwargs = calls[0]
    assert token_rows == ((10, 11),)
    assert sessions == (session,)
    assert kwargs == {
        "full_prompt_lengths": [2],
        "return_logits": True,
        "return_hidden_seeds": False,
    }
    assert row.slot is not None
    assert row.slot.generated_ids == [1]
    assert row.slot.native_compact_prefill is True
    assert runner._route_counts["native_sampled_prefill_rows"] == 1


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

        def close(self):
            calls.append(("runner_close",))

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

        def decode_graph_min_replay_steps(self):
            return 1

        def capture_decode_graph(self, **kwargs):
            session = self
            calls.append(("capture_decode_graph", self.slot_id, dict(kwargs)))

            class FakeGraph:
                closed = False
                replayed_steps = 0
                steps_per_replay = 1

                def replay(self, steps):
                    assert int(steps) == 1
                    self.replayed_steps += 1
                    session.position += 1
                    calls.append(("graph_replay", session.slot_id, self.replayed_steps))

                def read_sample(self, **kwargs):
                    return SimpleNamespace(token_id=2)

                def close(self):
                    self.closed = True
                    calls.append(("graph_close", session.slot_id, self.replayed_steps))

            return FakeGraph()

        def step_batch_native(self, token_ids, *, sessions, positions, **kwargs):
            physical_rows = int(kwargs.get("physical_rows", len(token_ids)))
            active_slots = tuple(
                int(index)
                for index in kwargs.get("active_slot_indices", range(len(token_ids)))
            )
            self.last_packed_execution_manifest = {
                "schema": 1,
                "kind": "gguf_packed_ar_execution_manifest",
                "rows": physical_rows,
                "physical_rows": physical_rows,
                "active_rows": len(token_ids),
                "active_mask": [index in active_slots for index in range(physical_rows)],
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

        def discard_packed_decode_state(self):
            was_dirty = self._packed_decode_state_dirty
            self._packed_decode_sessions = ()
            self._packed_decode_state_dirty = False
            return was_dirty

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
    prepared = PreparedPromptInput(
        source_text="first",
        token_ids=(10, 11),
        tokenize_ms=1.25,
        render_ms=2.5,
        admission_prepare_ms=3.75,
    )
    second = adapter.generate_detailed(_request(prompts=(prepared,), max_tokens=2))
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
    for output in (*first, *second, *sampled, *zero):
        assert output.telemetry is not None
        assert output.telemetry.timing is not None
        assert "tokenize_ms" in output.telemetry.timing
        assert output.telemetry.timing["tokenize_ms"] >= 0.0
    assert second[0].telemetry is not None
    assert second[0].telemetry.timing is not None
    assert second[0].telemetry.timing["tokenize_ms"] == 1.25
    assert second[0].telemetry.timing["prompt_encode_ms"] == 1.25
    assert second[0].telemetry.timing["render_ms"] == 2.5
    assert second[0].telemetry.timing["admission_prepare_ms"] == 3.75
    assert zero[0].telemetry is not None
    assert zero[0].telemetry.timing is not None
    assert zero[0].telemetry.timing["prompt_encode_ms"] >= 0.0
    assert zero[0].telemetry.timing["render_ms"] == 0.0
    assert zero[0].telemetry.timing["admission_prepare_ms"] == 0.0
    assert [call[0] for call in calls].count("runner_init") == 1
    assert [call[0] for call in calls].count("session_init") == 4
    packed_calls = [call for call in calls if call[0] == "step_batch_native"]
    assert [call[2] for call in packed_calls] == [(1, 1), (2, 2)]
    assert [call[5]["physical_rows"] for call in packed_calls] == [2, 2]
    assert [call[5]["active_slot_indices"] for call in packed_calls] == [
        (0, 1),
        (0, 1),
    ]
    assert [call for call in calls if call[0] == "graph_replay"] == [
        ("graph_replay", 2, 1)
    ]
    capture_calls = [call for call in calls if call[0] == "capture_decode_graph"]
    assert len(capture_calls) == 1
    assert capture_calls[0][2]["input_token_id"] == 1
    assert [call for call in calls if call[0] == "step"][-1][2] == 1
    assert runner.active_request_ids == ()
    assert runner.available_session_count == 2
    assert greedy_last["path"] == "gguf_packed_ar_server_decode"
    assert greedy_last["serial_decode_fallback"] is False
    assert greedy_last["native_c1_decode_steps"] == 1
    assert generator.last_batch_generation is not None
    assert generator.last_batch_generation["path"] == "gguf_resident_model_loop"
    assert generator.last_batch_generation["serial_decode_fallback"] is False

    observability = adapter.live_loop_snapshot()["runner"]
    assert observability["model_runner"] == {
        "capacity": 2,
        "active_request_ids": [],
        "active_requests": 0,
        "available_sessions": 2,
        "packed_workspace_current_bytes": 0,
        "packed_workspace_release_events": 0,
        "packed_workspace_released_bytes": 0,
        "kv_layout_audits": [],
        "persistent_int8_payload_bytes": 0,
        "persistent_bf16_payload_bytes": 0,
        "persistent_scale_bytes": 0,
        "persistent_bf16_mirror_bytes": 0,
        "persistent_kv_total_bytes": 0,
    }
    assert observability["routes"]["counts"] == {
        # Grouped calls are counted separately from rows: this scenario has no native
        # batch entry point, so three rows prefill serially and no group is formed.
        "native_full_prefill_groups": 0,
        "native_full_prefill_rows": 3,
        "native_incremental_prefill_chunks": 0,
        "native_incremental_prefill_unsampled_chunks": 0,
        "native_packed_decode_steps": 2,
        "native_packed_graph_captures": 0,
        "native_packed_graph_replays": 0,
        "native_c1_decode_steps": 2,
        "native_sampled_prefill_rows": 1,
        "native_sampler_requests": 0,
        "native_sampler_batch_launches": 0,
        "native_sampler_row_launches": 0,
        "host_sampler_requests": 1,
        "serial_decode_fallback_steps": 0,
        "serial_c1_row_steps": 0,
        "resident_fallback_requests": 1,
    }
    assert observability["routes"]["physical_width_decode_steps"] == {
        "1": 2,
        "2": 2,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
    }
    physical_group = {
        "logical_c": 1,
        "group_index": 0,
        "group_count": 1,
        "physical_slot_base": 0,
        "physical_slot_extent": 1,
        "physical_rows": 1,
        "active_rows": 1,
        "request_ids": [3],
        "global_slot_indices": [0],
        "active_slot_indices": [0],
        "active_mask": [True],
        "execution_row_mapping": "dense_active_rows",
    }
    assert observability["routes"]["last_execution_manifest"] == {
        "schema": 1,
        "kind": "gguf_ar_c1_execution_manifest",
        "mode": "native_c1_eager",
        "rows": 1,
        "physical_rows": 1,
        "active_rows": 1,
        "active_mask": [True],
        "model_step": {
            "complete_c1_session_replays": 0,
            "complete_c1_layer_replays": 0,
            "host_model_row_loop_sites": 0,
            "host_model_row_iterations": 0,
        },
        "graph": {
            "captured": False,
            "replay_count": 0,
        },
        "logical_c": 1,
        "physical_group": physical_group,
    }
    assert observability["routes"]["last_physical_group_plan"] == {
        "schema": 1,
        "kind": "gguf_ar_physical_group_plan",
        "logical_c": 1,
        "physical_bucket_widths": [1, 2, 3, 4, 5, 6, 7, 8],
        "policy": "occupancy_adaptive_dense_execution",
        "group_count": 1,
        "groups": [{**physical_group, "execution_path": "native_c1_eager"}],
    }
    assert observability["routes"]["fallback_reasons"]
    assert len(observability["routes"]["recent_completed"]) == 5
    assert observability["graph_buckets"]["captures_total"] == 0
    assert observability["graph_buckets"]["buckets"] == {}

    adapter.close()
    adapter.close()
    assert len([call for call in calls if call[0] == "close"]) == 4
    assert [call[0] for call in calls][-1] == "runner_close"


def test_gguf_c1_graph_seeds_survivor_token_after_packed_width_transition() -> None:
    captures: list[dict[str, object]] = []

    class FakeSession:
        feedback_token = 9709  # Last packed physical row, not the surviving owner row.

        def decode_graph_min_replay_steps(self):
            return 1

        def capture_decode_graph(self, **kwargs):
            captures.append(dict(kwargs))
            self.feedback_token = int(kwargs.get("input_token_id", self.feedback_token))
            session = self

            class FakeGraph:
                def replay(self, steps):
                    assert int(steps) == 1

                def read_sample(self, *, return_logits=False):
                    assert return_logits is False
                    return SimpleNamespace(
                        token_id=9710 if session.feedback_token == 9710 else 2
                    )

            return FakeGraph()

    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    slot = SimpleNamespace(
        session=FakeSession(),
        c1_decode_graph=None,
        seq_position=536,
        generated_ids=[9710] * 24,
        prev_token=9710,
    )
    row = SimpleNamespace(slot=slot, request=SimpleNamespace(max_tokens=48))

    result = runner._step_native_c1_graph(row)

    assert result.token_id == 9710
    assert captures == [
        {
            "position": 536,
            "steps_per_replay": 1,
            "max_replay_steps": 24,
            "attention_max_context_len": 560,
            "input_token_id": 9710,
        }
    ]
    assert slot.c1_decode_graph is not None


def test_gguf_c1_edge_disables_graph_while_peer_requests_are_resident() -> None:
    events: list[object] = []

    class FakeGraph:
        closed = False

        def close(self) -> None:
            self.closed = True
            events.append("graph_close")

    class FakeSession:
        def step(self, token_id: int, *, return_logits: bool):
            events.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(token_id=17)

    graph = FakeGraph()
    slot = SimpleNamespace(
        session=FakeSession(),
        c1_decode_graph=graph,
        prev_token=16,
    )
    row = SimpleNamespace(slot=slot, lease=None)
    peer = SimpleNamespace(slot=SimpleNamespace())
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner._rows = {1: row, 2: peer}

    result = runner._step_native_c1_graph(row)

    assert result.token_id == 17
    assert events == ["graph_close", ("step", 16, False)]
    assert slot.c1_decode_graph is None


def test_gguf_resident_runner_lowers_c13_to_declared_physical_groups(monkeypatch) -> None:
    calls: list[tuple] = []
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner._last_execution_manifest = {}
    runner._last_physical_group_plan = {}

    def step_native_chunk(
        rows,
        *,
        physical_rows=None,
        active_slot_indices=(),
        allow_graph=True,
    ):
        assert allow_graph is False
        calls.append(
            (
                tuple(int(row.request_id) for row in rows),
                int(physical_rows),
                tuple(int(index) for index in active_slot_indices),
            )
        )
        runner._last_execution_manifest = {
            "schema": 1,
            "kind": "gguf_packed_ar_execution_manifest",
            "physical_rows": int(physical_rows),
        }
        return True

    runner._step_native_chunk = step_native_chunk
    runner._step_native_serial = lambda rows: pytest.fail(
        f"unexpected serial fallback for {[row.request_id for row in rows]}"
    )
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")
    rows = [SimpleNamespace(request_id=request_id) for request_id in range(100, 113)]
    request_ids = tuple(row.request_id for row in rows)
    work = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=request_ids,
        row_to_request=request_ids,
        slot_ids=tuple(range(13)),
        active_mask=(True,) * 13,
    )

    runner._step_native_rows(rows, work=work)

    assert calls == [
        (tuple(range(100, 108)), 8, tuple(range(8))),
        (tuple(range(108, 113)), 5, tuple(range(5))),
    ]
    assert runner._last_physical_group_plan == {
        "schema": 1,
        "kind": "gguf_ar_physical_group_plan",
        "logical_c": 13,
        "physical_bucket_widths": [1, 2, 3, 4, 5, 6, 7, 8],
        "policy": "occupancy_adaptive_dense_execution",
        "group_count": 2,
        "groups": [
            {
                "logical_c": 13,
                "group_index": 0,
                "group_count": 2,
                "physical_slot_base": 0,
                "physical_slot_extent": 8,
                "physical_rows": 8,
                "active_rows": 8,
                "request_ids": list(range(100, 108)),
                "global_slot_indices": list(range(8)),
                "active_slot_indices": list(range(8)),
                "active_mask": [True] * 8,
                "execution_row_mapping": "dense_active_rows",
                "execution_path": "packed_native",
            },
            {
                "logical_c": 13,
                "group_index": 1,
                "group_count": 2,
                "physical_slot_base": 8,
                "physical_slot_extent": 5,
                "physical_rows": 5,
                "active_rows": 5,
                "request_ids": list(range(108, 113)),
                "global_slot_indices": list(range(8, 13)),
                "active_slot_indices": list(range(5)),
                "active_mask": [True, True, True, True, True],
                "execution_row_mapping": "dense_active_rows",
                "execution_path": "packed_native",
            },
        ],
    }
    assert runner._last_execution_manifest["logical_c"] == 13
    assert runner._last_execution_manifest["physical_group"]["group_index"] == 1
    assert runner._last_execution_manifest["physical_group"]["physical_rows"] == 5


def test_gguf_compact_serial_capability_is_artifact_scoped_and_bounded() -> None:
    qualified = SimpleNamespace(
        kv_capability_provenance={
            "status": "qualified",
            "runtime_action": "admit",
            "promotion_eligible": True,
            "effective_kv_storage": "int8_per_token_head",
            "evidence": {
                "max_direct_rows": 1,
                "max_serial_resident_rows": 4,
                "persistent_bf16_mirror": False,
            },
        }
    )
    diagnostic = SimpleNamespace(
        kv_capability_provenance={
            **qualified.kv_capability_provenance,
            "runtime_action": "diagnostic_override",
            "promotion_eligible": False,
        }
    )

    assert qwen35_gguf._qualified_compact_serial_int8_max_rows(qualified) == 4
    assert qwen35_gguf._qualified_compact_serial_int8_max_rows(diagnostic) == 0


def test_gguf_resident_request_diagnostics_retain_kv_layout_audit() -> None:
    audit = {
        "kv_attention_source": "int8_direct",
        "persistent_int8_payload_bytes": 33_554_432,
        "persistent_scale_bytes": 524_288,
        "persistent_bf16_mirror_bytes": 0,
    }
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner._prefix_request_telemetry = lambda row: {"mode": "off"}
    row = SimpleNamespace(
        slot=SimpleNamespace(
            session=SimpleNamespace(device_kv_layout_audit=lambda: audit)
        )
    )

    assert runner._request_diagnostics(row, include_kv_layout=False) == {
        "prefix_cache": {"mode": "off"}
    }
    diagnostics = runner._request_diagnostics(row)
    audit["persistent_bf16_mirror_bytes"] = 1

    assert diagnostics == {
        "prefix_cache": {"mode": "off"},
        "kv_layout": {
            "kv_attention_source": "int8_direct",
            "persistent_int8_payload_bytes": 33_554_432,
            "persistent_scale_bytes": 524_288,
            "persistent_bf16_mirror_bytes": 0,
        },
    }


def test_gguf_resident_direct_int8_c4_declares_serial_physical_width_one(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], str]] = []
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = SimpleNamespace(
        kv_capability_provenance={
            "status": "qualified",
            "runtime_action": "admit",
            "promotion_eligible": True,
            "effective_kv_storage": "int8_per_token_head",
            "evidence": {
                "max_direct_rows": 1,
                "max_serial_resident_rows": 4,
                "persistent_bf16_mirror": False,
            },
        }
    )
    runner._last_execution_manifest = {}
    runner._last_physical_group_plan = {}
    runner._step_native_chunk = lambda *args, **kwargs: pytest.fail(
        "compact direct INT8 C4 must not enter packed decode before IKV-C2"
    )
    runner._step_native_serial = lambda rows, *, fallback_reason: calls.append(
        (tuple(int(row.request_id) for row in rows), str(fallback_reason))
    )
    rows = [
        SimpleNamespace(
            request_id=100 + index,
            native_sampler=False,
            native_sampled=False,
            slot=SimpleNamespace(
                session=SimpleNamespace(
                    kv_attention_source="int8_direct",
                    packed_decode_max_rows=1,
                )
            ),
        )
        for index in range(4)
    ]
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")

    runner._step_native_rows(rows)

    assert calls == [((100, 101, 102, 103), "packed_decode_width_unqualified")]
    assert runner._last_execution_manifest["kv_attention_source"] == "int8_direct"
    assert runner._last_execution_manifest["logical_c"] == 4
    assert runner._last_execution_manifest["physical_execution_width"] == 1
    assert runner._last_execution_manifest["serial_decode_fallback"] is True
    assert runner._last_execution_manifest["throughput_claim_eligible"] is False
    assert runner._last_physical_group_plan["groups"][0]["planned_physical_rows"] == 4
    assert runner._last_physical_group_plan["groups"][0]["physical_execution_width"] == 1


def test_gguf_resident_direct_int8_c4_uses_capability_qualified_packed_width(
    monkeypatch,
) -> None:
    calls: list[tuple[tuple[int, ...], int, tuple[int, ...]]] = []
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = SimpleNamespace(
        kv_capability_provenance={
            "status": "qualified",
            "runtime_action": "admit",
            "promotion_eligible": True,
            "effective_kv_storage": "int8_per_token_head",
            "evidence": {
                "max_direct_rows": 1,
                "max_serial_resident_rows": 4,
                "persistent_bf16_mirror": False,
            },
        }
    )
    runner._last_execution_manifest = {"mode": "native_int8_batch"}
    runner._last_physical_group_plan = {}

    def step_chunk(rows, *, physical_rows, active_slot_indices, allow_graph=True):
        calls.append(
            (
                tuple(int(row.request_id) for row in rows),
                int(physical_rows),
                tuple(int(index) for index in active_slot_indices),
            )
        )
        return True

    runner._step_native_chunk = step_chunk
    runner._step_native_serial = lambda *args, **kwargs: pytest.fail(
        "qualified direct INT8 c4 must not execute serial c1 rows"
    )
    rows = [
        SimpleNamespace(
            request_id=200 + index,
            native_sampler=False,
            native_sampled=False,
            slot=SimpleNamespace(
                session=SimpleNamespace(
                    kv_attention_source="int8_direct",
                    packed_decode_max_rows=4,
                )
            ),
        )
        for index in range(4)
    ]
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")

    runner._step_native_rows(rows)

    assert calls == [((200, 201, 202, 203), 4, (0, 1, 2, 3))]
    group = runner._last_physical_group_plan["groups"][0]
    assert group["execution_path"] == "packed_native"
    assert group["physical_rows"] == 4
    assert "serial_decode_fallback" not in runner._last_execution_manifest


def test_gguf_resident_runner_compaction_flushes_and_invalidates_slot_bound_graphs() -> None:
    events: list[tuple] = []

    class FakeGraph:
        closed = False

    graph = FakeGraph()

    class FakeSession:
        def __init__(self, session_id: int) -> None:
            self.session_id = int(session_id)
            self.allocation = SimpleNamespace(base_ptr=0xA000 + session_id)
            self.state_identity = object()

        def invalidate_device_kv_graphs(self) -> int:
            events.append(("invalidate", self.session_id))
            if graph.closed:
                return 0
            graph.closed = True
            return 1

    sessions = (FakeSession(0), FakeSession(1))
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = SimpleNamespace(target_arch="gfx1100")
    runner._rows = {
        10: SimpleNamespace(lease=SimpleNamespace(session=sessions[0])),
        11: SimpleNamespace(lease=SimpleNamespace(session=sessions[1])),
    }
    runner._kv_graph_invalidation_count = 0
    runner._flush_all_packed_owners = lambda: events.append(("flush",))
    runner._graph_handles_for_sessions = lambda observed: (
        events.append(("handles", tuple(session.session_id for session in observed)))
        or (graph,)
    )
    runner._observe_graph_handles = lambda observed: events.append(
        ("observe", tuple(session.session_id for session in observed))
    )
    runner._record_graph_invalidations = lambda handles, count: events.append(
        ("record", tuple(handles), int(count))
    )
    identities_before = tuple(
        (id(session), id(session.allocation), session.allocation.base_ptr, id(session.state_identity))
        for session in sessions
    )

    runner.compact_batch(
        (
            SlotMove(request_id=10, old_slot=2, new_slot=0),
            SlotMove(request_id=11, old_slot=4, new_slot=1),
        )
    )

    assert events[0] == ("flush",)
    assert ("handles", (0, 1)) in events
    assert ("observe", (0, 1)) in events
    assert [event for event in events if event[0] == "invalidate"] == [
        ("invalidate", 0),
        ("invalidate", 1),
    ]
    assert [event for event in events if event[0] == "record"] == [
        ("record", (graph,), 1)
    ]
    assert graph.closed
    assert runner._kv_graph_invalidation_count == 1
    assert tuple(
        (id(session), id(session.allocation), session.allocation.base_ptr, id(session.state_identity))
        for session in sessions
    ) == identities_before

    events.clear()
    runner.compact_batch((SlotMove(request_id=10, old_slot=0, new_slot=0),))
    assert events == []


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

    with pytest.raises(GenerationAdmissionRejected, match="high-water") as rejected:
        runner.reserve_admission(SimpleNamespace(request_id=3))

    assert rejected.value.resource == "device_kv_pool"
    assert rejected.value.requested_units == 1
    assert rejected.value.current_units == 2
    assert rejected.value.capacity_units == 2
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


def test_gguf_resident_runner_retains_packed_workspace_across_reclaim() -> None:
    """Owner-shared packed workspace survives row release at any capacity.

    CONCURRENCY2.md: workspaces are ledger-reserved and reused by
    non-overlapping work; hot-path HIP allocation is disabled in the promoted
    server mode. Releasing the ~1 GiB owner slab per reclaim forces a
    same-size reallocation on the next packed step (the accepted C2-6 packet
    recorded 246 releases / 242.39 GiB cumulative churn).
    """

    events: list[str] = []

    class FakeSession:
        _decode_graphs: list[object] = []
        _device_kv_graph_handles: dict[int, object] = {}

        def invalidate_device_kv_graphs(self) -> int:
            events.append("invalidate")
            return 0

        def reset(self) -> None:
            events.append("reset")

        def release_idle_packed_workspace(self) -> int:
            events.append("release_packed")
            return 1234

    def _release_row(capacity: int, request_id: int, prompt: str):
        session = FakeSession()
        row = qwen35_gguf._GGUFResidentLoopRow(
            request_id=request_id,
            batch_id=request_id - 1,
            row_index=0,
            request=_request(prompts=(prompt,), max_tokens=1, ignore_eos=True),
            prompt_ids=(10, 11),
            native_greedy=True,
            native_sampled=False,
            submitted_at=0.0,
            lease=qwen35_gguf._GGUFResidentSessionLease(
                session=session,
                pool_key=("continuous_ar_dynamic_kv", True, True, 256),
            ),
        )
        runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
            qwen35_gguf.Qwen35GGUFResidentModelRunner
        )
        runner.capacity = capacity
        runner._prefix_cache = None
        runner._prefix_state_snapshots = {}
        runner._available = []
        runner._kv_pool = None
        runner._kv_graph_invalidation_count = 0
        runner._packed_workspace_release_events = 0
        runner._packed_workspace_released_bytes = 0
        runner._graph_handles_for_sessions = lambda sessions: ()
        runner._observe_graph_handles = lambda sessions: None
        runner._sample_kv_hip_memory = lambda: None
        runner._release_row_resources(row)
        return runner, row

    for capacity in (2, 4, 8):
        events.clear()
        runner, row = _release_row(capacity, capacity, f"prompt-{capacity}")
        assert events == ["invalidate", "reset"]
        assert "release_packed" not in events
        assert row.lease is None
        assert len(runner._available) == 1
        assert runner._packed_workspace_release_events == 0
        assert runner._packed_workspace_released_bytes == 0


def test_gguf_resident_runner_waits_for_stable_membership_before_graph_capture() -> None:
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    pending = SimpleNamespace(native_greedy=True, native_sampled=False, slot=None)
    runner._rows = {
        1: SimpleNamespace(native_greedy=True, native_sampled=False, slot=object()),
        2: pending,
    }

    assert runner._packed_graph_capture_membership_stable() is False
    pending.slot = object()
    assert runner._packed_graph_capture_membership_stable() is True


@pytest.mark.parametrize(
    ("done_values", "expected_events"),
    [
        ((True, True), ("close", "invalidate", "discard")),
        ((True, False), ("flush", "close", "invalidate", "owner_flush")),
    ],
    ids=("all_done", "live_survivor"),
)
def test_gguf_resident_runner_discards_only_terminal_packed_state(
    done_values: tuple[bool, bool],
    expected_events: tuple[str, ...],
) -> None:
    events: list[str] = []
    owner = SimpleNamespace(
        discard_packed_decode_state=lambda: events.append("discard"),
    )

    class FakeGraph:
        sessions = ()
        closed = False

        def flush_packed_state(self) -> bool:
            events.append("flush")
            return True

        def close(self) -> None:
            self.closed = True
            events.append("close")

    graph = FakeGraph()
    slots = [
        SimpleNamespace(
            packed_decode_owner=owner,
            packed_decode_graph=graph,
            done=done,
        )
        for done in done_values
    ]
    rows = [SimpleNamespace(slot=slot) for slot in slots]
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner._rows = {index: row for index, row in enumerate(rows)}
    runner._kv_graph_invalidation_count = 0
    runner._record_graph_invalidations = lambda handles, count: events.append(
        "invalidate"
    )

    def flush_owners(concrete) -> None:
        events.append("owner_flush")
        for slot in concrete:
            slot.packed_decode_owner = None

    runner.generator = SimpleNamespace(
        _flush_ar_packed_decode_owners=flush_owners,
    )

    runner._flush_row_owner(rows[0])

    assert events == list(expected_events)
    assert all(slot.packed_decode_graph is None for slot in slots)
    assert all(slot.packed_decode_owner is None for slot in slots)


def test_gguf_resident_runner_captures_replays_and_closes_packed_graph() -> None:
    events: list[tuple] = []

    class FakeGraph:
        def __init__(self, sessions) -> None:
            self.sessions = tuple(sessions)
            self.closed = False
            self.flushed = False
            self.replay_count = 0
            self.replayed_steps = 0
            self.execution_manifest = {
                "schema": 1,
                "kind": "gguf_packed_ar_execution_manifest",
                "mode": "decode_graph_replay",
                "graph": {
                    "captured": True,
                    "replay_count": 0,
                    "transport": {
                        "transport": "pm4",
                        "executable": {
                            "nodes": 2,
                            "module_records": [{"index": 0}],
                            "dispatch_records": [{"index": 0}, {"index": 1}],
                        },
                    },
                },
            }

        def replay(self, steps: int) -> None:
            self.replay_count += int(steps)
            self.replayed_steps += int(steps)
            self.execution_manifest["graph"]["replay_count"] = self.replay_count
            for session in self.sessions:
                session.position += int(steps)
            events.append(("replay", int(steps)))

        def read_latest_generated_token_ids(self):
            events.append(("read_latest",))
            return [41, 42]

        def flush_packed_state(self) -> bool:
            self.flushed = True
            events.append(("flush",))
            return True

        def close(self) -> None:
            self.closed = True
            events.append(("close",))

    class FakeSession:
        def __init__(self, position: int) -> None:
            self.position = int(position)
            self._decode_graphs = []
            self.step_batch_native = lambda *args, **kwargs: pytest.fail(
                "eager packed decode ran after graph admission"
            )

        @staticmethod
        def decode_graph_min_replay_steps() -> int:
            return 128

        @staticmethod
        def packed_decode_graph_min_replay_steps(physical_rows: int) -> int:
            assert physical_rows == 2
            return 23

    sessions = (FakeSession(512), FakeSession(512))
    graph = FakeGraph(sessions)
    capture_kwargs: dict[str, object] = {}

    def capture(token_ids, **kwargs):
        capture_kwargs["token_ids"] = tuple(token_ids)
        capture_kwargs.update(kwargs)
        for session in sessions:
            session._decode_graphs.append(graph)
        return graph

    sessions[0].capture_packed_decode_graph = capture
    slots = [
        SimpleNamespace(
            session=session,
            prev_token=10 + index,
            seq_position=512,
            generated_ids=[20 + index],
            c1_decode_graph=None,
            packed_decode_graph=None,
            packed_decode_graph_unavailable=False,
            packed_decode_owner=None,
            native_decode_steps=0,
        )
        for index, session in enumerate(sessions)
    ]
    request = _request(prompts=("first", "second"), max_tokens=24, ignore_eos=True)
    rows = [
        SimpleNamespace(
            request_id=index + 1,
            request=request,
            native_greedy=True,
            native_sampled=False,
            native_sampler=False,
            slot=slot,
            samples=[],
            sampling_request=None,
            sampling_state=None,
        )
        for index, slot in enumerate(slots)
    ]
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner.generator = SimpleNamespace(
        tokenizer=_FakeTokenizer(),
        _flush_ar_packed_decode_owners_if_chunk_changed=lambda chunk: None,
        _flush_ar_packed_decode_owners=lambda chunk: None,
    )
    runner._route_counts = Counter()
    runner._last_execution_manifest = {}
    runner._rows = {row.request_id: row for row in rows}
    runner._kv_graph_invalidation_count = 0
    runner._close_c1_decode_graph = lambda row: None
    runner._observe_graph_handles = lambda sessions: events.append(
        ("observe", tuple(sessions))
    )
    runner._record_graph_invalidations = lambda handles, count: events.append(
        ("invalidate", tuple(handles), int(count))
    )

    assert runner._step_native_chunk(
        rows,
        physical_rows=2,
        active_slot_indices=(0, 1),
    ) is True
    assert capture_kwargs == {
        "token_ids": (10, 11),
        "sessions": sessions,
        "physical_rows": 2,
        "active_slot_indices": (0, 1),
        "steps_per_replay": 1,
        "max_replay_steps": 23,
        "record_steps": 23,
    }
    assert [slot.generated_ids[-1] for slot in slots] == [41, 42]
    assert all(slot.packed_decode_graph is graph for slot in slots)
    assert runner._route_counts["native_packed_graph_captures"] == 1
    assert runner._route_counts["native_packed_graph_replays"] == 1
    observed_executable = runner._last_execution_manifest["graph"]["transport"]["executable"]
    assert observed_executable == {
        "nodes": 2,
        "module_record_count": 1,
        "dispatch_record_count": 2,
        "records_omitted": True,
    }
    assert graph.execution_manifest["graph"]["transport"]["executable"][
        "dispatch_records"
    ] == [{"index": 0}, {"index": 1}]

    runner._close_packed_decode_graphs(rows)

    assert graph.flushed is True
    assert graph.closed is True
    assert all(slot.packed_decode_graph is None for slot in slots)
    assert runner._kv_graph_invalidation_count == 1
    assert events == [
        ("replay", 1),
        ("read_latest",),
        ("observe", sessions),
        ("observe", sessions),
        ("flush",),
        ("close",),
        ("invalidate", (graph,), 1),
    ]


def test_gguf_resident_runner_counts_c1_graph_close_as_invalidation() -> None:
    events: list[tuple] = []

    class FakeGraph:
        closed = False

        def close(self) -> None:
            self.closed = True
            events.append(("close",))

    graph = FakeGraph()
    session = SimpleNamespace()
    row = SimpleNamespace(
        slot=SimpleNamespace(c1_decode_graph=graph),
        lease=SimpleNamespace(session=session),
    )
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner._kv_graph_invalidation_count = 0
    runner._graph_handles_for_sessions = lambda sessions: (graph,)
    runner._observe_graph_handles = lambda sessions: events.append(
        ("observe", tuple(sessions))
    )
    runner._record_graph_invalidations = lambda handles, count: events.append(
        ("record", tuple(handles), int(count))
    )

    runner._close_c1_decode_graph(row)

    assert events == [
        ("observe", (session,)),
        ("close",),
        ("record", (graph,), 1),
    ]
    assert runner._kv_graph_invalidation_count == 1
    assert row.slot.c1_decode_graph is None


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


def test_gguf_resident_runner_skips_sampling_for_nonfinal_prefill_chunks() -> None:
    calls: list[dict[str, object]] = []

    class FakeSession:
        def prefill_batch_native(self, prompts, **kwargs):
            calls.append({"prompts": prompts, **kwargs})
            return [None]

    row = qwen35_gguf._GGUFResidentLoopRow(
        request_id=1,
        batch_id=1,
        row_index=0,
        request=_request(prompts=("long",), max_tokens=8, ignore_eos=True),
        prompt_ids=(10, 11, 12, 13),
        native_greedy=True,
        native_sampled=False,
        submitted_at=0.0,
        lease=qwen35_gguf._GGUFResidentSessionLease(
            session=FakeSession(),
            pool_key=("continuous_ar_dynamic_kv", True, True, 256),
        ),
    )
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner._route_counts = Counter()
    runner._refresh_prefix_cache = lambda row: None

    runner._prefill_native_chunk(row, (10, 11), final_chunk=False)

    assert calls == [
        {
            "prompts": [(10, 11)],
            "sessions": [row.lease.session],
            "full_prompt_lengths": [4],
            "return_logits": False,
            "return_hidden_seeds": False,
            "sample_output": False,
        }
    ]
    assert runner._route_counts["native_incremental_prefill_chunks"] == 1
    assert runner._route_counts["native_incremental_prefill_unsampled_chunks"] == 1
    assert row.slot is None


def test_gguf_resident_direct_int8_buffers_scheduler_chunks_for_exact_full_prefill() -> None:
    events: list[str] = []

    class FakeSession:
        kv_attention_source = "int8_direct"

        def reset(self) -> None:
            events.append("reset")

        def prefill_batch_native(self, *args, **kwargs):
            pytest.fail("direct INT8 must not release its exact oracle between chunks")

    row = qwen35_gguf._GGUFResidentLoopRow(
        request_id=1,
        batch_id=1,
        row_index=0,
        request=_request(prompts=("long",), max_tokens=8, ignore_eos=True),
        prompt_ids=(10, 11, 12, 13),
        native_greedy=True,
        native_sampled=False,
        submitted_at=0.0,
        incremental_prefill=True,
        lease=qwen35_gguf._GGUFResidentSessionLease(
            session=FakeSession(),
            pool_key=("continuous_ar_dynamic_kv", True, True, 256),
        ),
    )
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner._fallback_reasons = Counter()

    runner._prefill_native_chunk(row, (10, 11), final_chunk=False)

    assert events == ["reset"]
    assert row.incremental_prefill is False
    assert row.prefill_chunk_count == 0
    assert runner._fallback_reasons["int8_direct_full_prompt_prefill"] == 1


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
    assert outputs[0].telemetry is not None
    assert outputs[0].telemetry.timing is not None
    assert outputs[0].telemetry.timing["tokenize_ms"] == 0.0


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


def test_gguf_ar_has_no_rejected_stream_prefill_route() -> None:
    assert not hasattr(
        qwen35_gguf.Qwen35GGUFBringupGenerator,
        "_try_prefill_ar_serving_slots_streams",
    )
    assert not hasattr(qwen35_gguf.Qwen35GGUFResidentSession, "prefill_async_top1")
    generation_source = Path(qwen35_gguf.__file__).read_text()
    assert "HIPENGINE_GGUF_AR_STREAM_PREFILL" not in generation_source


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


def test_gguf_ar_packed_prefill_batches_unequal_prompt_lengths(monkeypatch) -> None:
    """Mixed-length prompts must group into one native prefill call.

    Measured context: C1-C8 admission is flat at ~305 ms per lane while llama.cpp
    batches its wave (1.41x behind at C1, 2.88x at C8), and admission parity alone
    wins every AR cell.

    This pins the *serving* route (``generate_detailed`` -> the packed prefill call
    site around ``qwen35_gguf.py:3025``), which groups eight prompts of three
    distinct lengths into one ``prefill_batch_native`` call today. It exists so a
    future "equal lengths only" narrowing of that route fails loudly: the resident
    AR route's ``_try_prefill_native_work_batch`` guard *does* require equal chunk
    lengths, and that stricter policy - not a runner limitation - is what keeps the
    C1-C8 matrix on one-prompt-at-a-time prefill. Keep this test green.
    """

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
            calls.append(("scalar_prefill", self.slot_id, tuple(int(t) for t in token_ids)))
            self.position = len(token_ids)
            return SimpleNamespace(token_id=1)

        def prefill_batch_native(self, prompt_token_ids, *, sessions, return_logits=False):
            calls.append(
                (
                    "prefill_batch",
                    self.slot_id,
                    tuple(tuple(int(token) for token in prompt) for prompt in prompt_token_ids),
                    tuple(session.slot_id for session in sessions),
                    return_logits,
                )
            )
            for session, prompt in zip(sessions, prompt_token_ids, strict=True):
                session.position = len(prompt)
            return [SimpleNamespace(token_id=1) for _session in sessions]

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", FakeFullStackRunner)
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_PREFILL", "1")

    generator = _generator()
    generator.prepare()
    mixed = ("long", "first", "second", "long", "first", "second", "long", "first")
    outputs = generator.generate_detailed(_request(prompts=mixed, max_tokens=1))

    assert [output.text for output in outputs] == ["B"] * 8
    batch_calls = [call for call in calls if call[0] == "prefill_batch"]
    scalar_calls = [call for call in calls if call[0] == "scalar_prefill"]
    grouped_rows = {len(prompt) for call in batch_calls for prompt in call[2]}

    # Target contract, currently unmet: one grouped call carrying every lane.
    assert len(batch_calls) == 1 and len(batch_calls[0][2]) == 8, (
        f"mixed-length wave was not grouped: {len(batch_calls)} batch call(s), "
        f"{len(scalar_calls)} scalar call(s)"
    )
    assert grouped_rows == {4, 2, 1}, f"grouped call did not carry all three lengths: {grouped_rows}"


def test_resident_ar_packed_prefill_survives_one_over_long_lane(monkeypatch) -> None:
    """RED: one chunked lane must not cost grouping for every other lane.

    ``next_prefill_batch_work`` selects the first ``max_rows`` active requests
    regardless of length and advances every prefill cursor, then
    ``_try_prefill_native_work_batch`` refuses the **whole** work item if any row
    fails its per-row checks - including ``chunk == row.prompt_ids``, which a prompt
    longer than ``DEFAULT_MAX_PREFILL_CHUNK_TOKENS`` (256) always violates because its
    chunk is truncated. A single long lane therefore drops a 7-lane grouped prefill
    back to serial, and the standard 10-prompt suite (35-67 tokens) cannot observe
    it. Target contract: the compatible subset groups and the long lane is left to
    the chunked path.
    """

    from hipengine.dispatch.batch import WorkItem, WorkKind
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

    long_prompt = tuple(range(1000, 1300))          # 300 tokens > 256-token chunk
    short_prompt = (10, 11, 12, 13, 14, 15, 16, 17)
    prompts = {i: (long_prompt if i == 0 else short_prompt) for i in range(8)}

    class FakeSession:
        def __init__(self, slot_id):
            self.slot_id = slot_id
            self.position = 0

    class FakeLease:
        def __init__(self, slot_id):
            self.session = FakeSession(slot_id)

    class FakeOwner:
        def __init__(self):
            self.calls: list[tuple] = []

        def prefill_batch_native(
            self, prompt_token_ids, *, sessions, full_prompt_lengths=None, **kwargs
        ):
            self.calls.append(
                tuple(tuple(int(t) for t in prompt) for prompt in prompt_token_ids)
            )
            return [SimpleNamespace(token_id=1) for _ in sessions]

    rows = {}
    for request_id, prompt in prompts.items():
        rows[request_id] = SimpleNamespace(
            request_id=request_id,
            prompt_ids=prompt,
            request=SimpleNamespace(deadline_at=None, cancellation_token=None),
            lease=FakeLease(request_id),
            native_greedy=True,
            slot=None,
            prefill_tokens_seen=0,
            prefix_reused_tokens=0,
            mtp2_candidate_budget=0,
            incremental_prefill=True,
            prefill_ms=0.0,
            prefill_chunk_count=0,
        )

    owner = FakeOwner()
    runner = Qwen35GGUFResidentModelRunner.__new__(Qwen35GGUFResidentModelRunner)
    runner.packed_prefill_max_rows = 8
    runner._route_counts = Counter(native_full_prefill_rows=0)
    runner._row = lambda request_id: rows[int(request_id)]
    runner._packed_execution_owner = lambda session: owner
    runner._begin_mtp2_prompt_streaming = lambda _rows: [None] * len(prompts)
    runner._finish_mtp2_prompt_streaming = lambda *args, **kwargs: None
    runner._refresh_prefix_cache = lambda _row: None
    runner._finish_native_prefill = lambda *args, **kwargs: None
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_PREFILL", "1")

    # What the scheduler emits: the long lane arrives as a truncated 256-token chunk.
    token_rows = tuple(
        prompts[r][:256] if r == 0 else prompts[r] for r in prompts
    )
    work = WorkItem(
        kind=WorkKind.PREFILL,
        request_ids=tuple(prompts),
        row_to_request=tuple(prompts),
        token_rows=token_rows,
    )
    handled = runner._try_prefill_native_work_batch(work)

    assert handled == frozenset(range(1, 8)), (
        "a single over-long lane refused the whole item; every compatible lane fell "
        f"back to serial prefill (handled={handled!r})"
    )
    assert len(owner.calls) == 1 and len(owner.calls[0]) == 7, (
        f"expected the 7 compatible lanes to group, got {len(owner.calls)} call(s)"
    )
    assert rows[0].prefill_tokens_seen == 0, (
        "the over-long lane was consumed by the grouped call; it must stay on the "
        "chunked serial path"
    )
    assert runner._route_counts["native_full_prefill_rows"] == 7
    assert runner._route_counts["native_full_prefill_groups"] == 1


def test_gguf_gfx1100_packed_prefill_capability_is_declared() -> None:
    """gfx1100 ships grouped prefill; this replaces the earlier "undeclared" tripwire.

    The tripwire existed because ``engine_loop`` selects ``next_prefill_batch_work``
    only when ``runner.packed_prefill_max_rows > 1``, and no backend package declared
    ``GGUF_C2_PACKED_PREFILL_MAX_ROWS``, so the C1-C8 matrix prefilled one request at a
    time. The tripwire said to flip it only after qualifying grouping against the strict
    per-request chain; that qualification is now measured on W7900 / Qwen3.8-27B
    Q4_K_M with the canonical mtpbench suite (protocol byte-identical to the retained
    pre-declaration packet): 432 cross-packet per-row id comparisons with 0 mismatches,
    80/80 correctness cells, width 1 unchanged (-0.4% AR, acceptance identical at
    0.789), C8 AR 45.68 -> 78.67 tok/s, and the width-dependent draft-acceptance
    collapse removed (0.467-0.614 at C2-C8 -> 0.789 at every width). Grouping is also
    directly observed: ``native_full_prefill_groups`` is 10 at C4/C8 and 0 at C1.

    The protective half of the tripwire is kept: the lookup still defaults to 1, so a
    package that has *not* qualified grouping stays lane-by-lane.
    """

    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.backends import backend_package_capability

    assert backend_package_capability(
        "hip_gfx1100", "GGUF_C2_PACKED_PREFILL_MAX_ROWS", 1
    ) == 8, "grouped prefill regressed to one row per call"
    assert "GGUF_C2_PACKED_PREFILL_MAX_ROWS" in hip_gfx1100.__all__
    # Unqualified packages must keep the lane-by-lane default, not inherit 8.
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_C2_PACKED_PREFILL_MAX_ROWS_NOT_A_REAL_CAPABILITY", 1
    ) == 1


def test_gguf_packed_prefill_without_native_owner_returns_a_container(monkeypatch) -> None:
    """Regression: the no-native-owner path must return a container, never a bool.

    ``prefill_batch`` consumes the result as ``int(request_id) in handled``, so the
    ``return False`` that survived the subset-grouping change raised
    ``TypeError: argument of type 'bool' is not iterable`` for any wave whose packed
    execution owner does not expose ``prefill_batch_native`` - a live path now that
    gfx1100 declares ``GGUF_C2_PACKED_PREFILL_MAX_ROWS`` and the batch route is reached
    by default. The correct behaviour is to handle nothing and let the caller prefill
    each request serially.
    """

    from hipengine.dispatch.batch import WorkItem, WorkKind
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

    prompts = {0: (10, 11, 12, 13), 1: (20, 21), 2: (30,)}

    class FakeSession:
        def __init__(self, slot_id):
            self.slot_id = slot_id
            self.position = 0

    class FakeLease:
        def __init__(self, slot_id):
            self.session = FakeSession(slot_id)

    class OwnerWithoutNativeBatch:
        """Deliberately has no ``prefill_batch_native`` attribute."""

    rows = {}
    for request_id, prompt in prompts.items():
        rows[request_id] = SimpleNamespace(
            request_id=request_id,
            prompt_ids=prompt,
            request=SimpleNamespace(deadline_at=None, cancellation_token=None),
            lease=FakeLease(request_id),
            native_greedy=True,
            slot=None,
            prefill_tokens_seen=0,
            prefix_reused_tokens=0,
            mtp2_candidate_budget=0,
            incremental_prefill=True,
            prefill_ms=0.0,
            prefill_chunk_count=0,
        )

    runner = Qwen35GGUFResidentModelRunner.__new__(Qwen35GGUFResidentModelRunner)
    runner.packed_prefill_max_rows = 8
    runner._route_counts = Counter(native_full_prefill_rows=0)
    runner._row = lambda request_id: rows[int(request_id)]
    runner._packed_execution_owner = lambda session: OwnerWithoutNativeBatch()
    runner._begin_mtp2_prompt_streaming = lambda _rows: [None] * len(prompts)
    runner._finish_mtp2_prompt_streaming = lambda *args, **kwargs: None
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_PREFILL", "1")

    work = WorkItem(
        kind=WorkKind.PREFILL,
        request_ids=tuple(prompts),
        row_to_request=tuple(prompts),
        token_rows=tuple(prompts[r] for r in prompts),
    )
    handled = runner._try_prefill_native_work_batch(work)

    assert isinstance(handled, frozenset), f"returned {type(handled).__name__}: {handled!r}"
    assert handled == frozenset(), f"claimed to handle rows it never touched: {handled!r}"
    # This is the expression the caller uses; it must not raise.
    assert all(int(request_id) not in handled for request_id in prompts)
    assert runner._route_counts["native_full_prefill_groups"] == 0


def test_resident_ar_packed_prefill_groups_mixed_prompt_lengths(monkeypatch) -> None:
    """RED: the resident AR route must group a mixed-length wave like the serving route.

    The C1-C8 matrix prefills one request per step (measured: 305-335 ms per lane,
    flat in width, vs llama.cpp's 216-436 ms for the whole wave), which is why
    admission leaves 1.41x at C1 and 2.88x at C8 - and admission parity alone flips
    every AR cell. Two independent switches keep the matrix serial: the undeclared
    ``GGUF_C2_PACKED_PREFILL_MAX_ROWS`` capability (covered by the tripwire above)
    and this function's equal-chunk-length requirement. The serving route reaches the
    same entry point with arbitrary lengths, and the call already forwards
    ``full_prompt_lengths`` per row, so the guard is stricter than the ABI.

    Target contract: a lease-backed wave of three different prompt lengths groups
    into one ``prefill_batch_native`` call. Today the guard refuses and the route
    falls back to per-request prefill, which is what this test currently records.
    """

    from hipengine.dispatch.batch import WorkItem, WorkKind
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

    prompts = {
        0: (10, 11, 12, 13),
        1: (20, 21),
        2: (30,),
    }

    class FakeSession:
        def __init__(self, slot_id):
            self.slot_id = slot_id
            self.position = 0

    class FakeLease:
        def __init__(self, slot_id):
            self.session = FakeSession(slot_id)

    class FakeOwner:
        def __init__(self):
            self.calls: list[tuple] = []

        def prefill_batch_native(
            self, prompt_token_ids, *, sessions, full_prompt_lengths=None, **kwargs
        ):
            self.calls.append(
                (
                    tuple(tuple(int(t) for t in prompt) for prompt in prompt_token_ids),
                    tuple(session.slot_id for session in sessions),
                    None if full_prompt_lengths is None else tuple(full_prompt_lengths),
                )
            )
            return [SimpleNamespace(token_id=1) for _ in sessions]

    rows = {}
    for request_id, prompt in prompts.items():
        rows[request_id] = SimpleNamespace(
            request_id=request_id,
            prompt_ids=prompt,
            request=SimpleNamespace(deadline_at=None, cancellation_token=None),
            lease=FakeLease(request_id),
            native_greedy=True,
            slot=None,
            prefill_tokens_seen=0,
            prefix_reused_tokens=0,
            mtp2_candidate_budget=0,
            incremental_prefill=True,
            prefill_ms=0.0,
            prefill_chunk_count=0,
        )

    owner = FakeOwner()
    runner = Qwen35GGUFResidentModelRunner.__new__(Qwen35GGUFResidentModelRunner)
    runner.packed_prefill_max_rows = 8
    runner._route_counts = Counter(native_full_prefill_rows=0)
    runner._row = lambda request_id: rows[int(request_id)]
    runner._packed_execution_owner = lambda session: owner
    runner._begin_mtp2_prompt_streaming = lambda _rows: [None] * len(prompts)
    runner._finish_mtp2_prompt_streaming = lambda *args, **kwargs: None
    runner._refresh_prefix_cache = lambda _row: None
    runner._finish_native_prefill = lambda *args, **kwargs: None
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_PREFILL", "1")

    work = WorkItem(
        kind=WorkKind.PREFILL,
        request_ids=tuple(prompts),
        row_to_request=tuple(prompts),
        token_rows=tuple(prompts[r] for r in prompts),
    )
    handled = runner._try_prefill_native_work_batch(work)

    assert handled == frozenset(prompts), (
        f"mixed-length wave was refused and fell back to serial prefill: {handled!r}"
    )
    assert len(owner.calls) == 1, f"expected one grouped call, got {len(owner.calls)}"
    carried, slots, lengths = owner.calls[0]
    assert carried == tuple(prompts[r] for r in prompts)
    assert slots == (0, 1, 2)
    assert lengths == (4, 2, 1), f"per-row lengths were not forwarded: {lengths}"
    assert runner._route_counts["native_full_prefill_rows"] == 3
    # The group counter is what lets a packet prove a wave grouped: rows alone is also
    # bumped by single-request prefill.
    assert runner._route_counts["native_full_prefill_groups"] == 1


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
    generator.server_plain_ar_max_active_requests = 4
    result = generator.prepare_request_scratch(max_prompt_tokens=4, max_batch_size=8)

    assert result["max_batch_size"] == 8
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


def test_specdec2_streaming_prompt_operator_rollback_is_default_on(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_SPECDEC2_STREAMING_PROMPT", raising=False)
    assert qwen35_gguf._gguf_specdec2_streaming_prompt_enabled()

    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_STREAMING_PROMPT", "0")
    assert not qwen35_gguf._gguf_specdec2_streaming_prompt_enabled()


def test_specdec2_streaming_prompt_rollback_selects_k0_before_prefill(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_STREAMING_PROMPT", "0")
    row = SimpleNamespace(
        mtp2_candidate_budget=2,
        prefix_reused_tokens=0,
        mtp2_prompt_fallback_reason=None,
    )
    runner = object.__new__(qwen35_gguf.Qwen35GGUFResidentModelRunner)

    assert runner._begin_mtp2_prompt_streaming((row,)) == (None,)
    assert row.mtp2_candidate_budget == 0
    assert row.mtp2_prompt_fallback_reason == "operator_disabled_streaming_prompt_k0"


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


def test_gguf_tool_constraint_masks_disabled_thinking_and_undeclared_tool(monkeypatch) -> None:
    class ToolTokenizer(_FakeTokenizer):
        def encode(self, prompt: str) -> list[int]:
            if prompt in {"first", "second", "long", "long2", "{", "}"}:
                return super().encode(prompt)
            return []

        def decode(self, ids) -> str:
            table = {
                0: "<think>",
                1: "<tool_call>",
                2: '{"name":"read","arguments":',
                3: '{"path":"README.md"}',
                4: "}</tool_call>",
                5: '{"name":"write","arguments":',
                99: "<eos>",
            }
            return "".join(table[int(token)] for token in ids)

    class FakeSession:
        step_index = 0

        def __init__(self, model_path, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits=True):
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            logits[0, 0] = 10.0
            logits[0, 1] = 9.0
            return SimpleNamespace(token_id=0, logits=logits)

        def step(self, token_id: int, *, return_logits=True):
            self.step_index += 1
            logits = np.full((1, 100), -10.0, dtype=np.float32)
            if self.step_index == 1:
                logits[0, 5] = 10.0
                logits[0, 2] = 9.0
            elif self.step_index == 2:
                logits[0, 3] = 9.0
            else:
                logits[0, 4] = 9.0
            return SimpleNamespace(token_id=int(np.argmax(logits[0])), logits=logits)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()
    generator.tokenizer = ToolTokenizer()
    constraint = ToolCallConstraintSpec(
        tool_names=("read",),
        mode="auto",
        forbidden_text_prefixes=("<think>",),
    )

    outputs = generator.generate_detailed(
        _request(
            max_tokens=4,
            tool_call_constraint=constraint,
            stop_token_sequences=((4,),),
        )
    )

    assert outputs[0].text == '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'
    assert outputs[0].generated_token_ids == (1, 2, 3, 4)
    decode_state = _decode_state(outputs[0])
    assert decode_state["sampler_mode"] == "processed_argmax"
    assert decode_state["active_processors"] == ["stop_token_sequences", "tool_call_constraint"]
    assert decode_state["sampler_fast_path_blockers"] == ["stop_token_sequences", "tool_call_constraint"]


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
        "native_c1_decode_steps": 0,
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


def test_gguf_generic_quant_keeps_multi_prompt_serial_fallback(monkeypatch) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            calls.append(("init", str(model_path), kwargs))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill(self, token_ids, *, return_logits):
            calls.append(("prefill", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(token_id=1)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()
    generator.native_batch_decode = False

    outputs = generator.generate_detailed(
        _request(prompts=("first", "second"), max_tokens=1)
    )

    assert [output.text for output in outputs] == ["B", "B"]
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
        ("prefill", (10, 11), False),
        ("prefill", (20,), False),
    ]
    assert generator.last_batch_generation["path"] == "gguf_serial_greedy_decode"
    assert generator.last_batch_generation["serial_decode_fallback"] is True


def test_gguf_generate_detailed_uses_native_compact_rows_for_greedy_prompts(
    monkeypatch,
) -> None:
    calls = []

    class FakeSession:
        def __init__(self, model_path, **kwargs):
            calls.append(("init", kwargs))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill_slot(self, token_ids, *, slot, return_logits):
            calls.append(
                (
                    "prefill_slot",
                    tuple(token_ids),
                    int(slot),
                    bool(return_logits),
                )
            )
            return SimpleNamespace(token_id=1)

        def step_rows_native(self, token_ids, *, return_logits):
            calls.append(("step_rows_native", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_ids=(2, 2),
                execution_paths={
                    "linear_attention": "indexed_conv_gdn",
                    "full_attention": "kv_live_spans_batch_c1_exact",
                    "moe": "selected_rows_batch",
                    "lm_head": "row_linear_f32",
                    "sampler": "argmax_rows_i32",
                },
            )

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()
    generator.native_batch_decode = True

    outputs = generator.generate_detailed(
        _request(prompts=("first", "second"), max_tokens=2)
    )

    assert [output.text for output in outputs] == ["BC", "BC"]
    assert calls == [
        ("init", {"max_sequence_length": 256, "max_batch_size": 2}),
        ("prefill_slot", (10, 11), 0, False),
        ("prefill_slot", (20,), 1, False),
        ("step_rows_native", (1, 1), False),
    ]
    batch = generator.last_batch_generation
    assert batch is not None
    assert batch["path"] == "gguf_native_continuous_decode"
    assert batch["native_decode_steps"] == 1
    assert batch["serial_decode_fallback"] is False
    assert batch["native_caware_decode"] is True
    assert batch["native_sampler_rows"] is True
    assert batch["throughput_claim_eligible"] is True
    assert batch["native_execution_paths"]["linear_attention"] == "indexed_conv_gdn"
    assert all(
        chunk["chunk"]["telemetry"]["decode_state"]["native_caware_decode"]
        for chunk in batch["scheduler_token_chunks"]
    )


def test_gguf_greedy_prompt_batch_replays_native_row_graph(monkeypatch) -> None:
    calls = []

    class FakeGraph:
        def __enter__(self):
            calls.append(("graph_enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("graph_exit", exc_type is None))

        def step(self, token_ids):
            calls.append(("graph_step", tuple(token_ids)))
            return SimpleNamespace(token_ids=(3, 3), execution_paths={"sampler": "argmax_rows_i32_graph"})

    class FakeSession:
        host_token_embedding_enabled = False
        target_layout = SimpleNamespace(max_sequence_length=256)

        def __init__(self, model_path, **kwargs):
            calls.append(("init", kwargs))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill_slot(self, token_ids, *, slot, return_logits):
            calls.append(("prefill_slot", tuple(token_ids), int(slot), bool(return_logits)))
            return SimpleNamespace(token_id=1)

        def step_rows_native(self, token_ids, *, return_logits):
            calls.append(("step_rows_native", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(token_ids=(2, 2), execution_paths={"sampler": "argmax_rows_i32"})

        def capture_native_rows_graph(self, *, rows, max_context_len):
            calls.append(("capture_native_rows_graph", int(rows), int(max_context_len)))
            return FakeGraph()

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()
    generator.native_batch_decode = True

    outputs = generator.generate_detailed(
        _request(prompts=("first", "second"), max_tokens=3)
    )

    assert [output.text for output in outputs] == ["BCD", "BCD"]
    assert ("step_rows_native", (1, 1), False) in calls
    assert ("capture_native_rows_graph", 2, 256) in calls
    assert ("graph_step", (2, 2)) in calls
    assert generator.last_batch_generation["native_decode_steps"] == 2


def test_gguf_native_scheduler_reclaims_compacts_and_readmits(monkeypatch) -> None:
    calls = []

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("graph_exit", exc_type is None))

        def step(self, token_ids):
            tokens = tuple(int(token) for token in token_ids)
            calls.append(("graph_step", tokens))
            outputs = {
                (2, 1): (3, 99),
                (3, 1): (4, 2),
            }[tokens]
            return SimpleNamespace(
                token_ids=outputs,
                execution_paths={"sampler": "argmax_rows_i32_graph"},
            )

    class FakeSession:
        host_token_embedding_enabled = False
        target_layout = SimpleNamespace(max_sequence_length=256)

        def __init__(self, model_path, **kwargs):
            calls.append(("init", kwargs))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill_slot(self, token_ids, *, slot, return_logits):
            calls.append(("prefill_slot", tuple(token_ids), int(slot), bool(return_logits)))
            return SimpleNamespace(token_id=1)

        def step_rows_native(self, token_ids, *, return_logits):
            calls.append(("step_rows_native", tuple(token_ids), bool(return_logits)))
            return SimpleNamespace(
                token_ids=(99, 2),
                execution_paths={
                    "linear_attention": "indexed_conv_gdn",
                    "full_attention": "kv_live_spans_batch_c1_exact",
                    "moe": "selected_rows_batch",
                    "lm_head": "row_linear_f32",
                    "sampler": "argmax_rows_i32",
                },
            )

        def capture_native_rows_graph(self, *, rows, max_context_len):
            calls.append(("capture_native_rows_graph", int(rows), int(max_context_len)))
            return FakeGraph()

        def compact_target_slots(self, source_slots):
            calls.append(("compact_target_slots", tuple(source_slots)))

        def step(self, token_id, *, return_logits):
            calls.append(("step", int(token_id), bool(return_logits)))
            return SimpleNamespace(token_id=int(token_id) + 1)

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _generator()
    generator.native_batch_decode = True
    generator.native_batch_capacity = 2

    outputs = generator.generate_detailed(
        _request(
            prompts=("first", "second", "third", "fourth"),
            max_tokens=4,
        )
    )

    assert [output.text for output in outputs] == ["B<eos>", "BCD}", "B<eos>", "BCD}"]
    assert calls == [
        ("init", {"max_sequence_length": 256, "max_batch_size": 2}),
        ("prefill_slot", (10, 11), 0, False),
        ("prefill_slot", (20,), 1, False),
        ("step_rows_native", (1, 1), False),
        ("compact_target_slots", (1,)),
        ("prefill_slot", (30,), 1, False),
        ("capture_native_rows_graph", 2, 256),
        ("graph_step", (2, 1)),
        ("compact_target_slots", (0,)),
        ("prefill_slot", (40,), 1, False),
        ("graph_step", (3, 1)),
        ("compact_target_slots", (1,)),
        ("step", 2, False),
        ("step", 3, False),
        ("graph_exit", True),
    ]
    batch = generator.last_batch_generation
    assert batch is not None
    assert batch["path"] == "gguf_native_continuous_decode"
    assert batch["serial_decode_fallback"] is False
    assert batch["native_decode_steps"] == 3
    scheduler = batch["continuous_scheduler"]
    assert scheduler["continuous_batching"] is True
    assert scheduler["capacity"] == 2
    assert scheduler["admission_count"] == 4
    assert scheduler["admission_waves"] == 3
    assert scheduler["reclaim_count"] == 4
    assert scheduler["compaction_events"] == 3
    assert scheduler["compacted_slot_moves"] == 2
    assert scheduler["mixed_prefill_decode_admissions"] == 2
    assert scheduler["active_c_histogram"] == {"1": 2, "2": 3}
    assert scheduler["single_row_tail_steps"] == 2
    assert set(scheduler["request_observability"]) == {"0", "1", "2", "3"}
    assert all(
        row["submitted_timestamp"] <= row["admitted_timestamp"] <= row["completion_timestamp"]
        for row in scheduler["request_observability"].values()
    )
    assert scheduler["graph_bucket_stats"] == {
        "entries": 1,
        "hits": 1,
        "misses": 1,
        "replay_kernel_hits": 2,
        "replay_hit_rate": 0.5,
        "miss_reasons": {"gguf_native_shape_absent": 1},
        "kernel_time_histogram_ns": {
            "le_10us": 0,
            "le_100us": 0,
            "le_1ms": 0,
            "le_10ms": 0,
            "gt_10ms": 0,
        },
    }
    assert scheduler["final_request_to_slot"] == {}


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
    assert [chunk.generated_token_ids for chunk in chunks] == [None, (1, 2)]
    for chunk in chunks:
        assert chunk.telemetry is not None
        assert chunk.telemetry.timing is not None
        assert chunk.telemetry.timing["tokenize_ms"] >= 0.0
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


def test_gguf_host_sampler_stops_on_request_eos_token_id(monkeypatch) -> None:
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
    out = generator.generate(
        _request(temperature=0.7, top_k=1, eos_token_id=1)
    )

    assert out == ["B"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "eos",
        "eos_token_id": 1,
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


def test_shared_slot_runner_declares_same_round_prefill_decode_only() -> None:
    runner_type = qwen35_gguf.Qwen35GGUFResidentModelRunner

    assert runner_type.supports_prefill_decode_same_round is True
    assert runner_type.supports_multiple_prefill_quanta_per_round is False


def test_shared_slot_runner_lowers_logical_width_to_registered_c2_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner._resident_batch_owner = object()
    runner._shared_runner = SimpleNamespace(backend="hip_gfx1100")
    runner._last_execution_manifest = {}
    runner._last_physical_group_plan = {}
    packed_calls: list[tuple[tuple[int, ...], int]] = []
    serial_calls: list[tuple[int, ...]] = []
    runner._step_native_chunk = lambda rows, *, physical_rows, active_slot_indices, allow_graph: (
        packed_calls.append(
            (
                tuple(int(row.request_id) for row in rows),
                int(physical_rows),
            )
        )
        or True
    )
    runner._step_native_serial = lambda rows, *, fallback_reason: serial_calls.append(
        tuple(int(row.request_id) for row in rows)
    )
    monkeypatch.setattr(
        qwen35_gguf,
        "backend_package_capability",
        lambda backend, key, default=None: (
            (1, 2) if key == "GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS" else default
        ),
    )
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")
    rows = [
        SimpleNamespace(
            request_id=request_id,
            slot=SimpleNamespace(
                session=SimpleNamespace(kv_attention_source="bf16"),
                c1_decode_graph=None,
            ),
        )
        for request_id in range(5)
    ]
    work = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=tuple(range(5)),
        row_to_request=tuple(range(5)),
        slot_ids=tuple(range(5)),
        active_mask=(True,) * 5,
    )

    runner._step_native_rows(rows, work=work)

    assert packed_calls == [((0, 1), 2), ((2, 3), 2)]
    assert serial_calls == [(4,)]
    assert runner._last_physical_group_plan["physical_bucket_widths"] == [1, 2]
    assert runner._last_physical_group_plan["group_count"] == 3


def test_gfx1100_registers_shared_slot_ar_physical_widths_through_c8() -> None:
    """Host gate for the promoted direct-width shared-slot promotion.

    GREEN once the kernel package registers ``(1, 2, 3, 4, 5, 6, 7, 8)``
    (direct c3/c5/c6/c7 promoted after #36 lifecycle certification).
    """
    from hipengine.kernels.backends import backend_package_capability

    registered = backend_package_capability(
        "hip_gfx1100", "GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", (1,)
    )
    assert tuple(int(width) for width in registered) == (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    )


def test_shared_slot_runner_lowers_logical_width_to_registered_c4_c8_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mechanism coverage for the promoted ``(1, 2, 4, 8)`` lowering.

    Six active rows must lower to one width-8 group with two masked lanes,
    nine to one width-8 group plus a width-1 edge, and three to width 4.
    """
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner._resident_batch_owner = object()
    runner._shared_runner = SimpleNamespace(backend="hip_gfx1100")
    runner._last_execution_manifest = {}
    runner._last_physical_group_plan = {}
    packed_calls: list[tuple[tuple[int, ...], int, tuple[int, ...]]] = []
    serial_calls: list[tuple[int, ...]] = []
    runner._step_native_chunk = lambda rows, *, physical_rows, active_slot_indices, allow_graph: (
        packed_calls.append(
            (
                tuple(int(row.request_id) for row in rows),
                int(physical_rows),
                tuple(int(index) for index in active_slot_indices),
            )
        )
        or True
    )
    runner._step_native_serial = lambda rows, *, fallback_reason: serial_calls.append(
        tuple(int(row.request_id) for row in rows)
    )
    monkeypatch.setattr(
        qwen35_gguf,
        "backend_package_capability",
        lambda backend, key, default=None: (
            (1, 2, 4, 8) if key == "GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS" else default
        ),
    )
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")

    def _rows(count: int) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                request_id=request_id,
                slot=SimpleNamespace(
                    session=SimpleNamespace(kv_attention_source="bf16"),
                    c1_decode_graph=None,
                ),
            )
            for request_id in range(count)
        ]

    runner._step_native_rows(
        _rows(6),
        work=WorkItem(
            kind=WorkKind.DECODE,
            request_ids=tuple(range(6)),
            row_to_request=tuple(range(6)),
            slot_ids=tuple(range(6)),
            active_mask=(True,) * 6,
        ),
    )
    assert packed_calls == [((0, 1, 2, 3, 4, 5), 8, (0, 1, 2, 3, 4, 5))]

    packed_calls.clear()
    serial_calls.clear()
    runner._step_native_rows(
        _rows(9),
        work=WorkItem(
            kind=WorkKind.DECODE,
            request_ids=tuple(range(9)),
            row_to_request=tuple(range(9)),
            slot_ids=tuple(range(9)),
            active_mask=(True,) * 9,
        ),
    )
    assert packed_calls == [((0, 1, 2, 3, 4, 5, 6, 7), 8, (0, 1, 2, 3, 4, 5, 6, 7))]
    assert serial_calls == [(8,)]

    packed_calls.clear()
    serial_calls.clear()
    runner._step_native_rows(
        _rows(3),
        work=WorkItem(
            kind=WorkKind.DECODE,
            request_ids=tuple(range(3)),
            row_to_request=tuple(range(3)),
            slot_ids=tuple(range(3)),
            active_mask=(True,) * 3,
        ),
    )
    assert packed_calls == [((0, 1, 2), 4, (0, 1, 2))]
    assert runner._last_physical_group_plan["physical_bucket_widths"] == [1, 2, 4, 8]


def test_resident_runner_follows_d2_composition_across_c1_to_c32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the actual resident owner lowers every logical c1-c32 to the D2
    artifact-backed composition when a cost table is configured.

    Ceiling remains the fail-closed fallback when no cost table is set; with one
    set, the plan's group physical widths must equal ``d2_partition``.
    """
    step_ms = {
        1: 33.1701, 2: 37.5209, 3: 40.0602, 4: 43.2973,
        5: 48.0149, 6: 52.7025, 7: 57.8864, 8: 63.5257,
    }
    cost_table = CostTable(
        tuple(
            PhysicalWidthCost(
                active_rows=w,
                physical_width=w,
                mask_class="dense_all_active",
                model_step_ms=ms,
                workspace_bytes=0,
                route_manifest_sha256="a" * 64,
                correctness_sha256="b" * 64,
                source="post-promotion-fixture",
            )
            for w, ms in step_ms.items()
        )
    )
    runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
        qwen35_gguf.Qwen35GGUFResidentModelRunner
    )
    runner._resident_batch_owner = object()
    runner._shared_runner = SimpleNamespace(backend="hip_gfx1100")
    runner.generator = SimpleNamespace(
        model_path="/models/fixture.gguf",
        _kv_weight_quant_key=lambda: "gguf_q4_k_m",
    )
    runner._last_execution_manifest = {}
    runner._last_physical_group_plan = {}
    runner._gguf_ar_cost_table = cost_table
    runner._step_native_chunk = lambda rows, **kwargs: True
    runner._step_native_serial = lambda rows, **kwargs: None
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")
    # Default-D2 is production-active for the configured-owner loop below; for
    # the ceiling-fallback half, patch the resolver to fail closed to None so it
    # never reaches the fake runner's missing generator.
    monkeypatch.setattr(
        qwen35_gguf, "_gguf_ar_resolve_cost_table", lambda *a, **k: None
    )

    for c in range(1, 33):
        rows = [
            SimpleNamespace(
                request_id=request_id,
                slot=SimpleNamespace(
                    session=SimpleNamespace(kv_attention_source="bf16"),
                    c1_decode_graph=None,
                ),
            )
            for request_id in range(c)
        ]
        runner._step_native_rows(
            rows,
            work=WorkItem(
                kind=WorkKind.DECODE,
                request_ids=tuple(range(c)),
                row_to_request=tuple(range(c)),
                slot_ids=tuple(range(c)),
                active_mask=(True,) * c,
            ),
        )
        plan = runner._last_physical_group_plan
        got = sorted(
            (int(group["physical_rows"]) for group in plan["groups"]),
            reverse=True,
        )
        expected = list(d2_partition(c, cost_table))
        assert got == expected, f"c{c}: plan {got} != D2 {expected}"
        assert plan["policy"] == "artifact_backed_d2"
        assert plan["d2"]["width_sequence"] == expected
        # Composition must cover every active row exactly.
        assert sum(got) == c

    # Without a cost table the owner fails closed to ceiling (not D2).
    runner._gguf_ar_cost_table = None
    runner._step_native_rows(
        [
            SimpleNamespace(
                request_id=r,
                slot=SimpleNamespace(
                    session=SimpleNamespace(kv_attention_source="bf16"),
                    c1_decode_graph=None,
                ),
            )
            for r in range(13)
        ],
        work=WorkItem(
            kind=WorkKind.DECODE,
            request_ids=tuple(range(13)),
            row_to_request=tuple(range(13)),
            slot_ids=tuple(range(13)),
            active_mask=(True,) * 13,
        ),
    )
    plan = runner._last_physical_group_plan
    got = sorted(
        (int(group["physical_rows"]) for group in plan["groups"]),
        reverse=True,
    )
    assert got == [8, 5]  # ceiling fallback, not D2 (7,6)
    assert plan["policy"] == "occupancy_adaptive_dense_execution"
    assert "d2" not in plan


def test_resident_runner_d2_artifact_env_loads_cost_table(monkeypatch) -> None:
    path = Path(
        "benchmarks/results/2026-08-20-concurrency2-qwen38-d2-cost-map.json"
    ).resolve()
    monkeypatch.setenv("HIPENGINE_GGUF_AR_D2_COST_ARTIFACT", str(path))
    monkeypatch.setattr(
        qwen35_gguf,
        "collect_model_identity",
        lambda _path: {
            "fingerprint": {
                "exists": True,
                "value": "2512f262273074db82860f1f3d6c15b4d9054b29b3c4babb0e2c770d6474c850",
            }
        },
    )
    monkeypatch.setattr(qwen35_gguf.socket, "gethostname", lambda: "epyc")
    monkeypatch.setattr(
        qwen35_gguf,
        "detect_device_name",
        lambda: "AMD Radeon Pro W7900",
    )
    qwen35_gguf._GGUF_AR_D2_COST_CACHE.clear()
    try:
        ct = qwen35_gguf._gguf_ar_resolve_cost_table(
            "hip_gfx1100",
            target_arch="gfx1100",
            model_path="/models/fixture.gguf",
            quant="gguf_q4_k_m",
            kv_dtype="bf16",
            physical_widths=tuple(range(1, 9)),
        )
        assert ct is not None
        assert ct.widths == (1, 2, 3, 4, 5, 6, 7, 8)
        monkeypatch.setattr(
            qwen35_gguf,
            "detect_device_name",
            lambda: "AMD Radeon RX 7900 XTX",
        )
        with pytest.raises(ValueError, match="identity mismatch"):
            qwen35_gguf._gguf_ar_resolve_cost_table(
                "hip_gfx1100",
                target_arch="gfx1100",
                model_path="/models/fixture.gguf",
                quant="gguf_q4_k_m",
                kv_dtype="bf16",
                physical_widths=tuple(range(1, 9)),
            )
        monkeypatch.setattr(
            qwen35_gguf,
            "detect_device_name",
            lambda: "AMD Radeon Pro W7900",
        )
        with pytest.raises(ValueError, match="identity mismatch"):
            qwen35_gguf._gguf_ar_resolve_cost_table(
                "hip_gfx1100",
                target_arch="gfx1100",
                model_path="/models/fixture.gguf",
                quant="gguf_q4_k_s",
                kv_dtype="bf16",
                physical_widths=tuple(range(1, 9)),
            )

        runner = qwen35_gguf.Qwen35GGUFResidentModelRunner.__new__(
            qwen35_gguf.Qwen35GGUFResidentModelRunner
        )
        runner._resident_batch_owner = SimpleNamespace(
            kv_storage_dtype=SimpleNamespace(value="bf16")
        )
        runner._shared_runner = SimpleNamespace(
            backend="hip_gfx1100", target_arch="gfx1100"
        )
        runner.generator = SimpleNamespace(
            model_path="/models/fixture.gguf",
            _kv_weight_quant_key=lambda: "gguf_q4_k_m",
        )
        runner._last_execution_manifest = {}
        runner._last_physical_group_plan = {}
        runner._step_native_chunk = lambda rows, **kwargs: True
        runner._step_native_serial = lambda rows, **kwargs: None
        monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")
        runner._step_native_rows(
            [
                SimpleNamespace(
                    request_id=r,
                    slot=SimpleNamespace(
                        session=SimpleNamespace(kv_attention_source="bf16"),
                        c1_decode_graph=None,
                    ),
                )
                for r in range(13)
            ],
            work=WorkItem(
                kind=WorkKind.DECODE,
                request_ids=tuple(range(13)),
                row_to_request=tuple(range(13)),
                slot_ids=tuple(range(13)),
                active_mask=(True,) * 13,
            ),
        )
        got = sorted(
            int(group["physical_rows"])
            for group in runner._last_physical_group_plan["groups"]
        )
        assert got == [6, 7]  # D2 composition loaded from the artifact
        assert runner._last_physical_group_plan["policy"] == "artifact_backed_d2"
        assert runner._last_physical_group_plan["d2"]["identity"]["backend"] == "hip_gfx1100"
    finally:
        qwen35_gguf._GGUF_AR_D2_COST_CACHE.clear()


def test_resident_runner_d2_absent_uses_ceiling_and_invalid_explicit_fails(monkeypatch) -> None:
    """D2 is explicit-config until the actual-server promotion gate passes."""
    monkeypatch.delenv("HIPENGINE_GGUF_AR_D2_COST_ARTIFACT", raising=False)
    qwen35_gguf._GGUF_AR_D2_COST_CACHE.clear()
    try:
        assert qwen35_gguf._gguf_ar_resolve_cost_table(
            "hip_gfx1100",
            target_arch="gfx1100",
            model_path="/missing/model.gguf",
            quant="gguf_q4_k_m",
            kv_dtype="bf16",
            physical_widths=tuple(range(1, 9)),
        ) is None
        # An explicitly requested but invalid artifact still raises.
        monkeypatch.setenv(
            "HIPENGINE_GGUF_AR_D2_COST_ARTIFACT", "/missing/artifact.json"
        )
        with pytest.raises(ValueError, match="does not exist"):
            qwen35_gguf._gguf_ar_resolve_cost_table(
                "hip_gfx1100",
                target_arch="gfx1100",
                model_path="/missing/model.gguf",
                quant="gguf_q4_k_m",
                kv_dtype="bf16",
                physical_widths=tuple(range(1, 9)),
            )
    finally:
        qwen35_gguf._GGUF_AR_D2_COST_CACHE.clear()
