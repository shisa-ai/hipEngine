from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
from hipengine.loading.qwen35_gguf_materialize import plan_qwen35_gguf_materialization
from hipengine.runtime.prefill import PrefillConfig, resolve_prefill_config_for_sequence
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def _planning_session(max_sequence_length: int) -> Qwen35GGUFResidentSession:
    config = plan_qwen35_gguf_materialization(
        build_qwen35_gguf_tensor_map(GGUFReader(MODEL).info)
    ).config
    runner = SimpleNamespace(
        backend="hip_gfx1151",
        weights=SimpleNamespace(config=config),
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        ffn_size=config.feed_forward_length,
        q_width=config.head_count * config.key_length,
        kv_width=config.head_count_kv * config.value_length,
        linear_qkv_width=(
            2 * config.ssm_group_count * config.ssm_state_size
            + config.ssm_inner_size
        ),
        ssm_value_dim=config.ssm_inner_size // config.ssm_time_step_rank,
    )
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = runner
    session.max_sequence_length = int(max_sequence_length)
    session.max_batch_size = 1
    session.kv_storage_dtype = DType.BF16
    session.kv_storage_layout = "uniform"
    session.kv_scale_dtype = DType.FP16
    session.kv_scale_granularity = "per_token_head"
    session.int8_kv_value_bf16 = False
    session.int8_bf16_prefix_full_attention_layers = 0
    session.int8_bf16_full_attention_layer_indices = ()
    session.defer_kv_allocation = False
    session.use_expert_sidecar = False
    session.prefill_chunk_size = 0
    session.backend = "hip_gfx1151"
    session._runtime_state_library = None
    rounded = ((int(max_sequence_length) + 255) // 256) * 256
    session.prefill_config, session.prefill_chunk_tuning = (
        resolve_prefill_config_for_sequence(
            PrefillConfig(),
            max_sequence_length=rounded,
            total_memory_bytes=128 * (1 << 30),
        )
    )
    return session


@pytest.mark.parametrize(
    ("prompt_tokens", "requested_bytes", "capacity_bytes"),
    (
        (512, 164_287_744, 164_478_976),
        (4096, 577_259_576, 577_441_792),
        (32768, 1_271_136_760, 1_271_320_576),
        (65536, 2_064_139_256, 2_064_322_560),
    ),
)
def test_private_c1_session_arena_plan_matches_instrumented_inventory(
    prompt_tokens: int,
    requested_bytes: int,
    capacity_bytes: int,
) -> None:
    # Canonical pp/tg128 sessions reserve prompt + generation capacity.
    session = _planning_session(prompt_tokens + 128)

    plan = session._plan_initial_session_arena()

    assert plan.alignment == 4096
    assert plan.allocation_count == 189
    assert plan.requested_bytes == requested_bytes
    assert plan.capacity_bytes == capacity_bytes
