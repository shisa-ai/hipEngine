"""Two-GPU integration: the head-sharded attention substitution on the real model.

Builds a real rank runner with the attention and MLP slots excluded from
residency - the allowlist the head-sharded route uses - applies the session's own
``_apply_attention_shard``, and checks that the runner's derived geometry and its
resident scratch plan both follow the rank's slice.

The load-bearing claim is that the payloads and the geometry cannot disagree.
The payloads are the planner's slices; the geometry is what the runner derives
from its config and launches its kernels with. If the halved config described a
different shape than the shards hold, the kernels would read the wrong rows
silently, because both halves would be internally consistent. So the assertions
here compare the runner's own widths against the shapes the materializer
produced, not against a restatement of the same arithmetic.
"""

from __future__ import annotations

import ctypes
import pathlib

import pytest

GGUF_PATH = pathlib.Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
MAX_SEQ = 4096


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _device_count() -> int:
    if not _hip_available():
        return 0
    from hipengine.core.hip import get_hip_runtime

    try:
        return int(get_hip_runtime().device_count())
    except Exception:  # noqa: BLE001 - no usable device
        return 0


needs_two_gpus = pytest.mark.skipif(
    _device_count() < 2,
    reason="requires two HIP devices (W7900 + RX 7900 XTX target host)",
)
needs_model = pytest.mark.skipif(
    not GGUF_PATH.exists(), reason=f"GGUF model file not available: {GGUF_PATH}"
)


@needs_two_gpus
@needs_model
def test_head_sharded_attention_substitutes_rank_geometry_and_scratch() -> None:
    import types

    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free
    from hipengine.distributed.shard_weights import (
        materialize_attention_shards,
        upload_shard_weights,
    )
    from hipengine.distributed.tp2_generate import (
        MlpTP2GenerationSession,
        TP2GroupError,
        rank_slot_allowlist_from_records,
    )
    from hipengine.loading.gguf import scan_gguf
    from hipengine.loading.qwen35_gguf import (
        FULL_ATTENTION,
        LINEAR_ATTENTION,
    )
    from hipengine.loading.qwen35_gguf_admission import (
        build_qwen35_gguf_role_manifest,
    )
    from hipengine.loading.qwen35_gguf_materialize import (
        build_qwen35_gguf_tensor_map,
    )
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFFullStackRunner,
        _FullStackScratch,
    )

    runtime = get_hip_runtime()
    info = scan_gguf(str(GGUF_PATH))
    manifest = build_qwen35_gguf_role_manifest(build_qwen35_gguf_tensor_map(info))
    records = tuple(manifest.records)
    allowlist = rank_slot_allowlist_from_records(
        records, families=("mlp", "attention")
    )
    # The attention and MLP leaves are gone from residency, so the shards are the
    # only copies of them.
    for record in records:
        leaf = str(record[0]).rsplit(".", 1)[-1]
        if leaf in {"attn_q", "attn_k", "attn_v", "attn_output", "ssm_out", "ffn_down"}:
            assert str(record[0]) not in allowlist

    with scoped_current_device(runtime, 0):
        runner = Qwen35GGUFFullStackRunner(
            str(GGUF_PATH),
            backend="hip_gfx1100",
            execution_routes=("eager",),
            runtime=runtime,
            deferred_device_slots=("root.lm_head",),
            selected_slots=allowlist,
        )
        full_config = runner.weights.config
        full_widths = {
            "q_width": runner.q_width,
            "kv_width": runner.kv_width,
            "linear_qkv_width": runner.linear_qkv_width,
            "ssm_value_dim": runner.ssm_value_dim,
            "hidden_size": runner.hidden_size,
        }
        full_scratch = _FullStackScratch.allocate(
            runner, runtime=runtime, max_sequence_length=MAX_SEQ
        )
        full_total = sum(int(buffer.nbytes) for buffer in full_scratch.buffers)
        full_sizes = sorted(int(buffer.nbytes) for buffer in full_scratch.buffers)
        for buffer in reversed(full_scratch.buffers):
            free(buffer, runtime=runtime)

        shards = materialize_attention_shards(str(GGUF_PATH), world_size=2)
        uploaded = upload_shard_weights(runtime, shards, devices=(0, 1))
        # The session's own method, bound to just the state it reads, so this
        # exercises the real substitution rather than a copy of it.
        stub = types.SimpleNamespace(
            devices=(0, 1), _attention_shard_weights=uploaded, attention_shard=True
        )
        MlpTP2GenerationSession._apply_attention_shard(stub, 0, runner)

        sharded = runner.weights.config
        # The head axes halve and the widths the kernels launch with follow.
        assert sharded.head_count == full_config.head_count // 2
        assert sharded.head_count_kv == full_config.head_count_kv // 2
        assert sharded.ssm_inner_size == full_config.ssm_inner_size // 2
        assert sharded.ssm_time_step_rank == full_config.ssm_time_step_rank // 2
        assert runner.q_width == full_widths["q_width"] // 2
        assert runner.kv_width == full_widths["kv_width"] // 2
        assert runner.linear_qkv_width == full_widths["linear_qkv_width"] // 2
        # The residual stream is replicated and the value-head dim is per head, so
        # neither may move.
        assert runner.hidden_size == full_widths["hidden_size"]
        assert runner.ssm_value_dim == full_widths["ssm_value_dim"]

        # Each rank's attention slots are its own shard weights, and their shapes
        # are the widths the runner just derived. The logical shape comes from the
        # materializer's own ``local_shape`` - the uploaded weight holds the
        # repacked payload, whose shape is flat, so the planner is the authority
        # on the slice geometry.
        for layer_type in (FULL_ATTENTION, LINEAR_ATTENTION):
            layer_id = full_config.layer_types.index(layer_type)
            layer = runner.weights.layer(layer_id)
            rank_roles = uploaded[layer_id][0]
            rank_payloads = shards[layer_id].rank_payloads(0)
            assert set(rank_roles) <= set(layer.weights)
            for role, weight in rank_roles.items():
                assert layer.weights[role] is weight, role
                local_shape = tuple(int(v) for v in rank_payloads[role].local_shape)
                if role == "attn_q":
                    # The fused query and its output gate.
                    assert local_shape == (2 * runner.q_width, runner.hidden_size)
                elif role in {"attn_k", "attn_v"}:
                    assert local_shape == (runner.kv_width, runner.hidden_size)
                elif role == "attn_output":
                    assert local_shape == (runner.hidden_size, runner.q_width)
                elif role == "ssm_out":
                    assert local_shape == (runner.hidden_size, sharded.ssm_inner_size)
                elif role == "attn_gate":
                    assert local_shape == (sharded.ssm_inner_size, runner.hidden_size)
            # A slot the shard does not own is untouched, so the substitution
            # replaced the attention copies rather than the whole layer map.
            for role in ("attn_norm", "post_attention_norm"):
                assert role in layer.weights
                assert role not in rank_roles

        # The scratch stack is sized from the substituted geometry, which is what
        # makes it per-rank rather than replicated.
        sharded_scratch = _FullStackScratch.allocate(
            runner, runtime=runtime, max_sequence_length=MAX_SEQ
        )
        try:
            sharded_total = sum(int(buffer.nbytes) for buffer in sharded_scratch.buffers)
            sharded_sizes = sorted(int(buffer.nbytes) for buffer in sharded_scratch.buffers)
            assert sharded_sizes != full_sizes, "the scratch plan did not follow the geometry"
            assert 0.4 < sharded_total / full_total < 0.6, (
                f"scratch went {full_total} -> {sharded_total} bytes"
            )
        finally:
            for buffer in reversed(sharded_scratch.buffers):
                free(buffer, runtime=runtime)

        # A head-sharded layer must refuse to run while the reduce is unwired.
        with pytest.raises(TP2GroupError, match="attention-output reduce"):
            MlpTP2GenerationSession._require_attention_reduce(stub)
