"""CPU tests for the TP2 model-owning generation driver.

No ROCm is required: the driver's HIP runtime, resident runners, scratch,
shard group and every launcher are fakes/spies, so the tests pin the
schedule - the per-layer rank interleave, the single residual add per rank,
the control rank's head/sampling, position and token ownership, EOS, reset,
poison-on-failure, and teardown - without touching hardware.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.distributed.tp2_generate as tg
import hipengine.runtime.gguf_linear as gguf_linear
from hipengine.runtime.gguf_linear import resident_session_wmma_prefill_default
from hipengine.distributed.tp2_generate import (
    LINEAR_ATTENTION,
    MlpTP2GenerationSession,
    TP2GroupError,
)
from tests.test_unit_distributed_staged_and_shard import FakeHipRuntime

HIDDEN = 16
VOCAB = 32
FFN = 24
EPS = 1e-5
LAYER_TYPES = (LINEAR_ATTENTION, "full_attention", LINEAR_ATTENTION)


class FakeBuffer:
    def __init__(self, ptr: int, nbytes: int) -> None:
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)


class FakeWeight:
    def __init__(self, name: str) -> None:
        self.name = name

    def allocation(self):
        return self

    @property
    def tensor(self):
        return self

    @property
    def ptr(self) -> int:
        return 0x7000 + (hash(self.name) % 0x1000)


class FakeLayer:
    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id

    def weight(self, slot: str) -> FakeWeight:
        return FakeWeight(f"layers.{self.layer_id}.{slot}")


class FakeConfig:
    def __init__(self, layer_types) -> None:
        self.layer_types = tuple(layer_types)
        self.rms_norm_eps = EPS
        self.block_count = len(self.layer_types)
        self.feed_forward_length = FFN
        self.hidden_size = HIDDEN


class FakeWeights:
    def __init__(self, layer_types) -> None:
        self.config = FakeConfig(layer_types)

    def root(self, slot: str) -> FakeWeight:
        return FakeWeight(f"root.{slot}")

    def layer(self, layer_id: int) -> FakeLayer:
        return FakeLayer(layer_id)

    def free(self, *, runtime=None) -> None:
        if getattr(self, "freed", False):
            raise AssertionError("weights freed twice")
        self.freed = True


class FakeScratch:
    def __init__(self) -> None:
        self.max_positions = 4096
        self.attn_out = FakeBuffer(0x4000, HIDDEN * 2)
        self.post_norm = FakeBuffer(0x4100, HIDDEN * 2)
        self.residual = FakeBuffer(0x4200, HIDDEN * 2)
        self.norm = FakeBuffer(0x4300, HIDDEN * 2)
        self.cos_table_buf = FakeBuffer(0x4400, 64)
        self.sin_table_buf = FakeBuffer(0x4500, 64)
        self.full_cache_calls: list[int] = []
        self.positions: list[int] = []
        self.zeroed = 0

    def full_cache(self, layer_id: int):
        self.full_cache_calls.append(int(layer_id))
        return (
            FakeBuffer(0x4600 + int(layer_id), 1024),
            FakeBuffer(0x4700 + int(layer_id), 1024),
        )

    def set_full_attention_position(self, position: int, runtime) -> None:
        self.positions.append(int(position))

    def zero_states(self, runtime, *, stream: int = 0, set_position: bool = True) -> None:
        self.zeroed += 1
        self.positions = []


class RunnerSpy:
    def __init__(self, rt: FakeHipRuntime, layer_types) -> None:
        self.rt = rt
        self.calls: list[tuple[int, str, int]] = []
        self.weights = FakeWeights(layer_types)
        # ``resident_prefill_dispatch_session`` resolves the shape-gated owners
        # from ``runner.backend`` + the weight geometry identity.
        self.backend = "hip_gfx1100"
        self.hidden_size = HIDDEN
        self.vocab_size = VOCAB
        self.ffn_size = FFN
        self.fail_on_layer: int | None = None
        self.scratch: FakeScratch | None = None
        # The full-attention prefill helper derives its row count from
        # ``scratch.rows`` (it has no explicit ``rows`` parameter), so the
        # scratch handed to it must carry the ACTIVE prompt rows, not the
        # bulk capacity.
        self.prefill_attn_scratch_rows: list[int] = []
        self.prefill_attn_rows_arg: list[int] = []
        # The session-scoped GGUF linear dispatch toggles that were active when
        # the prefill attention helper ran. The resident bulk prefill
        # establishes these for the whole request; a caller that runs the same
        # helper outside the context resolves different kernels for the same
        # weight and rows.
        self.prefill_dispatch_context: list[dict] = []
        self.prefill_norm_dispatch_context: list[dict] = []
        self.head_dispatch_context: list[dict] = []
        self.head_launches: list[tuple[str, int]] = []

    def _run_linear_attention_attn_only(self, layer_id, hidden_ptr, attn_out, scratch, **kwargs):
        self._attn(layer_id)

    def _run_full_attention_attn_only(self, layer_id, hidden_ptr, attn_out, scratch, **kwargs):
        self._attn(layer_id)

    def _run_linear_attention_prefill_attn_rows(
        self, layer_id, hidden_ptr, scratch, *, rows, decode_scratch, stream=0, **kwargs
    ):
        self.prefill_attn_scratch_rows.append(int(scratch.rows))
        self.prefill_attn_rows_arg.append(int(rows))
        self.prefill_dispatch_context.append(
            dict(gguf_linear.gguf_prefill_dispatch_context())
        )
        self._prefill_attn(layer_id, "linear_prefill_attn")
        return None

    def _run_full_attention_prefill_attn_rows(
        self,
        layer_id,
        hidden_ptr,
        scratch,
        *,
        cos_table_ptr,
        sin_table_ptr,
        max_positions,
        stream=0,
        **kwargs,
    ):
        self.prefill_attn_scratch_rows.append(int(scratch.rows))
        self.prefill_dispatch_context.append(
            dict(gguf_linear.gguf_prefill_dispatch_context())
        )
        self._prefill_attn(layer_id, "full_prefill_attn")
        return True

    def _run_post_attention_norm_residual_rows(
        self, layer_id, hidden_ptr, attn_out_ptr, scratch, *, rows, stream=0, **kwargs
    ):
        self.prefill_norm_dispatch_context.append(
            dict(gguf_linear.gguf_prefill_dispatch_context())
        )
        if self.fail_on_layer is not None and layer_id == self.fail_on_layer:
            raise RuntimeError("simulated prefill norm failure")
        self.calls.append((self.rt.get_device(), "prefill_norm", int(layer_id)))
        return None

    def _prefill_attn(self, layer_id: int, name: str) -> None:
        if self.fail_on_layer is not None and layer_id == self.fail_on_layer:
            raise RuntimeError("simulated attention failure")
        self.calls.append((self.rt.get_device(), name, int(layer_id)))

    def _attn(self, layer_id: int) -> None:
        if self.fail_on_layer is not None and layer_id == self.fail_on_layer:
            raise RuntimeError("simulated attention failure")
        self.calls.append((self.rt.get_device(), "attn", int(layer_id)))


class FakeShardGroup:
    def __init__(self, *, devices=(0, 1), rows: int = 1) -> None:
        self.devices = tuple(int(device) for device in devices)
        self.rows = int(rows)
        self.owns_weights = True
        self.forwards: list[int] = []
        self.forward_inputs: list[dict] = []
        self.closed = 0
        self.exchange_walls_s: list[float] = []
        self.fail_on_layer: int | None = None
        self.chains: list[tuple[int, int]] = []
        self.reduces: list[tuple[tuple[int, ...], int | None]] = []
        # Session-scoped GGUF linear dispatch context active at each group
        # forward: the shard chain's own GGUF linears resolve from it.
        self.forward_dispatch_context: list[dict] = []

    def forward(self, layer_id: int, inputs, *, rows=None):
        if self.fail_on_layer is not None and layer_id == self.fail_on_layer:
            raise RuntimeError("simulated shard failure")
        self.forward_dispatch_context.append(
            dict(gguf_linear.gguf_prefill_dispatch_context())
        )
        self.forwards.append(int(layer_id))
        self.forward_inputs.append(dict(inputs))
        self.exchange_walls_s.append(0.0)
        return {device: 0x5000 + device for device in self.devices}

    def enqueue_rank_chain(self, layer_id: int, device: int, input_ptr: int) -> int:
        if self.fail_on_layer is not None and layer_id == self.fail_on_layer:
            raise RuntimeError("simulated shard failure")
        self.chains.append((int(layer_id), int(device)))
        return 0x5200 + int(device) + 64 * int(layer_id)

    def reduce_partials(self, partial_ptrs, *, slot=None):
        self.reduces.append(
            (tuple(int(partial_ptrs[d]) for d in sorted(partial_ptrs)), slot)
        )
        self.exchange_walls_s.append(0.0)
        return {0: 0x5400, 1: 0x5500}

    def output_ptr(self, device: int) -> int:
        return {0: 0x5000, 1: 0x5100}[int(device)]

    def reduced_payload_ptr(self, slot: int) -> int:
        return 0x6000 + int(slot)

    def reset_exchange_walls(self) -> None:
        self.exchange_walls_s.clear()

    def close(self) -> None:
        self.closed += 1


class FakeDeviceExchange:
    def __init__(self) -> None:
        self.step_begins = 0
        self.enqueues: list[tuple[int, int, int]] = []
        self.waits = 0
        self.closed = 0

    def step_begin(self) -> None:
        self.step_begins += 1

    def enqueue_rank(self, rank, own_partial, slot, out_payload):
        self.enqueues.append((int(rank), int(slot), int(own_partial)))
        return int(out_payload)

    def wait(self) -> None:
        self.waits += 1

    def close(self) -> None:
        self.closed += 1


@pytest.fixture()
def env(monkeypatch):
    rt = FakeHipRuntime(device_count=2)
    rt.device_synchronize = lambda: rt.calls.append(("device_sync", rt.get_device()))
    created_streams: list[int] = []
    destroyed_streams: list[int] = []

    def fake_stream_create(*, nonblocking=True, priority=None):
        handle = 0xB000 + len(created_streams) * 0x10
        created_streams.append(handle)
        return handle

    rt.stream_create = fake_stream_create
    rt.stream_destroy = lambda handle: destroyed_streams.append(int(handle))
    device_exchanges: list[FakeDeviceExchange] = []

    class FakeDeviceExchangeFactory(FakeDeviceExchange):
        def __init__(self, runtime, **kwargs):
            super().__init__()
            device_exchanges.append(self)

    monkeypatch.setattr(tg, "CompiledDeviceExchange", FakeDeviceExchangeFactory)
    graph_counter = [0]

    def fake_begin_capture(stream, mode=2):
        rt.calls.append(("begin_capture", int(stream)))

    def fake_end_capture(stream):
        rt.calls.append(("end_capture", int(stream)))
        graph_counter[0] += 1
        return 0x9000 + graph_counter[0]

    def fake_instantiate(graph):
        return 0xA000 + int(graph)

    def fake_graph_launch(graph_exec, stream):
        rt.calls.append(("graph_launch", rt.get_device(), int(graph_exec), int(stream)))

    def fake_graph_destroy(graph):
        rt.calls.append(("graph_destroy", int(graph)))

    def fake_exec_destroy(graph_exec):
        rt.calls.append(("graph_exec_destroy", int(graph_exec)))

    rt.stream_begin_capture = fake_begin_capture
    rt.stream_end_capture = fake_end_capture
    rt.graph_instantiate = fake_instantiate
    rt.graph_launch = fake_graph_launch
    rt.graph_destroy = fake_graph_destroy
    rt.graph_exec_destroy = fake_exec_destroy
    runner_spies: list[RunnerSpy] = []
    groups: list[FakeShardGroup] = []
    logit_rows: list[np.ndarray] = []
    injected_rows: list[np.ndarray] = []
    launch_log: list[tuple[int, str, str]] = []
    # The session-scoped GGUF linear dispatch context observed at every bulk
    # launch site, in launch order: the head projection goes through
    # ``launch_gguf_linear`` directly, so it is the only one the fake launch
    # helper can see.
    linear_dispatch_context: list[dict] = []

    def fake_runtime():
        return rt

    def fake_runner(model_path, *, backend, execution_routes, runtime, **kwargs):
        spy = RunnerSpy(rt, LAYER_TYPES)
        spy.deferred_device_slots = tuple(kwargs.get("deferred_device_slots", ()))
        runner_spies.append(spy)
        return spy

    class FakeScratchFactory:
        @staticmethod
        def allocate(runner, *, runtime, max_sequence_length=None, **kwargs):
            scratch = FakeScratch()
            if runner_spies:
                runner_spies[-1].scratch = scratch
            return scratch

    def fake_norm_kernel(runner, *, layer, rows, hidden_size, fallback=None):
        def kernel(*args, **kwargs):
            launch_log.append((rt.get_device(), layer, "leaf"))
        return kernel

    def fake_launch(name):
        def launch(*args, **kwargs):
            launch_log.append((rt.get_device(), name, "launch"))
        return launch

    def fake_linear_launch(*args, **kwargs):
        launch_log.append((rt.get_device(), "linear", "launch"))
        linear_dispatch_context.append(
            dict(gguf_linear.gguf_prefill_dispatch_context())
        )

    def fake_add(a_ptr, b_ptr, out_ptr, n, **kwargs):
        launch_log.append((rt.get_device(), "bf16_add", "add"))

    def fake_f32_to_bf16(src_ptr, out_ptr, n, **kwargs):
        launch_log.append((rt.get_device(), "f32_to_bf16", "cast"))

    def fake_copy_d2h(host_ptr, buffer, nbytes=None, *, runtime=None):
        if injected_rows:
            row = injected_rows.pop(0)
        else:
            row = logit_rows.pop(0)
        # The real API copies exactly nbytes from the buffer; the queued
        # rows may be wider than one rank's shard readback, so honor the
        # request instead of the row's own width.
        count = row.nbytes if nbytes is None else min(int(nbytes), row.nbytes)
        ctypes.memmove(host_ptr, row.ctypes.data, count)

    def fake_malloc(nbytes, *, runtime=None, device=None):
        return FakeBuffer(0x8000 + len(launch_log) * 0x40, nbytes)

    def fake_free(buffer, *, runtime=None):
        launch_log.append((rt.get_device(), "free", "free"))

    class FakeGroupFactory(FakeShardGroup):
        def __init__(self, runtime, **kwargs):
            super().__init__(
                devices=kwargs.get("devices", (0, 1)), rows=kwargs.get("rows", 1)
            )
            self.owns_weights = bool(kwargs.get("owns_weights", True))
            self.staging_dtype = kwargs.get("staging_dtype")
            groups.append(self)

    def fake_resolve(model_path, *, world_size, backend="hip_gfx1100"):
        return None, {}, None, FakeConfig(LAYER_TYPES)

    def fake_materialize(model_path, *, world_size, layer_ids=None, backend="hip_gfx1100"):
        return {}

    def fake_upload(runtime, shards, *, devices):
        return {}

    head_plans: list[Any] = []
    head_uploads: list[tuple[int, str, str]] = []
    head_freed: list[int] = []

    class FakeHeadAllocation:
        def __init__(self, device):
            self.device = device
            self.tensor = SimpleNamespace(ptr=0x9000 + device)

        def free(self):
            head_freed.append(self.device)

    class FakeHeadWeight:
        def __init__(self, device):
            self.device = device
            self._allocation = FakeHeadAllocation(device)

        def allocation(self, name=None):
            return self._allocation

    def fake_materialize_head(model_path, *, world_size, backend="hip_gfx1100"):
        from hipengine.distributed.head_shard import HeadShardPlan

        rows_per_rank = VOCAB // world_size
        plan = HeadShardPlan(
            world_size=world_size,
            vocab_rows=VOCAB,
            hidden=HIDDEN,
            layout="gguf_q6_k_t16_qmicro_planar_v1",
            quant_key="gguf_q6_k_t16_qmicro_planar_v1",
            rows_per_rank=rows_per_rank,
            blocks_per_rank=rows_per_rank // 256,
            source_row_bytes=4200,
        )
        head_plans.append(plan)
        return plan, {
            rank: {"tiles": np.zeros(8, dtype=np.uint8)} for rank in range(world_size)
        }

    def fake_upload_shard_weight(runtime, *, device, name, layout, quant_key, payload):
        head_uploads.append((device, layout, quant_key))
        return FakeHeadWeight(device)

    monkeypatch.setattr(tg, "get_hip_runtime", fake_runtime)
    monkeypatch.setattr(tg, "Qwen35GGUFFullStackRunner", fake_runner)
    monkeypatch.setattr(tg, "_FullStackScratch", FakeScratchFactory)
    monkeypatch.setattr(tg, "_gguf_norm_residual_decode_kernel", fake_norm_kernel)
    monkeypatch.setattr(tg, "launch_gguf_embedding", fake_launch("embedding"))
    monkeypatch.setattr(tg, "launch_gguf_linear", fake_linear_launch)
    monkeypatch.setattr(tg, "gguf_bf16_add", fake_add)
    monkeypatch.setattr(tg, "gguf_rmsnorm_bf16_f32_weight", fake_launch("rmsnorm"))
    monkeypatch.setattr(tg, "f32_to_bf16", fake_f32_to_bf16)
    monkeypatch.setattr(tg, "silu_mul_separate_out_bf16", fake_launch("silu"))
    monkeypatch.setattr(tg, "copy_device_to_host", fake_copy_d2h)
    monkeypatch.setattr(tg, "malloc", fake_malloc)
    monkeypatch.setattr(tg, "free", fake_free)
    monkeypatch.setattr(tg, "MlpShardGroup", FakeGroupFactory)

    from dataclasses import dataclass as _dc, replace as _dc_replace

    bulk_scratches: list[FakeBulkScratch] = []
    bulk_for_chunk_calls: list[tuple[int, int, int, int]] = []

    @_dc(frozen=True)
    class FakeBulkScratch:
        rows: int
        norm: FakeBuffer
        attn_out: FakeBuffer
        post_norm: FakeBuffer
        residual: FakeBuffer
        key_cache: object | None = None
        value_cache: object | None = None
        retained_key_cache: object | None = None
        retained_value_cache: object | None = None
        retained_append_spans: object | None = None
        int8_kv_value_bf16: bool = False
        max_positions: int = 4096
        start: int = 0
        chunk: tuple | None = None
        buffers: tuple = ()
        full_attn_split_growth_buffers: tuple = ()

        def for_chunk(self, start, rows, total_tokens, *, runtime, stream=0):
            bulk_for_chunk_calls.append(
                (int(self.rows), int(start), int(rows), int(total_tokens))
            )
            return _dc_replace(self, start=int(start), rows=int(rows),
                               chunk=(int(start), int(rows), int(total_tokens)))

    bulk_scratches: list[FakeBulkScratch] = []

    class FakeBulkScratchFactory:
        @staticmethod
        def allocate(runner, *, rows, capacity=None, allocate_kv_cache=True,
                     runtime=None, **kwargs):
            scratch = FakeBulkScratch(
                rows=int(rows),
                norm=FakeBuffer(0x4800, int(rows) * HIDDEN * 2),
                attn_out=FakeBuffer(0x4900, int(rows) * HIDDEN * 2),
                post_norm=FakeBuffer(0x4A00, int(rows) * HIDDEN * 2),
                residual=FakeBuffer(0x4B00, int(rows) * HIDDEN * 2),
            )
            bulk_scratches.append(scratch)
            return scratch

    monkeypatch.setattr(tg, "_GGUFFullAttentionPrefillScratch", FakeBulkScratchFactory)
    monkeypatch.setattr(tg, "resolve_mlp_shard_context", fake_resolve)
    monkeypatch.setattr(tg, "materialize_mlp_shards", fake_materialize)
    monkeypatch.setattr(tg, "upload_mlp_shard_weights", fake_upload)
    monkeypatch.setattr(tg, "materialize_head_shards", fake_materialize_head)
    monkeypatch.setattr(tg, "upload_shard_weight", fake_upload_shard_weight)

    def queue_logits(preferred_ids):
        # Each finish step reads one f32 row per rank (the sharded head is
        # the tp2 default): queue the row's two halves in rank order, so the
        # concatenation of the two readbacks reproduces the row the
        # replicated head would have returned.
        for i in preferred_ids:
            full = _logits_row(i)
            half = VOCAB // 2
            logit_rows.append(np.ascontiguousarray(full[:half]))
            logit_rows.append(np.ascontiguousarray(full[half:]))

    return {
        "rt": rt,
        "runners": runner_spies,
        "groups": groups,
        "launch_log": launch_log,
        "linear_dispatch_context": linear_dispatch_context,
        "queue_logits": queue_logits,
        "created_streams": created_streams,
        "destroyed_streams": destroyed_streams,
        "device_exchanges": device_exchanges,
        "head_plans": head_plans,
        "head_uploads": head_uploads,
        "head_freed": head_freed,
        "injected_rows": injected_rows,
        "bulk_scratches": bulk_scratches,
        "bulk_for_chunk_calls": bulk_for_chunk_calls,
    }


def _logits_row(preferred: int) -> np.ndarray:
    row = np.zeros(VOCAB, dtype="<f4")
    row[preferred] = 10.0
    return np.ascontiguousarray(row)


def _session(env, *, devices=(0, 1), mode="tp2") -> MlpTP2GenerationSession:
    # The eager per-launch schedule pinned: these tests assert the eager
    # per-layer launch recipe; the graphed schedule has its own tests below.
    return MlpTP2GenerationSession(
        "fake.gguf",
        devices=devices,
        mode=mode,
        max_sequence_length=64,
        schedule="eager",
    )


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def test_tp2_interleaves_both_ranks_per_layer(env) -> None:
    # Rows: the sample after each forward. [7, 8] prefill, then two decode
    # steps feed the previous samples.
    env["queue_logits"]([3, 5, 6, 9])
    session = _session(env)
    result = session.generate([7, 8], max_new_tokens=2, eos_token_id=None)
    runners = env["runners"]
    assert len(runners) == 2
    # Every rank ran attention for every layer of every position.
    assert len(runners[0].calls) == len(runners[1].calls) == 12
    assert [c[2] for c in runners[0].calls] == [0, 1, 2] * 4
    # The shard group saw one forward per layer per position.
    group = env["groups"][0]
    assert group.forwards == [0, 1, 2] * 4
    assert group.closed == 0
    # The generated tokens are the samples that decode positions consumed.
    assert result.token_ids == (5, 6)
    session.close()


def test_tp2_per_layer_order_is_attn_norm_group_add(env) -> None:
    env["queue_logits"]([1, 1])
    session = _session(env)
    session.generate([9], max_new_tokens=1, eos_token_id=None)
    kinds = [name for _, name, _ in env["launch_log"]]
    # Two positions x three layers x two ranks: both ranks' add+norm leaves
    # run before the sharded MLP, and both ranks add the residual once.
    assert kinds.count("embedding") == 4
    assert kinds.count("add_rmsnorm") == 12
    assert kinds.count("bf16_add") == 12
    # The final norm + head are the last launches, on the control rank.
    assert kinds[-2:] == ["rmsnorm", "linear"]
    session.close()


def test_tp1_mode_runs_the_local_mlp_chain_and_no_group(env) -> None:
    env["queue_logits"]([2, 2])
    session = _session(env, devices=(0,), mode="tp1")
    session.generate([4], max_new_tokens=1, eos_token_id=None)
    assert env["groups"] == [], "tp1 mode builds no shard group"
    kinds = [name for _, name, _ in env["launch_log"]]
    # Per layer per step: gate, up, silu, down (unfused full-width chain).
    assert kinds.count("silu") == 6
    assert kinds.count("bf16_add") == 6
    session.close()


def test_tp1_local_mlp_launches_under_its_rank_device(env) -> None:
    # Regression: _local_mlp had no scoped_current_device, so for a rank-1
    # session the ambient device - restored to 0 by the preceding scoped
    # attention/add-norm blocks - leaked into the full-width MLP launches.
    session = _session(env, devices=(1,), mode="tp1")
    env["rt"].set_device(0)
    env["launch_log"].clear()
    session._local_mlp(1, 0)
    devices = [dev for dev, _, _ in env["launch_log"]]
    assert devices, "expected full-width MLP launches"
    assert all(dev == 1 for dev in devices), devices
    assert env["rt"].get_device() == 0, "ambient device must be restored"
    session.close()


def test_tp1_local_mlp_uses_the_rank_stream(env, monkeypatch) -> None:
    session = _session(env, devices=(1,), mode="tp1")
    monkeypatch.setattr(session, "_rank_stream", lambda device: 0xB0 if device == 1 else 0)
    streams: list[int] = []

    def record_stream(*args, **kwargs):
        streams.append(int(kwargs.get("stream", -1)))

    monkeypatch.setattr(tg, "launch_gguf_linear", record_stream)
    session._local_mlp(1, 0)
    assert streams and all(stream == 0xB0 for stream in streams), streams
    session.close()


def test_tp1_local_mlp_restores_ambient_device_on_failure(env, monkeypatch) -> None:
    session = _session(env, devices=(1,), mode="tp1")
    env["rt"].set_device(0)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated mlp failure")

    monkeypatch.setattr(tg, "silu_mul_separate_out_bf16", boom)
    with pytest.raises(RuntimeError):
        session._local_mlp(1, 0)
    assert env["rt"].get_device() == 0, "ambient device must be restored on failure"
    session.close()


def test_tp1_generate_keeps_every_mlp_launch_on_the_rank_device(env) -> None:
    # The full _enqueue_layer path, not just _local_mlp: a rank-1 session with
    # ambient device 0 must still launch the MLP on device 1.
    env["queue_logits"]([2, 2])
    session = _session(env, devices=(1,), mode="tp1")
    env["rt"].set_device(0)
    env["launch_log"].clear()
    session.generate([4], max_new_tokens=1, eos_token_id=None)
    mlp = [
        dev
        for dev, name, kind in env["launch_log"]
        if kind == "launch" and name in {"linear", "silu"}
    ]
    assert mlp, "expected full-width MLP launches"
    assert all(dev == 1 for dev in mlp), mlp
    session.close()



# ---------------------------------------------------------------------------
# Ownership, lifecycle, failure
# ---------------------------------------------------------------------------


def test_positions_and_tokens_are_owned_by_the_loop(env) -> None:
    env["queue_logits"]([2, 3, 4, 5, 6])
    session = _session(env)
    result = session.generate([11, 12, 13], max_new_tokens=2, eos_token_id=None)
    assert result.token_ids == (4, 5)
    assert result.finished_on_eos is False
    # One trace per position: three prefill, two decode, positions in order.
    assert [t.kind for t in result.step_traces] == [
        "prefill", "prefill", "prefill", "decode", "decode",
    ]
    assert [t.position for t in result.step_traces] == [0, 1, 2, 3, 4]
    # Positions were published to both ranks' full-attention scratch.
    for spy in env["runners"]:
        assert spy.scratch is not None
        assert spy.scratch.positions == [0, 1, 2, 3, 4]
    session.close()


def test_eos_stops_generation_and_is_reported(env) -> None:
    # The prefill sample (21) and the two decode samples (22, 31) each feed
    # one finish step; the queue holds their per-rank readbacks, with spare
    # rows in case the loop runs past the EOS position.
    env["queue_logits"]([21, 22, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    session = _session(env)
    result = session.generate([1], max_new_tokens=8, eos_token_id=31)
    assert result.token_ids == (21, 22, 31)
    assert result.finished_on_eos is True
    session.close()


def test_max_new_tokens_bounds_the_generation(env) -> None:
    env["queue_logits"]([1] * 6)
    session = _session(env)
    result = session.generate([1], max_new_tokens=5, eos_token_id=None)
    assert len(result.token_ids) == 5
    assert result.finished_on_eos is False
    session.close()


def test_reset_zeroes_every_rank_state(env) -> None:
    session = _session(env)
    session.reset()
    session.close()
    for spy in env["runners"]:
        assert spy.scratch is not None
        assert spy.scratch.zeroed == 2, "construction plus reset"


def test_a_rank_failure_poisons_the_group_and_refuses_reuse(env) -> None:
    session = _session(env)
    env["runners"][0].fail_on_layer = 1
    env["queue_logits"]([1])
    with pytest.raises(TP2GroupError, match="simulated attention failure"):
        session.generate([1], max_new_tokens=1, eos_token_id=None)
    assert session.poisoned is True
    env["queue_logits"]([1])
    with pytest.raises(TP2GroupError, match="poisoned"):
        session.generate([1], max_new_tokens=1, eos_token_id=None)
    session.close()


def test_a_shard_failure_is_reported_as_a_group_error(env) -> None:
    session = _session(env)
    env["groups"][0].fail_on_layer = 2
    env["queue_logits"]([1])
    with pytest.raises(TP2GroupError, match="simulated shard failure"):
        session.generate([1], max_new_tokens=1, eos_token_id=None)
    session.close()


def test_teacher_forced_logits_return_one_full_row_per_token(env) -> None:
    env["queue_logits"]([0, 1, 2])
    session = _session(env)
    logits = session.teacher_forced_logits([5, 6, 7])
    assert logits.shape == (3, VOCAB)
    assert int(np.argmax(logits[0])) == 0
    assert int(np.argmax(logits[2])) == 2
    session.close()


def test_close_frees_weights_once_and_refuses_further_use(env) -> None:
    session = _session(env)
    group = env["groups"][0]
    weights = [spy.weights for spy in env["runners"]]
    session.close()
    session.close()
    assert group.closed == 1
    for w in weights:
        assert w.freed is True
    with pytest.raises(TP2GroupError, match="closed"):
        session.generate([1], max_new_tokens=1, eos_token_id=None)


def test_tp2_requires_two_devices(env) -> None:
    with pytest.raises(ValueError, match="two rank devices"):
        _session(env, devices=(0,), mode="tp2")


def test_the_step_trace_separates_stages_and_totals(env) -> None:
    env["queue_logits"]([1, 1])
    session = _session(env)
    result = session.generate([2], max_new_tokens=1, eos_token_id=None)
    trace = result.step_traces[-1]
    assert set(trace.stages.keys()) == {
        "embedding",
        "attention",
        "add_norm",
        "mlp",
        "residual_add",
        "head_sample",
    }
    assert trace.total_s >= sum(trace.stages.values()) - 1e-9
    assert trace.exchange_layers == 3
    session.close()


def test_capacity_is_validated_before_generation(env) -> None:
    session = _session(env)
    with pytest.raises(ValueError, match="capacity"):
        session.generate(list(range(60)), max_new_tokens=16, eos_token_id=None)
    session.close()


# ---------------------------------------------------------------------------
# Graphed schedule
# ---------------------------------------------------------------------------


def test_graphed_schedule_knob_is_validated_before_any_hip_init() -> None:
    with pytest.raises(ValueError, match="tp2-only"):
        tg.MlpTP2GenerationSession(
            "fake.gguf", devices=(0,), mode="tp1", schedule="graphed"
        )


def test_graphed_schedule_needs_the_compiled_driver() -> None:
    with pytest.raises(ValueError, match="compiled staged-exchange driver"):
        tg.MlpTP2GenerationSession(
            "fake.gguf", devices=(0, 1), mode="tp2", driver="python", schedule="graphed"
        )


def _graphed_session(env) -> MlpTP2GenerationSession:
    # The device-side exchange is the graphed default (see the host opt-out
    # test below).
    return MlpTP2GenerationSession(
        "fake.gguf",
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=64,
        schedule="graphed",
        reduce_mode="device",
    )


def test_graphed_schedule_captures_one_graph_per_layer_rank(env) -> None:
    env["queue_logits"]([0, 1, 2, 3])
    session = _graphed_session(env)
    session.generate([2], max_new_tokens=1, eos_token_id=None)
    rt = env["rt"]
    begins = [c for c in rt.calls if c[0] == "begin_capture"]
    ends = [c for c in rt.calls if c[0] == "end_capture"]
    # One capture per (layer, rank): 3 layers x 2 ranks, plus the warmup eager
    # token runs before any capture.
    assert len(begins) == 6
    assert len(ends) == 6
    warmup_attn = [
        c for c in env["runners"][0].calls if c[1] == "attn"
    ]
    # The warmup token is eager (3 layers) and each capture enqueues through
    # the runner once per layer (3 more); every replayed step launches graphs
    # instead, so exactly 6 eager attention calls per rank.
    assert len(warmup_attn) == 6
    session.close()


def test_graphed_steps_launch_graphs_and_reduce_on_device(env) -> None:
    env["queue_logits"]([0, 1, 2, 3])
    session = _graphed_session(env)
    session.generate([2], max_new_tokens=1, eos_token_id=None)
    rt = env["rt"]
    launches = [c for c in rt.calls if c[0] == "graph_launch"]
    # One graph per (layer, rank) per token step; the run is two steps
    # (one prefill, one decode) x 3 layers x 2 ranks.
    assert len(launches) == 12
    # Every launch runs on that rank's created non-blocking capture stream.
    rank_streams = set(env["created_streams"])
    assert {c[3] for c in launches} == rank_streams
    per_device = {c[1] for c in launches}
    assert per_device == {0, 1}
    group = env["groups"][0]
    # Device mode: the host never reduces per layer - the exchange runs
    # inside the captured graphs (one enqueue per prior layer per rank at
    # capture: layers 1,2 x 2 ranks) plus the eager tail (slot 2 x 2 ranks).
    assert group.reduces == []
    exchange = env["device_exchanges"][0]
    # Two steps, each bumping both ranks' counters once.
    assert exchange.step_begins == 2
    assert exchange.waits == 2
    slots = sorted(slot for _rank, slot, _ptr in exchange.enqueues)
    assert slots == [0, 0, 1, 1, 2, 2, 2, 2], (
        "capture-time enqueues for prior slots 0,1 (x2 ranks, once) and the tail's slot 2 (x2 ranks per step)"
    )
    ranks = sorted(rank for rank, _slot, _ptr in exchange.enqueues)
    assert ranks == [0, 0, 0, 0, 1, 1, 1, 1]
    session.close()


def test_graphed_replay_owns_position_and_token_metadata(env) -> None:
    env["queue_logits"]([0, 1, 2, 3, 4, 5, 6, 7])
    session = _graphed_session(env)
    session.generate([2, 3], max_new_tokens=2, eos_token_id=None)
    scratch = env["runners"][0].scratch
    # Capture sets the bound position (63) after the post-warmup state reset;
    # the generate's four steps (prefill 0, 1, decode 2, 3) refresh the pinned
    # position metadata eagerly every token.
    assert scratch.positions == [63, 0, 1, 2, 3]
    session.close()


def test_graphed_capture_failure_poisons_the_session(env) -> None:
    env["queue_logits"]([0, 1, 2, 3])
    session = _graphed_session(env)
    group = env["groups"][0]
    original_chain = group.enqueue_rank_chain
    chain_calls = [0]

    def failing_chain(layer_id: int, device: int, input_ptr: int) -> int:
        chain_calls[0] += 1
        if chain_calls[0] == 3:  # the third capture: (layer 1, device 0)
            raise RuntimeError("simulated capture failure")
        return original_chain(layer_id, device, input_ptr)

    group.enqueue_rank_chain = failing_chain
    with pytest.raises(tg.TP2GroupError, match="graph schedule capture failed"):
        session.generate([2], max_new_tokens=1, eos_token_id=None)
    assert session.poisoned
    # The half-captured graph is destroyed, not leaked.
    rt = env["rt"]
    assert any(c[0] == "graph_destroy" for c in rt.calls)
    session.close()


def test_graphed_close_destroys_every_captured_graph(env) -> None:
    env["queue_logits"]([0, 1, 2, 3])
    session = _graphed_session(env)
    session.generate([2], max_new_tokens=1, eos_token_id=None)
    session.close()
    rt = env["rt"]
    destroyed_execs = [c[1] for c in rt.calls if c[0] == "graph_exec_destroy"]
    destroyed_graphs = [c[1] for c in rt.calls if c[0] == "graph_destroy"]
    assert len(destroyed_execs) == 6
    assert len(destroyed_graphs) == 6
    # The created rank streams are destroyed exactly once too.
    assert sorted(env["destroyed_streams"]) == sorted(env["created_streams"])


def test_tp2_sessions_default_to_the_graphed_schedule(env) -> None:
    session = MlpTP2GenerationSession("fake.gguf", devices=(0, 1), mode="tp2")
    assert session.schedule == "graphed"
    # The graphed default reduces on device inside the captured graphs; the
    # host-summed transport is the explicit opt-out.
    assert session.reduce_mode == "device"
    session.close()


def test_tp1_controls_default_to_the_eager_schedule(env) -> None:
    session = _session(env, devices=(0,), mode="tp1")
    assert session.schedule == "eager"
    session.close()


def test_reduce_mode_defaults_and_validation() -> None:
    # Graphed tp2 defaults to the device-side reduction; eager to the host
    # transport. Device requires graphed.
    with pytest.raises(ValueError, match="device-side reduction needs the graphed"):
        tg.MlpTP2GenerationSession(
            "fake.gguf", devices=(0,), mode="tp1", schedule="eager", reduce_mode="device"
        )
    with pytest.raises(ValueError, match="unknown reduce_mode"):
        tg.MlpTP2GenerationSession(
            "fake.gguf", devices=(0, 1), mode="tp2", reduce_mode="p2p"
        )


def test_graphed_host_reduce_opt_out_reduces_per_layer(env) -> None:
    env["queue_logits"]([0, 1, 2, 3])
    session = MlpTP2GenerationSession(
        "fake.gguf",
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=64,
        schedule="graphed",
        reduce_mode="host",
    )  # the explicit opt-out
    session.generate([2], max_new_tokens=1, eos_token_id=None)
    group = env["groups"][0]
    # Host mode keeps the per-layer host-driven transport reduction, with the
    # captured prefix casting the fixed mapped payload slot.
    assert [slot for _ptrs, slot in group.reduces] == [0, 1, 2] * 2
    assert env["device_exchanges"] == []
    session.close()


def test_head_shard_validation_and_default_on(env) -> None:
    # The sharded head is the tp2 production default; the replicated head is
    # the explicit opt-out and bisection control.
    session = MlpTP2GenerationSession("fake.gguf", devices=(0, 1), mode="tp2")
    assert session.head_shard is True
    assert len(env["head_plans"]) == 1
    session.close()
    session = MlpTP2GenerationSession(
        "fake.gguf", devices=(0, 1), mode="tp2", head_shard=False
    )
    assert session.head_shard is False
    assert len(env["head_plans"]) == 1  # no new plan for the opt-out
    session.close()
    # The sharded head is tp2-only.
    with pytest.raises(ValueError, match="tp2-only"):
        MlpTP2GenerationSession(
            "fake.gguf", devices=(0,), mode="tp1", head_shard=True
        )


def test_head_shard_builds_per_rank_shards_and_defers_the_replica(env) -> None:
    session = MlpTP2GenerationSession(
        "fake.gguf",
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=64,
        head_shard=True,
    )
    # The runners defer the replicated head; each rank uploads its shard from
    # the runtime-resolved layout, never a guessed one.
    assert [runner.deferred_device_slots for runner in env["runners"]] == [
        ("root.lm_head",),
        ("root.lm_head",),
    ]
    assert [entry[0] for entry in env["head_uploads"]] == [0, 1]
    assert all(
        layout == "gguf_q6_k_t16_qmicro_planar_v1"
        for _device, layout, _quant in env["head_uploads"]
    )
    assert session._head_plan.rows_per_rank == VOCAB // 2
    session.close()
    assert sorted(env["head_freed"]) == [0, 1]


def test_head_shard_finish_concatenates_shards_with_exact_tie_break(env) -> None:
    # Rank 0's shard holds the maximum at vocab index 3 (rank 0's range), and
    # rank 1's shard holds an equal value at index VOCAB // 2 + 3; the exact
    # global first-maximum tie-break must pick index 3 - the same token the
    # replicated head's argmax would return.
    row0 = np.zeros(VOCAB // 2, dtype="<f4")
    row0[3] = 5.0
    row1 = np.zeros(VOCAB // 2, dtype="<f4")
    row1[3] = 5.0  # rank 1's local index 3 = global index VOCAB // 2 + 3
    # Two readbacks per finish step, and the eager schedule has no capture
    # warmup: exactly a prefill pair then the decode pair. The generated
    # token is the prefill's argmax, so the marked pair comes first.
    env["injected_rows"].extend([row0, row1])
    env["injected_rows"].extend(
        [np.zeros(VOCAB // 2, dtype="<f4"), np.zeros(VOCAB // 2, dtype="<f4")]
    )
    session = MlpTP2GenerationSession(
        "fake.gguf",
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=64,
        schedule="eager",
        head_shard=True,
    )
    result = session.generate([2], max_new_tokens=1, eos_token_id=None)
    assert result.token_ids == (3,)
    # Both ranks enqueue their shard GEMV in the tail (rmsnorm + linear per
    # rank), and the returned logits row is the concatenation.
    tail_linears = [
        entry for entry in env["launch_log"] if entry[1] == "linear" and entry[2] == "launch"
    ]
    assert {entry[0] for entry in tail_linears} == {0, 1}
    full_row = result.logits[0] if result.logits is not None else None
    assert full_row is None or float(full_row[3]) == 5.0
    session.close()
    assert sorted(env["head_freed"]) == [0, 1]


# ---------------------------------------------------------------------------
# Opt-in rank-local bulk prefill
# ---------------------------------------------------------------------------


def _bulk_session(
    env,
    *,
    rows=8,
    head_shard=False,
    schedule="eager",
    mode="tp2",
):
    return MlpTP2GenerationSession(
        "fake.gguf",
        devices=(0, 1),
        mode=mode,
        max_sequence_length=64,
        schedule=schedule,
        head_shard=head_shard,
        bulk_prefill=True,
        bulk_prefill_rows=rows,
    )


def _bulk_group(env):
    # The decode group is built first, then the bulk (weight-sharing) group.
    assert len(env["groups"]) == 2
    return env["groups"][0], env["groups"][1]


def test_decode_partial_dtype_selects_the_decode_group_only(env) -> None:
    """The f32 down partial is a decode/serial-prefill choice.

    The bulk prefill group's rows>1 f32-out leaves are not registered, so it
    must keep its own dtype; a single knob must not silently move both routes.
    """

    default = _session(env)
    assert env["groups"][0].staging_dtype == "bf16"
    assert default.decode_partial_dtype == "bf16"

    env["groups"].clear()
    session = MlpTP2GenerationSession(
        "fake.gguf",
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=64,
        schedule="eager",
        bulk_prefill=True,
        bulk_prefill_rows=8,
        decode_partial_dtype="f32",
    )
    decode_group, bulk_group = _bulk_group(env)
    assert session.decode_partial_dtype == "f32"
    assert decode_group.staging_dtype == "f32"
    assert bulk_group.staging_dtype == "bf16"


@pytest.mark.parametrize("value", ["fp16", "", "F32"])
def test_decode_partial_dtype_rejects_unknown_values(env, value) -> None:
    with pytest.raises(ValueError, match="decode_partial_dtype"):
        MlpTP2GenerationSession(
            "fake.gguf",
            devices=(0, 1),
            mode="tp2",
            max_sequence_length=64,
            schedule="eager",
            decode_partial_dtype=value,
        )
    assert env["groups"] == []


def test_f32_partial_dtype_requires_the_host_reduction(env) -> None:
    """The device exchange stages bf16 rows; f32 must not reach it.

    Silent corruption, not a slowdown: the driver's ``row_bytes = hidden * 2``
    would read half of each f32 row and spin-sum the halves as bf16 values.
    """

    with pytest.raises(ValueError, match="reduce_mode='host'"):
        MlpTP2GenerationSession(
            "fake.gguf",
            devices=(0, 1),
            mode="tp2",
            max_sequence_length=64,
            schedule="graphed",
            decode_partial_dtype="f32",
        )
    assert env["groups"] == []

    # The graphed schedule with the host reduction is the supported f32 route.
    session = MlpTP2GenerationSession(
        "fake.gguf",
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=64,
        schedule="graphed",
        reduce_mode="host",
        decode_partial_dtype="f32",
    )
    assert session.reduce_mode == "host"
    assert session.decode_partial_dtype == "f32"
    assert env["groups"][0].staging_dtype == "f32"


def test_bulk_prefill_is_off_by_default(env) -> None:
    session = _session(env)
    assert session.bulk_prefill_enabled is False
    assert session.prefill_schedule == "token-serial"
    assert len(env["groups"]) == 1
    with pytest.raises(TP2GroupError, match="not enabled"):
        session.bulk_prefill([1, 2])
    session.close()


def test_bulk_prefill_requires_tp2(env) -> None:
    with pytest.raises(ValueError, match="tp2-only"):
        MlpTP2GenerationSession(
            "fake.gguf",
            devices=(0,),
            mode="tp1",
            max_sequence_length=64,
            schedule="eager",
            bulk_prefill=True,
        )


def test_bulk_prefill_validates_its_capacity(env) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _bulk_session(env, rows=0)
    with pytest.raises(ValueError, match="exceeds the session capacity"):
        _bulk_session(env, rows=128)


def test_bulk_prefill_schedule_is_reported_as_bulk(env) -> None:
    session = _bulk_session(env, rows=4)
    assert session.prefill_schedule == "bulk-tp2"
    session.close()


def test_bulk_prefill_rejects_over_capacity_before_any_launch(env) -> None:
    session = _bulk_session(env, rows=4)
    _decode, bulk = _bulk_group(env)
    env["launch_log"].clear()
    with pytest.raises(ValueError, match="exceeds the bulk prefill capacity"):
        session.bulk_prefill([1, 2, 3, 4, 5])
    assert env["launch_log"] == []
    assert bulk.forwards == []
    session.close()


def test_bulk_prefill_rejects_tokens_outside_the_vocabulary(env) -> None:
    session = _bulk_session(env, rows=4)
    env["launch_log"].clear()
    with pytest.raises(ValueError, match="outside"):
        session.bulk_prefill([1, VOCAB])
    assert env["launch_log"] == []
    session.close()


def test_bulk_prefill_runs_attention_norm_one_mlp_and_one_residual(env) -> None:
    session = _bulk_session(env, rows=4)
    env["injected_rows"].append(np.zeros(4 * VOCAB, dtype="<f4"))
    session.bulk_prefill([1, 2, 3, 4])
    expected = [
        "linear_prefill_attn",
        "prefill_norm",
        "full_prefill_attn",
        "prefill_norm",
        "linear_prefill_attn",
        "prefill_norm",
    ]
    for runner in env["runners"]:
        assert [call[1] for call in runner.calls] == expected
    _decode, bulk = _bulk_group(env)
    # Exactly one sharded MLP exchange per layer, and one residual add per
    # rank per layer (the group never adds the residual itself).
    assert bulk.forwards == [0, 1, 2]
    adds = [entry for entry in env["launch_log"] if entry[1] == "bf16_add"]
    assert len(adds) == 6
    session.close()


def test_bulk_prefill_full_attention_reuses_the_decode_kv_cache(env) -> None:
    session = _bulk_session(env, rows=4)
    env["injected_rows"].append(np.zeros(4 * VOCAB, dtype="<f4"))
    session.bulk_prefill([1, 2, 3, 4])
    for runner in env["runners"]:
        # Only the single full-attention layer reads the shared decode cache.
        assert runner.scratch.full_cache_calls == [1]
    session.close()


def test_bulk_prefill_builds_the_decode_schedule_before_writing_state(
    env, monkeypatch
) -> None:
    session = _bulk_session(env, rows=4)
    order: list[str] = []
    original_ensure = session._ensure_graph_schedule

    def spy_ensure():
        order.append("schedule")
        original_ensure()

    monkeypatch.setattr(session, "_ensure_graph_schedule", spy_ensure)
    original_copy = tg.copy_host_to_device

    def spy_copy(*args, **kwargs):
        order.append("upload")
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(tg, "copy_host_to_device", spy_copy)
    env["injected_rows"].append(np.zeros(4 * VOCAB, dtype="<f4"))
    session.bulk_prefill([1, 2, 3, 4])
    assert order[0] == "schedule"
    assert "upload" in order
    session.close()


def test_bulk_prefill_graphed_capture_cannot_overwrite_prefill_state(env) -> None:
    session = _bulk_session(env, rows=4, schedule="graphed")
    # The graph warmup runs one eager decode step (one replicated-head row)
    # before the bulk prefill's own whole-batch readback.
    env["injected_rows"].append(_logits_row(0))
    env["injected_rows"].append(np.zeros(4 * VOCAB, dtype="<f4"))
    before = [runner.scratch.zeroed for runner in env["runners"]]
    session.bulk_prefill([1, 2, 3, 4])
    assert session._graph_schedule_ready is True
    # The capture zeroes state during warmup; the bulk path then zeroes it
    # again and writes the prefill KV/GDN state after every capture.
    for runner, baseline in zip(env["runners"], before):
        assert runner.scratch.zeroed > baseline
    session.close()


def test_bulk_prefill_reuses_weights_without_double_free(env) -> None:
    session = _bulk_session(env, rows=4)
    decode, bulk = _bulk_group(env)
    assert decode.owns_weights is True
    assert bulk.owns_weights is False
    session.close()
    assert decode.closed == 1
    assert bulk.closed == 1


def test_bulk_prefill_is_repeatable_and_resets_state(env) -> None:
    session = _bulk_session(env, rows=4)
    env["injected_rows"].extend(
        [
            np.zeros(4 * VOCAB, dtype="<f4"),
            np.zeros(4 * VOCAB, dtype="<f4"),
        ]
    )
    before = [runner.scratch.zeroed for runner in env["runners"]]
    session.bulk_prefill([1, 2, 3, 4])
    session.bulk_prefill([5, 6, 7, 8])
    for runner, baseline in zip(env["runners"], before):
        assert runner.scratch.zeroed == baseline + 2
    session.close()


def test_bulk_prefill_returns_per_position_logits(env) -> None:
    session = _bulk_session(env, rows=3)
    full = np.arange(3 * VOCAB, dtype="<f4").reshape(3, VOCAB)
    env["injected_rows"].append(np.ascontiguousarray(full).reshape(-1))
    logits = session.bulk_prefill([1, 2, 3])
    assert logits.shape == (3, VOCAB)
    np.testing.assert_array_equal(logits, full)
    session.close()


def test_bulk_prefill_sharded_head_concatenates_per_rank_shards(env) -> None:
    session = _bulk_session(env, rows=2, head_shard=True)
    half = VOCAB // 2
    # The tail reads rank 0's then rank 1's whole-batch shard, in device order.
    env["injected_rows"].append(np.full(2 * half, 1.0, dtype="<f4"))
    env["injected_rows"].append(np.full(2 * half, 2.0, dtype="<f4"))
    logits = session.bulk_prefill([1, 2])
    assert logits.shape == (2, VOCAB)
    np.testing.assert_array_equal(logits[:, :half], np.full((2, half), 1.0))
    np.testing.assert_array_equal(logits[:, half:], np.full((2, half), 2.0))
    session.close()


def test_bulk_prefill_transitions_to_decode_at_the_prompt_length(env) -> None:
    session = _bulk_session(env, rows=4)
    env["injected_rows"].append(np.zeros(4 * VOCAB, dtype="<f4"))
    session.bulk_prefill([1, 2, 3, 4])
    env["injected_rows"].append(_logits_row(7))
    logits, _trace = session._forward_token(9, 4, kind="decode")
    assert int(np.argmax(logits)) == 7
    for runner in env["runners"]:
        assert 4 in runner.scratch.positions
    session.close()


def test_bulk_prefill_failure_poisons_the_session(env) -> None:
    session = _bulk_session(env, rows=4)
    env["runners"][0].fail_on_layer = 1
    with pytest.raises(TP2GroupError, match="bulk prefill failed"):
        session.bulk_prefill([1, 2, 3, 4])
    assert session.poisoned is True
    with pytest.raises(TP2GroupError, match="poisoned"):
        session.bulk_prefill([1, 2, 3, 4])
    session.close()


def test_bulk_prefill_sets_the_active_chunk_metadata(env) -> None:
    session = _bulk_session(env, rows=8)
    env["injected_rows"].append(np.zeros(4 * VOCAB, dtype="<f4"))
    session.bulk_prefill([1, 2, 3, 4])
    # The bulk scratch is allocated at capacity and must be narrowed to the
    # active prompt rows exactly like the resident bulk caller's for_chunk.
    assert env["bulk_for_chunk_calls"] == [(8, 0, 4, 4), (8, 0, 4, 4)]
    session.close()


def test_bulk_full_attention_uses_the_active_prompt_rows(env) -> None:
    session = _bulk_session(env, rows=8)
    env["injected_rows"].append(np.zeros(4 * VOCAB, dtype="<f4"))
    session.bulk_prefill([1, 2, 3, 4])
    for runner in env["runners"]:
        # linear-attention layers pass rows explicitly
        assert runner.prefill_attn_rows_arg == [4, 4]
        # the full-attention helper has no explicit rows argument; it reads
        # scratch.rows, so the scratch it receives must carry the active
        # prompt rows (4), never the bulk capacity (8)
        assert runner.prefill_attn_scratch_rows == [4, 4, 4]
    session.close()


# --------------------------------------------------------------------------
# dispatch-context contract
# --------------------------------------------------------------------------

#: The session-scoped GGUF linear dispatch owners the resident bulk prefill
#: establishes for this model and request shape. ``wmma_prefill`` is the one
#: that changes the layer-0 Q6_K ``attn_qkv`` leaf (measured: the Q4_K
#: ``attn_gate`` projection from the same launch group is already on its WMMA
#: variant and stays bit-identical). ``q8_t16_two_wave_prefill`` and
#: ``q4_t16_unequal_pair_prefill`` are admitted for this geometry by the
#: backend package; the two dual-WMMA policies have no admitted rows.
RESIDENT_BULK_DISPATCH_CONTEXT = {
    "wmma_prefill": True,
    "gemv_decode": True,
    "q8_t16_two_wave_prefill": True,
    "q8_t16_dual_wmma_prefill": False,
    "q4_pack8_dual_wmma_silu_prefill": False,
    "q4_t16_unequal_pair_prefill": False,
    "t16_f16_rocblas_prefill": False,
}


def test_bulk_prefill_establishes_the_resident_dispatch_context(env) -> None:
    """The bulk prefill must select the same kernels as the resident teacher.

    ``Qwen35GGUFResidentSession`` wraps its whole bulk prefill in the
    session-scoped dispatch owners (``wmma_prefill_session``,
    ``gemv_decode_session``, the q8_t16/q4 pair sessions). The shared layer
    helper reads those process-global toggles when it resolves a GGUF linear
    kernel, so a route that calls the helper outside them runs different
    arithmetic on the same weights. The GPU layer-0 comparison shows exactly
    that: with identical inputs, weights and initial state, the Q6_K
    ``attn_qkv`` projection differs (rel 3.7e-03) while the Q4_K ``attn_gate``
    projection from the same launch group is bit-identical.

    The owners are process-global, so every rank must set them around its own
    work - and restore them - rather than relying on one entry around the whole
    layer loop.
    """

    session = _bulk_session(env, rows=4)
    env["injected_rows"].append(np.zeros(4 * VOCAB, dtype="<f4"))
    session.bulk_prefill([1, 2, 3, 4])
    _group, bulk_group = _bulk_group(env)
    for runner in env["runners"]:
        assert runner.prefill_dispatch_context, "helper never ran"
        for context in runner.prefill_dispatch_context:
            assert context == RESIDENT_BULK_DISPATCH_CONTEXT, context
        assert runner.prefill_norm_dispatch_context
        for context in runner.prefill_norm_dispatch_context:
            assert context == RESIDENT_BULK_DISPATCH_CONTEXT, context
    # The shard chain launches its own GGUF linears; it must run under the same
    # owners even though the group forwards once for both ranks.
    assert bulk_group.forward_dispatch_context
    for context in bulk_group.forward_dispatch_context:
        assert context == RESIDENT_BULK_DISPATCH_CONTEXT, context
    # The head projection is the one bulk launch that is NOT a resident layer
    # helper: the resident samples through its dedicated lm_head kernel, and
    # the f32-out WMMA variant is not registered. It must keep its registered
    # ambient route.
    assert env["linear_dispatch_context"]
    for context in env["linear_dispatch_context"]:
        assert context["wmma_prefill"] is False, context
    session.close()


def test_bulk_prefill_restores_the_dispatch_context_after_each_rank(env) -> None:
    """The process-global owners must be back to their prior state after bulk."""

    before = dict(gguf_linear.gguf_prefill_dispatch_context())
    session = _bulk_session(env, rows=4)
    env["injected_rows"].append(np.zeros(4 * VOCAB, dtype="<f4"))
    session.bulk_prefill([1, 2, 3, 4])
    assert dict(gguf_linear.gguf_prefill_dispatch_context()) == before
    session.close()
    assert dict(gguf_linear.gguf_prefill_dispatch_context()) == before


def test_bulk_prefill_restores_the_dispatch_context_after_a_failure(env) -> None:
    """A mid-layer failure must not leave the owners set for the next caller."""

    before = dict(gguf_linear.gguf_prefill_dispatch_context())
    session = _bulk_session(env, rows=4)
    env["runners"][1].fail_on_layer = 0
    with pytest.raises(tg.TP2GroupError):
        session.bulk_prefill([1, 2, 3, 4])
    assert dict(gguf_linear.gguf_prefill_dispatch_context()) == before
    session.close()


def test_bulk_prefill_context_is_per_rank_not_per_layer_loop(env) -> None:
    """Every rank's helper must see the context on that rank's own device.

    The owners are process-global, so establishing them once around the whole
    multi-rank layer loop would set one rank's context while the other rank
    ran. Each spy records the logical device of every call it served, so this
    pins that each rank both ran and observed the resident policy.
    """

    session = _bulk_session(env, rows=4)
    env["injected_rows"].append(np.zeros(4 * VOCAB, dtype="<f4"))
    session.bulk_prefill([1, 2, 3, 4])
    assert len(env["runners"]) == 2
    all_devices: set[int] = set()
    for runner in env["runners"]:
        devices = {device for device, _name, _layer in runner.calls}
        assert len(devices) == 1, runner.calls
        all_devices |= devices
        assert runner.prefill_dispatch_context
        for context in runner.prefill_dispatch_context:
            assert context == RESIDENT_BULK_DISPATCH_CONTEXT, context
    assert all_devices == {0, 1}
    session.close()


def test_bulk_prefill_wmma_override_changes_only_the_bulk_context(env) -> None:
    """The route A/B override is read only while bulk prefill runs."""

    before = dict(gguf_linear.gguf_prefill_dispatch_context())
    session = MlpTP2GenerationSession(
        "fake.gguf",
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=64,
        schedule="eager",
        head_shard=False,
        bulk_prefill=True,
        bulk_prefill_rows=4,
        use_wmma_prefill=False,
        use_gemv_decode=False,
    )
    assert session.use_wmma_prefill is False
    assert session.use_gemv_decode is False
    env["injected_rows"].append(np.zeros(4 * VOCAB, dtype="<f4"))
    session.bulk_prefill([1, 2, 3, 4])
    expected = dict(RESIDENT_BULK_DISPATCH_CONTEXT)
    expected["wmma_prefill"] = False
    expected["gemv_decode"] = False
    for runner in env["runners"]:
        assert runner.prefill_dispatch_context
        for context in runner.prefill_dispatch_context:
            assert context == expected, context
    # ... and the override never leaks out of the bulk call.
    assert dict(gguf_linear.gguf_prefill_dispatch_context()) == before
    session.close()


def test_bulk_prefill_defaults_to_the_shipped_resident_prefill_policy() -> None:
    """A bulk session with no override uses the resident session's policy."""

    assert (
        resident_session_wmma_prefill_default()
        is True
    ), "the shipped resident prefill route changed"


def test_single_row_decode_does_not_enter_the_bulk_dispatch_context(env) -> None:
    """The single-row route keeps its own per-launch arguments."""

    session = _session(env)
    for _ in range(8):
        env["injected_rows"].append(np.zeros(VOCAB, dtype="<f4"))
    session.generate([1, 2], max_new_tokens=1)
    # Decode passes ``use_gemv_decode`` explicitly per launch and never enters
    # the bulk owners: every launch it makes sees the ambient context, not the
    # resident bulk policy.
    assert env["linear_dispatch_context"]
    for context in env["linear_dispatch_context"]:
        assert context["wmma_prefill"] is False, context
        assert context != RESIDENT_BULK_DISPATCH_CONTEXT, context
    assert not any(
        runner.prefill_dispatch_context for runner in env["runners"]
    )
    session.close()
