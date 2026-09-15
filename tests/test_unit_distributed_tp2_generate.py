"""CPU tests for the TP2 model-owning generation driver.

No ROCm is required: the driver's HIP runtime, resident runners, scratch,
shard group and every launcher are fakes/spies, so the tests pin the
schedule - the per-layer rank interleave, the single residual add per rank,
the control rank's head/sampling, position and token ownership, EOS, reset,
poison-on-failure, and teardown - without touching hardware.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import hipengine.distributed.tp2_generate as tg
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
        self.positions: list[int] = []
        self.zeroed = 0

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
        self.hidden_size = HIDDEN
        self.vocab_size = VOCAB
        self.ffn_size = FFN
        self.fail_on_layer: int | None = None
        self.scratch: FakeScratch | None = None

    def _run_linear_attention_attn_only(self, layer_id, hidden_ptr, attn_out, scratch, **kwargs):
        self._attn(layer_id)

    def _run_full_attention_attn_only(self, layer_id, hidden_ptr, attn_out, scratch, **kwargs):
        self._attn(layer_id)

    def _attn(self, layer_id: int) -> None:
        if self.fail_on_layer is not None and layer_id == self.fail_on_layer:
            raise RuntimeError("simulated attention failure")
        self.calls.append((self.rt.get_device(), "attn", int(layer_id)))


class FakeShardGroup:
    def __init__(self) -> None:
        self.forwards: list[int] = []
        self.closed = 0
        self.exchange_walls_s: list[float] = []
        self.fail_on_layer: int | None = None
        self.chains: list[tuple[int, int]] = []
        self.reduces: list[tuple[tuple[int, ...], int | None]] = []

    def forward(self, layer_id: int, inputs):
        if self.fail_on_layer is not None and layer_id == self.fail_on_layer:
            raise RuntimeError("simulated shard failure")
        self.forwards.append(int(layer_id))
        self.exchange_walls_s.append(0.0)
        return {0: 0x5000, 1: 0x5100}

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
    launch_log: list[tuple[int, str, str]] = []

    def fake_runtime():
        return rt

    def fake_runner(model_path, *, backend, execution_routes, runtime):
        spy = RunnerSpy(rt, LAYER_TYPES)
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

    def fake_add(a_ptr, b_ptr, out_ptr, n, **kwargs):
        launch_log.append((rt.get_device(), "bf16_add", "add"))

    def fake_f32_to_bf16(src_ptr, out_ptr, n, **kwargs):
        launch_log.append((rt.get_device(), "f32_to_bf16", "cast"))

    def fake_copy_d2h(host_ptr, buffer, nbytes=None, *, runtime=None):
        row = logit_rows.pop(0)
        ctypes.memmove(host_ptr, row.tobytes(), row.nbytes)

    def fake_malloc(nbytes, *, runtime=None, device=None):
        return FakeBuffer(0x8000 + len(launch_log) * 0x40, nbytes)

    def fake_free(buffer, *, runtime=None):
        launch_log.append((rt.get_device(), "free", "free"))

    class FakeGroupFactory(FakeShardGroup):
        def __init__(self, runtime, **kwargs):
            super().__init__()
            groups.append(self)

    def fake_resolve(model_path, *, world_size, backend="hip_gfx1100"):
        return None, {}, None, FakeConfig(LAYER_TYPES)

    def fake_materialize(model_path, *, world_size, layer_ids=None, backend="hip_gfx1100"):
        return {}

    def fake_upload(runtime, shards, *, devices):
        return {}

    monkeypatch.setattr(tg, "get_hip_runtime", fake_runtime)
    monkeypatch.setattr(tg, "Qwen35GGUFFullStackRunner", fake_runner)
    monkeypatch.setattr(tg, "_FullStackScratch", FakeScratchFactory)
    monkeypatch.setattr(tg, "_gguf_norm_residual_decode_kernel", fake_norm_kernel)
    monkeypatch.setattr(tg, "launch_gguf_embedding", fake_launch("embedding"))
    monkeypatch.setattr(tg, "launch_gguf_linear", fake_launch("linear"))
    monkeypatch.setattr(tg, "gguf_bf16_add", fake_add)
    monkeypatch.setattr(tg, "gguf_rmsnorm_bf16_f32_weight", fake_launch("rmsnorm"))
    monkeypatch.setattr(tg, "f32_to_bf16", fake_f32_to_bf16)
    monkeypatch.setattr(tg, "silu_mul_separate_out_bf16", fake_launch("silu"))
    monkeypatch.setattr(tg, "copy_device_to_host", fake_copy_d2h)
    monkeypatch.setattr(tg, "malloc", fake_malloc)
    monkeypatch.setattr(tg, "free", fake_free)
    monkeypatch.setattr(tg, "MlpShardGroup", FakeGroupFactory)
    monkeypatch.setattr(tg, "resolve_mlp_shard_context", fake_resolve)
    monkeypatch.setattr(tg, "materialize_mlp_shards", fake_materialize)
    monkeypatch.setattr(tg, "upload_mlp_shard_weights", fake_upload)

    def queue_logits(preferred_ids):
        logit_rows.extend(_logits_row(i) for i in preferred_ids)

    return {
        "rt": rt,
        "runners": runner_spies,
        "groups": groups,
        "launch_log": launch_log,
        "queue_logits": queue_logits,
        "created_streams": created_streams,
        "destroyed_streams": destroyed_streams,
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
    env["queue_logits"]([21, 22, 31, 0])
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
    return MlpTP2GenerationSession(
        "fake.gguf",
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=64,
        schedule="graphed",
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


def test_graphed_steps_launch_one_graph_per_layer_rank_and_reduce_per_layer(env) -> None:
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
    # Every layer's exchange publishes into that layer's fixed payload slot,
    # on both steps.
    assert [slot for _ptrs, slot in group.reduces] == [0, 1, 2] * 2
    assert [ptrs for ptrs, _slot in group.reduces] == [
        (0x5200 + 0 + 64 * l, 0x5200 + 1 + 64 * l) for l in range(3)
    ] * 2
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
    session.close()


def test_tp1_controls_default_to_the_eager_schedule(env) -> None:
    session = _session(env, devices=(0,), mode="tp1")
    assert session.schedule == "eager"
    session.close()
