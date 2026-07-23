from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.loading.laguna_gguf_materialize import LAYOUT_DENSE_F16, LAYOUT_RAW_GGUF
from hipengine.runtime import laguna_gguf_runner as runner_module
from hipengine.runtime.laguna_gguf_runner import (
    LAGUNA_DFLASH_CAPTURE_DEPTHS,
    LagunaEagerScratch,
    LagunaHiddenCaptureTargets,
    LagunaRowsScratch,
    capture_laguna_hidden_rows,
    capture_laguna_hidden_tap,
    capture_laguna_routing_rows,
    resolve_laguna_eager_kernel_plan,
)
from tests._laguna_synthetic import make_laguna_info


class _FakeRuntime:
    def __init__(self, *, fail_malloc_at: int | None = None) -> None:
        self.next_ptr = 0x40000000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, int, HipMemcpyKind, int]] = []
        self.fail_malloc_at = fail_malloc_at
        self.malloc_calls = 0

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        if self.fail_malloc_at == self.malloc_calls:
            raise MemoryError("synthetic Laguna eager scratch failure")
        ptr = self.next_ptr
        self.next_ptr += max(0x1000, int(nbytes) + 0x100)
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))
        self.allocations.pop(int(ptr), None)

    def memcpy_async(
        self,
        dst: int,
        src: int,
        nbytes: int,
        kind: HipMemcpyKind,
        stream: int,
    ) -> None:
        self.copies.append((int(dst), int(src), int(nbytes), kind, int(stream)))


def _config():
    return laguna_gguf_config_from_metadata(make_laguna_info())


def test_laguna_eager_plan_resolves_only_concrete_gfx1151_keys() -> None:
    plan = resolve_laguna_eager_kernel_plan(_config(), backend="hip_gfx1151")

    assert plan.backend == "hip_gfx1151"
    assert all(key.backend == "hip_gfx1151" for key in plan.kernel_keys)
    assert (
        plan.rmsnorm_key.layer,
        plan.rmsnorm_key.quant,
        plan.rmsnorm_key.variant,
    ) == ("rmsnorm", "gguf_f32_weight", "bf16_out")
    assert (
        plan.add_rmsnorm_key.layer,
        plan.add_rmsnorm_key.quant,
        plan.add_rmsnorm_key.variant,
    ) == ("add_rmsnorm", "gguf_f32_weight", "bf16_out")
    assert (
        plan.attention_gate_key.layer,
        plan.attention_gate_key.quant,
        plan.attention_gate_key.variant,
    ) == ("attention_gate", "f32", "softplus_broadcast_bf16_out")
    assert (plan.argmax_key.layer, plan.argmax_key.quant, plan.argmax_key.variant) == (
        "argmax",
        "f32",
        "top1_i64",
    )


def test_laguna_eager_scratch_is_bounded_by_max_head_width_and_frees() -> None:
    runtime = _FakeRuntime()
    scratch = LagunaEagerScratch.allocate(_config(), runtime=runtime)

    assert scratch.max_query_width == 72 * 128
    assert scratch.query.nbytes == scratch.max_query_width * DType.FP32.itemsize
    assert scratch.gate_logits.nbytes == 72 * DType.FP32.itemsize
    assert scratch.dense_gate.nbytes == 12_288 * DType.BF16.itemsize
    assert scratch.logits.nbytes == 100_352 * DType.FP32.itemsize
    assert scratch.nbytes == sum(buffer.nbytes for buffer in scratch.buffers)
    assert len(runtime.allocations) == len(scratch.buffers)

    scratch.free(runtime=runtime)
    assert runtime.allocations == {}
    scratch.free(runtime=runtime)


def test_laguna_eager_scratch_cleans_partial_allocation_failure() -> None:
    runtime = _FakeRuntime(fail_malloc_at=9)

    with pytest.raises(MemoryError, match="synthetic Laguna eager"):
        LagunaEagerScratch.allocate(_config(), runtime=runtime)

    assert runtime.allocations == {}


def test_laguna_rows_scratch_is_bounded_and_frees() -> None:
    runtime = _FakeRuntime()
    scratch = LagunaRowsScratch.allocate(_config(), max_rows=8, runtime=runtime)

    assert scratch.max_rows == 8
    assert scratch.token_ids.nbytes == 8 * DType.INT64.itemsize
    assert scratch.positions.nbytes == 8 * DType.INT64.itemsize
    assert scratch.hidden.nbytes == 8 * 3_072 * DType.BF16.itemsize
    assert scratch.query.nbytes == 8 * 72 * 128 * DType.FP32.itemsize
    assert scratch.logits.nbytes == 8 * 100_352 * DType.FP32.itemsize
    assert scratch.nbytes == sum(buffer.nbytes for buffer in scratch.buffers)

    scratch.free(runtime=runtime)
    assert runtime.allocations == {}

    with pytest.raises(ValueError, match="max_rows"):
        LagunaRowsScratch.allocate(_config(), max_rows=0, runtime=runtime)


def test_laguna_routing_replay_copies_each_sparse_layer_to_a_bounded_plane() -> None:
    runtime = _FakeRuntime()
    rows = 3
    top_k = 10
    layers = 47
    layer_nbytes = rows * top_k * DType.INT64.itemsize
    capture = DeviceBuffer(0x71000000, layers * layer_nbytes)

    capture_laguna_routing_rows(
        0x12340000,
        layer_id=1,
        leading_dense_layers=1,
        sparse_layers=layers,
        rows=rows,
        top_k=top_k,
        capture=capture,
        runtime=runtime,
        stream=5,
    )
    capture_laguna_routing_rows(
        0x22340000,
        layer_id=47,
        leading_dense_layers=1,
        sparse_layers=layers,
        rows=rows,
        top_k=top_k,
        capture=capture,
        runtime=runtime,
        stream=5,
    )

    assert runtime.copies == [
        (
            capture.ptr,
            0x12340000,
            layer_nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            5,
        ),
        (
            capture.ptr + 46 * layer_nbytes,
            0x22340000,
            layer_nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            5,
        ),
    ]

    with pytest.raises(ValueError, match="capture buffer"):
        capture_laguna_routing_rows(
            0x12340000,
            layer_id=1,
            leading_dense_layers=1,
            sparse_layers=layers,
            rows=rows,
            top_k=top_k,
            capture=DeviceBuffer(capture.ptr, capture.nbytes - 8),
            runtime=runtime,
        )
    with pytest.raises(ValueError, match="sparse layer range"):
        capture_laguna_routing_rows(
            0x12340000,
            layer_id=0,
            leading_dense_layers=1,
            sparse_layers=layers,
            rows=rows,
            top_k=top_k,
            capture=capture,
            runtime=runtime,
        )


def test_laguna_hidden_taps_are_caller_owned_exact_bf16_depths() -> None:
    hidden_size = 3_072
    row_nbytes = hidden_size * DType.BF16.itemsize
    targets = LagunaHiddenCaptureTargets(
        hidden_size=hidden_size,
        buffers={
            depth: DeviceBuffer(0x50000000 + index * 0x10000, row_nbytes)
            for index, depth in enumerate(LAGUNA_DFLASH_CAPTURE_DEPTHS)
        },
    )
    runtime = _FakeRuntime()

    capture_laguna_hidden_tap(
        0x12340000,
        depth=11,
        targets=None,
        hidden_size=hidden_size,
        runtime=runtime,
        stream=7,
    )
    assert runtime.copies == []

    capture_laguna_hidden_tap(
        0x12340000,
        depth=11,
        targets=targets,
        hidden_size=hidden_size,
        runtime=runtime,
        stream=7,
    )
    assert runtime.copies == [
        (
            targets.buffers[11].ptr,
            0x12340000,
            row_nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            7,
        )
    ]

    with pytest.raises(ValueError, match="configured DFlash depths"):
        LagunaHiddenCaptureTargets(
            hidden_size=hidden_size,
            buffers={3: DeviceBuffer(0x60000000, row_nbytes)},
        )
    with pytest.raises(ValueError, match="exactly one BF16 hidden row"):
        LagunaHiddenCaptureTargets(
            hidden_size=hidden_size,
            buffers={2: DeviceBuffer(0x60000000, row_nbytes - 2)},
        )
    with pytest.raises(ValueError, match="hidden_size"):
        capture_laguna_hidden_tap(
            0x12340000,
            depth=2,
            targets=targets,
            hidden_size=hidden_size // 2,
            runtime=runtime,
        )

    row_targets = LagunaHiddenCaptureTargets(
        hidden_size=hidden_size,
        buffers={2: DeviceBuffer(0x70000000, 3 * row_nbytes)},
        rows=3,
    )
    capture_laguna_hidden_rows(
        0x22340000,
        depth=2,
        rows=3,
        targets=row_targets,
        hidden_size=hidden_size,
        runtime=runtime,
        stream=9,
    )
    assert runtime.copies[-1] == (
        row_targets.buffers[2].ptr,
        0x22340000,
        3 * row_nbytes,
        HipMemcpyKind.DEVICE_TO_DEVICE,
        9,
    )
    with pytest.raises(ValueError, match="capture rows"):
        capture_laguna_hidden_rows(
            0x22340000,
            depth=2,
            rows=2,
            targets=row_targets,
            hidden_size=hidden_size,
            runtime=runtime,
        )


def test_laguna_projection_dispatches_by_resident_layout(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    libraries = SimpleNamespace(
        f16_linear={"fp16_weight": object()},
        linear={"gguf_q5_k": object()},
    )

    def f16_launch(weight, *args, **kwargs):
        del args
        calls.append(("f16", (weight, kwargs)))

    def raw_launch(weight, *args, **kwargs):
        del args
        calls.append(("raw", (weight, kwargs)))

    monkeypatch.setattr(runner_module, "launch_f16_weight_linear", f16_launch)
    monkeypatch.setattr(runner_module, "launch_gguf_linear", raw_launch)
    f16_weight = SimpleNamespace(spec=SimpleNamespace(layout=LAYOUT_DENSE_F16))
    raw_weight = SimpleNamespace(spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF))

    runner_module.launch_laguna_weight_linear(
        f16_weight,
        1,
        2,
        3,
        4,
        5,
        output_dtype="f32",
        backend="hip_gfx1100",
        stream=7,
        libraries=libraries,
        runtime=object(),
    )
    runner_module.launch_laguna_weight_linear(
        raw_weight,
        1,
        2,
        3,
        4,
        5,
        output_dtype="bf16",
        backend="hip_gfx1100",
        stream=7,
        libraries=libraries,
        runtime=object(),
    )

    assert [name for name, _ in calls] == ["f16", "raw"]
    assert calls[0][1][1]["libraries"] is libraries.f16_linear
    assert calls[1][1][1]["libraries"] is libraries.linear
    with pytest.raises(ValueError, match="resident layout"):
        runner_module.launch_laguna_weight_linear(
            SimpleNamespace(spec=SimpleNamespace(layout="unsupported")),
            1,
            2,
            3,
            4,
            5,
            libraries=libraries,
        )


def test_laguna_session_constructor_failure_frees_partial_state_in_reverse(
    monkeypatch,
) -> None:
    events: list[str] = []
    config = _config()

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def free(self, **kwargs) -> None:
            del kwargs
            events.append(self.name)

    shared_weights = SimpleNamespace(config=config, backend="hip_gfx1151")
    monkeypatch.setattr(
        runner_module.LagunaGGUFResidentSession,
        "_validate_resident_weights",
        lambda self: None,
    )
    monkeypatch.setattr(
        runner_module,
        "load_laguna_eager_libraries",
        lambda **kwargs: Resource("libraries"),
    )
    ropes = iter((Resource("full_rope"), Resource("swa_rope")))
    monkeypatch.setattr(
        runner_module,
        "materialize_laguna_rope_tables",
        lambda *args, **kwargs: next(ropes),
    )
    monkeypatch.setattr(
        runner_module,
        "allocate_laguna_kv_cache",
        lambda *args, **kwargs: Resource("kv"),
    )
    monkeypatch.setattr(
        runner_module.LagunaEagerScratch,
        "allocate",
        lambda *args, **kwargs: Resource("scratch"),
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_laguna_moe_plan",
        lambda *args, **kwargs: Resource("moe_plan"),
    )
    monkeypatch.setattr(
        runner_module,
        "allocate_laguna_moe_scratch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            MemoryError("synthetic session allocation failure")
        ),
    )

    with pytest.raises(MemoryError, match="synthetic session"):
        runner_module.LagunaGGUFResidentSession(
            resident_weights=shared_weights,
            backend="hip_gfx1151",
            runtime=SimpleNamespace(),
        )

    assert events == ["scratch", "kv", "swa_rope", "full_rope"]
    assert "libraries" not in events


def test_laguna_owned_session_close_frees_weights_and_is_idempotent(monkeypatch) -> None:
    events: list[str] = []
    config = _config()

    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def free(self, **kwargs) -> None:
            del kwargs
            events.append(self.name)

    weights = Resource("weights")
    weights.config = config
    weights.backend = "hip_gfx1151"
    materialize_kwargs = {}
    monkeypatch.setattr(runner_module, "GGUFReader", lambda path: SimpleNamespace(info=object()))
    monkeypatch.setattr(
        runner_module,
        "laguna_gguf_config_from_metadata",
        lambda info: config,
    )

    def materialize(*args, **kwargs):
        del args
        materialize_kwargs.update(kwargs)
        return weights

    monkeypatch.setattr(runner_module, "materialize_laguna_gguf_weights", materialize)
    monkeypatch.setattr(
        runner_module.LagunaGGUFResidentSession,
        "_validate_resident_weights",
        lambda self: None,
    )
    monkeypatch.setattr(
        runner_module,
        "load_laguna_eager_libraries",
        lambda **kwargs: Resource("libraries"),
    )
    ropes = iter((Resource("full_rope"), Resource("swa_rope")))
    monkeypatch.setattr(
        runner_module,
        "materialize_laguna_rope_tables",
        lambda *args, **kwargs: next(ropes),
    )
    monkeypatch.setattr(
        runner_module,
        "allocate_laguna_kv_cache",
        lambda *args, **kwargs: Resource("kv"),
    )
    monkeypatch.setattr(
        runner_module.LagunaEagerScratch,
        "allocate",
        lambda *args, **kwargs: Resource("scratch"),
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_laguna_moe_plan",
        lambda *args, **kwargs: Resource("moe_plan"),
    )
    monkeypatch.setattr(
        runner_module.LagunaRowsScratch,
        "allocate",
        lambda *args, **kwargs: Resource("rows_scratch"),
    )
    moe_scratches = iter((Resource("moe_scratch"), Resource("rows_moe_scratch")))
    monkeypatch.setattr(
        runner_module,
        "allocate_laguna_moe_scratch",
        lambda *args, **kwargs: next(moe_scratches),
    )

    session = runner_module.LagunaGGUFResidentSession(
        "/synthetic/laguna.gguf",
        backend="hip_gfx1151",
        runtime=SimpleNamespace(),
        repacked_cache="/synthetic/laguna-repacked-v1",
        model_sha256="synthetic-sha256",
        safety_reserve_nbytes=4 * 2**30,
    )
    assert session.prefill_chunk_size == 128
    assert session.swa_prefill_variant == "swa_context_rows_wave32_exact_spans"
    assert session.selected_down_mode == "adaptive_grouped_smallm"
    assert session.verifier_scratch is None
    session.close()
    session.close()

    assert events == [
        "rows_moe_scratch",
        "rows_scratch",
        "moe_scratch",
        "scratch",
        "kv",
        "swa_rope",
        "full_rope",
        "weights",
    ]
    assert session.closed
    assert materialize_kwargs["repacked_cache"] == "/synthetic/laguna-repacked-v1"
    assert materialize_kwargs["repacked_cache_source_sha256"] == "synthetic-sha256"
    assert materialize_kwargs["safety_reserve_nbytes"] == 4 * 2**30


def test_laguna_borrowed_session_rejects_loader_cache_options() -> None:
    weights = SimpleNamespace(config=_config(), backend="hip_gfx1151")
    with pytest.raises(ValueError, match="apply only when the session owns"):
        runner_module.LagunaGGUFResidentSession(
            resident_weights=weights,
            backend="hip_gfx1151",
            runtime=SimpleNamespace(),
            repacked_cache="/synthetic/laguna-repacked-v1",
        )


def test_laguna_eager_plan_rejects_non_s21_shapes() -> None:
    config = _config()
    with pytest.raises(ValueError, match="48 layers"):
        resolve_laguna_eager_kernel_plan(replace(config, block_count=4), backend="hip_gfx1151")
    with pytest.raises(ValueError, match="query-head sequence"):
        resolve_laguna_eager_kernel_plan(
            replace(config, head_counts=(48,) * 48),
            backend="hip_gfx1151",
        )
    with pytest.raises(ValueError, match="HIP backend"):
        resolve_laguna_eager_kernel_plan(config, backend="cpu_reference")
