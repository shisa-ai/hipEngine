from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.core.build import plan_hip_build
from hipengine.generation import register_builtin_generators, resolve_text_generator
from hipengine.kernels.backends import (
    CPU_BACKEND,
    backend_package_capability,
    configure_hip_process_environment,
    hip_target_arch_for_backend,
    resolve_backend,
    select_backend,
)
from hipengine.kernels.hip_gfx1100.norm import (
    paro_rmsnorm_out_fp16,
    register_qwen35_rmsnorm_kernels,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w_auto_256,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
    gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100 import (
    GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS as GFX1100_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS,
    GGUF_GDN_INDEXED_SINGLETON_DECODE as GFX1100_GGUF_GDN_INDEXED_SINGLETON_DECODE,
    GGUF_GDN_PREFILL_AUTO_MODE as GFX1100_GGUF_GDN_PREFILL_AUTO_MODE,
    GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS as GFX1100_GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS,
    GGUF_Q8_T16_DECODE_ROWTILE_ALL as GFX1100_GGUF_Q8_T16_DECODE_ROWTILE_ALL,
    GGUF_GDN_PREFILL_EXACT_MODE as GFX1100_GGUF_GDN_PREFILL_EXACT_MODE,
    GGUF_PAGED_ATTN_PARALLEL_REDUCE as GFX1100_GGUF_PAGED_ATTN_PARALLEL_REDUCE,
    GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT as GFX1100_GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT,
    GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS as GFX1100_GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS,
    GGUF_PREFILL_ROUTER_SELECT_THREADS as GFX1100_GGUF_PREFILL_ROUTER_SELECT_THREADS,
    GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE as GFX1100_GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE,
    GGUF_Q8_T16_PREFILL_TWO_WAVE as GFX1100_GGUF_Q8_T16_PREFILL_TWO_WAVE,
    GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS as GFX1100_GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS,
    GGUF_ROUTER_F32_BF16_HIDDEN_THREADS as GFX1100_GGUF_ROUTER_F32_BF16_HIDDEN_THREADS,
)
from hipengine.kernels.hip_gfx1151 import (
    GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS,
    GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS,
    GGUF_PAGED_ATTN_PARALLEL_REDUCE,
    GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT,
    GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS,
    GGUF_PREFILL_ROUTER_SELECT_THREADS,
    GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS,
    GGUF_Q8_T16_DECODE_ROWTILE_ALL,
    GGUF_Q8_T16_PREFILL_FOUR_WAVE,
    GGUF_Q8_T16_PREFILL_TWO_WAVE,
    GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS,
    GGUF_ROUTER_F32_BF16_HIDDEN_THREADS,
    GGUF_GDN_INDEXED_SINGLETON_DECODE,
    GGUF_GDN_PREFILL_AUTO_MODE,
    GGUF_GDN_PREFILL_EXACT_MODE,
    GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE,
    TARGET_ARCH,
    register_gfx1151_kernels,
)
from hipengine.kernels.registry import KernelKey, resolve


def test_auto_backend_selects_supported_hip_arches() -> None:
    assert select_backend("auto", detected_arches=["gfx1100"]).backend == "hip_gfx1100"
    assert (
        select_backend("auto", detected_arches=["gfx1151:sramecc+:xnack-"]).backend == "hip_gfx1151"
    )


def test_auto_backend_honors_force_env_override() -> None:
    selection = select_backend(
        "auto",
        detected_arches=["gfx1151"],
        env={"HIPENGINE_BACKEND": "hip_gfx1100"},
    )

    assert selection.backend == "hip_gfx1100"
    assert selection.source == "HIPENGINE_BACKEND"


def test_auto_backend_warns_and_falls_back_for_unknown_arch() -> None:
    selection = select_backend("auto", detected_arches=["gfx1102"], env={})

    assert selection.backend == CPU_BACKEND
    assert selection.detected_arches == ("gfx1102",)
    assert selection.warning is not None
    assert "gfx1102" in selection.warning
    assert "HIPENGINE_BACKEND=hip_gfx1100" in selection.warning

    with pytest.warns(RuntimeWarning, match="gfx1102"):
        assert resolve_backend("auto", detected_arches=["gfx1102"], env={}) == CPU_BACKEND


def test_explicit_backend_is_not_autodetected() -> None:
    selection = select_backend("custom_backend", detected_arches=["gfx1151"], env={})

    assert selection.backend == "custom_backend"
    assert selection.source == "explicit"
    assert selection.detected_arches == ()


def test_gfx1151_hip_process_environment_defaults_to_one_hardware_queue() -> None:
    env: dict[str, str] = {}

    applied = configure_hip_process_environment(
        detected_arches=["gfx1151:sramecc+:xnack-"],
        env=env,
    )

    assert applied == {"GPU_MAX_HW_QUEUES": "1"}
    assert env["GPU_MAX_HW_QUEUES"] == "1"


def test_gfx1151_hip_process_environment_preserves_explicit_queue_override() -> None:
    env = {"GPU_MAX_HW_QUEUES": "4"}

    applied = configure_hip_process_environment(detected_arches=["gfx1151"], env=env)

    assert applied == {}
    assert env["GPU_MAX_HW_QUEUES"] == "4"


def test_gfx1100_hip_process_environment_does_not_change_queue_policy() -> None:
    env: dict[str, str] = {}

    applied = configure_hip_process_environment(detected_arches=["gfx1100"], env=env)

    assert applied == {}
    assert "GPU_MAX_HW_QUEUES" not in env


def test_mixed_hip_arches_do_not_receive_a_process_wide_queue_default() -> None:
    env: dict[str, str] = {}

    applied = configure_hip_process_environment(
        detected_arches=["gfx1151", "gfx1100"],
        env=env,
    )

    assert applied == {}
    assert "GPU_MAX_HW_QUEUES" not in env


def test_explicit_gfx1151_backend_hint_applies_when_arch_detection_is_empty() -> None:
    env = {"HIPENGINE_BACKEND": "hip_gfx1151"}

    applied = configure_hip_process_environment(detected_arches=[], env=env)

    assert applied == {"GPU_MAX_HW_QUEUES": "1"}


def test_gfx1151_backend_does_not_alias_unvalidated_native_spec_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.kernels.hip_gfx1151 as backend

    source_keys = (
        KernelKey(
            "hip_gfx1100",
            "speculative_cycle",
            "w4_gguf",
            "native_v1_b2_target_graph",
        ),
        KernelKey(
            "hip_gfx1100",
            "speculative_cycle",
            "w4_gguf",
            "native_v1_b2_proposal_graph",
        ),
    )
    registered: list[KernelKey] = []
    monkeypatch.setattr(backend, "import_module", lambda _name: None)
    monkeypatch.setattr(backend, "registered_keys", lambda: source_keys)
    monkeypatch.setattr(backend, "is_registered", lambda _key: False)
    monkeypatch.setattr(backend, "resolve", lambda **_kwargs: object())
    monkeypatch.setattr(
        backend,
        "register",
        lambda key, _kernel, *, replace=False: registered.append(key),
    )

    backend.register_gfx1151_kernels()

    assert registered == []


def test_gfx1151_backend_aliases_gfx1100_kernel_keys() -> None:
    register_qwen35_rmsnorm_kernels()
    register_gfx1151_kernels()

    assert TARGET_ARCH == "gfx1151"
    assert GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS == 128
    assert GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS == 4096
    assert GFX1100_GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS == 4096
    assert GGUF_PREFILL_ROUTER_SELECT_THREADS == 128
    assert GFX1100_GGUF_PREFILL_ROUTER_SELECT_THREADS == 128
    assert GGUF_Q8_T16_PREFILL_FOUR_WAVE is True
    assert GGUF_Q8_T16_PREFILL_TWO_WAVE is True
    assert GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS == 65536
    assert GGUF_ROUTER_F32_BF16_HIDDEN_THREADS == 256
    assert GFX1100_GGUF_ROUTER_F32_BF16_HIDDEN_THREADS == 256
    assert GFX1100_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS == 4096
    assert GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS == 0
    assert GFX1100_GGUF_GDN_INDEXED_SINGLETON_DECODE is False
    assert GGUF_GDN_INDEXED_SINGLETON_DECODE is True
    assert GFX1100_GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS == 0
    assert GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS == 8
    assert GFX1100_GGUF_Q8_T16_DECODE_ROWTILE_ALL is False
    assert GGUF_Q8_T16_DECODE_ROWTILE_ALL is False
    assert GFX1100_GGUF_GDN_PREFILL_AUTO_MODE == "chain_peer_wave32"
    assert GFX1100_GGUF_GDN_PREFILL_EXACT_MODE == "chain_lds32_direct_nonvolatile"
    assert GGUF_GDN_PREFILL_AUTO_MODE == "chain_lds32_direct_nonvolatile"
    assert GGUF_GDN_PREFILL_EXACT_MODE == "chain_lds32_direct_nonvolatile"
    assert GFX1100_GGUF_PAGED_ATTN_PARALLEL_REDUCE is True
    assert GFX1100_GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT == 32768
    assert GGUF_PAGED_ATTN_PARALLEL_REDUCE is False
    assert GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT == 32768
    assert GFX1100_GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE == "shared_x"
    assert GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE == "shared_x"
    assert GFX1100_GGUF_Q8_T16_PREFILL_TWO_WAVE is True
    assert GFX1100_GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS == 4096
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS",
        )
        == 4096
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_PREFILL_ROUTER_SELECT_THREADS",
        )
        == 128
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q8_T16_PREFILL_FOUR_WAVE",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q8_T16_PREFILL_TWO_WAVE",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
        )
        == 65536
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="linear",
            quant="gguf_q8_0_t16_v1",
            variant="t16_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q8_0_t16_v1",
            variant="t16_wmma_prefill_bf16_bf16_out",
        )
        is gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="router_logits",
            quant="f32",
            variant="bf16_hidden",
        )
        is qwen35_router_logits_bf16_f32w_auto_256
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="router_logits",
            quant="f32",
            variant="bf16_hidden",
        )
        is qwen35_router_logits_bf16_f32w_auto_256
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS",
        )
        == 128
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_GDN_INDEXED_SINGLETON_DECODE",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_GDN_INDEXED_SINGLETON_DECODE",
        )
        is False
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
        )
        == 0
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
        )
        == 8
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
        )
        is False
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
        )
        is False
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_GDN_PREFILL_AUTO_MODE",
        )
        == "chain_peer_wave32"
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
        )
        == "shared_x"
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_PAGED_ATTN_PARALLEL_REDUCE",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT",
        )
        == 32768
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_PAGED_ATTN_PARALLEL_REDUCE",
        )
        is False
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q8_T16_PREFILL_TWO_WAVE",
        )
        is True
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
        )
        == 4096
    )
    assert (
        backend_package_capability(
            "hip_gfx1100",
            "GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS",
        )
        == 4096
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS",
        )
        == 0
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_GDN_PREFILL_AUTO_MODE",
        )
        == "chain_lds32_direct_nonvolatile"
    )
    assert (
        backend_package_capability(
            "hip_gfx1151",
            "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
        )
        == "shared_x"
    )
    assert hip_target_arch_for_backend("hip_gfx1151") == "gfx1151"
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="rmsnorm",
            quant="w4_paro",
            variant="paro_out_fp16",
        )
        is paro_rmsnorm_out_fp16
    )


def test_plan_hip_build_target_arch_is_in_flags_and_cache_key(tmp_path: Path) -> None:
    source = tmp_path / "smoke.hip"
    source.write_text('extern "C" __global__ void smoke() {}\n')

    gfx1100 = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
        target_arch="gfx1100",
    )
    gfx1151 = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
        target_arch="gfx1151",
    )

    assert gfx1100.cache_key != gfx1151.cache_key
    assert gfx1100.target_arch == "gfx1100"
    assert gfx1151.target_arch == "gfx1151"
    assert "--offload-arch=gfx1100" in gfx1100.flags
    assert "--offload-arch=gfx1151" in gfx1151.flags
    assert "--offload-arch=gfx1151" in gfx1151.command


def test_plan_hip_build_reads_target_arch_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "smoke.hip"
    source.write_text('extern "C" __global__ void smoke() {}\n')
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")

    artifact = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
    )

    assert artifact.target_arch == "gfx1151"
    assert "--offload-arch=gfx1151" in artifact.flags


def test_plan_hip_build_includes_device_lib_path_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "smoke.hip"
    source.write_text('extern "C" __global__ void smoke() {}\n')
    device_lib_path = tmp_path / "amdgcn" / "bitcode"
    monkeypatch.setenv("HIP_DEVICE_LIB_PATH", str(device_lib_path))

    artifact = plan_hip_build(
        sources=[source],
        family="smoke",
        profile="baseline",
        cache_root=tmp_path / "cache",
        compiler_version="hipcc test version",
        target_arch="gfx1151",
    )

    assert f"--rocm-device-lib-path={device_lib_path}" in artifact.flags
    assert f"--rocm-device-lib-path={device_lib_path}" in artifact.command


def test_qwen35_paro_gfx1151_generation_factory_sets_backend() -> None:
    register_builtin_generators()
    factory = resolve_text_generator(
        model="qwen3_5_moe_paro",
        backend="hip_gfx1151",
        quant="w4_paro",
    )

    generator = factory(model_path="/tmp/fake", weight_index=object(), model_plugin=object())

    assert getattr(generator, "backend") == "hip_gfx1151"


def test_qwen35_gguf_gfx1151_generation_factory_sets_backend(monkeypatch) -> None:
    import hipengine.generation.qwen35_gguf as qwen35_gguf

    monkeypatch.setattr(
        qwen35_gguf.Qwen35GGUFTokenizer,
        "from_gguf_info",
        classmethod(lambda cls, weight_index: object()),
    )
    register_builtin_generators()
    factory = resolve_text_generator(
        model="qwen3_5_moe_gguf",
        backend="hip_gfx1151",
        quant="gguf_q4_k_m",
    )

    generator = factory(
        model_path="/tmp/fake.gguf",
        weight_index=object(),
        model_plugin=object(),
    )

    assert getattr(generator, "backend") == "hip_gfx1151"
    assert generator.target_arch == "gfx1151"
    assert generator.engine_loop_config_defaults == {
        "prefill_decode_policy": "fair",
        "max_prefill_chunk_tokens": 256,
        "fair_prefill_burst_chunks": 2,
    }
    assert generator.server_plain_ar_max_active_requests == 8

    other_quant_factory = resolve_text_generator(
        model="qwen3_5_moe_gguf",
        backend="hip_gfx1151",
        quant="gguf_q8_0",
    )
    other_quant_generator = other_quant_factory(
        model_path="/tmp/fake-q8.gguf",
        weight_index=object(),
        model_plugin=object(),
    )
    assert other_quant_generator.engine_loop_config_defaults == {}
    assert other_quant_generator.server_plain_ar_max_active_requests is None


def test_gguf_weight_backend_drives_embedding_and_linear_dispatch() -> None:
    from hipengine.loading.qwen35_gguf_materialize import LAYOUT_DENSE_BF16
    from hipengine.runtime.gguf_embedding import resolve_gguf_embedding_dispatch
    from hipengine.runtime.gguf_linear import resolve_gguf_linear_dispatch

    weight = SimpleNamespace(
        backend="hip_gfx1151",
        spec=SimpleNamespace(layout=LAYOUT_DENSE_BF16, quant_key="bf16"),
    )

    assert resolve_gguf_embedding_dispatch(weight).key.backend == "hip_gfx1151"
    assert resolve_gguf_linear_dispatch(weight).key.backend == "hip_gfx1151"


def test_gguf_router_resolve_uses_weight_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import hipengine.runtime.qwen35_gguf_runner as qwen35_gguf_runner

    resolved: list[str] = []

    def fake_resolve(*, backend, layer, quant, variant, **kwargs):
        del layer, quant, variant, kwargs
        resolved.append(backend)
        return lambda *args, **launch_kwargs: None

    monkeypatch.setattr(qwen35_gguf_runner, "resolve", fake_resolve)
    weight = SimpleNamespace(
        backend="hip_gfx1151",
        spec=SimpleNamespace(quant_key="f32"),
        allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=22)),
    )

    qwen35_gguf_runner._launch_qwen35_router_logits_bf16_hidden(
        11,
        weight,
        33,
        1,
        2048,
        256,
    )

    assert resolved == ["hip_gfx1151"]


def test_gguf_gdn_plan_resolves_every_key_for_runner_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as qwen35_gguf_runner

    resolved: list[str] = []

    def fake_resolve(*, backend, layer, quant, variant, **kwargs):
        del layer, quant, variant, kwargs
        resolved.append(backend)
        return object()

    monkeypatch.setattr(qwen35_gguf_runner, "resolve", fake_resolve)
    monkeypatch.setattr(
        qwen35_gguf_runner,
        "register_qwen35_linear_attn_gdn_kernels",
        lambda: None,
    )
    runner = object.__new__(qwen35_gguf_runner.Qwen35GGUFFullStackRunner)
    runner.backend = "hip_gfx1151"

    plan = runner._gdn_prefill_plan()

    assert plan.has_chain
    assert plan.has_exact_chain
    assert plan.has_fused
    assert len(resolved) == 30
    assert set(resolved) == {"hip_gfx1151"}


def test_gguf_runner_loads_backend_aliases_and_tags_resident_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as qwen35_gguf_runner

    loaded: list[str] = []
    materialized: list[str] = []
    fake_weights = object()
    monkeypatch.setattr(
        qwen35_gguf_runner,
        "load_backend_kernel_package",
        lambda backend: loaded.append(backend),
    )

    def fake_materialize(model_path, *, runtime, backend):
        del model_path, runtime
        materialized.append(backend)
        return fake_weights

    monkeypatch.setattr(
        qwen35_gguf_runner,
        "materialize_qwen35_gguf_weights",
        fake_materialize,
    )

    runner = qwen35_gguf_runner.Qwen35GGUFFullStackRunner(
        "/tmp/fake.gguf",
        runtime=object(),
        backend="hip_gfx1151",
    )

    assert runner.backend == "hip_gfx1151"
    assert runner.target_arch == "gfx1151"
    assert runner.weights is fake_weights
    assert loaded == ["hip_gfx1151"]
    assert materialized == ["hip_gfx1151"]


def test_gguf_fused_linear_matching_uses_resident_backend() -> None:
    from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
    from hipengine.runtime.gguf_linear import _resolve_gguf_linear_pair_kind

    def weight():
        return SimpleNamespace(
            backend="hip_gfx1151",
            spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0"),
        )

    assert (
        _resolve_gguf_linear_pair_kind(
            weight(),
            weight(),
            rows=1,
            in_features=2048,
            out_features=4096,
            out_features_b=4096,
            backend="hip_gfx1151",
            use_wmma=False,
        )
        == "q8_raw_dual"
    )


def test_gguf_runtime_has_no_literal_gfx1100_resolver_backend() -> None:
    import hipengine.runtime.gguf_embedding as gguf_embedding
    import hipengine.runtime.gguf_linear as gguf_linear
    import hipengine.runtime.qwen35_gguf_runner as qwen35_gguf_runner

    resolver_names = {
        "resolve",
        "resolve_gguf_embedding_dispatch",
        "resolve_gguf_linear_dispatch",
    }
    violations: list[str] = []
    for module in (gguf_embedding, gguf_linear, qwen35_gguf_runner):
        tree = ast.parse(inspect.getsource(module))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Name):
                name = call.func.id
            elif isinstance(call.func, ast.Attribute):
                name = call.func.attr
            else:
                continue
            if name not in resolver_names:
                continue
            for keyword in call.keywords:
                if (
                    keyword.arg == "backend"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "hip_gfx1100"
                ):
                    violations.append(f"{module.__name__}:{call.lineno}:{name}")

    assert violations == []


def test_gfx1151_gguf_lazy_registration_rebinds_source_kernels() -> None:
    from hipengine.kernels.registry import KernelKey, clear_registry_for_tests, resolve
    from hipengine.runtime.gguf_embedding import _ensure_embedding_kernel_registered
    from hipengine.runtime.gguf_linear import _ensure_linear_kernel_registered

    embedding_key = KernelKey(
        "hip_gfx1151",
        "embedding",
        "gguf_q6_k",
        "lookup_bf16_out",
    )
    linear_key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q8_0",
        "gemv_bf16_bf16_out",
    )
    clear_registry_for_tests()

    _ensure_embedding_kernel_registered(embedding_key)
    _ensure_linear_kernel_registered(linear_key)

    assert callable(
        resolve(
            backend=embedding_key.backend,
            layer=embedding_key.layer,
            quant=embedding_key.quant,
            variant=embedding_key.variant,
        )
    )
    assert callable(
        resolve(
            backend=linear_key.backend,
            layer=linear_key.layer,
            quant=linear_key.quant,
            variant=linear_key.variant,
        )
    )
