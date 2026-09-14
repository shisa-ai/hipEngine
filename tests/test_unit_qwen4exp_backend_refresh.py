import pytest

from hipengine.runtime import qwen4_exp_runner as runner


@pytest.fixture
def registry(monkeypatch):
    state = {"generation": 10, "calls": []}
    monkeypatch.setattr(runner, "_MOE_BACKEND_REGISTRY_GENERATIONS", {})
    monkeypatch.setattr(runner, "_registry_generation", lambda: state["generation"])

    def load(backend):
        state["calls"].append(backend)

    monkeypatch.setattr(runner, "load_backend_kernel_package", load)
    return state


def test_stable_registry_refreshes_once(registry):
    for _ in range(3):
        runner._ensure_qwen4_exp_moe_backend("hip_gfx1151")
    assert registry["calls"] == ["hip_gfx1151"]


def test_registry_mutation_and_backend_keys_are_independent(registry):
    runner._ensure_qwen4_exp_moe_backend("hip_gfx1151")
    registry["generation"] += 1
    runner._ensure_qwen4_exp_moe_backend("hip_gfx1151")
    runner._ensure_qwen4_exp_moe_backend("hip_gfx1100")
    runner._ensure_qwen4_exp_moe_backend("hip_gfx1151")
    assert registry["calls"] == [
        "hip_gfx1151", "hip_gfx1151", "hip_gfx1100"]


def test_mutation_during_refresh_does_not_certify_changed_generation(registry, monkeypatch):
    def load(backend):
        registry["calls"].append(backend)
        if len(registry["calls"]) == 1:
            registry["generation"] += 1

    monkeypatch.setattr(runner, "load_backend_kernel_package", load)
    runner._ensure_qwen4_exp_moe_backend("hip_gfx1151")
    assert runner._MOE_BACKEND_REGISTRY_GENERATIONS == {}
    runner._ensure_qwen4_exp_moe_backend("hip_gfx1151")
    runner._ensure_qwen4_exp_moe_backend("hip_gfx1151")
    assert registry["calls"] == ["hip_gfx1151", "hip_gfx1151"]


def test_failed_refresh_is_not_cached(registry, monkeypatch):
    def fail(backend):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(runner, "load_backend_kernel_package", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="registration failed"):
            runner._ensure_qwen4_exp_moe_backend("hip_gfx1151")
    assert runner._MOE_BACKEND_REGISTRY_GENERATIONS == {}


def test_retired_flag_does_not_disable_the_qualified_cache(registry, monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_MOE_BACKEND_CACHE", "0")
    runner._ensure_qwen4_exp_moe_backend("hip_gfx1151")
    runner._ensure_qwen4_exp_moe_backend("hip_gfx1151")
    assert registry["calls"] == ["hip_gfx1151"]


def test_real_registry_restores_missing_key_without_overwriting_wrapper(monkeypatch):
    from hipengine.kernels.registry import KernelKey, register, resolve, unregister

    monkeypatch.setattr(runner, "_MOE_BACKEND_REGISTRY_GENERATIONS", {})
    key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                    "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out")
    runner._ensure_qwen4_exp_moe_backend(key.backend)

    def wrapper(*args, **kwargs):
        pass

    register(key, wrapper, replace=True)
    runner._ensure_qwen4_exp_moe_backend(key.backend)
    assert resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                   variant=key.variant) is wrapper
    unregister(key)
    runner._ensure_qwen4_exp_moe_backend(key.backend)
    assert resolve(backend=key.backend, layer=key.layer, quant=key.quant,
                   variant=key.variant) is not wrapper
