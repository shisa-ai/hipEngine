"""CPU-only consumer geometry qualification, independent of resident storage."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from test_gguf_ud_admission import _synthetic_model_map
from hipengine.loading.qwen35_gguf_admission import (
    preflight_qwen35_gguf_artifact, certificate_covers_artifact,
)

SLOT = "layers.0.ssm_a"


@pytest.fixture(autouse=True)
def forbid_payload_and_allocation(monkeypatch):
    from hipengine.loading.gguf import GGUFReader
    import hipengine.loading.qwen35_gguf_materialize as loader
    def forbidden(*args, **kwargs):
        pytest.fail("payload/allocation during cold geometry qualification")
    monkeypatch.setattr(GGUFReader, "tensor_data", forbidden)
    monkeypatch.setattr(loader, "malloc", forbidden)


def report(config, operation="ar_decode_c1", slot=SLOT, state="f32"):
    model = replace(_synthetic_model_map(), config=config)
    return preflight_qwen35_gguf_artifact(model, backend="hip_gfx1100",
        operations=(operation,), slot_filter=(slot,), recurrent_state_dtype=state)


@pytest.mark.parametrize("changes,reason", [
    ({"ssm_group_count": 3}, "divisible"),
    ({"ssm_inner_size": 16 * 129}, "<= 128"),
    ({"ssm_group_count": 0}, "num_k_heads must be positive"),
    ({"ssm_time_step_rank": 0}, "num_v_heads must be positive"),
    ({"ssm_state_size": 0}, "head_k_dim must be positive"),
    ({"ssm_inner_size": 0}, "ssm_inner_size must be positive"),
    ({"ssm_inner_size": 16 * 128 + 1}, "ssm_inner_size must be divisible"),
])
def test_invalid_gdn_geometry_refuses_and_keeps_expected_invocation(changes, reason):
    cfg = _synthetic_model_map().config
    valid = report(cfg)
    assert valid.supported
    invalid = report(replace(cfg, **changes))
    assert not invalid.supported
    assert any(reason in item.reason for item in invalid.unsupported)
    assert (SLOT, "ar_decode_c1") in invalid.plan_contract.required_invocations
    assert SLOT in invalid.plan_contract.required_plan_slots
    assert not invalid.plan_contract.is_complete()
    with pytest.raises(ValueError):
        invalid.certificate()
    assert not certificate_covers_artifact(valid.certificate(),
        manifest_fingerprint=invalid.manifest_fingerprint,
        plan_contract=invalid.plan_contract, slot_filter=(SLOT,))


@pytest.mark.parametrize("operation,state", [
    ("ar_decode_c1", "f32"), ("ar_decode_rows", "f32"),
    ("ar_decode_native_rows", "f32"), ("ar_prefill", "f32"),
    ("ar_decode_c1", "fp16"), ("ar_decode_native_rows", "fp16"),
])
@pytest.mark.parametrize("heads,value_dim", [(1, 1), (2, 127), (16, 128), (3, 128), (2, 129)])
def test_actual_wrapper_and_admission_geometry_parity(operation, state, heads, value_dim, monkeypatch):
    from hipengine.loading.qwen35_gguf_consumer_surface import resolve_gdn_operation_contract
    import hipengine.kernels.hip_gfx1100.linear_attn.gdn as wrappers
    cfg = replace(_synthetic_model_map().config, ssm_group_count=heads,
                  ssm_inner_size=16 * value_dim)
    admission = report(cfg, operation, state=state)
    consumer = resolve_gdn_operation_contract(operation, state)
    calls = []
    def native(*args):
        calls.append(args)
        return 0
    library = NS(**{consumer.symbol: native})
    def forbidden(*args, **kwargs):
        pytest.fail("device runtime/build access in CPU geometry test")
    monkeypatch.setattr(wrappers, "build_qwen35_linear_attn_gdn", forbidden)
    monkeypatch.setattr(wrappers, "get_hip_runtime", forbidden)
    fn = getattr(wrappers, consumer.symbol.removeprefix("hipengine_"))
    values = {name: i + 1 for i, (name, _, _) in enumerate(consumer.operands)}
    values.update(num_k_heads=heads, num_v_heads=16, head_k_dim=cfg.ssm_state_size,
                  head_v_dim=value_dim, eps=cfg.rms_norm_eps, tokens=2,
                  total_tokens=2, segments=1)
    kwargs = {name: values[name] for name, _, _ in consumer.operands}
    kwargs.update({name: values[name] for name, dtype in consumer.scalars if dtype != "stream"})
    valid = 16 % heads == 0 and (value_dim <= 128 or operation in {"ar_prefill", "ar_decode_native_rows"})
    if valid:
        fn(**kwargs, library=library, runtime=NS(check=forbidden))
        assert len(calls) == 1
        assert admission.supported, admission.render_refusals()
        assert admission.plan_contract.is_complete()
        admission.certificate()
    else:
        with pytest.raises(ValueError):
            fn(**kwargs, library=library, runtime=NS(check=forbidden))
        assert not calls
        assert not admission.supported


@pytest.mark.parametrize("inner,heads,expected", [(16, 16, 1), (2048, 16, 128), (2049, 16, None), (0, 16, None), (256, 0, None)])
def test_actual_runner_value_head_geometry(inner, heads, expected):
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner
    runner = NS(weights=NS(config=NS(ssm_inner_size=inner, ssm_time_step_rank=heads)))
    getter = Qwen35GGUFFullStackRunner.ssm_value_dim.fget
    if expected is None:
        with pytest.raises(ValueError):
            getter(runner)
    else:
        assert getter(runner) == expected


@pytest.mark.parametrize("kernel_size", [0, -1])
def test_conv_composite_rejects_invalid_kernel_geometry(kernel_size, monkeypatch):
    cfg = replace(_synthetic_model_map().config, ssm_conv_kernel=kernel_size)
    invalid = report(cfg, slot="layers.0.ssm_conv1d")
    assert not invalid.supported
    assert any("kernel_size must be positive" in item.reason for item in invalid.unsupported)
    assert not invalid.plan_contract.is_complete()
    import hipengine.kernels.hip_gfx1100.linear_attn.conv as wrappers
    def forbidden(*args, **kwargs):
        pytest.fail("build before invalid conv geometry refusal")
    monkeypatch.setattr(wrappers, "build_qwen35_linear_attn_conv", forbidden)
    with pytest.raises(ValueError, match="kernel_size must be positive"):
        wrappers.qwen35_linear_attn_conv_decode_bf16(1, 2, 3, 4, 272, kernel_size)
