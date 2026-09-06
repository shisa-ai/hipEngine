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
    # Native valid_gdn_shape caps both scalar and segmented recurrence;
    # only the baseline prefill export uses valid_prefill_shape (see below).
    valid = 16 % heads == 0 and (value_dim <= 128 or operation == "ar_prefill")
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


@pytest.mark.parametrize("state", ["f32", "fp16"])
def test_native_rows_value_head_limit_blocks_mint_and_transfer(state):
    op = "ar_decode_native_rows"
    cfg = replace(_synthetic_model_map().config, ssm_inner_size=16 * 128)
    valid = report(cfg, op, state=state)
    invalid = report(replace(cfg, ssm_inner_size=16 * 129), op, state=state)
    assert valid.supported and valid.plan_contract.is_complete()
    assert not invalid.supported
    assert any("head_v_dim must be <= 128" in item.reason for item in invalid.unsupported)
    assert (SLOT, op) in invalid.plan_contract.required_invocations
    assert SLOT in invalid.plan_contract.required_plan_slots
    assert not invalid.plan_contract.invocations
    assert not invalid.plan_contract.is_complete()
    with pytest.raises(ValueError):
        invalid.certificate()
    assert not certificate_covers_artifact(valid.certificate(),
        manifest_fingerprint=invalid.manifest_fingerprint,
        plan_contract=invalid.plan_contract, slot_filter=(SLOT,))


@pytest.mark.parametrize("name", [
    "qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_state_rows_bf16",
    "qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_fp16",
    "qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16",
    "qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16_fp16state",
    "qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16",
    "qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16_fp16state",
])
@pytest.mark.parametrize("value_dim", [128, 129])
def test_sibling_recurrent_wrappers_apply_native_limit(name, value_dim, monkeypatch):
    import inspect
    import hipengine.kernels.hip_gfx1100.linear_attn.gdn as wrappers
    fn = getattr(wrappers, name)
    calls = []
    def native(*args):
        calls.append(args)
        return 0
    def forbidden(*args, **kwargs):
        pytest.fail("device/build access in segmented geometry test")
    monkeypatch.setattr(wrappers, "build_qwen35_linear_attn_gdn", forbidden)
    monkeypatch.setattr(wrappers, "get_hip_runtime", forbidden)
    pointers = {key: 1 for key in inspect.signature(fn).parameters if key.endswith("_ptr")}
    rows = {"rows": 8} if "rows" in inspect.signature(fn).parameters else {"total_tokens": 2, "segments": 1}
    kwargs = dict(**pointers, **rows, eps=1e-6,
                  num_k_heads=2, num_v_heads=16, head_k_dim=4, head_v_dim=value_dim,
                  library=NS(**{"hipengine_" + name: native}), runtime=NS(check=forbidden))
    if value_dim == 129:
        with pytest.raises(ValueError, match="head_v_dim must be <= 128"):
            fn(**kwargs)
        assert not calls
    else:
        fn(**kwargs)
        assert len(calls) == 1


def test_native_export_predicates_are_the_geometry_oracle():
    """Native source evidence independent of Python descriptors/mock returns."""
    import re
    from pathlib import Path
    source = Path("hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip").read_text()
    def body(name):
        # Native functions begin at column zero; nested braces are indented.
        return re.search(r"\b" + name + r"\([^{}]*\) \{(.*?)\n\}", source, re.S).group(1)
    common = ("num_k_heads > 0 && num_v_heads > 0 && (num_v_heads % num_k_heads) == 0 && "
              "head_k_dim > 0 && head_v_dim > 0")
    assert " ".join(body("valid_gdn_shape").split()) == "return " + common + " && head_v_dim <= 128;"
    assert " ".join(body("valid_prefill_shape").split()) == "return tokens > 0 && " + common + ";"
    prefix = "hipengine_qwen35_gdn_"
    for suffix in ("recurrent_rmsnorm_gate_lowp_bf16",
                   "recurrent_rmsnorm_gate_lowp_bf16_fp16state",
                   "recurrent_rmsnorm_gate_segments_lowp_bf16",
                   "recurrent_rmsnorm_gate_segments_lowp_bf16_fp16state",
                   "recurrent_rmsnorm_gate_segments_lowp_state_rows_bf16",
                   "recurrent_rmsnorm_gate_segments_lowp_fp16",
                   "recurrent_rmsnorm_gate_indexed_lowp_bf16",
                   "recurrent_rmsnorm_gate_indexed_lowp_bf16_fp16state",
                   "recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16",
                   "recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16_fp16state"):
        guard = body(prefix + suffix).split("hipLaunchKernelGGL", 1)[0]
        assert "!valid_gdn_shape(num_k_heads, num_v_heads, head_k_dim, head_v_dim)" in guard
        assert "hipErrorInvalidValue" in guard
    guard = body(prefix + "prefill_recurrent_rmsnorm_gate_bf16_decode_order").split("hipLaunchKernelGGL", 1)[0]
    assert "!valid_prefill_shape(tokens, num_k_heads, num_v_heads, head_k_dim, head_v_dim)" in guard
    assert "hipErrorInvalidValue" in guard


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
