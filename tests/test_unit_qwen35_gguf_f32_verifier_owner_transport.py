"""R0137 owner transport for verifier F32 residual/post-norm and R28.

The profile owns these arithmetic selectors on the generator/runner/session
chain. The process environment remains the unbound/direct diagnostic input and
must not carry a decision from one owner to another.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)

RESIDUAL_ENV = "HIPENGINE_GGUF_VERIFY_F32_RESIDUAL"
POST_NORM_ENV = "HIPENGINE_GGUF_VERIFY_F32_POST_NORM"
R28_ENV = "HIPENGINE_GGUF_Q4_T16_DUAL_SILU_PRODUCTION_R28"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    for name in (RESIDUAL_ENV, POST_NORM_ENV, R28_ENV):
        monkeypatch.delenv(name, raising=False)
    yield
    clear_runtime_profile_registry_for_tests()
    for name in (RESIDUAL_ENV, POST_NORM_ENV, R28_ENV):
        os.environ.pop(name, None)


def _new_generator():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    return object.__new__(Qwen35GGUFBringupGenerator)


def _bare_runner(
    *,
    residual: bool | None = None,
    post_norm: bool | None = None,
    r28: bool | None = None,
):
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    runner = object.__new__(runner_mod.Qwen35GGUFFullStackRunner)
    runner.target_verify_f32_residual = residual
    runner.target_verify_f32_post_norm = post_norm
    runner.target_q4_t16_dual_silu_r28 = r28
    runner.verify_f32_residual = residual
    runner.verify_f32_post_norm = post_norm
    runner.q4_t16_dual_silu_r28 = r28
    return runner


# --------------------------------------------------------------- resolution


def test_f32_owner_precedence_is_explicit_then_bound_then_ambient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    monkeypatch.setenv(RESIDUAL_ENV, "1")
    monkeypatch.setenv(POST_NORM_ENV, "1")
    bound = _bare_runner(residual=True, post_norm=True)

    assert runner_mod._verify_f32_residual_for_call(bound, False) is False
    assert runner_mod._verify_f32_residual_for_call(bound, True) is True
    assert runner_mod._verify_f32_post_norm_for_call(bound, False) is False
    assert runner_mod._verify_f32_post_norm_for_call(bound, True) is True

    # A bound owner wins over a contradictory ambient value.
    monkeypatch.setenv(RESIDUAL_ENV, "0")
    monkeypatch.setenv(POST_NORM_ENV, "0")
    assert runner_mod._verify_f32_residual_for_call(bound) is True
    assert runner_mod._verify_f32_post_norm_for_call(bound) is True

    # Unbound owners keep the historical per-call ambient read.
    unbound = _bare_runner()
    assert runner_mod._verify_f32_residual_for_call(unbound) is False
    assert runner_mod._verify_f32_post_norm_for_call(unbound) is False
    monkeypatch.setenv(RESIDUAL_ENV, "1")
    monkeypatch.setenv(POST_NORM_ENV, "1")
    assert runner_mod._verify_f32_residual_for_call(unbound) is True
    assert runner_mod._verify_f32_post_norm_for_call(unbound) is True


def test_f32_bound_owner_never_reads_ambient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "_gguf_verify_f32_residual_enabled",
        lambda: (_ for _ in ()).throw(AssertionError("ambient residual read")),
    )
    monkeypatch.setattr(
        runner_mod,
        "_gguf_verify_f32_post_norm_enabled",
        lambda: (_ for _ in ()).throw(AssertionError("ambient post-norm read")),
    )
    bound = _bare_runner(residual=False, post_norm=False)
    assert runner_mod._verify_f32_residual_for_call(bound) is False
    assert runner_mod._verify_f32_post_norm_for_call(bound) is False


def test_f32_residual_keeps_moe_c2_narrowing(monkeypatch: pytest.MonkeyPatch) -> None:
    from hipengine.core.specdec2_scope import moe_physical_c2_numerics_session
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    runner = _bare_runner(residual=True, post_norm=True)
    with moe_physical_c2_numerics_session(True):
        assert runner_mod._verify_f32_residual_for_call(runner) is False
        assert runner_mod._verify_f32_residual_for_call(runner, True) is False
        # Post-norm is only consumed inside the residual path; its own owner
        # value is not narrowed by the C2 residual scope.
        assert runner_mod._verify_f32_post_norm_for_call(runner) is True


def test_f32_construction_order_and_interleaving_do_not_retarget_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Resolver-level only. The actual profile binder matrix lives in
    # tests/test_unit_qwen35_gfx1100_profiles.py.
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    strict = _bare_runner(residual=False, post_norm=False)
    monkeypatch.setenv(RESIDUAL_ENV, "1")
    monkeypatch.setenv(POST_NORM_ENV, "1")
    production = _bare_runner(residual=True, post_norm=True)

    assert runner_mod._verify_f32_residual_for_call(strict) is False
    assert runner_mod._verify_f32_post_norm_for_call(strict) is False
    assert runner_mod._verify_f32_residual_for_call(production) is True
    assert runner_mod._verify_f32_post_norm_for_call(production) is True

    monkeypatch.setenv(RESIDUAL_ENV, "0")
    monkeypatch.setenv(POST_NORM_ENV, "0")
    assert runner_mod._verify_f32_residual_for_call(production) is True
    assert runner_mod._verify_f32_post_norm_for_call(production) is True
    assert runner_mod._verify_f32_residual_for_call(strict) is False
    assert runner_mod._verify_f32_post_norm_for_call(strict) is False


def test_f32_session_methods_delegate_to_the_bound_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    monkeypatch.setenv(RESIDUAL_ENV, "1")
    monkeypatch.setenv(POST_NORM_ENV, "1")
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = _bare_runner(residual=False, post_norm=False)

    assert session._verify_f32_residual_for_call(None) is False
    assert session._verify_f32_residual_for_call(True) is True
    assert session._verify_f32_post_norm_for_call(None) is False
    assert session._verify_f32_post_norm_for_call(True) is True


# ------------------------------------------------------ generator transport


def test_f32_generator_session_kwargs_carry_only_bound_values() -> None:
    generator = _new_generator()
    generator.target_verify_f32_residual = None
    generator.target_verify_f32_post_norm = None
    generator.target_q4_t16_dual_silu_r28 = None
    assert generator._target_verify_f32_session_kwargs() == {}
    assert generator._target_q4_t16_dual_silu_r28_session_kwargs() == {}

    generator.target_verify_f32_residual = False
    generator.target_verify_f32_post_norm = True
    generator.target_q4_t16_dual_silu_r28 = True
    assert generator._target_verify_f32_session_kwargs() == {
        "target_verify_f32_residual": False,
        "target_verify_f32_post_norm": True,
    }
    assert generator._target_q4_t16_dual_silu_r28_session_kwargs() == {
        "target_q4_t16_dual_silu_r28": True
    }


def test_f32_generator_transports_to_shared_runner_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import qwen35_gguf as gguf

    runner_kwargs: dict[str, object] = {}
    session_kwargs: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            runner_kwargs.update(kwargs)

    class FakeSession:
        runtime = None
        max_sequence_length = None

        def __init__(self, *args, **kwargs) -> None:
            session_kwargs.update(kwargs)

    monkeypatch.setattr(gguf, "Qwen35GGUFFullStackRunner", FakeRunner)
    monkeypatch.setattr(gguf, "Qwen35GGUFResidentSession", FakeSession)
    generator = _new_generator()
    generator.model_path = "unused"
    generator.backend = "hip_gfx1151"
    generator.native_batch_decode = False
    generator._serving_capacity_bound = None
    generator._prepared_kv_signature = None
    generator.dms_metadata_path = None
    generator._auto_context_estimate = None
    generator.target_verify_f32_residual = False
    generator.target_verify_f32_post_norm = True
    generator.target_q4_t16_dual_silu_r28 = True

    generator._get_shared_runner()
    generator._construct_shared_session(
        SimpleNamespace(runtime=None),
        max_sequence_length=None,
        max_batch_size=1,
        defer_kv_allocation=False,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    )

    for kwargs in (runner_kwargs, session_kwargs):
        assert kwargs["target_verify_f32_residual"] is False
        assert kwargs["target_verify_f32_post_norm"] is True
        assert kwargs["target_q4_t16_dual_silu_r28"] is True


@pytest.mark.parametrize("requested", [True, False, None])
def test_f32_nextn_transports_owner(
    monkeypatch: pytest.MonkeyPatch,
    requested: bool | None,
) -> None:
    from hipengine.runtime import qwen35_gguf_nextn as nextn

    runner_kwargs: dict[str, object] = {}
    session_kwargs: dict[str, object] = {}

    class StopNextN(Exception):
        pass

    class FakeWeights:
        backend = "hip_gfx1151"

        def as_full_stack_weights(self):
            return object()

    class FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            runner_kwargs.update(kwargs)
            self.hidden_size = 4
            self.vocab_size = 8
            self.runtime = kwargs.get("runtime")

    class FakeSession:
        def __init__(self, *args, **kwargs) -> None:
            session_kwargs.update(kwargs)
            raise StopNextN

    monkeypatch.setattr(nextn, "register_gguf_q3_k_gemv_kernels", lambda: None)
    monkeypatch.setattr(nextn, "Qwen35GGUFFullStackRunner", FakeRunner)
    monkeypatch.setattr(nextn, "Qwen35GGUFResidentSession", FakeSession)

    with pytest.raises(StopNextN):
        nextn.Qwen35GGUFNextNDraftProvider.from_model(
            "unused",
            max_positions=8,
            max_requests=1,
            runtime=object(),
            resident_weights=FakeWeights(),
            target_verify_f32_residual=requested,
            target_verify_f32_post_norm=requested,
            target_q4_t16_dual_silu_r28=requested,
        )

    for kwargs in (runner_kwargs, session_kwargs):
        assert kwargs["target_verify_f32_residual"] is requested
        assert kwargs["target_verify_f32_post_norm"] is requested
        assert kwargs["target_q4_t16_dual_silu_r28"] is requested


# ------------------------------------------------------------- FFN seam


class _Ptr:
    def __init__(self, ptr: int, nbytes: int = 1 << 20) -> None:
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)


def _ffn_fixture(post_norm_calls: list[tuple[object, ...]]):
    from hipengine.runtime.qwen35_gguf_layer_ffn import run_post_attention_ffn_rows

    class Weight:
        def __init__(self, ptr: int) -> None:
            self._ptr = ptr

        def allocation(self, *_args, **_kwargs):
            return SimpleNamespace(tensor=SimpleNamespace(ptr=self._ptr))

    class Layer:
        def __init__(self) -> None:
            self._weights = {
                "post_attention_norm": Weight(0x101),
                "ffn_gate": Weight(0x102),
                "ffn_up": Weight(0x103),
                "ffn_down": Weight(0x104),
            }

        def weight(self, name: str):
            return self._weights[name]

    scratch = SimpleNamespace(
        post_norm=_Ptr(0x201),
        residual=_Ptr(0x202),
        post_norm_f32=_Ptr(0x203),
        ffn_intermediate=_Ptr(0x204),
        ffn_gate_up=_Ptr(0x205, 1 << 24),
        ffn_down=_Ptr(0x206),
    )
    kernels = SimpleNamespace(
        gguf_add_rmsnorm_f32_bf16_f32_weight=lambda *a, **k: post_norm_calls.append(
            ("add_norm", a, k)
        ),
        gguf_add_rmsnorm_f32_f32_f32_weight=lambda *a, **k: post_norm_calls.append(
            ("add_norm_f32", a, k)
        ),
        gguf_rmsnorm_f32_f32_weight_out_f32=lambda *a, **k: post_norm_calls.append(
            ("post_norm_f32", a, k)
        ),
        launch_gguf_linear_pair_silu=lambda *a, **k: True,
        launch_gguf_linear_pair=lambda *a, **k: False,
        launch_gguf_linear=lambda *a, **k: None,
        gguf_f32_bf16_add_out_f32=lambda *a, **k: None,
        f32_to_bf16=lambda *a, **k: None,
        gguf_bf16_add=lambda *a, **k: None,
        gguf_rmsnorm_bf16_f32_weight=lambda *a, **k: None,
        silu_mul_separate_out_bf16=lambda *a, **k: None,
    )
    ops = SimpleNamespace(
        mark_sync_stage=lambda *a, **k: 0.0,
        norm_residual_kernel=lambda **k: (lambda *a, **kw: None),
        dense_pair_silu_variant=lambda **k: None,
        dense_down_residual_fused=lambda **k: False,
        optional_q8_1_workspace_ptr=lambda *a, **k: None,
        cast_library=lambda: object(),
        rounded_next_rmsnorm=lambda **k: None,
    )
    flags = SimpleNamespace(
        verify_f32_attn_out_enabled=lambda: False,
        verify_f32_post_norm_enabled=lambda: False,
    )
    config = SimpleNamespace(is_moe=False, rms_norm_eps=1e-6)
    return run_post_attention_ffn_rows, Layer(), scratch, kernels, ops, flags, config


@pytest.mark.parametrize("requested", [None, False, True])
def test_ffn_explicit_post_norm_input_wins_over_owner_flag(requested) -> None:
    calls: list[tuple[object, ...]] = []
    fn, layer, scratch, kernels, ops, flags, config = _ffn_fixture(calls)
    flags.verify_f32_post_norm_enabled = lambda: True

    fn(
        layer,
        config,
        SimpleNamespace(hidden_size=8, ffn_size=16, backend="hip_gfx1150"),
        kernels,
        ops,
        flags,
        0,
        0x301,
        0x302,
        0x303,
        scratch,
        rows=1,
        hidden_f32_ptr=0x401,
        out_f32_ptr=0x402,
        dense_down_residual_decode_c1=False,
        force_bulk_rows=True,
        runtime=object(),
        verify_f32_post_norm=requested,
    )

    called = any(call[0] == "post_norm_f32" for call in calls)
    assert called is (True if requested is None else requested)


# --------------------------------------------------------- resource ABI


def test_f32_resource_validation_checks_supplied_buffers_only() -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=8)
    session._verify_hidden_f32_a = _Ptr(0x501, 2 * 8 * 4)
    session._verify_hidden_f32_b = _Ptr(0x502, 2 * 8 * 4)
    session.scratch = SimpleNamespace(post_norm_f32=_Ptr(0x503, 2 * 8 * 4))

    session._validate_f32_verifier_resources(rows=2, residual=True, post_norm=True)
    # Post-norm is irrelevant when the F32 residual path is not running.
    session.scratch = SimpleNamespace()
    session._validate_f32_verifier_resources(rows=2, residual=False, post_norm=True)

    session._verify_hidden_f32_b = _Ptr(0x502, 2 * 8 * 4 - 1)
    with pytest.raises(RuntimeError, match="hidden_f32_b"):
        session._validate_f32_verifier_resources(rows=2, residual=True, post_norm=False)

    session._verify_hidden_f32_b = _Ptr(0x502, 2 * 8 * 4)
    session.scratch = SimpleNamespace(post_norm_f32=_Ptr(0x503, 2 * 8 * 4 - 1))
    with pytest.raises(RuntimeError, match="post_norm_f32"):
        session._validate_f32_verifier_resources(rows=2, residual=True, post_norm=True)


def test_f32_resource_validation_uses_supplied_graph_buffers() -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=8)
    # The resident pair is absent; only the graph-owned pair is valid.
    session._verify_hidden_f32_a = None
    session._verify_hidden_f32_b = None
    session.scratch = SimpleNamespace(post_norm_f32=_Ptr(0x603, 2 * 8 * 4 - 1))

    session._validate_f32_verifier_resources(
        rows=2,
        residual=True,
        post_norm=True,
        hidden_f32_a=_Ptr(0x601, 2 * 8 * 4),
        hidden_f32_b=_Ptr(0x602, 2 * 8 * 4),
        scratch=SimpleNamespace(post_norm_f32=_Ptr(0x604, 2 * 8 * 4)),
    )


@pytest.mark.parametrize("rows", [1, 2, 5])
@pytest.mark.parametrize("closed", [False, True])
def test_f32_resource_validation_accepts_real_graph_tensor_handles(rows, closed) -> None:
    from hipengine.core.device import Device
    from hipengine.core.dtype import DType
    from hipengine.core.tensor import Tensor
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=8)
    session._verify_hidden_f32_a = None
    session._verify_hidden_f32_b = None
    session.scratch = SimpleNamespace()
    first = Tensor.from_handle(0 if closed else 0x601, (rows, 8), DType.FP32, Device("cpu"))
    second = Tensor.from_handle(0x602, (rows, 8), DType.FP32, Device("cpu"))
    kwargs = dict(rows=rows, residual=True, post_norm=False, hidden_f32_a=first, hidden_f32_b=second)
    if closed:
        with pytest.raises(RuntimeError, match="hidden_f32_a"):
            session._validate_f32_verifier_resources(**kwargs)
    else:
        session._validate_f32_verifier_resources(**kwargs)


def test_f32_resource_validation_rejects_undersized_graph_tensor_handle() -> None:
    from hipengine.core.device import Device
    from hipengine.core.dtype import DType
    from hipengine.core.tensor import Tensor
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=8)
    with pytest.raises(RuntimeError, match="hidden_f32_b"):
        session._validate_f32_verifier_resources(
            rows=2, residual=True, post_norm=False,
            hidden_f32_a=Tensor.from_handle(0x601, (2, 8), DType.FP32, Device("cpu")),
            hidden_f32_b=Tensor.from_handle(0x602, (1, 8), DType.FP32, Device("cpu")),
        )


def test_f32_resource_validation_rejects_undersized_supplied_graph_buffer() -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=8)
    # The resident pair is valid; the graph pair selected for this call is not.
    session._verify_hidden_f32_a = _Ptr(0x501, 2 * 8 * 4)
    session._verify_hidden_f32_b = _Ptr(0x502, 2 * 8 * 4)
    session.scratch = SimpleNamespace(post_norm_f32=_Ptr(0x503, 2 * 8 * 4))

    with pytest.raises(RuntimeError, match="hidden_f32_b"):
        session._validate_f32_verifier_resources(
            rows=2,
            residual=True,
            post_norm=False,
            hidden_f32_a=_Ptr(0x601, 2 * 8 * 4),
            hidden_f32_b=_Ptr(0x602, 2 * 8 * 4 - 1),
        )


def test_verify_target_block_validates_graph_selected_f32_resources() -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    class _StopValidation(RuntimeError):
        pass

    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        hidden_size=8,
        vocab_size=16,
        weights=object(),
    )
    session.scratch = SimpleNamespace(max_positions=8)
    session._position = 0
    session.runtime = object()
    session._prefill_token_buf = _Ptr(0x10)
    session._prefill_hidden_a = _Ptr(0x11)
    session._prefill_hidden_b = _Ptr(0x12)
    session._bulk_prefill_scratch = SimpleNamespace(rows=2)
    # Resident verifier F32 buffers are absent; graph-owned buffers are valid.
    session._verify_hidden_f32_a = None
    session._verify_hidden_f32_b = None
    session._ensure_verify_block_buffers = lambda *args, **kwargs: None
    session._normalize_layer_output_capture = lambda value: ()

    graph_scratch = SimpleNamespace(start=0, rows=2)
    graph_hidden_seed = _Ptr(0x21)
    graph_hidden_a = _Ptr(0x22, 2 * 8 * 4)
    graph_hidden_b = _Ptr(0x23, 2 * 8 * 4)
    seen: dict[str, object] = {}

    def validate(**kwargs):
        seen.update(kwargs)
        raise _StopValidation

    session._validate_f32_verifier_resources = validate

    with pytest.raises(_StopValidation):
        session.verify_target_block(
            [1, 2],
            stream=7,
            verify_f32_residual=True,
            verify_f32_post_norm=True,
            _pre_staged_token_ids_ptr=0x31,
            _target_top1_i64_ptr=0x32,
            _enqueue_only=True,
            _prebuilt_bulk_scratch=graph_scratch,
            _dynamic_cursor_advance=True,
            _graph_hidden_seed_buf=graph_hidden_seed,
            _graph_hidden_f32_a=graph_hidden_a,
            _graph_hidden_f32_b=graph_hidden_b,
        )

    assert seen["hidden_f32_a"] is graph_hidden_a
    assert seen["hidden_f32_b"] is graph_hidden_b
    assert seen["scratch"] is graph_scratch
    assert seen["residual"] is True
    assert seen["post_norm"] is True


def test_layer_boundary_capture_uses_explicit_post_norm_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM", "1")
    monkeypatch.setattr(
        runner_mod,
        "_copy_bf16_ptr_to_host_f32",
        lambda ptr, elements, *, runtime: np.zeros(elements, dtype=np.float32),
    )
    monkeypatch.setattr(
        runner_mod,
        "_copy_f32_ptr_to_host",
        lambda ptr, elements, *, runtime: np.zeros(elements, dtype=np.float32),
    )
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        hidden_size=8,
        weights=SimpleNamespace(config=SimpleNamespace(is_moe=False)),
    )
    session.scratch = SimpleNamespace(
        norm=_Ptr(0x101),
        attn_out=_Ptr(0x102),
        residual=_Ptr(0x103),
        post_norm=_Ptr(0x104),
        post_norm_f32=_Ptr(0x105),
    )
    # The frozen owner says post-norm is on; this operation explicitly says off.
    session._verify_f32_post_norm_for_call = (
        lambda requested=None: True if requested is None else bool(requested)
    )

    arrays = session._capture_verify_layer_boundary_rows(
        0,
        "full_attention",
        hidden_in_ptr=0x201,
        hidden_in_f32_ptr=0x202,
        layer_out_ptr=0x203,
        layer_out_f32_ptr=0x204,
        scratch=session.scratch,
        rows=1,
        runtime=object(),
        verify_f32_post_norm=False,
    )

    assert "attn_norm_f32_scratch" in arrays
    assert "attn_post_norm_f32" not in arrays


# ------------------------------------------------------ native identity


def test_capture_admission_uses_bound_session_owner_over_opposite_ambient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    monkeypatch.setenv(RESIDUAL_ENV, "1")
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING", "1")
    monkeypatch.setattr(cycle_mod, "resolve", lambda **kwargs: object())
    session = object.__new__(runner_mod.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        verify_f32_residual=False,
        verify_f32_post_norm=False,
    )
    session.scratch = SimpleNamespace(max_positions=4096)
    session.backend = "hip_gfx1100"
    session.kv_storage_dtype = "bf16"
    session._position = 8
    admission = dict(
        context_limit=512,
        bulk_attention_mode="bulk",
        use_wmma_prefill=False,
        capture_lm_head_logits=False,
        record_stage_timings=False,
        sync_stage_timings=False,
    )

    # The bound session owner is False; opposite ambient True must not retarget
    # graph admission.
    cycle_mod._validate_capture_admission(session, [1, 2, 3], **admission)

    # An explicit operation override still wins over the bound owner.
    with pytest.raises(
        cycle_mod.NativeSpecTargetGraphUnsupportedError,
        match="token-embedding",
    ):
        cycle_mod._validate_capture_admission(
            session,
            [1, 2, 3],
            verify_f32_residual=True,
            **admission,
        )


def test_native_configuration_key_separates_f32_and_r28_owner_values() -> None:
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod

    common = dict(
        bulk_attention_mode="native",
        use_wmma_prefill=False,
        capture_linear_state_rows=True,
        capture_prefill_gdn=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True,
        device_accept_commit=True,
    )
    baseline = cycle_mod._native_target_configuration_key(**common)
    f32 = cycle_mod._native_target_configuration_key(
        **common, verify_f32_residual=True, verify_f32_post_norm=True
    )
    r28 = cycle_mod._native_target_configuration_key(
        **common, q4_t16_dual_silu_r28=True
    )
    assert baseline != f32
    assert baseline != r28
    assert f32 != r28


def test_native_route_identity_reads_owner_aware_session_values() -> None:
    from hipengine.runtime import gguf_native_spec_cycle as cycle_mod

    session = SimpleNamespace(
        runner=SimpleNamespace(q4_t16_dual_silu_r28=True),
        _verify_f32_residual_for_call=lambda requested: (
            False if requested is None else bool(requested)
        ),
        _verify_f32_post_norm_for_call=lambda requested: (
            True if requested is None else bool(requested)
        ),
    )
    identity = cycle_mod._native_target_route_identity(session)
    assert identity == {
        "router_f32w_coop": None,
        "router_f32w_persistent_counter": None,
        "q8_t16_rowtile_all": None,
        "verify_f32_residual": False,
        "verify_f32_post_norm": True,
        "q4_t16_dual_silu_r28": True,
    }
    overridden = cycle_mod._native_target_route_identity(
        session, verify_f32_residual=True, verify_f32_post_norm=False
    )
    assert overridden["verify_f32_residual"] is True
    assert overridden["verify_f32_post_norm"] is False


def test_k1_disable_is_explicit_input_without_environment_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime.gguf_native_spec_cycle import (
        run_qwen35_gguf_native_mtp_cycle,
    )

    seen: list[tuple[dict[str, object], dict[str, str | None]]] = []

    class Draft:
        def propose_chain_from_device_seed(self, hidden_seed_ptr, **kwargs):
            del hidden_seed_ptr
            return [201], [[201]], int(kwargs["dense_cache_len"]) + 1

    class Context:
        pending_seed = SimpleNamespace(hidden_ptr=0xA000)

        def record_verify_seeds(self, seeds):
            self.seeds = tuple(seeds)

        def accept(self, accepted):
            self.pending_seed = self.seeds[int(accepted)]
            return self.pending_seed

    class Session:
        position = 17
        verifier_prefill_gdn_capture = False

        def verify_target_block_native_cycle(self, input_token_ids, **kwargs):
            seen.append(
                (
                    dict(kwargs),
                    {
                        name: os.environ.get(name)
                        for name in (RESIDUAL_ENV, POST_NORM_ENV)
                    },
                )
            )
            self.position = 18
            return SimpleNamespace(
                token_ids=[909],
                accepted_draft_tokens=0,
                start_position=17,
                end_position=18,
                hidden_seed_rows_ptr=0xB000,
                hidden_seed_row_count=2,
                device_accept_commit=True,
            )

        def mtp_verify_seed(self, row, *, token_id, position, **kwargs):
            return SimpleNamespace(
                token_id=int(token_id),
                position=int(position),
                hidden_ptr=int(kwargs["hidden_seed_base_ptr"]) + int(row) * 16,
                hidden_contract=SimpleNamespace(
                    ready_for_mtp=True, rows=1, hidden_size=4
                ),
            )

    run_qwen35_gguf_native_mtp_cycle(
        Session(),
        Draft(),
        Context(),
        root_token=101,
        root_position=17,
        candidate_budget=1,
        remaining_decode=7,
        rope_cos=np.ones((32, 4), dtype=np.float32),
        rope_sin=np.zeros((32, 4), dtype=np.float32),
        draft_key_cache=DeviceBuffer(0xC000, 4096),
        draft_value_cache=DeviceBuffer(0xD000, 4096),
        draft_cache_len=7,
        k1_disable_f32_verifier=True,
    )

    assert seen[0][0]["verify_f32_residual"] is False
    assert seen[0][0]["verify_f32_post_norm"] is False
    assert seen[0][1] == {RESIDUAL_ENV: None, POST_NORM_ENV: None}
    assert RESIDUAL_ENV not in os.environ
    assert POST_NORM_ENV not in os.environ


def test_k1_disable_operation_failure_leaves_environment_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.runtime.gguf_native_spec_cycle import (
        run_qwen35_gguf_native_mtp_cycle,
    )

    class Draft:
        def propose_chain_from_device_seed(self, hidden_seed_ptr, **kwargs):
            del hidden_seed_ptr
            return [201], [[201]], int(kwargs["dense_cache_len"]) + 1

    class Context:
        pending_seed = SimpleNamespace(hidden_ptr=0xA000)

    class Session:
        position = 17
        verifier_prefill_gdn_capture = False

        def verify_target_block_native_cycle(self, input_token_ids, **kwargs):
            del input_token_ids
            assert kwargs["verify_f32_residual"] is False
            assert kwargs["verify_f32_post_norm"] is False
            assert RESIDUAL_ENV not in os.environ
            assert POST_NORM_ENV not in os.environ
            raise RuntimeError("target failure")

    with pytest.raises(RuntimeError, match="target failure"):
        run_qwen35_gguf_native_mtp_cycle(
            Session(),
            Draft(),
            Context(),
            root_token=101,
            root_position=17,
            candidate_budget=1,
            remaining_decode=7,
            rope_cos=np.ones((32, 4), dtype=np.float32),
            rope_sin=np.zeros((32, 4), dtype=np.float32),
            draft_key_cache=DeviceBuffer(0xC000, 4096),
            draft_value_cache=DeviceBuffer(0xD000, 4096),
            draft_cache_len=7,
            k1_disable_f32_verifier=True,
        )
    assert RESIDUAL_ENV not in os.environ
    assert POST_NORM_ENV not in os.environ
