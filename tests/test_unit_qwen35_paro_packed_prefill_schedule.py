"""Boundary and pre-move control for the PARO packed-prefill layer schedule."""

from __future__ import annotations

import ast
import inspect
import json
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from hipengine.core.dtype import DType
from hipengine.runtime import qwen35_paro
from hipengine.runtime.qwen35_paro import (
    Qwen35ParoDecodeState,
    Qwen35ParoGroupedMoeScratch,
)
from hipengine.runtime import qwen35_paro_packed_prefill as owner

FIXTURE = Path(__file__).parent / "fixtures" / "qwen35_paro_packed_prefill_control.json"
CONTROL = json.loads(FIXTURE.read_text())
CASES = {case["name"]: case for case in CONTROL["cases"]}

OWNER_CASES = [
    name
    for name in CASES
    if name.startswith("provided_")
    or name
    in {
        "input_failure",
        "linear_failure",
        "post_norm_failure",
        "grouped_moe_failure",
    }
]
ADAPTER_CASES = [
    name for name in CASES if name.startswith("fallback_") or name == "invalid_tokens_zero"
]
FAILURE_OPS = {
    "input_failure": "input_rmsnorm_fp16",
    "linear_failure": "run_linear_attention_prefill_out_proj_segments_fp16",
    "post_norm_failure": "post_attention_add_rmsnorm_fp16",
    "grouped_moe_failure": "run_moe_grouped_compact_fp16",
}


@dataclass(frozen=True)
class FakeTensor:
    name: str
    ptr: int


class FakeLinearScratch:
    def __init__(self, name: str, attn_input: FakeTensor) -> None:
        self.name = name
        self.attn_input = attn_input


class FalsyLinearScratch(FakeLinearScratch):
    def __bool__(self) -> bool:
        return False


class FakeMoeScratch:
    def __init__(self, name: str) -> None:
        self.name = name
        self.normed = FakeTensor(f"{name}.normed", 0x9000)
        self.residual = FakeTensor(f"{name}.residual", 0x9100)
        self.moe_out = FakeTensor(f"{name}.moe_out", 0x9200)


def _real_moe(name: str) -> Qwen35ParoGroupedMoeScratch:
    scratch = object.__new__(Qwen35ParoGroupedMoeScratch)
    object.__setattr__(scratch, "name", name)
    object.__setattr__(scratch, "normed", FakeTensor(f"{name}.normed", 0x9000))
    object.__setattr__(scratch, "residual", FakeTensor(f"{name}.residual", 0x9100))
    object.__setattr__(scratch, "moe_out", FakeTensor(f"{name}.moe_out", 0x9200))
    return scratch


def _serialize(value):
    if isinstance(value, FakeTensor):
        return {"tensor": value.name, "ptr": value.ptr}
    if isinstance(value, FakeLinearScratch):
        return {"scratch": value.name, "kind": "linear"}
    if isinstance(value, (FakeMoeScratch, Qwen35ParoGroupedMoeScratch)):
        return {"scratch": value.name, "kind": "grouped_moe"}
    if isinstance(value, DType):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"repr": repr(value)}


class Recorder:
    def __init__(
        self,
        *,
        linear_scratch: FakeLinearScratch,
        moe_scratch: FakeMoeScratch | Qwen35ParoGroupedMoeScratch,
        failure: str | None = None,
    ) -> None:
        self.linear_scratch = linear_scratch
        self.moe_scratch = moe_scratch
        self.failure = failure
        self.failure_error = None if failure is None else RuntimeError(f"{failure} failed")
        self.events = []
        self.raw_calls = []

    def _record(self, op, args, kwargs):
        self.raw_calls.append((op, args, kwargs))
        self.events.append(
            {
                "op": op,
                "args": [_serialize(value) for value in args],
                "kwargs": [[key, _serialize(value)] for key, value in kwargs.items()],
            }
        )

    def _raise(self, op):
        if self.failure == op:
            assert self.failure_error is not None
            raise self.failure_error

    def reserve_linear_attention_scratch(self, *args, **kwargs):
        self._record("reserve_linear_attention_scratch", args, kwargs)
        self._raise("reserve_linear_attention_scratch")
        return self.linear_scratch

    def reserve_moe_grouped_prefill_scratch(self, *args, **kwargs):
        self._record("reserve_moe_grouped_prefill_scratch", args, kwargs)
        self._raise("reserve_moe_grouped_prefill_scratch")
        return self.moe_scratch

    def input_rmsnorm_fp16(self, *args, **kwargs):
        self._record("input_rmsnorm_fp16", args, kwargs)
        self._raise("input_rmsnorm_fp16")
        return None

    def run_linear_attention_prefill_out_proj_segments_fp16(self, *args, **kwargs):
        self._record("run_linear_attention_prefill_out_proj_segments_fp16", args, kwargs)
        self._raise("run_linear_attention_prefill_out_proj_segments_fp16")
        return FakeTensor("attn_out", 0x7000)

    def post_attention_add_rmsnorm_fp16(self, *args, **kwargs):
        self._record("post_attention_add_rmsnorm_fp16", args, kwargs)
        self._raise("post_attention_add_rmsnorm_fp16")
        return self.moe_scratch.normed, self.moe_scratch.residual

    def run_moe_grouped_compact_fp16(self, *args, **kwargs):
        self._record("run_moe_grouped_compact_fp16", args, kwargs)
        self._raise("run_moe_grouped_compact_fp16")
        return self.moe_scratch.moe_out


def _linear_scratch(case_name: str) -> FakeLinearScratch:
    if case_name == "fallback_both_t3_s2_g128":
        return FakeLinearScratch("linear.reserve", FakeTensor("linear.attn_input", 0x6000))
    return FakeLinearScratch(
        "linear.provided", FakeTensor("linear.provided.attn_input", 0x6100)
    )


def _fake_moe(case_name: str) -> FakeMoeScratch:
    if case_name in {"fallback_both_t3_s2_g128", "fallback_moe_only_t4_s2_g128"}:
        return FakeMoeScratch("moe.reserve")
    return FakeMoeScratch("moe.provided")


def _real_moe_for(case_name: str) -> Qwen35ParoGroupedMoeScratch:
    if case_name in {"fallback_both_t3_s2_g128", "fallback_moe_only_t4_s2_g128"}:
        return _real_moe("moe.reserve")
    return _real_moe("moe.provided")


def _tensors():
    return {
        "hidden": FakeTensor("hidden", 0x1000),
        "conv_state": FakeTensor("conv_state", 0x2000),
        "recurrent_state": FakeTensor("recurrent_state", 0x3000),
        "cu_seqlens": FakeTensor("cu_seqlens", 0x4000),
        "state_indices": FakeTensor("state_indices", 0x5000),
    }


def _call_owner(case_name: str, recorder: Recorder, tensors):
    case = CASES[case_name]
    inputs = case["inputs"]
    return owner.run_linear_attention_moe_packed_prefill_layer(
        tensors["hidden"],
        conv_state=tensors["conv_state"],
        recurrent_state=tensors["recurrent_state"],
        cu_seqlens=tensors["cu_seqlens"],
        state_indices=tensors["state_indices"],
        geometry=owner.Qwen35ParoPackedPrefillGeometry(
            tokens=inputs["tokens"],
            segments=inputs["segments"],
            group_size=inputs["group_size"],
        ),
        scratch=owner.Qwen35ParoPackedPrefillScratch(
            linear=recorder.linear_scratch,
            moe=recorder.moe_scratch,
        ),
        input_rmsnorm=recorder.input_rmsnorm_fp16,
        linear_prefill_segments=recorder.run_linear_attention_prefill_out_proj_segments_fp16,
        post_attention_add_rmsnorm=recorder.post_attention_add_rmsnorm_fp16,
        grouped_moe=recorder.run_moe_grouped_compact_fp16,
        library="lib",
        stream=7,
    )


def _observe(call):
    try:
        returned = call()
    except BaseException as exc:  # noqa: BLE001 - exact exception identity is asserted elsewhere
        return None, {"type": type(exc).__name__, "message": str(exc)}, exc
    return returned, None, None


def _state(recorder: Recorder) -> Qwen35ParoDecodeState:
    state = object.__new__(Qwen35ParoDecodeState)

    def bind(func):
        def bound(_self, *args, **kwargs):
            return func(*args, **kwargs)

        return types.MethodType(bound, state)

    state.reserve_linear_attention_scratch = bind(recorder.reserve_linear_attention_scratch)
    state.reserve_moe_grouped_prefill_scratch = bind(
        recorder.reserve_moe_grouped_prefill_scratch
    )
    state.input_rmsnorm_fp16 = bind(recorder.input_rmsnorm_fp16)
    state.run_linear_attention_prefill_out_proj_segments_fp16 = bind(
        recorder.run_linear_attention_prefill_out_proj_segments_fp16
    )
    state.post_attention_add_rmsnorm_fp16 = bind(
        recorder.post_attention_add_rmsnorm_fp16
    )
    state.run_moe_grouped_compact_fp16 = bind(recorder.run_moe_grouped_compact_fp16)
    return state


def _call_adapter(case_name: str, recorder: Recorder, tensors):
    case = CASES[case_name]
    inputs = case["inputs"]
    state = _state(recorder)
    linear = None if inputs["linear_scratch"] is None else recorder.linear_scratch
    moe = None if inputs["moe_scratch"] is None else recorder.moe_scratch
    return state.run_linear_attention_moe_packed_prefill_layer_fp16(
        tensors["hidden"],
        conv_state=tensors["conv_state"],
        recurrent_state=tensors["recurrent_state"],
        cu_seqlens=tensors["cu_seqlens"],
        state_indices=tensors["state_indices"],
        segments=inputs["segments"],
        linear_scratch=linear,
        moe_scratch=moe,
        tokens=inputs["tokens"],
        group_size=inputs["group_size"],
        library="lib",
        stream=7,
    )


def test_control_fixture_provenance() -> None:
    assert CONTROL["schema"] == 1
    assert CONTROL["base_commit"] == "d1dd4c4af9fc0318fb3c3af2a2a9fcc63635ad5b"
    assert CONTROL["method"] == (
        "Qwen35ParoDecodeState.run_linear_attention_moe_packed_prefill_layer_fp16"
    )


@pytest.mark.parametrize("case_name", OWNER_CASES)
def test_owner_matches_pre_move_control(case_name: str) -> None:
    case = CASES[case_name]
    recorder = Recorder(
        linear_scratch=_linear_scratch(case_name),
        moe_scratch=_fake_moe(case_name),
        failure=case["inputs"]["failure"],
    )
    tensors = _tensors()
    returned, error, _ = _observe(lambda: _call_owner(case_name, recorder, tensors))
    assert recorder.events == case["events"]
    assert (_serialize(returned) if returned is not None else None) == case["returned"]
    assert error == case["error"]


@pytest.mark.parametrize("case_name", ADAPTER_CASES)
def test_adapter_matches_pre_move_control(case_name: str) -> None:
    case = CASES[case_name]
    recorder = Recorder(
        linear_scratch=_linear_scratch(case_name),
        moe_scratch=_real_moe_for(case_name),
        failure=case["inputs"]["failure"],
    )
    tensors = _tensors()
    returned, error, exc = _observe(lambda: _call_adapter(case_name, recorder, tensors))
    assert recorder.events == case["events"]
    assert (_serialize(returned) if returned is not None else None) == case["returned"]
    assert error == case["error"]
    if case["inputs"]["failure"] is not None:
        assert exc is recorder.failure_error


def test_owner_passes_raw_tensor_identity_and_returns_grouped_moe_tensor() -> None:
    recorder = Recorder(
        linear_scratch=_linear_scratch("provided_t3_s2_g128"),
        moe_scratch=_fake_moe("provided_t3_s2_g128"),
    )
    tensors = _tensors()
    returned = _call_owner("provided_t3_s2_g128", recorder, tensors)

    assert returned is recorder.moe_scratch.moe_out
    assert [call[0] for call in recorder.raw_calls] == [
        "input_rmsnorm_fp16",
        "run_linear_attention_prefill_out_proj_segments_fp16",
        "post_attention_add_rmsnorm_fp16",
        "run_moe_grouped_compact_fp16",
    ]
    input_call = recorder.raw_calls[0]
    assert input_call[1] == (tensors["hidden"], recorder.linear_scratch.attn_input)
    linear_call = recorder.raw_calls[1]
    assert linear_call[1] == (recorder.linear_scratch.attn_input,)
    assert linear_call[2]["conv_state"] is tensors["conv_state"]
    assert linear_call[2]["recurrent_state"] is tensors["recurrent_state"]
    assert linear_call[2]["cu_seqlens"] is tensors["cu_seqlens"]
    assert linear_call[2]["state_indices"] is tensors["state_indices"]
    assert linear_call[2]["scratch"] is recorder.linear_scratch
    post_call = recorder.raw_calls[2]
    assert post_call[1][0] is tensors["hidden"]
    assert post_call[1][1] is not None
    assert post_call[1][2] is recorder.moe_scratch
    grouped_call = recorder.raw_calls[3]
    assert grouped_call[1] == (
        recorder.moe_scratch.normed,
        recorder.moe_scratch.residual,
    )
    assert grouped_call[2]["scratch"] is recorder.moe_scratch


@pytest.mark.parametrize("case_name,op", FAILURE_OPS.items())
def test_owner_preserves_exact_exception_and_stops(case_name: str, op: str) -> None:
    recorder = Recorder(
        linear_scratch=_linear_scratch(case_name),
        moe_scratch=_fake_moe(case_name),
        failure=op,
    )
    tensors = _tensors()
    _, _, error = _observe(lambda: _call_owner(case_name, recorder, tensors))
    assert error is recorder.failure_error
    assert recorder.events[-1]["op"] == op
    assert len(recorder.events) == len(CASES[case_name]["events"])


def test_adapter_validation_precedes_scratch_fallback() -> None:
    recorder = Recorder(
        linear_scratch=_linear_scratch("fallback_both_t3_s2_g128"),
        moe_scratch=_real_moe_for("fallback_both_t3_s2_g128"),
    )
    tensors = _tensors()
    _, error, _ = _observe(lambda: _call_adapter("invalid_tokens_zero", recorder, tensors))
    assert error == {"type": "ValueError", "message": "tokens must be positive"}
    assert recorder.events == []


def test_adapter_truthy_linear_scratch_fallback_and_grouped_moe_isinstance() -> None:
    provided_linear = FalsyLinearScratch(
        "linear.falsy", FakeTensor("linear.falsy.attn_input", 0x6200)
    )
    reserved_linear = FakeLinearScratch(
        "linear.reserve", FakeTensor("linear.attn_input", 0x6000)
    )
    reserved_moe = _real_moe("moe.reserve")
    recorder = Recorder(linear_scratch=reserved_linear, moe_scratch=reserved_moe)
    state = _state(recorder)
    tensors = _tensors()

    returned = state.run_linear_attention_moe_packed_prefill_layer_fp16(
        tensors["hidden"],
        conv_state=tensors["conv_state"],
        recurrent_state=tensors["recurrent_state"],
        cu_seqlens=tensors["cu_seqlens"],
        state_indices=tensors["state_indices"],
        segments=2,
        linear_scratch=provided_linear,
        moe_scratch=object(),
        tokens=3,
        group_size=128,
        library="lib",
        stream=7,
    )

    assert returned is reserved_moe.moe_out
    assert [event["op"] for event in recorder.events[:2]] == [
        "reserve_linear_attention_scratch",
        "reserve_moe_grouped_prefill_scratch",
    ]
    assert recorder.raw_calls[2][1] == (tensors["hidden"], reserved_linear.attn_input)
    assert recorder.raw_calls[3][2]["scratch"] is reserved_linear


def test_adapter_delegates_four_separate_callables(monkeypatch) -> None:
    recorder = Recorder(
        linear_scratch=_linear_scratch("provided_t3_s2_g128"),
        moe_scratch=_real_moe_for("provided_t3_s2_g128"),
    )
    state = _state(recorder)
    tensors = _tensors()
    observed = {}
    sentinel = object()

    def fake_owner(hidden, **kwargs):
        observed["hidden"] = hidden
        observed["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(qwen35_paro, "run_linear_attention_moe_packed_prefill_layer", fake_owner)
    returned = state.run_linear_attention_moe_packed_prefill_layer_fp16(
        tensors["hidden"],
        conv_state=tensors["conv_state"],
        recurrent_state=tensors["recurrent_state"],
        cu_seqlens=tensors["cu_seqlens"],
        state_indices=tensors["state_indices"],
        segments=2,
        linear_scratch=recorder.linear_scratch,
        moe_scratch=recorder.moe_scratch,
        tokens=3,
        group_size=64,
        library="lib",
        stream=7,
    )

    assert returned is sentinel
    assert observed["hidden"] is tensors["hidden"]
    kwargs = observed["kwargs"]
    assert "ops" not in kwargs
    assert set(kwargs) == {
        "conv_state",
        "recurrent_state",
        "cu_seqlens",
        "state_indices",
        "geometry",
        "scratch",
        "input_rmsnorm",
        "linear_prefill_segments",
        "post_attention_add_rmsnorm",
        "grouped_moe",
        "library",
        "stream",
    }
    assert kwargs["conv_state"] is tensors["conv_state"]
    assert kwargs["recurrent_state"] is tensors["recurrent_state"]
    assert kwargs["cu_seqlens"] is tensors["cu_seqlens"]
    assert kwargs["state_indices"] is tensors["state_indices"]
    assert kwargs["geometry"] == owner.Qwen35ParoPackedPrefillGeometry(
        tokens=3, segments=2, group_size=64
    )
    assert kwargs["scratch"].linear is recorder.linear_scratch
    assert kwargs["scratch"].moe is recorder.moe_scratch
    for name in (
        "input_rmsnorm",
        "linear_prefill_segments",
        "post_attention_add_rmsnorm",
        "grouped_moe",
    ):
        callable_ = kwargs[name]
        assert callable_.__self__ is state
    assert kwargs["library"] == "lib"
    assert kwargs["stream"] == 7


def test_public_state_method_signature_is_unchanged() -> None:
    signature = inspect.signature(
        Qwen35ParoDecodeState.run_linear_attention_moe_packed_prefill_layer_fp16
    )
    assert list(signature.parameters) == [
        "self",
        "hidden",
        "conv_state",
        "recurrent_state",
        "cu_seqlens",
        "state_indices",
        "segments",
        "linear_scratch",
        "moe_scratch",
        "tokens",
        "group_size",
        "library",
        "stream",
    ]
    assert signature.parameters["linear_scratch"].default is None
    assert signature.parameters["moe_scratch"].default is None
    assert signature.parameters["group_size"].default == 128
    assert signature.parameters["library"].default is None
    assert signature.parameters["stream"].default == 0


def test_owner_has_no_runtime_state_import_or_forbidden_effects() -> None:
    source = Path(owner.__file__).read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert node.module != "hipengine.runtime.qwen35_paro"
    for forbidden in ("malloc(", "memcpy", "synchronize", "resolve_", "get_hip_runtime"):
        assert forbidden not in source
