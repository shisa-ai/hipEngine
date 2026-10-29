"""Keep the halfdot/exact comparison runnable across resident layouts."""

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime import laguna_moe as moe
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from scripts import laguna_selected_natural_decode_leaf as leaf


@pytest.mark.parametrize("backend,default", [("hip_gfx1100", False), ("hip_gfx1151", True)])
@pytest.mark.parametrize("override", [None, False, True])
def test_halfdot_constructor_policy_and_explicit_rollback(backend, default, override):
    session = object.__new__(LagunaGGUFResidentSession)
    with pytest.raises(ValueError, match="model_path is required"):
        session.__init__(backend=backend, runtime=object(), use_selected_halfdot_decode=override)
    assert session.use_selected_halfdot_decode is (default if override is None else override)
    assert not hasattr(LagunaGGUFResidentSession, "set_selected_halfdot_decode")


@pytest.mark.parametrize("rows", [1, 2, 17])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("halfdot", [False, True])
def test_halfdot_only_rewrites_paired_singleton_route(monkeypatch, rows, paired, halfdot):
    calls = []
    def route(label, allocation):
        return SimpleNamespace(label=label, abi="capture", allocation_name=allocation)
    retained = route("ordinary", "tiles")
    exact = route("paired-exact", "tiles_dual")
    half = route("paired-halfdot", "tiles_dual")
    plan = SimpleNamespace(
        selected_gate_up_routes={"q4": retained}, c1_selected_gate_up_routes={"q4": retained},
        interleaved_natural_selected_gate_up_route=exact,
        interleaved_natural_selected_gate_up_halfdot_route=half,
    )
    gate = SimpleNamespace(
        spec=SimpleNamespace(quant_key="q4"), allocations={"tiles_dual": object()} if paired else {},
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
    )
    up = SimpleNamespace(allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=12)))
    layer = SimpleNamespace(weight=lambda name: gate if name == "ffn_gate_exps" else up)
    scratch = SimpleNamespace(plan=plan, selected_experts=SimpleNamespace(ptr=13))
    monkeypatch.setattr(moe, "_SELECTED_GATE_UP_ABIS", {"capture": lambda route, *a, **kw: calls.append((route, a, kw))})
    moe._launch_selected_gate_up(10, layer, scratch, x_rows=rows, lanes=rows*10,
        stream=7, runtime=None, libraries=None, use_interleaved_halfdot=halfdot)
    selected, args, kwargs = calls[0]
    assert selected is (half if paired and halfdot and rows == 1 else exact if paired else retained)
    assert args[1:5] == (11, 11 if paired else 12, 10, 13)
    assert kwargs["x_rows"] == rows and kwargs["lanes"] == rows*10 and kwargs["stream"] == 7


@pytest.fixture
def leaf_harness(monkeypatch):
    allocated, freed, calls = [], [], []
    def malloc(nbytes, **kwargs):
        buffer = SimpleNamespace(ptr=len(allocated)+1, nbytes=nbytes)
        allocated.append(buffer)
        return buffer
    monkeypatch.setattr(leaf, "malloc", malloc)
    monkeypatch.setattr(leaf, "free", lambda buffer, **kwargs: freed.append(buffer.ptr))
    monkeypatch.setattr(leaf, "copy_host_to_device", lambda *a, **kw: None)
    monkeypatch.setattr(leaf, "_event_ms", lambda runtime, fn, **kw: (fn(), 1.0)[1])
    monkeypatch.setattr(leaf, "_read_bf16", lambda runtime, buffer, shape: np.zeros(shape, dtype=np.uint16))
    prefix = "gguf_q4_k_t16_selected_dual_"
    for name in (
        "natural_tile8_parallel_silu_gemv_bf16_bf16_out",
        "natural_tile8_parallel_silu_paircoeff_gemv_bf16_bf16_out",
        "natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out",
        "interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out",
        "interleaved_natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out",
    ):
        monkeypatch.setattr(leaf, prefix+name, lambda *a, _name=name, **kw: calls.append((_name, a)))
    args = dict(runtime=SimpleNamespace(device_synchronize=lambda: None), library=object(),
        x=np.zeros((1, 3072), dtype=np.uint16), selected=np.zeros(10, dtype=np.int32),
        tiles_a=np.zeros(2, dtype=np.uint8), tiles_b=np.zeros(2, dtype=np.uint8),
        tiles_dual=np.zeros(4, dtype=np.uint8), in_features=3072, out_features=1024,
        warmups=0, samples=2, burst=1)
    return SimpleNamespace(allocated=allocated, freed=freed, calls=calls, malloc=malloc, args=args)


@pytest.mark.parametrize("fail_at", [None, 2, 5, 6])
def test_ordinary_halfdot_leaf_preserves_reference_and_partial_cleanup(leaf_harness, monkeypatch, fail_at):
    h = leaf_harness
    args = {name: value for name, value in h.args.items() if name != "tiles_dual"}
    def failing_malloc(nbytes, **kwargs):
        if len(h.allocated)+1 == fail_at:
            raise MemoryError("synthetic ordinary halfdot allocation failure")
        return h.malloc(nbytes, **kwargs)
    monkeypatch.setattr(leaf, "malloc", failing_malloc)
    if fail_at is None:
        result = leaf._screen_pair_silu_halfdot(**args)
        control = "natural_tile8_parallel_silu_paircoeff_gemv_bf16_bf16_out"
        candidate = "natural_tile8_parallel_silu_halfdot_gemv_bf16_bf16_out"
        assert [name for name, _ in h.calls] == [control, candidate, candidate, control, control, candidate]
        assert all(args[2:4] == (3, 4) for _, args in h.calls)
        assert result["bf16_mismatch"] == 0 and result["max_abs_error"] == 0
    else:
        with pytest.raises(MemoryError, match="synthetic ordinary halfdot"):
            leaf._screen_pair_silu_halfdot(**args)
    assert h.freed == list(reversed([buf.ptr for buf in h.allocated]))


@pytest.mark.parametrize("halfdot", [False, True])
def test_interleaved_leaf_preserves_distinct_exact_and_halfdot_controls(leaf_harness, halfdot):
    h = leaf_harness
    result = leaf._screen_pair_silu_interleaved(**h.args, halfdot=halfdot)
    ordinary = "natural_tile8_parallel_silu_gemv_bf16_bf16_out"
    exact = "interleaved_" + ordinary
    candidate = exact.replace("silu_gemv", "silu_halfdot_gemv") if halfdot else exact
    control = exact if halfdot else ordinary
    assert [name for name, _ in h.calls] == [control, candidate, candidate, control, control, candidate]
    for name, args in h.calls:
        assert args[2:4] == ((5, 5) if name.startswith("interleaved_") else (3, 4))
    assert result["bf16_mismatch"] == 0
    assert h.freed == list(reversed([buf.ptr for buf in h.allocated]))


@pytest.mark.parametrize("fail_at", [1, 4])
def test_interleaved_leaf_frees_uploaded_buffer_when_copy_fails(leaf_harness, monkeypatch, fail_at):
    h = leaf_harness
    copies = []
    def copy(*args, **kwargs):
        copies.append(None)
        if len(copies) == fail_at:
            raise RuntimeError("synthetic halfdot copy failure")
    monkeypatch.setattr(leaf, "copy_host_to_device", copy)
    with pytest.raises(RuntimeError, match="synthetic halfdot copy"):
        leaf._screen_pair_silu_interleaved(**h.args, halfdot=True)
    assert h.freed == list(reversed([buf.ptr for buf in h.allocated]))


@pytest.mark.parametrize("failure", ["stop_create", "body", "stop_destroy", None])
def test_leaf_timing_releases_events_on_setup_body_and_destroy_failure(failure):
    created, destroyed = [], []
    def create():
        if failure == "stop_create" and created:
            raise RuntimeError("synthetic timing failure")
        created.append(len(created)+1)
        return created[-1]
    def destroy(event):
        destroyed.append(event)
        if failure == "stop_destroy" and event == 2:
            raise RuntimeError("synthetic timing failure")
    def body():
        if failure == "body":
            raise RuntimeError("synthetic timing failure")
    runtime = SimpleNamespace(event_create=create, event_destroy=destroy,
        event_record=lambda event: None, event_synchronize=lambda event: None,
        event_elapsed_time_ms=lambda *a: 4.0)
    if failure:
        with pytest.raises(RuntimeError, match="synthetic timing"):
            leaf._event_ms(runtime, body, burst=2)
    else:
        assert leaf._event_ms(runtime, body, burst=2) == 2.0
    assert destroyed == list(reversed(created))


@pytest.mark.parametrize("failure", ["kernel", "timing"])
def test_interleaved_halfdot_leaf_frees_buffers_after_body_failure(leaf_harness, monkeypatch, failure):
    h = leaf_harness
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic halfdot body failure")
    if failure == "kernel":
        monkeypatch.setattr(leaf, "gguf_q4_k_t16_selected_dual_interleaved_natural_tile8_parallel_silu_gemv_bf16_bf16_out", fail)
    else:
        monkeypatch.setattr(leaf, "_event_ms", fail)
    with pytest.raises(RuntimeError, match="synthetic halfdot body"):
        leaf._screen_pair_silu_interleaved(**h.args, halfdot=True)
    assert h.freed == list(reversed([buf.ptr for buf in h.allocated]))


@pytest.mark.parametrize("fail_at", [2, 5, 7])
def test_interleaved_halfdot_leaf_frees_partial_allocations(leaf_harness, monkeypatch, fail_at):
    h = leaf_harness
    def failing_malloc(nbytes, **kwargs):
        if len(h.allocated)+1 == fail_at:
            raise MemoryError("synthetic halfdot allocation failure")
        return h.malloc(nbytes, **kwargs)
    monkeypatch.setattr(leaf, "malloc", failing_malloc)
    with pytest.raises(MemoryError, match="synthetic halfdot"):
        leaf._screen_pair_silu_interleaved(**h.args, halfdot=True)
    assert h.freed == list(reversed([buf.ptr for buf in h.allocated]))
