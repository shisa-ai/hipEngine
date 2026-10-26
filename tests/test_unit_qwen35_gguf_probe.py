"""Focused CPU tests for the standalone Qwen3.5 GGUF one-layer probe owner.

Every device interaction is replaced by a recorder so the tests can assert the
probe's launch/copy order, its allocation sizes, and its cleanup on every
failure point without a GPU, model file, or kernel build.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION
from hipengine.runtime import qwen35_gguf_probe as probe_module
from hipengine.runtime.qwen35_gguf_probe import Qwen35GGUFOneLayerProbe
from hipengine.runtime.qwen35_gguf_results import Qwen35GGUFNextTokenProbeResult

HIDDEN = 8
SSM_INNER = 4
VOCAB = 16
RMS_EPS = 1.0e-6
MODEL_PATH = Path("/models/fake/qwen35-probe.gguf")

_ATTN_NORM_PTR = 0xA000
_ATTN_GATE_PTR = 0xA100
_SSM_OUT_PTR = 0xA200


class _Events:
    def __init__(self) -> None:
        self.items: list[tuple] = []

    def record(self, *item: object) -> None:
        self.items.append(item)

    def kinds(self) -> list[str]:
        return [str(item[0]) for item in self.items]


class _FakeRuntime:
    def __init__(self, events: _Events) -> None:
        self.events = events
        self.synchronize_calls = 0

    def device_synchronize(self) -> None:
        self.synchronize_calls += 1
        self.events.record("synchronize")


class _FakeDevice:
    """Record allocations, frees, and copies; optionally fail one allocation."""

    def __init__(self, events: _Events, *, fail_malloc_at: int | None = None) -> None:
        self.events = events
        self.fail_malloc_at = fail_malloc_at
        self.malloc_calls = 0
        self.mallocs: list[DeviceBuffer] = []
        self.frees: list[DeviceBuffer] = []
        self._next_ptr = 0x1000

    def malloc(self, nbytes: int, *, runtime: object) -> DeviceBuffer:
        index = self.malloc_calls
        self.malloc_calls += 1
        if self.fail_malloc_at is not None and index == self.fail_malloc_at:
            raise RuntimeError(f"injected malloc failure at allocation {index}")
        buffer = DeviceBuffer(ptr=self._next_ptr, nbytes=int(nbytes))
        self._next_ptr += 0x100
        self.mallocs.append(buffer)
        self.events.record("malloc", buffer.ptr, buffer.nbytes)
        return buffer

    def free(self, buffer: DeviceBuffer, *, runtime: object) -> None:
        self.frees.append(buffer)
        self.events.record("free", buffer.ptr)

    def copy_host_array_to_device(
        self, buffer: DeviceBuffer, array: object, nbytes: int | None = None, *, runtime: object
    ) -> None:
        count = buffer.nbytes if nbytes is None else int(nbytes)
        self.events.record("copy_h2d", buffer.ptr, int(array.ctypes.data), count)

    def copy_device_to_host(
        self, host_ptr: int, buffer: DeviceBuffer, nbytes: int | None = None, *, runtime: object
    ) -> None:
        count = buffer.nbytes if nbytes is None else int(nbytes)
        self.events.record("copy_d2h", int(host_ptr), buffer.ptr, count)


def _fake_weight(name: str, ptr: int) -> SimpleNamespace:
    allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=ptr))
    return SimpleNamespace(name=name, allocation=lambda: allocation)


class _FakeResidentWeights:
    def __init__(
        self,
        *,
        layer_types: tuple[str, ...] = (LINEAR_ATTENTION,) * 8,
        layer_id: int = 0,
    ) -> None:
        self.config = SimpleNamespace(
            hidden_size=HIDDEN,
            ssm_inner_size=SSM_INNER,
            vocab_size=VOCAB,
            rms_norm_eps=RMS_EPS,
            layer_types=tuple(layer_types),
        )
        self.free_calls: list[object] = []
        self._root = {
            "token_embedding": _fake_weight("token_embedding", 0xB000),
            "lm_head": _fake_weight("lm_head", 0xB100),
        }
        self._layers = {
            layer_id: {
                "attn_norm": _fake_weight("attn_norm", _ATTN_NORM_PTR),
                "attn_gate": _fake_weight("attn_gate", _ATTN_GATE_PTR),
                "ssm_out": _fake_weight("ssm_out", _SSM_OUT_PTR),
            }
        }

    def root(self, slot: str) -> SimpleNamespace:
        return self._root[slot]

    def layer(self, layer_id: int) -> SimpleNamespace:
        weights = self._layers.setdefault(
            layer_id,
            {
                "attn_norm": _fake_weight("attn_norm", _ATTN_NORM_PTR),
                "attn_gate": _fake_weight("attn_gate", _ATTN_GATE_PTR),
                "ssm_out": _fake_weight("ssm_out", _SSM_OUT_PTR),
            },
        )
        return SimpleNamespace(weight=weights.__getitem__)

    def free(self, *, runtime: object = None) -> None:
        self.free_calls.append(runtime)


class _Harness:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        fail_malloc_at: int | None = None,
        layer_types: tuple[str, ...] = (LINEAR_ATTENTION,) * 8,
    ) -> None:
        self.events = _Events()
        self.runtime = _FakeRuntime(self.events)
        self.device = _FakeDevice(self.events, fail_malloc_at=fail_malloc_at)
        self.weights = _FakeResidentWeights(layer_types=layer_types)
        self.materialize_calls: list[tuple] = []
        self.linear_calls: list[tuple] = []
        self.embed_calls: list[tuple] = []
        self.rmsnorm_calls: list[tuple] = []

        def fake_materialize(model_path, *, selected_slots, runtime, selected_call_intents):
            assert selected_call_intents == (), "projection probe must not request full-model expert calls"
            self.materialize_calls.append((model_path, tuple(selected_slots), runtime))
            return self.weights

        def fake_embed(weight, token_ids_ptr, out_ptr, *, rows, hidden_size, vocab_size, runtime):
            call = (
                "embed",
                weight.name,
                int(token_ids_ptr),
                int(out_ptr),
                rows,
                hidden_size,
                vocab_size,
                runtime,
            )
            self.embed_calls.append(call)
            self.events.record(*call[:-1])

        def fake_rmsnorm(x_ptr, weight_ptr, out_ptr, *, rows, hidden_size, eps, runtime, **kwargs):
            call = (
                "rmsnorm",
                int(x_ptr),
                int(weight_ptr),
                int(out_ptr),
                rows,
                hidden_size,
                eps,
                runtime,
            )
            self.rmsnorm_calls.append(call)
            self.events.record(*call[:-1])

        def fake_linear(
            weight, x_ptr, out_ptr, *, rows, in_features, out_features, runtime, **kwargs
        ):
            call = (
                "linear",
                weight.name,
                int(x_ptr),
                int(out_ptr),
                rows,
                in_features,
                out_features,
                kwargs,
                runtime,
            )
            self.linear_calls.append(call)
            self.events.record(
                "linear",
                weight.name,
                int(x_ptr),
                int(out_ptr),
                rows,
                in_features,
                out_features,
                tuple(sorted(kwargs.items())),
            )

        monkeypatch.setattr(probe_module, "materialize_qwen35_gguf_weights", fake_materialize)
        monkeypatch.setattr(probe_module, "get_hip_runtime", lambda: self.runtime)
        monkeypatch.setattr(probe_module, "malloc", self.device.malloc)
        monkeypatch.setattr(probe_module, "free", self.device.free)
        monkeypatch.setattr(
            probe_module, "copy_host_array_to_device", self.device.copy_host_array_to_device
        )
        monkeypatch.setattr(probe_module, "copy_device_to_host", self.device.copy_device_to_host)
        monkeypatch.setattr(probe_module, "launch_gguf_embedding", fake_embed)
        monkeypatch.setattr(probe_module, "gguf_rmsnorm_bf16_f32_weight", fake_rmsnorm)
        monkeypatch.setattr(probe_module, "launch_gguf_linear", fake_linear)

    def probe(self, *, layer_id: int = 0, runtime: object = None) -> Qwen35GGUFOneLayerProbe:
        return Qwen35GGUFOneLayerProbe(
            MODEL_PATH, layer_id=layer_id, runtime=runtime or self.runtime
        )


def _make_probe(
    monkeypatch: pytest.MonkeyPatch, *, layer_id: int = 0
) -> tuple[_Harness, Qwen35GGUFOneLayerProbe]:
    harness = _Harness(monkeypatch)
    return harness, harness.probe(layer_id=layer_id)


def test_probe_module_owns_the_class() -> None:
    assert probe_module.__all__ == ["Qwen35GGUFOneLayerProbe"]
    assert Qwen35GGUFOneLayerProbe.__module__ == "hipengine.runtime.qwen35_gguf_probe"


def test_constructor_requests_probe_slots_for_requested_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch)

    probe = harness.probe(layer_id=3)

    assert len(harness.materialize_calls) == 1
    model_path, selected_slots, runtime = harness.materialize_calls[0]
    assert model_path == MODEL_PATH
    assert selected_slots == (
        "root.token_embedding",
        "root.lm_head",
        "layers.3.attn_norm",
        "layers.3.attn_gate",
        "layers.3.ssm_out",
    )
    assert runtime is harness.runtime
    assert probe.weights is harness.weights
    assert probe.hidden_size == HIDDEN
    assert probe.ssm_inner_size == SSM_INNER
    assert probe.vocab_size == VOCAB


def test_constructor_resolves_default_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)
    default_runtime = _FakeRuntime(harness.events)
    monkeypatch.setattr(probe_module, "get_hip_runtime", lambda: default_runtime)

    probe = Qwen35GGUFOneLayerProbe(MODEL_PATH)

    assert probe.runtime is default_runtime
    assert harness.materialize_calls[0][2] is default_runtime


def test_constructor_rolls_back_weights_for_non_linear_attention_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(monkeypatch, layer_types=(FULL_ATTENTION,))

    with pytest.raises(ValueError, match="is not a linear_attention layer"):
        harness.probe()

    assert harness.weights.free_calls == [harness.runtime]


def test_constructor_keeps_weights_when_layer_is_linear_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, probe = _make_probe(monkeypatch)

    assert harness.weights.free_calls == []
    assert probe.weights is harness.weights


def test_run_token_launch_and_copy_order(monkeypatch: pytest.MonkeyPatch) -> None:
    harness, probe = _make_probe(monkeypatch)

    out_bits = probe.run_token(760)

    token_nbytes = np.dtype(np.int64).itemsize
    hidden_nbytes = 2 * HIDDEN
    gate_nbytes = 2 * SSM_INNER
    token_buf, hidden_buf, norm_buf, gate_buf, out_buf = harness.device.mallocs
    assert [buffer.nbytes for buffer in harness.device.mallocs] == [
        token_nbytes,
        hidden_nbytes,
        hidden_nbytes,
        gate_nbytes,
        hidden_nbytes,
    ]
    events = list(harness.events.items)
    # The token-id staging array is a call-local temporary, so its host pointer is
    # pinned only by the copy itself; every other field is asserted literally.
    assert events[5][0] == "copy_h2d"
    assert events[5][1] == token_buf.ptr
    assert events[5][3] == token_nbytes
    events[5] = ("copy_h2d", token_buf.ptr, "<token-ids>", token_nbytes)
    assert events == [
        ("malloc", token_buf.ptr, token_nbytes),
        ("malloc", hidden_buf.ptr, hidden_nbytes),
        ("malloc", norm_buf.ptr, hidden_nbytes),
        ("malloc", gate_buf.ptr, gate_nbytes),
        ("malloc", out_buf.ptr, hidden_nbytes),
        ("copy_h2d", token_buf.ptr, "<token-ids>", token_nbytes),
        ("embed", "token_embedding", token_buf.ptr, hidden_buf.ptr, 1, HIDDEN, VOCAB),
        ("rmsnorm", hidden_buf.ptr, _ATTN_NORM_PTR, norm_buf.ptr, 1, HIDDEN, RMS_EPS),
        ("linear", "attn_gate", norm_buf.ptr, gate_buf.ptr, 1, HIDDEN, SSM_INNER, ()),
        ("linear", "ssm_out", gate_buf.ptr, out_buf.ptr, 1, SSM_INNER, HIDDEN, ()),
        ("synchronize",),
        ("copy_d2h", out_bits.ctypes.data, out_buf.ptr, hidden_nbytes),
        ("free", out_buf.ptr),
        ("free", gate_buf.ptr),
        ("free", norm_buf.ptr),
        ("free", hidden_buf.ptr),
        ("free", token_buf.ptr),
    ]
    assert out_bits.shape == (1, HIDDEN)
    assert out_bits.dtype == np.uint16
    assert harness.runtime.synchronize_calls == 1
    assert harness.device.frees == list(reversed(harness.device.mallocs))


def test_run_token_stages_one_int64_token_id(monkeypatch: pytest.MonkeyPatch) -> None:
    harness, probe = _make_probe(monkeypatch)
    seen: list[int] = []

    def capture_host_token(
        buffer: DeviceBuffer, array: object, nbytes: int | None = None, *, runtime: object
    ) -> None:
        seen.append(int(array[0]))
        harness.device.copy_host_array_to_device(buffer, array, nbytes, runtime=runtime)

    monkeypatch.setattr(probe_module, "copy_host_array_to_device", capture_host_token)

    probe.run_token(760)

    assert seen == [760]
    assert harness.device.mallocs[0].nbytes == np.dtype(np.int64).itemsize


@pytest.mark.parametrize("fail_at", range(5))
def test_run_token_failure_at_each_allocation_frees_acquired_buffers(
    monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    harness = _Harness(monkeypatch, fail_malloc_at=fail_at)
    probe = harness.probe()

    with pytest.raises(RuntimeError, match=f"injected malloc failure at allocation {fail_at}"):
        probe.run_token(760)

    acquired = harness.device.mallocs
    assert len(acquired) == fail_at
    assert harness.device.frees == list(reversed(acquired))
    assert harness.events.kinds() == ["malloc"] * fail_at + ["free"] * fail_at
    assert harness.embed_calls == []
    assert harness.linear_calls == []
    assert harness.runtime.synchronize_calls == 0


def test_run_token_f32_converts_bf16_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    _, probe = _make_probe(monkeypatch)
    bits = np.asarray(
        [[0x3F80, 0x0000, 0xBF80, 0x4000, 0x0001, 0x0002, 0x0003, 0x0004]], dtype=np.uint16
    )
    monkeypatch.setattr(probe, "run_token", lambda token_id: bits)

    values = probe.run_token_f32(760)

    assert values.dtype == np.float32
    assert values.shape == (1, HIDDEN)
    assert values[0, 0] == 1.0
    assert values[0, 1] == 0.0
    assert values[0, 2] == -1.0
    assert values[0, 3] == 2.0


def test_logits_from_hidden_bits_launch_and_copy_order(monkeypatch: pytest.MonkeyPatch) -> None:
    harness, probe = _make_probe(monkeypatch)
    hidden = np.zeros((1, HIDDEN), dtype=np.uint16)

    logits = probe.logits_from_hidden_bits(hidden)

    hidden_nbytes = 2 * HIDDEN
    logits_nbytes = 4 * VOCAB
    hidden_buf, logits_buf = harness.device.mallocs
    assert [buffer.nbytes for buffer in harness.device.mallocs] == [hidden_nbytes, logits_nbytes]
    assert harness.events.items == [
        ("malloc", hidden_buf.ptr, hidden_nbytes),
        ("malloc", logits_buf.ptr, logits_nbytes),
        ("copy_h2d", hidden_buf.ptr, hidden.ctypes.data, hidden_nbytes),
        (
            "linear",
            "lm_head",
            hidden_buf.ptr,
            logits_buf.ptr,
            1,
            HIDDEN,
            VOCAB,
            (("output_dtype", "f32"),),
        ),
        ("synchronize",),
        ("copy_d2h", logits.ctypes.data, logits_buf.ptr, logits_nbytes),
        ("free", logits_buf.ptr),
        ("free", hidden_buf.ptr),
    ]
    assert logits.shape == (1, VOCAB)
    assert logits.dtype == np.float32
    assert harness.runtime.synchronize_calls == 1


def test_logits_from_hidden_bits_accepts_non_contiguous_bf16_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, probe = _make_probe(monkeypatch)
    source = np.zeros((1, 2 * HIDDEN), dtype=np.uint16)[:, ::2]
    assert not source.flags["C_CONTIGUOUS"]

    probe.logits_from_hidden_bits(source)

    assert [buffer.nbytes for buffer in harness.device.mallocs] == [2 * HIDDEN, 4 * VOCAB]
    assert harness.events.items[2][0] == "copy_h2d"
    assert harness.events.items[2][3] == 2 * HIDDEN


@pytest.mark.parametrize("shape", [(HIDDEN,), (2, HIDDEN), (1, HIDDEN + 1), (1, HIDDEN, 1)])
def test_logits_from_hidden_bits_rejects_wrong_shape_before_allocating(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    harness, probe = _make_probe(monkeypatch)

    with pytest.raises(ValueError, match=r"hidden_bits must have shape \(1, 8\)"):
        probe.logits_from_hidden_bits(np.zeros(shape, dtype=np.uint16))

    assert harness.device.malloc_calls == 0
    assert harness.linear_calls == []


@pytest.mark.parametrize("fail_at", range(2))
def test_logits_from_hidden_bits_failure_at_each_allocation_frees_acquired_buffers(
    monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    harness = _Harness(monkeypatch, fail_malloc_at=fail_at)
    probe = harness.probe()

    with pytest.raises(RuntimeError, match=f"injected malloc failure at allocation {fail_at}"):
        probe.logits_from_hidden_bits(np.zeros((1, HIDDEN), dtype=np.uint16))

    acquired = harness.device.mallocs
    assert len(acquired) == fail_at
    assert harness.device.frees == list(reversed(acquired))
    assert harness.events.kinds() == ["malloc"] * fail_at + ["free"] * fail_at
    assert harness.linear_calls == []
    assert harness.runtime.synchronize_calls == 0


def test_sample_next_token_returns_argmax_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _, probe = _make_probe(monkeypatch)
    logits = np.zeros((1, VOCAB), dtype=np.float32)
    logits[0, 5] = 2.5
    logits[0, 9] = 1.5
    seen: list[int] = []

    def fake_run_token(token_id: int) -> np.ndarray:
        seen.append(token_id)
        return np.zeros((1, HIDDEN), dtype=np.uint16)

    monkeypatch.setattr(probe, "run_token", fake_run_token)
    monkeypatch.setattr(probe, "logits_from_hidden_bits", lambda hidden_bits: logits)

    result = probe.sample_next_token(760)

    assert seen == [760]
    assert type(result) is Qwen35GGUFNextTokenProbeResult
    assert Qwen35GGUFNextTokenProbeResult.__module__ == "hipengine.runtime.qwen35_gguf_results"
    assert result.token_id == 5
    assert result.logit == pytest.approx(2.5)
    assert result.logits is logits


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_sample_next_token_rejects_non_finite_logits(
    monkeypatch: pytest.MonkeyPatch, bad_value: float
) -> None:
    _, probe = _make_probe(monkeypatch)
    logits = np.zeros((1, VOCAB), dtype=np.float32)
    logits[0, 3] = bad_value
    monkeypatch.setattr(probe, "run_token", lambda token_id: np.zeros((1, HIDDEN), dtype=np.uint16))
    monkeypatch.setattr(probe, "logits_from_hidden_bits", lambda hidden_bits: logits)

    with pytest.raises(FloatingPointError, match="logits contain NaN or Inf"):
        probe.sample_next_token(760)


def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    harness, probe = _make_probe(monkeypatch)

    probe.close()
    probe.close()

    assert harness.weights.free_calls == [harness.runtime]
    assert probe.weights is None


def test_context_manager_closes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _Harness(monkeypatch)

    with harness.probe() as probe:
        assert probe.weights is harness.weights
    assert probe.weights is None

    probe.close()
    assert harness.weights.free_calls == [harness.runtime]


def test_close_without_materialized_weights_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    harness, probe = _make_probe(monkeypatch)
    probe.weights = None

    probe.close()

    assert harness.weights.free_calls == []


def test_probe_module_imports_the_neutral_result_type_not_the_runner() -> None:
    source = Path(probe_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.append(module)
            imported.extend(f"{module}.{alias.name}" for alias in node.names)

    assert "hipengine.runtime.qwen35_gguf_results" in imported
    assert "hipengine.runtime.qwen35_gguf_results.Qwen35GGUFNextTokenProbeResult" in imported
    runner_imports = [
        name for name in imported if name.startswith("hipengine.runtime.qwen35_gguf_runner")
    ]
    assert runner_imports == []
