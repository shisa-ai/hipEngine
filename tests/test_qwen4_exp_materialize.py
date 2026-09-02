from __future__ import annotations

import ctypes
from math import prod
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.qwen4_exp_gguf import (
    Qwen4ExpGGUFTensorRef,
    build_qwen4_exp_gguf_tensor_map,
)
from hipengine.loading.qwen4_exp_materialize import (
    LAYOUT_RAW_GGUF,
    Qwen4ExpGGUFWeightSpec,
    materialize_qwen4_exp_raw_weight,
    materialize_qwen4_exp_weights,
    plan_qwen4_exp_residency,
)
from hipengine.quant.gguf import GGMLQuantizationType
from tests.test_qwen4_exp_gguf_mapping import _infos


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


class _Allocation:
    def __init__(self, name: str, closed: list[str]) -> None:
        self.name = name
        self.closed = closed

    def free(self, *, runtime=None) -> None:
        del runtime
        self.closed.append(self.name)


class _Reader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def tensor_data(self, name: str) -> np.ndarray:
        self.calls.append(name)
        return np.zeros((1, 90), dtype=np.uint8)


def test_qwen4_exp_materialize_owns_every_hot_weight_once_and_closes_reverse() -> None:
    model_map = build_qwen4_exp_gguf_tensor_map(_infos())
    plan = plan_qwen4_exp_residency(model_map, staging_token_capacity=1)
    readers = (_Reader(), _Reader())
    loaded: list[str] = []
    closed: list[str] = []

    def loader(spec, reader, *, runtime=None):
        del reader, runtime
        loaded.append(spec.slot_path)
        return _Allocation(spec.slot_path, closed)

    resident = materialize_qwen4_exp_weights(
        readers,
        plan=plan,
        device_loader=loader,
        pin_ple_staging=False,
    )

    assert len(loaded) == 1_223
    assert len(set(loaded)) == 1_223
    assert "ple.table" not in loaded
    assert readers[0].calls == []
    assert readers[1].calls == ["per_layer_token_embd.weight"]
    assert len(resident.device_weights) == 1_223
    assert resident.weight("root.token_embedding").allocation().name == (
        "root.token_embedding"
    )
    assert resident.weight("root.token_embedding").spec.quant_key == "f32"
    assert resident.ple_table.tensor.name == "per_layer_token_embd.weight"
    assert resident.ple_staging.row_capacity == 16

    resident.close()
    resident.close()
    assert closed == list(reversed(loaded))
    assert resident.closed


def test_qwen4_exp_materialize_drop_behind_follows_device_copy_and_excludes_ple(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    model_map = build_qwen4_exp_gguf_tensor_map(_infos())
    plan = plan_qwen4_exp_residency(model_map, staging_token_capacity=1)
    paths = (tmp_path / "part0.gguf", tmp_path / "part1.gguf")
    for path in paths:
        path.write_bytes(b"x")
    readers = (_Reader(), _Reader())
    readers[0].path, readers[1].path = paths
    events: list[tuple[str, str | int]] = []
    closed: list[str] = []

    def loader(spec, reader, *, runtime=None):
        del reader, runtime
        events.append(("load", spec.source.name))
        return _Allocation(spec.slot_path, closed)

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_LOAD_DROP_BEHIND", "1")
    monkeypatch.setattr(
        "hipengine.loading.qwen4_exp_materialize.os.posix_fadvise",
        lambda fd, offset, length, advice: events.append(("advice", advice)),
    )
    resident = materialize_qwen4_exp_weights(
        readers, plan=plan, device_loader=loader, pin_ple_staging=False
    )

    assert len([event for event in events if event[0] == "advice"]) == 1_223
    assert all(
        events[index][0:1] == ("load",) and events[index + 1] == ("advice", os.POSIX_FADV_DONTNEED)
        for index in range(0, len(events), 2)
    )
    assert all(event != ("load", "per_layer_token_embd.weight") for event in events)
    assert readers[1].calls == ["per_layer_token_embd.weight"]
    resident.close()


def test_qwen4_exp_materialize_frees_partial_owners_after_injected_failure() -> None:
    model_map = build_qwen4_exp_gguf_tensor_map(_infos())
    plan = plan_qwen4_exp_residency(model_map, staging_token_capacity=1)
    readers = (_Reader(), _Reader())
    loaded: list[str] = []
    closed: list[str] = []

    def loader(spec, reader, *, runtime=None):
        del reader, runtime
        if len(loaded) == 3:
            raise RuntimeError("injected materialization failure")
        loaded.append(spec.slot_path)
        return _Allocation(spec.slot_path, closed)

    with pytest.raises(RuntimeError, match="injected"):
        materialize_qwen4_exp_weights(
            readers,
            plan=plan,
            device_loader=loader,
            pin_ple_staging=False,
        )

    assert len(loaded) == 3
    assert closed == list(reversed(loaded))
    assert readers[1].calls == []


@pytest.mark.skipif(
    not _hip_available(),
    reason="HIP runtime is not available",
)
def test_qwen4_exp_raw_weight_loader_preserves_quant_bytes() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    runtime = get_hip_runtime()
    shape = (1, 32)
    tensor = GGUFTensorInfo(
        name="test.weight",
        shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(GGMLQuantizationType.Q8_0),
        ggml_type_name="Q8_0",
        n_elements=prod(shape),
        nbytes=34,
        offset=0,
        data_offset=0,
        byte_shape=(1, 34),
    )
    source_ref = Qwen4ExpGGUFTensorRef(0, Path("test.gguf"), tensor)
    spec = Qwen4ExpGGUFWeightSpec(
        slot_path="root.test",
        source_ref=source_ref,
        quant_key="gguf_q8_0",
        layout=LAYOUT_RAW_GGUF,
        allocation_names=("raw",),
        device_resident=True,
        device_nbytes=34,
    )
    raw = np.arange(34, dtype=np.uint8).reshape(1, 34)
    allocation = materialize_qwen4_exp_raw_weight(
        spec,
        SimpleNamespace(tensor_data=lambda name: raw),
        runtime=runtime,
    )
    try:
        actual = np.empty_like(raw)
        copy_device_to_host(host_array_ptr(actual), allocation.buffer, runtime=runtime)
    finally:
        allocation.free(runtime=runtime)
    np.testing.assert_array_equal(actual, raw)


def test_qwen4_exp_device_weight_implements_model_neutral_dispatch_abi() -> None:
    from hipengine.runtime.gguf_linear import (
        GGUF_ACTIVATION_F32,
        GGUF_OUTPUT_F32,
        resolve_gguf_linear_dispatch,
    )

    model_map = build_qwen4_exp_gguf_tensor_map(_infos())
    plan = plan_qwen4_exp_residency(model_map)
    spec = plan.root_specs["token_embedding"]
    weight = SimpleNamespace(spec=spec, backend="hip_gfx1151")

    dispatch = resolve_gguf_linear_dispatch(
        weight,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
    )
    assert dispatch.key.backend == "hip_gfx1151"
    assert dispatch.key.layer == "dense_gemv"
    assert dispatch.key.quant == "f32"
    assert dispatch.key.variant == "f32_hidden_f32_out"


def test_qwen4_exp_materialize_rejects_reader_part_mismatch() -> None:
    model_map = build_qwen4_exp_gguf_tensor_map(_infos())
    plan = plan_qwen4_exp_residency(model_map)

    with pytest.raises(ValueError, match="reader"):
        materialize_qwen4_exp_weights((_Reader(),), plan=plan, pin_ple_staging=False)
