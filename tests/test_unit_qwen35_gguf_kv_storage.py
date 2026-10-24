"""The KV storage owner needs geometry and layout, never an executor or session."""
from __future__ import annotations

import ast
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_kv as storage
from hipengine.runtime import qwen35_gguf_runner as execution


@pytest.fixture
def allocator(monkeypatch):
    live = {}
    sizes = []
    released = []
    state = SimpleNamespace(fail_at=None)

    def malloc(nbytes, *, runtime):
        sizes.append(int(nbytes))
        if len(sizes) == state.fail_at:
            raise MemoryError("injected KV allocation failure")
        buffer = DeviceBuffer(ptr=len(sizes) * 4096, nbytes=int(nbytes))
        live[buffer.ptr] = buffer
        return buffer

    def free(buffer, *, runtime):
        assert live.pop(buffer.ptr) is buffer
        released.append(buffer.ptr)

    monkeypatch.setattr(storage, "malloc", malloc)
    monkeypatch.setattr(storage, "free", free)
    return state, live, sizes, released


def geometry_and_layout():
    # Deliberately no weights, runner, model path or session attributes.
    config = SimpleNamespace(
        layer_types=("linear_attention", "full_attention"),
        head_count_kv=2,
        key_length=64,
    )
    layout = storage.Qwen35GGUFKVChunkLayout(
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        storage_layout="uniform",
        scale_dtype=DType.FP16,
        scale_granularity="per_token_head",
        int8_kv_value_bf16=False,
        layer_storage_dtypes=(None, DType.INT8_PER_TOKEN_HEAD),
        bf16_mirror_layer_indices=(1,),
    )
    return config, layout


def test_geometry_only_allocation_preserves_planes_accounting_and_release(allocator):
    _, live, sizes, released = allocator
    config, layout = geometry_and_layout()
    runtime = object()
    backing = storage._allocate_qwen35_gguf_kv_chunk(
        config, runtime=runtime, start_block_id=7, pages=2, layout=layout
    )
    assert sizes == [65536, 65536, 131072, 131072, 2048, 2048]
    assert backing.page_bytes == storage._qwen35_gguf_kv_page_bytes(config, layout)
    assert sum(size for _, size in storage._qwen35_gguf_kv_plane_specs(config, layout)) == backing.page_bytes
    backing.validate_bound_blocks((7, 8))
    with pytest.raises(ValueError, match="outside its backing chunk"):
        backing.validate_bound_blocks((9,))
    assert backing.page_pointer(1) == backing.full_key_caches[1].ptr + 32768
    storage._free_qwen35_gguf_kv_chunk(backing, runtime=runtime)
    assert released == [6 * 4096, 5 * 4096, 4 * 4096, 3 * 4096, 2 * 4096, 4096]
    assert not live


@pytest.mark.parametrize("fail_at", range(1, 7))
def test_geometry_only_allocation_unwinds_each_partial_plane_set(allocator, fail_at):
    state, live, _, released = allocator
    state.fail_at = fail_at
    config, layout = geometry_and_layout()
    with pytest.raises(MemoryError, match="injected KV allocation failure"):
        storage._allocate_qwen35_gguf_kv_chunk(
            config, runtime=object(), start_block_id=0, pages=2, layout=layout
        )
    assert not live
    assert released == [index * 4096 for index in range(fail_at - 1, 0, -1)]


def test_geometry_mismatch_is_rejected_before_allocation(allocator):
    _, live, sizes, _ = allocator
    config, layout = geometry_and_layout()
    config.layer_types = ("full_attention",)
    with pytest.raises(ValueError, match="model layer count"):
        storage._allocate_qwen35_gguf_kv_chunk(
            config, runtime=object(), start_block_id=0, pages=2, layout=layout
        )
    assert not live
    assert not sizes


def test_executor_keeps_the_closed_owner_guard_at_the_call_boundary():
    config, _ = geometry_and_layout()
    runner = SimpleNamespace(weights=SimpleNamespace(config=config))
    assert execution._kv_allocation_config(runner) is config
    runner.weights = None
    with pytest.raises(RuntimeError, match="GGUF full-stack runner is closed"):
        execution._kv_allocation_config(runner)


def test_existing_type_imports_are_aliases_not_duplicate_classes():
    for name in ("Qwen35GGUFKVChunkLayout", "Qwen35GGUFKVChunkBacking", "VmmPlaneBuffer"):
        assert getattr(execution, name) is getattr(storage, name)
    _, layout = geometry_and_layout()
    assert pickle.loads(pickle.dumps(layout)) == layout
    legacy = pickle.dumps(layout, protocol=0).replace(
        b"hipengine.runtime.qwen35_gguf_kv\n",
        b"hipengine.runtime.qwen35_gguf_runner\n",
    )
    assert pickle.loads(legacy) == layout


def test_storage_has_no_execution_or_kernel_import_dependency():
    # This is an architectural boundary, not a spelling check on implementation.
    tree = ast.parse(Path(storage.__file__).read_text())
    dependencies = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            dependencies.append(node.module or "")
        elif isinstance(node, ast.Import):
            dependencies.extend(alias.name for alias in node.names)
    forbidden = ("hipengine.runtime", "hipengine.generation", "hipengine.kernels")
    assert not [name for name in dependencies if name.startswith(forbidden)]
