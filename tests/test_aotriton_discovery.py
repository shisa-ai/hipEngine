from __future__ import annotations

import pytest

from hipengine.core.dtype import DType
from hipengine.kernels.hip_gfx1100.attention.aotriton import aotriton_runtime_tree, aotriton_source_tree
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import (
    AOTRITON_DTYPE_FP32,
    aotriton_attn_fwd_compact_varlen,
    aotriton_dtype,
    plan_aotriton_wrap_build,
    tensor4,
)
from hipengine.kernels.registry import resolve


def test_aotriton_source_tree_uses_env_root(monkeypatch, tmp_path) -> None:
    root = tmp_path / "aotriton"
    header = root / "include" / "aotriton" / "flash.h"
    header.parent.mkdir(parents=True)
    header.write_text("// fake flash header\n")
    monkeypatch.setenv("HIPENGINE_AOTRITON_SOURCE_ROOT", str(root))

    tree = aotriton_source_tree()

    assert tree.root == root.resolve()
    assert tree.include_dir == (root / "include").resolve()
    assert tree.flash_header == header.resolve()


def test_aotriton_runtime_tree_uses_env_root(monkeypatch, tmp_path) -> None:
    root = tmp_path / "aotriton"
    header = root / "include" / "aotriton" / "flash.h"
    images = root / "lib" / "aotriton.images"
    lib = root / "lib" / "libaotriton_v2.so.0.8.0"
    header.parent.mkdir(parents=True)
    images.mkdir(parents=True)
    header.write_text("// fake flash header\n")
    lib.write_bytes(b"fake so\n")
    monkeypatch.setenv("HIPENGINE_AOTRITON_RUNTIME_ROOT", str(root))

    tree = aotriton_runtime_tree()

    assert tree.root == root.resolve()
    assert tree.include_dir == (root / "include").resolve()
    assert tree.flash_header == header.resolve()
    assert tree.library == lib.resolve()
    assert tree.images_dir == images.resolve()


def test_aotriton_runtime_tree_requires_built_library(monkeypatch, tmp_path) -> None:
    root = tmp_path / "aotriton"
    (root / "lib").mkdir(parents=True)
    monkeypatch.setenv("HIPENGINE_AOTRITON_RUNTIME_ROOT", str(root))

    with pytest.raises(FileNotFoundError, match="runtime library"):
        aotriton_runtime_tree()


def test_aotriton_wrap_build_plan_links_runtime(monkeypatch, tmp_path) -> None:
    root = tmp_path / "aotriton"
    header = root / "include" / "aotriton" / "flash.h"
    images = root / "lib" / "aotriton.images"
    lib = root / "lib" / "libaotriton_v2.so"
    header.parent.mkdir(parents=True)
    images.mkdir(parents=True)
    header.write_text("// fake flash header\n")
    lib.write_bytes(b"fake so\n")
    monkeypatch.setenv("HIPENGINE_AOTRITON_RUNTIME_ROOT", str(root))

    artifact = plan_aotriton_wrap_build(cache_root=tmp_path / "build", compiler_version="hipcc-test")

    assert artifact.family == "aotriton_wrap"
    assert f"-I{root / 'include'}" in artifact.flags
    assert f"-L{root / 'lib'}" in artifact.flags
    assert "-laotriton_v2" in artifact.flags
    assert f"-Wl,-rpath,{root / 'lib'}" in artifact.flags
    assert artifact.output_path.name == "hipengine_aotriton_wrap.so"


def test_aotriton_tensor_descriptor_and_dtype_mapping() -> None:
    desc = tensor4(
        0x1234,
        sizes=(1, 16, 512, 128),
        strides=(16 * 512 * 128, 128, 16 * 128, 1),
        dtype=DType.FP32,
    )

    assert desc.data == 0x1234
    assert list(desc.sizes) == [1, 16, 512, 128]
    assert list(desc.strides) == [1048576, 128, 2048, 1]
    assert desc.dtype == AOTRITON_DTYPE_FP32
    assert aotriton_dtype("fp32") == AOTRITON_DTYPE_FP32


def test_aotriton_prefill_variant_is_registered() -> None:
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_prefill",
            quant="w4_paro",
            variant="aotriton_attn_fwd",
        )
        is aotriton_attn_fwd_compact_varlen
    )
