"""Tests for the AOTriton discovery + wrapper-build plan.

These tests are torch-free.  They construct fake AOTriton cache trees on disk
and verify that the lookup chain documented in ``aotriton.py`` and
``docs/PREFILL.md`` finds them.  The real vendored AOTriton tree is discovered
without dlopening it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.core.dtype import DType
from hipengine.kernels.hip_gfx1100.attention.aotriton import (
    AotritonNotInstalledError,
    aotriton_runtime_tree,
    load_manifest,
)
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import (
    AOTRITON_DTYPE_FP32,
    aotriton_attn_fwd_compact_varlen,
    aotriton_attn_fwd_compact_varlen_gqa_per_q_head,
    aotriton_attn_fwd_v3_compact_varlen,
    aotriton_dtype,
    plan_aotriton_wrap_build,
    tensor4,
)
from hipengine.kernels.registry import resolve


def _make_fake_tree(root: Path, version: str, so_name: str) -> Path:
    """Lay out a minimal AOTriton cache tree mirroring the release tarball."""

    version_dir = root / version
    lib_dir = version_dir / "lib"
    include_dir = version_dir / "include"
    images_dir = lib_dir / "aotriton.images" / "amd-gfx11xx" / "flash" / "attn_fwd"
    images_dir.mkdir(parents=True)
    flash_header = include_dir / "aotriton" / "flash.h"
    flash_header.parent.mkdir(parents=True)
    flash_header.write_text("// fake flash header\n")
    library = lib_dir / so_name
    # Embed the SONAME string literal so _read_soname succeeds.
    library.write_bytes(b"fake elf header\x00" + so_name.encode("ascii") + b"\x00")
    unversioned = lib_dir / "libaotriton_v2.so"
    unversioned.symlink_to(library.name)
    return version_dir


def test_manifest_pins_version_and_soname() -> None:
    pin = load_manifest()
    assert pin.version, "manifest must record an AOTriton version"
    assert pin.so_name.startswith("libaotriton_v2.so")


def test_discovery_finds_vendored_runtime_by_default(monkeypatch) -> None:
    pin = load_manifest()
    monkeypatch.delenv("HIPENGINE_AOTRITON_LIB", raising=False)
    monkeypatch.delenv("HIPENGINE_AOTRITON_HOME", raising=False)

    tree = aotriton_runtime_tree()

    assert tree.manifest == pin
    assert tree.source == "vendored"
    assert tree.library.name == "libaotriton_v2.so.0.11.2"
    assert tree.flash_header.is_file()
    assert (tree.images_dir / "amd-gfx11xx" / "flash" / "attn_fwd").is_dir()


def test_discovery_finds_cache_under_home_root(monkeypatch, tmp_path) -> None:
    pin = load_manifest()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    _make_fake_tree(cache_root, pin.version, pin.so_name)
    monkeypatch.delenv("HIPENGINE_AOTRITON_LIB", raising=False)
    monkeypatch.setenv("HIPENGINE_AOTRITON_HOME", str(cache_root))

    tree = aotriton_runtime_tree()

    assert tree.manifest == pin
    assert tree.library.parent == (cache_root / pin.version / "lib").resolve()
    assert tree.include_dir == (cache_root / pin.version / "include").resolve()
    assert tree.images_dir == (cache_root / pin.version / "lib" / "aotriton.images").resolve()
    assert tree.source == str(cache_root.resolve())


def test_discovery_developer_root_override(monkeypatch, tmp_path) -> None:
    pin = load_manifest()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    _make_fake_tree(cache_root, pin.version, pin.so_name)
    monkeypatch.delenv("HIPENGINE_AOTRITON_LIB", raising=False)
    monkeypatch.delenv("HIPENGINE_AOTRITON_HOME", raising=False)

    tree = aotriton_runtime_tree(cache_root)

    assert tree.library.parent == (cache_root / pin.version / "lib").resolve()


def test_discovery_lib_env_takes_precedence(monkeypatch, tmp_path) -> None:
    pin = load_manifest()
    override_root = tmp_path / "override"
    override_root.mkdir()
    version_dir = _make_fake_tree(override_root, pin.version, pin.so_name)
    library = version_dir / "lib" / pin.so_name
    decoy_root = tmp_path / "decoy"
    decoy_root.mkdir()
    monkeypatch.setenv("HIPENGINE_AOTRITON_LIB", str(library))
    monkeypatch.setenv("HIPENGINE_AOTRITON_HOME", str(decoy_root))

    tree = aotriton_runtime_tree()

    assert tree.library == library.resolve()
    assert tree.source == "HIPENGINE_AOTRITON_LIB"


def test_discovery_lib_env_must_exist(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HIPENGINE_AOTRITON_LIB", str(tmp_path / "does-not-exist.so"))
    monkeypatch.delenv("HIPENGINE_AOTRITON_HOME", raising=False)

    with pytest.raises(FileNotFoundError, match="HIPENGINE_AOTRITON_LIB"):
        aotriton_runtime_tree()


def test_discovery_rejects_unpulled_lfs_pointer(monkeypatch, tmp_path) -> None:
    pin = load_manifest()
    cache_root = tmp_path / "cache"
    version_dir = _make_fake_tree(cache_root, pin.version, pin.so_name)
    library = version_dir / "lib" / pin.so_name
    library.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef\n"
        "size 123\n"
    )
    monkeypatch.delenv("HIPENGINE_AOTRITON_LIB", raising=False)
    monkeypatch.setenv("HIPENGINE_AOTRITON_HOME", str(cache_root))

    with pytest.raises(AotritonNotInstalledError, match="git lfs pull"):
        aotriton_runtime_tree()


def test_discovery_missing_raises_with_install_hint(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HIPENGINE_AOTRITON_LIB", raising=False)
    monkeypatch.setenv("HIPENGINE_AOTRITON_HOME", str(tmp_path / "empty"))

    with pytest.raises(AotritonNotInstalledError, match="scripts/fetch_aotriton.sh"):
        aotriton_runtime_tree()


def test_wrap_build_plan_links_cache_library(monkeypatch, tmp_path) -> None:
    pin = load_manifest()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    version_dir = _make_fake_tree(cache_root, pin.version, pin.so_name)
    monkeypatch.delenv("HIPENGINE_AOTRITON_LIB", raising=False)
    monkeypatch.setenv("HIPENGINE_AOTRITON_HOME", str(cache_root))

    artifact = plan_aotriton_wrap_build(cache_root=tmp_path / "build", compiler_version="hipcc-test")

    assert artifact.family == "aotriton_wrap"
    assert f"-I{version_dir / 'include'}" in artifact.flags
    assert f"-L{version_dir / 'lib'}" in artifact.flags
    assert "-laotriton_v2" in artifact.flags
    assert f"-Wl,-rpath,{version_dir / 'lib'}" in artifact.flags
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
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_prefill",
            quant="w4_paro",
            variant="aotriton_attn_fwd_v3",
        )
        is aotriton_attn_fwd_v3_compact_varlen
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_prefill",
            quant="gguf_qwen35",
            variant="aotriton_attn_fwd_v3",
        )
        is aotriton_attn_fwd_v3_compact_varlen
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="full_attn_prefill",
            quant="w4_paro",
            variant="aotriton_attn_fwd_gqa_per_q_head",
        )
        is aotriton_attn_fwd_compact_varlen_gqa_per_q_head
    )
