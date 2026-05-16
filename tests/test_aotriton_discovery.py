from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.attention.aotriton import aotriton_runtime_tree, aotriton_source_tree


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
