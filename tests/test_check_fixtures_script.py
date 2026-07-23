from __future__ import annotations

from pathlib import Path

from scripts.check_fixtures import _iter_fixture_paths


def test_fixture_directory_discovery_is_non_recursive(tmp_path: Path) -> None:
    top_level = tmp_path / "layer.json"
    top_level.write_text("{}")
    specialized_dir = tmp_path / "moe"
    specialized_dir.mkdir()
    specialized = specialized_dir / "golden.json"
    specialized.write_text("{}")

    assert tuple(_iter_fixture_paths([tmp_path])) == (top_level,)


def test_explicit_nested_fixture_path_is_preserved(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested = nested_dir / "layer.json"
    nested.write_text("{}")

    assert tuple(_iter_fixture_paths([nested])) == (nested,)
