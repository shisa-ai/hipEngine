"""Keep generated metadata compatible with the pinned PyPI uploader."""

from pathlib import Path
import tomllib

import pytest


@pytest.mark.parametrize("target", ("wheel", "sdist"))
def test_release_metadata_version_matches_uploader_support(target: str) -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert project["tool"]["hatch"]["build"]["targets"][target]["core-metadata-version"] == "2.4"
