"""Unit coverage for the live VibeVoice-TTS checkpoint guard.

The live TTS modules resolve an external checkpoint in fixture setup. This
pins the guard's contract on CPU, without ROCm or the checkpoint itself:

- an available default snapshot is returned;
- a missing default snapshot skips instead of raising;
- an explicitly configured but missing path fails loudly;
- an explicitly configured available path is returned and takes precedence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._vibevoice_tts_guard import (
    MODEL_ENV,
    PINNED_MODEL_ID,
    resolve_tts_snapshot,
)


@pytest.fixture(autouse=True)
def _no_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_ENV, raising=False)


def _patch_resolver(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    def _resolve(model_ref: str | Path) -> Path:
        assert str(model_ref) == PINNED_MODEL_ID
        return path

    monkeypatch.setattr("hipengine.loading.hf_cache.resolve_model_path", _resolve)


def test_available_default_snapshot_is_returned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "VibeVoice-1.5B"
    snapshot.mkdir()
    _patch_resolver(monkeypatch, snapshot)

    assert resolve_tts_snapshot() == snapshot


def test_missing_default_snapshot_skips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_resolver(monkeypatch, tmp_path / "absent")

    with pytest.raises(pytest.skip.Exception) as excinfo:
        resolve_tts_snapshot()

    message = str(excinfo.value)
    assert PINNED_MODEL_ID in message
    assert MODEL_ENV in message


def test_explicit_missing_path_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "configured-but-absent"
    monkeypatch.setenv(MODEL_ENV, str(missing))

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_tts_snapshot()

    assert str(missing) in str(excinfo.value)


def test_explicit_available_path_is_returned_and_overrides_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "explicit-checkpoint"
    snapshot.mkdir()
    monkeypatch.setenv(MODEL_ENV, str(snapshot))

    # The default resolver would skip; the explicit path must win.
    def _unreachable(model_ref: str | Path) -> Path:
        raise AssertionError("explicit configuration must not consult the default resolver")

    monkeypatch.setattr("hipengine.loading.hf_cache.resolve_model_path", _unreachable)

    assert resolve_tts_snapshot() == snapshot
