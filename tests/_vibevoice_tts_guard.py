"""Model-snapshot guard shared by the live VibeVoice-TTS tests.

The pinned checkpoint ``microsoft/VibeVoice-1.5B`` is an external artifact that
is not committed to the repository. On a host that has not downloaded it, the
live TTS modules must skip in fixture setup instead of raising a
fixture-not-found error. A host that explicitly names a checkpoint through
``HIPENGINE_TEST_VIBEVOICE_TTS_MODEL`` has opted into the artifact, so a path
that does not resolve is a configuration error and must fail loudly rather than
skip.

The pinned Hugging Face repo id is resolved through
``hipengine.loading.hf_cache.resolve_model_path``, which uses local cache state
only and never downloads.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PINNED_MODEL_ID = "microsoft/VibeVoice-1.5B"
MODEL_ENV = "HIPENGINE_TEST_VIBEVOICE_TTS_MODEL"


def resolve_tts_snapshot(*, module_level: bool = False) -> Path:
    """Return the local VibeVoice-TTS checkpoint directory.

    Resolution order:

    1. ``HIPENGINE_TEST_VIBEVOICE_TTS_MODEL`` when set. A set-but-missing path
       raises ``FileNotFoundError``; an explicit request is never silently
       skipped.
    2. The pinned Hugging Face repo id in the local cache. When that snapshot is
       absent, skip: an unconfigured host has not opted into the artifact.

    Pass ``module_level=True`` only when calling during module import, where
    pytest requires the flag to skip the whole module.
    """

    configured = os.environ.get(MODEL_ENV)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"{MODEL_ENV}={configured!r} does not name a local VibeVoice-TTS "
                f"checkpoint directory"
            )
        return path

    from hipengine.loading.hf_cache import resolve_model_path

    path = resolve_model_path(PINNED_MODEL_ID)
    if not path.is_dir():
        pytest.skip(
            f"VibeVoice-TTS checkpoint {PINNED_MODEL_ID!r} is not in the local HF cache; "
            f"set {MODEL_ENV} to a local checkpoint directory to run this test",
            allow_module_level=module_level,
        )
    return path


__all__ = ["MODEL_ENV", "PINNED_MODEL_ID", "resolve_tts_snapshot"]
