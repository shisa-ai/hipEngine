"""Live admission of the real gemma-4 target and its MTP sidecar (punchlist M1).

Skipped where the local artifacts are absent. This is the ground-truth half of
the M1 evidence: the synthetic unit tests pin the contract, this pins reality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gemma4_mtp_gguf import (
    discover_gemma4_mtp_artifacts,
    load_gemma4_mtp_validation,
    require_gemma4_mtp,
)

_MODEL_DIR = Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF")
_TARGET = _MODEL_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"

pytestmark = pytest.mark.skipif(
    not _TARGET.exists(),
    reason="gemma-4 target artifact is local-only",
)


def test_discovery_finds_the_assistant_beside_the_target() -> None:
    found = discover_gemma4_mtp_artifacts(_TARGET)
    assert found, "MTP/*.gguf was not discovered next to the target"
    assert all(p.suffix == ".gguf" for p in found)
    assert all(p.parent.name == "MTP" for p in found)


def test_real_assistant_passes_capability_admission() -> None:
    validation = load_gemma4_mtp_validation(
        _TARGET, discover_gemma4_mtp_artifacts(_TARGET)[0]
    )
    assert validation.passed, validation
    # 3 sliding draft layers read the target's last sliding layer (28), the
    # single global draft layer reads its last global layer (29).
    assert validation.shared_kv_layers == (28, 28, 28, 29)
    assert validation.config.output_width == 2816
    assert validation.config.hidden_size == 1024
    assert validation.config.block_count == 4
    require_gemma4_mtp(_TARGET, discover_gemma4_mtp_artifacts(_TARGET)[0])