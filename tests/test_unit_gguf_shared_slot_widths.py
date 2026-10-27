from __future__ import annotations

import pytest

from hipengine.generation import qwen35_gguf


@pytest.mark.parametrize("legacy_value", ["1", "1,2,4,8", "1 3 2", "1 9"])
@pytest.mark.parametrize("use_capability", [False, True])
def test_shared_slot_widths_ignore_retired_environment(
    monkeypatch, legacy_value: str, use_capability: bool
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", legacy_value)
    monkeypatch.setattr(
        qwen35_gguf,
        "backend_package_capability",
        lambda backend, name, default: (1, 2),
    )

    widths = qwen35_gguf._gguf_ar_physical_widths(
        "hip_gfx1100", use_capability=use_capability
    )

    assert widths == ((1, 2) if use_capability else tuple(range(1, 9)))


@pytest.mark.parametrize("widths", [(), (2,), (1, 3, 2), (1, 9)])
def test_shared_slot_widths_validate_backend_declaration(monkeypatch, widths) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", raising=False)
    monkeypatch.setattr(
        qwen35_gguf,
        "backend_package_capability",
        lambda backend, name, default: widths,
    )

    with pytest.raises(RuntimeError, match="sorted registered AR widths"):
        qwen35_gguf._gguf_ar_physical_widths("hip_gfx1100", use_capability=True)
