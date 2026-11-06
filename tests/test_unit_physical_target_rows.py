"""Physical row policy freezes diagnostic inputs at owner construction."""

from dataclasses import FrozenInstanceError

import pytest

from hipengine.generation.physical_target_rows import resolve_physical_target_rows


@pytest.mark.parametrize("profile", ["strict", "production", "batch_invariant"])
def test_rows_plan_snapshots_rollbacks(monkeypatch, profile):
    import hipengine.generation.physical_target_rows as module

    def capability(backend, name, default):
        assert backend == "test_backend"
        return {
            "GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS": (6,),
            "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXACT_ROWTILE_ROWS": (8,),
            "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_ROWTILE_ROWS": (6,),
            "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_MIXED_ROWTILE_CHUNKS": {8: (6, 2)},
            "GGUF_SPECDEC2_EXACT_C7_TARGET_ROWS_POLICY": {
                "rows": (28,), "enabled_env": "TEST_EXACT_ROWS", "enabled_default": True,
            },
        }.get(name, default)

    monkeypatch.setattr(module, "backend_package_capability", capability)
    monkeypatch.delenv("HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS", raising=False)
    monkeypatch.delenv("TEST_EXACT_ROWS", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES", raising=False)
    initial = resolve_physical_target_rows("test_backend", profile)
    assert initial.exact_counts == ((8, 28) if profile == "production" else ())
    assert initial.pad_counts == ((6,) if profile == "production" else ())
    assert initial.mixed_q6_rowtiles is (profile == "production")
    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES", "0")
    assert not resolve_physical_target_rows("test_backend", profile).mixed_q6_rowtiles
    assert initial.mixed_q6_rowtiles is (profile == "production")
    monkeypatch.setenv("TEST_EXACT_ROWS", "0")
    subsequent = resolve_physical_target_rows("test_backend", profile)
    assert subsequent.exact_counts == ((8,) if profile == "production" else ())
    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS", "0")
    assert resolve_physical_target_rows("test_backend", profile).exact_counts == ()
    assert initial.exact_counts == ((8, 28) if profile == "production" else ())
    with pytest.raises(FrozenInstanceError):
        initial.exact_counts = ()
