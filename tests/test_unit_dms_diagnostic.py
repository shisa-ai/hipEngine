from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kvcache.dms_diagnostic import (
    DiagnosticInjection,
    adapter,
    build_injection,
    concentration,
    discarded_mass,
    frozen_decode_mask,
    load_injection,
    overlap,
    token_hash,
    write_injection,
)
from hipengine.runtime.qwen35_gguf_runner import _normalize_external_dms_decision_mode


def _shape() -> tuple[int, int, int]:
    return 10, 2, 2


def test_runtime_mode_normalization_preserves_ordinary_modes() -> None:
    assert _normalize_external_dms_decision_mode("sidecar") == "sidecar"
    assert _normalize_external_dms_decision_mode("no-evict") == "no_evict"
    assert _normalize_external_dms_decision_mode("diagnostic") == "diagnostic"
    with pytest.raises(ValueError, match="one of"):
        _normalize_external_dms_decision_mode("unknown")


def test_random_adapters_are_seeded_and_distinct() -> None:
    args = dict(tokens=10, layers=2, heads=2, positions=np.arange(10), window=1,
                ratio=2, current_position=9)
    assert np.array_equal(adapter("random", seed=0, **args), adapter("random", seed=0, **args))
    assert not np.array_equal(adapter("random", seed=0, **args), adapter("random", seed=1, **args))


def test_recency_evicts_oldest_and_oracle_evicts_lowest_mass() -> None:
    args = dict(tokens=10, layers=1, heads=1, positions=np.arange(10), window=1,
                ratio=2, current_position=9)
    recency = adapter("recency", **args)
    assert recency[:, 0, 0].tolist() == [True] * 4 + [False] * 6
    mass = np.arange(10, dtype=np.float32)[:, None, None]
    oracle = adapter("oracle_mass", mass=mass, **args)
    assert oracle[:, 0, 0].tolist() == [True] * 4 + [False] * 6


def test_exact_budget_protection_ties_and_noncanonical_fail_closed() -> None:
    args = dict(tokens=10, layers=1, heads=1, positions=np.arange(10), window=1,
                ratio=2, current_position=9)
    mask = adapter("random", seed=0, **args)
    assert int(mask[:, 0, 0].sum()) == 4
    assert not mask[8:, 0, 0].any()
    with pytest.raises(ValueError, match="canonical"):
        adapter("random", seed=0, **{**args, "positions": np.arange(10) + 1})
    with pytest.raises(ValueError, match="finite"):
        adapter("oracle_mass", mass=np.full((10, 1, 1), np.nan), **args)


def test_injection_binding_rejects_prompt_geometry_budget_and_protected_tokens() -> None:
    mask = adapter("random", tokens=10, layers=2, heads=2, positions=np.arange(10),
                    window=1, ratio=2, current_position=9, seed=0)
    inj = DiagnosticInjection(10, token_hash(range(10)), (3, 7), 2, tuple(range(10)),
                              "random", 0, "a" * 64, 1, 2, mask)
    common = dict(prompt=list(range(10)), physical_layer_ids=(3, 7), num_kv_heads=2,
                  window_size=1, target_compression_ratio=2)
    inj.validate(**common, decode_steps=0)
    with pytest.raises(ValueError, match="decode-retention"):
        inj.validate(**common, decode_steps=3)
    for bad in (
        {**common, "prompt": list(range(1, 11))},
        {**common, "physical_layer_ids": (3, 8)},
        {**common, "num_kv_heads": 1},
        {**common, "window_size": 2},
        {**common, "target_compression_ratio": 4},
    ):
        with pytest.raises(ValueError):
            inj.validate(**bad)
    protected = mask.copy()
    protected[9, 0, 0] = True
    with pytest.raises(ValueError, match="protected"):
        DiagnosticInjection(10, token_hash(range(10)), (3, 7), 2, tuple(range(10)),
                            "random", 0, "a" * 64, 1, 2, protected).validate(**common)


def test_freeze_retains_decoded_tokens_and_helpers() -> None:
    mask = np.zeros((10, 1, 1), dtype=bool)
    mask[0] = True
    frozen = frozen_decode_mask(mask, steps=3, layers=1, heads=1)
    assert len(frozen) == 3 and all(np.array_equal(row, mask) for row in frozen)
    assert overlap(mask, mask) == 1.0
    assert discarded_mass(np.ones_like(mask, dtype=float), mask) == pytest.approx(0.1)
    assert concentration(np.ones((2, 3, 4)), axis=(0, 1)).shape == (4,)


def test_builder_supports_scores_and_strict_masks() -> None:
    common = dict(
        prompt=list(range(10)),
        physical_layer_ids=(3, 7),
        num_kv_heads=2,
        selector_kind="scores",
        seed=0,
        source_sha256="a" * 64,
        window_size=1,
        target_compression_ratio=2,
        decode_retention_steps=3,
    )
    scores = np.broadcast_to(
        -np.arange(10, dtype=np.float32)[:, None, None], (10, 2, 2)
    )
    injection = build_injection(**common, scores=scores)
    injection.validate(
        prompt=common["prompt"],
        physical_layer_ids=(3, 7),
        num_kv_heads=2,
        window_size=1,
        target_compression_ratio=2,
        decode_steps=3,
    )
    assert not injection.eviction_mask.flags.writeable
    with pytest.raises(ValueError, match="exactly one"):
        build_injection(**common, scores=scores, eviction_mask=injection.eviction_mask)
    bad_mask = injection.eviction_mask.astype(np.int8)
    bad_mask[0, 0, 0] = 2
    with pytest.raises(ValueError, match="0/1"):
        build_injection(**common, eviction_mask=bad_mask)
    with pytest.raises(ValueError, match="SHA-256"):
        build_injection(**{**common, "source_sha256": "bad"}, scores=scores)


def test_loader_requires_explicit_sealed_mask(tmp_path: Path) -> None:
    mask = adapter(
        "random", tokens=10, layers=2, heads=2,
        positions=np.arange(10), window=1, ratio=2,
        current_position=9, seed=0,
    )
    injection = build_injection(
        prompt=list(range(10)),
        physical_layer_ids=(3, 7),
        num_kv_heads=2,
        selector_kind="random",
        seed=0,
        source_sha256="a" * 64,
        window_size=1,
        target_compression_ratio=2,
        decode_retention_steps=0,
        eviction_mask=mask,
    )
    path = tmp_path / "x.json"
    write_injection(path, injection)
    loaded = load_injection(path)
    loaded.validate(
        prompt=list(range(10)), physical_layer_ids=(3, 7), num_kv_heads=2,
        window_size=1, target_compression_ratio=2, decode_steps=0,
    )
    assert len(loaded.digest) == 64
    with pytest.raises(FileExistsError):
        write_injection(path, injection)

    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["eviction_mask"]
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit"):
        load_injection(malformed)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["eviction_mask_sha256"] = "b" * 64
    malformed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_injection(malformed)
