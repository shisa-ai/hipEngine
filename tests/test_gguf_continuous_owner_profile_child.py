from __future__ import annotations

from pathlib import Path

import pytest

from scripts.gguf_continuous_owner_profile_child import (
    build_parser,
    configure_build_environment,
    marker_name,
)


def test_continuous_owner_marker_is_width_and_transition_scoped() -> None:
    assert marker_name(8, 3) == "hipengine_c2_production_owner_c8_decode_transition_3"


def test_continuous_owner_profile_defaults_to_c8_graph_horizon() -> None:
    args = build_parser().parse_args(
        ["--model", "/models/model.gguf", "--out", "/tmp/result.json"]
    )

    assert args.backend == "hip_gfx1100"
    assert args.quant == "gguf_q4_k_m"
    assert args.concurrency == 8
    assert args.prompt_length == 512
    assert args.decode_tokens == 32
    assert args.marker_index == 3
    assert args.profile is False
    assert args.cache_root is None
    assert args.require_cached_build is False


def test_continuous_owner_cache_only_environment_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = tmp_path / "hipcc-version.txt"
    version.write_text("HIP version: test\n", encoding="utf-8")
    cache = tmp_path / "cache"
    args = build_parser().parse_args(
        [
            "--model",
            "/models/model.gguf",
            "--backend",
            "hip_gfx1151",
            "--compiler-version-file",
            str(version),
            "--cache-root",
            str(cache),
            "--require-cached-build",
            "--out",
            str(tmp_path / "result.json"),
        ]
    )
    monkeypatch.delenv("HIPENGINE_HIP_ARCH", raising=False)

    selected = configure_build_environment(args)

    assert selected["HIPENGINE_HIP_ARCH"] == "gfx1151"
    assert selected["HIPENGINE_COMPILER_VERSION_FILE"] == str(version.resolve())
    assert selected["HIPENGINE_BUILD_CACHE_ROOT"] == str(cache.resolve())
    assert selected["HIPENGINE_REQUIRE_CACHED_BUILD"] == "1"
