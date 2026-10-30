"""Current profiler ownership of promoted head/KV and split-attention controls."""

import sys
from types import SimpleNamespace

import pytest

from hipengine.kernels import hip_gfx1151
from scripts import laguna_long_context_profile as profiler
from hipengine.kernels.backends import backend_package_capability


@pytest.mark.parametrize("head", [None, False, True])
@pytest.mark.parametrize("split", [None, False, True])
def test_profile_keeps_explicit_decode_controls_distinct_from_package_defaults(monkeypatch, tmp_path, head, split):
    flags = []
    if head is not None:
        flags.append("--head-kv-fusion" if head else "--no-head-kv-fusion")
    if split is not None:
        flags.append("--decode-split-attention" if split else "--no-decode-split-attention")
    model = tmp_path / "synthetic.gguf"
    model.touch()
    monkeypatch.setattr(sys, "argv", [
        "laguna_long_context_profile.py", str(model), "--lengths", "512",
        "--context-length", "512", "--allow-dirty", *flags,
    ])
    args = profiler._parse_args()
    monkeypatch.setattr(profiler, "_repo_state", lambda: {"tracked_clean": True})
    monkeypatch.setattr(profiler, "get_hip_runtime", lambda: SimpleNamespace(mem_get_info=lambda: (10, 20)))
    monkeypatch.setattr(profiler, "collect_artifact_provenance", lambda **kw: {})
    monkeypatch.setattr(profiler, "GGUFReader", lambda path: SimpleNamespace(info=object()))
    monkeypatch.setattr(profiler.LagunaGGUFTokenizer, "from_gguf_info", lambda info: object())
    monkeypatch.setattr(profiler, "_load_prompts", lambda *a: [])
    monkeypatch.setattr(profiler, "_profile_token_stream", lambda *a: ([1] * 512, "synthetic"))
    monkeypatch.setattr(profiler, "memory_stats", lambda: {})
    monkeypatch.setattr(profiler, "_compiler_version", lambda path: "synthetic")
    captured = []

    def stop(*a, **kw):
        captured.append(kw)
        raise RuntimeError("profile reached session constructor")

    monkeypatch.setattr(profiler, "LagunaGGUFResidentSession", stop)
    with pytest.raises(RuntimeError, match="profile reached session constructor"):
        profiler.run(args)
    kw, = captured
    assert kw["use_head_kv_fusion"] is head
    assert kw["use_split_attention"] is split
    for key, threshold in (
        ("global_split_min_live", 127), ("swa_split_min_live", 65),
        ("swa_split_tile16_min_live", 257),
    ):
        assert kw[key] == (threshold if split is True else None)
    for key in ("use_swa_split_tile16", "use_split_gate_fusion", "use_swa_split_wave_local"):
        assert kw[key] is (True if split is True else None)


def test_gfx1151_transfer_capabilities_are_already_package_defaults():
    assert backend_package_capability("hip_gfx1151", "LAGUNA_HEAD_KV_FUSION")
    assert backend_package_capability("hip_gfx1151", "LAGUNA_GLOBAL_SPLIT_MIN_LIVE") == 127
    assert backend_package_capability("hip_gfx1151", "LAGUNA_SWA_SPLIT_MIN_LIVE") == 65
    assert backend_package_capability("hip_gfx1151", "LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE") == 257
    assert backend_package_capability("hip_gfx1151", "LAGUNA_SPLIT_GATE_FUSION")
    assert backend_package_capability("hip_gfx1151", "LAGUNA_SWA_SPLIT_WAVE_LOCAL")
