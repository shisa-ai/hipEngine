from __future__ import annotations

from scripts.gguf_continuous_owner_profile_child import build_parser, marker_name


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
