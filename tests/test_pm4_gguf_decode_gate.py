from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.pm4_gguf_decode_gate import _native_proof_passed, _read_logits, build_parser


def _run_proof(*, transport: str = "pm4", steps: int = 4) -> dict[str, object]:
    return {
        "transport_provenance": {
            "live": {
                "transport": transport,
                "source": "hipengine_in_tree_rocr_pm4",
                "native_fallbacks": 0,
                "launches": steps,
                "launch_attempts": steps,
                "node_count": 700,
                "hsaco_sha256": ["abc"],
                "context": {
                    "submissions": steps,
                    "unretired_submissions": 0,
                    "usable": True,
                },
                "executable": {
                    f"{transport}_submissions": steps,
                    "retired": True,
                },
            },
            "closed": {
                "transport": transport,
                "native_fallbacks": 0,
                "closed": True,
            },
        }
    }


def test_p5_gate_requires_exact_native_submission_and_retirement_proof() -> None:
    proof = _run_proof()
    assert _native_proof_passed(proof, transport="pm4", steps=4) is True

    proof["transport_provenance"]["live"]["native_fallbacks"] = 1  # type: ignore[index]
    assert _native_proof_passed(proof, transport="pm4", steps=4) is False


def test_p5_gate_flattens_single_row_final_logits() -> None:
    session = SimpleNamespace(
        _read_sample=lambda **kwargs: SimpleNamespace(
            logits=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        )
    )

    assert _read_logits(session).tolist() == [1.0, 2.0, 3.0]


def test_p5_gate_cli_defaults_to_safe_persistent_pm4_control() -> None:
    args = build_parser().parse_args([])

    assert args.backend == "hip_gfx1100"
    assert args.submission_transport == "pm4"
    assert args.prompt_length == 512
    assert args.steps == 4
