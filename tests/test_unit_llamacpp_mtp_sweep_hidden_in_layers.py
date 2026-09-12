from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from scripts.llamacpp_mtp_run_hidden_in_capture import sha256_bytes
from scripts.llamacpp_mtp_sweep_hidden_in_layers import (
    build_layer_sweep_artifact,
    conclude,
    parse_layers,
    rank_layers,
)


def test_parse_layers_supports_ranges_and_singletons() -> None:
    assert parse_layers("0-3,7,9-8") == [0, 1, 2, 3, 7, 9, 8]


def test_layer_sweep_finds_exact_matching_layer(tmp_path: Path) -> None:
    reference = [1.0, 2.0, 3.0, 4.0]
    compile_artifact = _write_compile_artifact(
        tmp_path,
        _write_layer_fake_harness(tmp_path, reference=reference, matching_layer=2),
    )
    reference_path = _write_reference_arrays(tmp_path, reference)
    expected = sha256_bytes(struct.pack("<4f", *reference))

    artifact = build_layer_sweep_artifact(
        compile_artifact_path=compile_artifact,
        model_path=tmp_path / "model.gguf",
        prompt_tokens="1,2,3,4",
        layers=[0, 1, 2, 3],
        position=3,
        expected_sha256=expected,
        reference_arrays_path=reference_path,
        reference_key="hidden_in_f32",
        output_dir=tmp_path / "sweep",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
    )

    assert artifact["status"] == "matched"
    assert artifact["conclusion"] == "found_exact_layer_match"
    assert artifact["ranking"]["best_selected"]["layer"] == 2
    assert artifact["ranking"]["exact_selected_matches"][0]["layer"] == 2
    assert "switch_llamacpp_oracle_layer" in artifact["next_action"]
    json.dumps(artifact)


def test_layer_sweep_reports_target_layer_best_but_mismatched(tmp_path: Path) -> None:
    reference = [1.0, 2.0, 3.0, 4.0]
    compile_artifact = _write_compile_artifact(
        tmp_path,
        _write_layer_fake_harness(tmp_path, reference=reference, matching_layer=None),
    )
    reference_path = _write_reference_arrays(tmp_path, reference)

    artifact = build_layer_sweep_artifact(
        compile_artifact_path=compile_artifact,
        model_path=tmp_path / "model.gguf",
        prompt_tokens="1,2,3,4",
        layers=[2, 3, 4],
        position=3,
        expected_sha256="0" * 64,
        reference_arrays_path=reference_path,
        reference_key="hidden_in_f32",
        output_dir=tmp_path / "sweep",
        timeout_seconds=30,
        env={"PATH": _prepend_path(tmp_path)},
    )

    assert artifact["status"] == "mismatched"
    assert artifact["ranking"]["best_selected"]["layer"] == 3
    assert artifact["conclusion"] == "target_layer_best_but_mismatched"
    assert artifact["next_action"] == "inspect_tap_placement_or_graph_path_for_layer3_hidden_in"


def test_rank_layers_detects_closest_non_target_layer() -> None:
    results = [
        _result(layer=2, rmse=0.1),
        _result(layer=3, rmse=0.2),
        _result(layer=4, rmse=0.3),
    ]

    ranking = rank_layers(results)

    assert ranking["best_selected"]["layer"] == 2
    assert conclude(ranking=ranking, target_layer=3) == "different_layer_is_closest"


def _result(*, layer: int, rmse: float) -> dict[str, object]:
    return {
        "layer": layer,
        "status": "mismatched",
        "returncode": 0,
        "capture_sha256": f"sha-{layer}",
        "matches_expected": False,
        "selected_row": 3,
        "selected_max_abs_diff": rmse,
        "selected_mean_abs_diff": rmse / 2,
        "selected_rmse": rmse,
        "all_rows_available": True,
        "all_rows_count": 4,
        "best_any_row": {
            "row": 3,
            "sha256": f"sha-{layer}",
            "max_abs_diff": rmse,
            "mean_abs_diff": rmse / 2,
            "rmse": rmse,
        },
        "exact_rows": [],
    }


def _write_reference_arrays(tmp_path: Path, values: list[float]) -> Path:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"arrays": {"hidden_in_f32": values}}))
    return path


def _write_compile_artifact(tmp_path: Path, executable: Path) -> Path:
    artifact = {
        "outputs": {"executable": str(executable)},
        "lib_dir": str(tmp_path / "lib"),
    }
    (tmp_path / "lib").mkdir(exist_ok=True)
    path = tmp_path / "compile.json"
    path.write_text(json.dumps(artifact))
    return path


def _write_layer_fake_harness(
    tmp_path: Path, *, reference: list[float], matching_layer: int | None
) -> Path:
    exe = tmp_path / "fake-layer-harness"
    reference_raw = repr(struct.pack("<4f", *reference))
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, struct, sys\n"
        "args = sys.argv[1:]\n"
        "layer = int(args[args.index('--layer') + 1])\n"
        "prefix = pathlib.Path(args[args.index('--output-prefix') + 1])\n"
        "prefix.parent.mkdir(parents=True, exist_ok=True)\n"
        f"reference = {reference_raw}\n"
        f"matching_layer = {repr(matching_layer)}\n"
        "if matching_layer is not None and layer == matching_layer:\n"
        "    raw = reference\n"
        "else:\n"
        "    offset = abs(layer - 3) + 1\n"
        "    raw = struct.pack('<4f', 1.0 + offset, 2.0, 3.0, 4.0)\n"
        "(prefix.with_suffix('.f32')).write_bytes(raw)\n"
        "(prefix.with_suffix('.all.f32')).write_bytes(raw * 4)\n"
        "(prefix.with_suffix('.json')).write_text(json.dumps({\n"
        "    'kind': 'llamacpp_hidden_in_capture',\n"
        "    'prompt_token_source': 'token_ids',\n"
        "    'prompt_token_count': 4,\n"
        "    'layer': layer,\n"
        "    'position': 3,\n"
        "    'n_embd': 4,\n"
        "    'binary': str(prefix.with_suffix('.f32')),\n"
        "    'all_rows': True,\n"
        "    'all_rows_binary': str(prefix.with_suffix('.all.f32')),\n"
        "}))\n"
        "print('{\"captured_hidden_in\":true}')\n"
    )
    exe.chmod(0o755)
    return exe


def _prepend_path(path: Path) -> str:
    return str(path) + os.pathsep + os.environ.get("PATH", "")
