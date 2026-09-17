"""The wide tile family's exports, its wrapper, and the shapes it is admitted on.

``gguf_q8_0_dense_wide.hip`` now carries two activation ABIs: the f32 staging
path that the promoted route uses, and an f16-input path for a caller that hoists
the conversion out of the K loop with ``hipengine_gguf_q8_0_f32_to_f16``. Both
put the same f16 bytes in LDS for the same values, so the second is only
interesting if it is wired to the same symbols the wrapper names -- an export
without a wrapper, or a wrapper without an export, is a route that cannot run.

The admitted-shape table in the measurement harness is pinned to the launch
census of the promoted route, so the shapes it reports on are the shapes the
shipped default actually runs rather than a convenient subset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide

REPO_ROOT = Path(__file__).resolve().parents[1]
HIP_SOURCE = REPO_ROOT / "hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_dense_wide.hip"
CENSUS = (
    REPO_ROOT
    / "benchmarks/results/2026-09-17-q8-dense-default-path-census/census.json"
)

_EXPORT = re.compile(
    r"^\s*HIPENGINE_DENSE_WIDE_EXPORT\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*,",
    re.MULTILINE,
)


def _exports() -> dict[str, str]:
    """Export symbol -> activation type, read from the kernel source.

    The macro's own ``#define`` is skipped: its ``(name, x_t, ...)`` parameter
    list has the same shape as a call site.
    """

    return {
        name: x_type
        for name, x_type in _EXPORT.findall(HIP_SOURCE.read_text())
        if name.startswith("hipengine_")
    }


def test_every_export_has_a_wrapper_variant() -> None:
    exports = _exports()
    assert exports, "no exports found in the kernel source"

    wrapped = set(wide._VARIANTS.values())
    assert set(exports) == wrapped, {
        "exported but unwrapped": sorted(set(exports) - wrapped),
        "wrapped but unexported": sorted(wrapped - set(exports)),
    }


def test_activation_abi_matches_the_variant_name() -> None:
    """``f16in`` names must be the ones taking a pre-converted activation."""

    for symbol, x_type in _exports().items():
        expects_f16 = "_f16in_" in symbol
        assert (x_type == "half_t") is expects_f16, (symbol, x_type)


def test_f16in_variants_are_registered() -> None:
    from hipengine.kernels.registry import KernelKey, is_registered

    for variant in (
        "dense_wide256_f16in_f32_f32_out",
        "dense_wide128x128_f16in_f32_f32_out",
    ):
        key = KernelKey(
            backend="hip_gfx1100", layer="linear", quant="gguf_q8_0", variant=variant
        )
        assert is_registered(key), variant


def test_converter_rejects_a_nonpositive_count_without_a_gpu() -> None:
    """The range check precedes any library or runtime work."""

    for n in (0, -1):
        with pytest.raises(ValueError):
            wide.f32_to_f16(0, 0, n)


def test_admitted_shapes_match_the_promoted_route_census() -> None:
    from scripts.qwen4exp_dense_wide_f16_activation import ADMITTED_SHAPES

    rows = [
        row
        for row in json.loads(CENSUS.read_text())["rows"]
        if "dense_wide256" in row["symbol"]
    ]
    assert rows, "the census has no wide-route rows"

    seen: dict[str, dict[str, object]] = {}
    for row in rows:
        role = row["role"].rsplit(".", 1)[-1]
        entry = seen.setdefault(role, {"shapes": set(), "layers": set()})
        entry["shapes"].add((row["in_features"], row["out_features"]))
        entry["layers"].add(int(row["role"].split("layers.", 1)[1].split(".", 1)[0]))

    declared = {
        role: (layer, in_features, out_features, layers)
        for role, layer, in_features, out_features, layers, _ in ADMITTED_SHAPES
    }
    assert set(declared) == set(seen), {
        "in the census but not the harness": sorted(set(seen) - set(declared)),
        "in the harness but not the census": sorted(set(declared) - set(seen)),
    }
    for role, (layer, in_features, out_features, layers) in declared.items():
        assert seen[role]["shapes"] == {(in_features, out_features)}, role
        assert len(seen[role]["layers"]) == layers, role
        assert layer in seen[role]["layers"], (role, layer)
