"""The prefill attribution has to classify every dense route this model selects.

``scripts/qwen4exp_comparator_role_map.map_hipengine`` buckets hipEngine's
prefill kernels into the shared family vocabulary, and the 2026-09-17 family
table (``benchmarks/results/2026-09-17-qwen4exp-per-role-cost/``) is read from
it. Its dense rules key on *device kernel symbols*, because that is what
``rocprofv3`` records -- the registry's dispatch labels never reach the trace.

The wide-row route was added to the kernel tree, the registry, the production
plan, and the launch census, but not to this mapper: every kernel in
``gguf_q8_0_dense_wide.hip`` was classified ``other``. A post-promotion
attribution would therefore have billed the route the promotion is *about* to
``other``. ``qwen4exp_shared_family_comparison.py`` runs ``--strict`` with
``--unmapped-floor-ms 1``, so that run fails rather than mis-reports -- but it
cannot complete, and the owed HEAD attribution is exactly that run.

These tests pin both directions from the kernel sources: every non-routed
``__global__`` kernel in the dense Q8_0 linear sources is a dense projection,
every routed (``selected``) kernel in the same files is not, and every dtype
conversion in them is ``elementwise_norm`` -- the bucket the comparator's
``mmb_cvt_*`` rule uses, so a converter that drifted back to ``other`` would
show up as a smaller elementwise bucket rather than as a failure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.qwen4exp_comparator_role_map import (
    DENSE_PROJECTION_STEMS,
    map_hipengine,
)
from scripts.qwen4exp_shared_family_comparison import load_hipengine

REPO_ROOT = Path(__file__).resolve().parents[1]
QUANT_SOURCE_DIR = REPO_ROOT / "hipengine/kernels/hip_gfx1100/quant"

# The Q8_0 sources whose kernels are dense (non-routed) linears. The routed
# kernels that share them carry ``selected`` in the name.
DENSE_Q8_0_SOURCES = (
    "gguf_q8_0_dense_wide.hip",
    "gguf_q8_0_prefill.hip",
    "gguf_q8_0_t16_prefill.hip",
    "gguf_q8_0_pack8_gemv.hip",
)

# ``__global__`` with the specifiers in any order, and the name on the next line.
_GLOBAL_KERNEL = re.compile(
    r"__global__\s*(?:(?:void|__launch_bounds__\s*\([^)]*\))\s*)*([A-Za-z_]\w*)\s*\("
)

# One representative per declared stem, in the spelling a trace carries.
STEM_EXAMPLES = {
    "dense_gemv": "dense_gemv_f32_bf16w_f32_out_kernel",
    "gguf_k_prefill_out_coltile_rowbatch":
        "void (anonymous namespace)::gguf_k_prefill_out_coltile_rowbatch_kernel"
        "<float, float, 8, 8, 4, true>(float const*, unsigned char const*, float*,"
        " int, int, int)",
    "gguf_k_pack8_prefill_out": "gguf_k_pack8_prefill_out_kernel<float, float, 8>",
    "gguf_q8_0_pack8": "gguf_q8_0_pack8_gemv_kernel<unsigned short>",
    "gguf_q8_0_rowvec8_dual_split_gemv":
        "gguf_q8_0_rowvec8_dual_split_gemv_kernel<unsigned short>",
    "dense_wide_kernel": "q8_0_dense_wide_kernel<128, 256, 64, 64>",
    "gguf_q8_0_prefill_wmma": "gguf_q8_0_prefill_wmma_kernel",
    "gguf_q8_0_prefill_dual_wmma": "gguf_q8_0_prefill_dual_wmma_kernel",
    "gguf_q8_0_t16_prefill_wmma": "gguf_q8_0_t16_prefill_wmma_kernel",
    "gguf_q8_0_t16_dual_prefill_wmma": "gguf_q8_0_t16_dual_prefill_wmma_kernel",
}

GRID = ("1", "1", "1")

# Dtype conversions share these sources but are neither dense linears nor routed
# projections. They are asserted positively rather than skipped.
CONVERSION_TOKENS = ("f32_to_f16", "f16_to_f32", "f32_to_bf16", "bf16_to_f32")

CANDIDATE_ARTIFACT = (
    REPO_ROOT
    / "benchmarks/results/2026-09-16-dense-wide-q8-prefill-candidate/artifact.json"
)


def _global_kernel_names(path: Path) -> list[str]:
    return sorted({m.group(1) for m in _GLOBAL_KERNEL.finditer(path.read_text())})


def test_declared_dense_stems_cover_the_route_family() -> None:
    """The stem table is the mapper's dense-route coverage, so it is explicit."""

    assert set(STEM_EXAMPLES) == set(DENSE_PROJECTION_STEMS)
    for stem, symbol in STEM_EXAMPLES.items():
        assert map_hipengine(symbol, GRID) == "dense_projection", stem

    # The pack8 stem covers two kernels, so pin the second one too.
    assert map_hipengine(
        "gguf_q8_0_pack8_dual_gate_up_gemv_kernel<unsigned short>", GRID
    ) == "dense_projection"


def test_promoted_wide_route_is_a_dense_projection() -> None:
    """The kernel the 2026-09-17 promotion made the default, as a trace sees it."""

    recorded = json.loads(CANDIDATE_ARTIFACT.read_text())["candidate"]["kernel"]
    assert recorded.startswith("q8_0_dense_wide_kernel")
    assert map_hipengine(recorded, GRID) == "dense_projection"

    # The tail instantiation, behind the anonymous-namespace prefix that a
    # profiled trace carries.
    assert map_hipengine(
        "void (anonymous namespace)::q8_0_dense_wide_kernel<128, 256, 64, 64, true>"
        "(float const*, unsigned char const*, float*, int, int, int)",
        GRID,
    ) == "dense_projection"


@pytest.mark.parametrize("filename", DENSE_Q8_0_SOURCES)
def test_dense_q8_0_sources_are_classified(filename: str) -> None:
    """Every kernel in these files is dense, routed-not-dense, or a conversion."""

    names = _global_kernel_names(QUANT_SOURCE_DIR / filename)
    assert names, f"{filename}: no __global__ kernels found"

    conversions = 0
    for name in names:
        family = map_hipengine(name, GRID)
        if any(tok in name for tok in CONVERSION_TOKENS):
            conversions += 1
            assert family == "elementwise_norm", (
                f"{filename}: {name} converts a dtype but is classified {family}"
            )
        elif "selected" in name:
            assert family != "dense_projection", (
                f"{filename}: {name} is routed but classified {family}"
            )
        else:
            assert family == "dense_projection", (
                f"{filename}: {name} is a dense Q8_0 linear but classified {family}"
            )

    if filename == "gguf_q8_0_dense_wide.hip":
        # The F16-activation route's converter. Pinned so that a rename or a
        # dropped kernel cannot silently remove the conversion coverage.
        assert conversions == 1, f"{filename}: expected 1 conversion, got {conversions}"


def test_routed_kernels_are_not_dense_projections() -> None:
    """No dense stem is broad enough to swallow a routed or foreign kernel."""

    for symbol in (
        "q8_0_selected_grouped_wmma_prefill_bf16_bf16_kernel",
        "q8_0_selected_sparse_repair_kernel",
        "gguf_k_selected_wmma_prefill_compact_kernel",
        "gguf_iq4_xs_selected_wmma_prefill_compact_kernel",
        "gguf_q4_k_selected_dual_wmma_iu8_prefill_kernel",
        # Dense by name, but the Q4_K t16 shared-expert family: it stays
        # ``other`` rather than being claimed by a Q8_0 stem.
        "gguf_q4_t16_dense_wmma_prefill_bf16_kernel",
    ):
        assert map_hipengine(symbol, GRID) != "dense_projection", symbol


def test_comparison_loader_leaves_no_unmapped_dense_route(tmp_path: Path) -> None:
    """The ``--strict`` gate the owed HEAD attribution runs under."""

    wide = json.loads(CANDIDATE_ARTIFACT.read_text())["candidate"]["kernel"]
    payload = {
        "label": "synthetic-head-route-set",
        "window_ms": 20000.0,
        "attributed_ms": 19900.0,
        "attributed_time_pct": 99.5,
        "exact_role_kernels": [
            {"role": "linear:attn_qkv", "kernel": wide, "api": "hipLaunchKernel",
             "ms": 5400.0, "rows": 12},
            {"role": "linear:ssm_out", "api": "hipLaunchKernel", "rows": 12,
             "kernel": STEM_EXAMPLES["gguf_k_prefill_out_coltile_rowbatch"],
             "ms": 4200.0},
            {"role": "moe:expert_gate", "api": "hipLaunchKernel", "rows": 12,
             "kernel": "void (anonymous namespace)::"
                       "q4_k_selected_dual_wmma_iu8_risk_prefill_kernel<true>"
                       "(float const*, unsigned char const*, float*, int, int, int)",
             "ms": 3414.0},
        ],
    }
    path = tmp_path / "role-analysis.json"
    path.write_text(json.dumps(payload))

    result = load_hipengine(path, unmapped_floor_ms=1.0)

    assert result["unmapped_over_floor"] == []
    assert result["by_family_ms"]["dense_projection"] == 9600.0
    assert result["by_family_ms"]["expert_gate_up"] == 3414.0
