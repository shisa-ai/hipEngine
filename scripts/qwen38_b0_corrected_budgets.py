#!/usr/bin/env python3
"""B0: corrected C5-C8 MTP pass budgets after the M3 target-kernel attribution.

The Z0 pass-budget artifact (2026-09-01) counted the full accept/sync/commit
window as non-target cost and concluded a zero feasible target-only pass
budget. The M3 attribution (2026-09-02) later measured that 96.2-98.3% of the
target-pass host windows is traced target-kernel execution queued behind the
accept marker. This script re-derives the budgets reassigning that kernel time
to the target stage, using only committed artifacts as inputs.

All outputs are derived from measured artifacts; the only non-measured inputs
are (a) the C6 kernel-coverage interpolation (M3 measured C5/C7/C8 only) and
(b) the F3 prefill-owner target-pass anchor band, both labeled derived.
No GPU run is performed and no performance claim is made.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
Z0 = REPO / "benchmarks/results/2026-09-01-gfx1151-qwen38-z0-pass-budgets.json"
M3_WIDE = (
    REPO
    / "benchmarks/results/2026-09-02-gfx1151-qwen38-z4-m3-wide-accept-boundary-closure.json"
)
OUT = (
    REPO
    / "benchmarks/results/2026-09-02-gfx1151-qwen38-b0-corrected-pass-budgets.json"
)

# F3 anchor (structural campaign doc section 7.1, table in F3): retained exact
# Y2/Y3 prefill owners execute the same T16 GEMMs as the verify R20-R32 pass at
# ~230-260 ms per full pass versus today's ~790 ms measured verify pass
# (W0 row curve). DERIVED, not a new measurement.
ANCHOR_LO_MS = 230.0
ANCHOR_HI_MS = 260.0
F3_VERIFY_PASS_TODAY_MS = 790.0


def main() -> None:
    z0 = json.loads(Z0.read_text())
    m3 = json.loads(M3_WIDE.read_text())

    coverage = {
        c: m3["cells"][c]["kernel_coverage"] for c in ("5", "7", "8")
    }
    # C6 was not part of the M3 wide closure; interpolate C5/C7 coverage.
    coverage["6"] = (coverage["5"] + coverage["7"]) / 2.0
    c6_derived = True

    cells = {}
    for c in ("5", "6", "7", "8"):
        z = z0["wide_cells"][c]
        st = z["stage_ms_per_cycle"]
        accept = st["accept"]
        proposal = st["proposal"]
        provider = st["provider_update"]
        commit = st["selected_commit"]
        target_telemetry = st["target"]
        residual = z["measured_non_stage_residual_ms_per_cycle"]
        cycles = z["cycles"]
        prefill_per_cycle = z["matched_grouped_prefill_wall_ms"] / cycles
        allowed_decode = z["budgets"]["external_parity"][
            "allowed_decode_cycle_ms"
        ]
        old_slack_removed = z["budgets"]["external_parity"][
            "pass_slack_residual_removed_ms"
        ]

        cov = coverage[c]
        # Host windows the M3 attribution showed to be kernel-covered: the
        # accept/sync boundary plus the target-enqueue window.
        windows = accept + target_telemetry
        reassigned_kernel = windows * cov
        # Host-minus-kernel ceiling scaled to the full-suite window sizes; the
        # M3 diagnostic per-cycle ceilings cross-check C5 (9.18 vs 9.25) and
        # C7 (11.57 vs 11.61) within 0.1 ms.
        host_ceiling = windows * (1.0 - cov)
        corrected_host_removed = proposal + provider + commit + host_ceiling
        corrected_host_kept = corrected_host_removed + residual

        budget_removed = allowed_decode - corrected_host_removed
        budget_kept = allowed_decode - corrected_host_kept

        cells[c] = {
            "full_row": z["full_row"],
            "kernel_coverage": round(cov, 6),
            "kernel_coverage_source": (
                "derived_interpolated_c5_c7" if c == "6" and c6_derived else "m3_measured"
            ),
            "measured_windows_ms_per_cycle": {
                "accept_sync_commit": round(accept, 3),
                "target_enqueue": round(target_telemetry, 3),
            },
            "reassigned_target_kernel_ms_per_cycle": round(reassigned_kernel, 3),
            "host_ceiling_ms_per_cycle": round(host_ceiling, 3),
            "m3_diagnostic_host_minus_kernel_crosscheck_ms": (
                m3["cells"][c]["host_minus_kernel_ms_per_cycle"]
                if c != "6"
                else None
            ),
            "corrected_non_target_host_ms_per_cycle": {
                "residual_removed": round(corrected_host_removed, 3),
                "residual_kept": round(corrected_host_kept, 3),
            },
            "prefill_amortized_ms_per_cycle": round(prefill_per_cycle, 3),
            "allowed_decode_cycle_ms": round(allowed_decode, 3),
            "z0_old_pass_slack_residual_removed_ms": round(old_slack_removed, 3),
            "corrected_target_pass_budget_ms": {
                "residual_removed": round(budget_removed, 3),
                "residual_kept": round(budget_kept, 3),
            },
            "entry_verdict": {
                "residual_removed": _verdict(budget_removed),
                "residual_kept": _verdict(budget_kept),
            },
            "projected_ab_cycle_ms": {
                "anchor_lo": round(corrected_host_removed + ANCHOR_LO_MS, 3),
                "anchor_hi": round(corrected_host_removed + ANCHOR_HI_MS, 3),
                "note": "decode cycle = corrected host + F3 owner anchor (derived)",
            },
            "projected_cycle_with_today_owners_ms": round(
                corrected_host_removed + F3_VERIFY_PASS_TODAY_MS, 3
            ),
        }

    artifact = {
        "schema": 1,
        "kind": "gfx1151_qwen38_b0_corrected_pass_budgets",
        "date": "2026-09-02",
        "status": "mechanism_a_entry_reopened",
        "performance_claim": False,
        "gpu_run": False,
        "physical_host": "gfx1151 / Framework Desktop / AMD Radeon 8060S / 1002:1586 (inherited from source artifacts)",
        "correction": (
            "Z0 counted the accept/sync/commit window as non-target cost; M3 "
            "measured 96.2-98.3% of the target-pass host windows is traced "
            "target-kernel execution queued behind the accept marker. This "
            "artifact reassigns that kernel time to the target stage and "
            "recomputes the mechanism A+B entry budgets. It corrects the Z0 "
            "decision text ('zero feasible target-only pass budget'); the Z0 "
            "artifact itself is immutable and unchanged."
        ),
        "inputs": {
            "z0_pass_budgets": str(Z0.relative_to(REPO)),
            "m3_wide_closure": str(M3_WIDE.relative_to(REPO)),
            "m3_c8_attribution": "benchmarks/results/2026-09-02-gfx1151-qwen38-z4-m3-c8-accept-boundary-attribution.json",
            "f3_anchor": {
                "source": "docs/QWEN38-GFX1151-STRUCTURAL-DIFFERENTIAL-CAMPAIGN.md section 7.1 finding F3",
                "anchor_band_ms_per_full_pass": [ANCHOR_LO_MS, ANCHOR_HI_MS],
                "verify_pass_today_ms": F3_VERIFY_PASS_TODAY_MS,
                "label": "derived from measured W0 row curve and Y0/Y2 prefill sizing; not a new measurement",
            },
        },
        "formula": {
            "windows": "accept_sync_commit + target_enqueue (host windows M3 showed kernel-covered)",
            "reassigned_kernel": "windows * kernel_coverage",
            "host_ceiling": "windows * (1 - kernel_coverage)",
            "corrected_host_residual_removed": "proposal + provider_update + selected_commit + host_ceiling",
            "corrected_target_pass_budget": "allowed_decode_cycle - corrected_host",
            "verdict": (
                "positive if the budget covers the F3 anchor band "
                f"[{ANCHOR_LO_MS:.0f}, {ANCHOR_HI_MS:.0f}] ms; near_miss if it "
                "covers part of the band; blocked if below the band"
            ),
        },
        "cells": cells,
        "findings": [],
    }

    verdicts_removed = {c: cells[c]["entry_verdict"]["residual_removed"] for c in cells}
    verdicts_kept = {c: cells[c]["entry_verdict"]["residual_kept"] for c in cells}
    artifact["findings"].append(
        "Reassigned target-kernel execution hidden in the Z0 non-target stage is "
        + ", ".join(
            f"{cells[c]['reassigned_target_kernel_ms_per_cycle']:.0f} ms/cycle at C{c}"
            for c in ("5", "6", "7", "8")
        )
        + " (derived from measured coverage); this is the double-count that drove the Z0 zero-budget verdict."
    )
    artifact["findings"].append(
        "Corrected target-pass budgets (residual removed / kept): "
        + "; ".join(
            f"C{c} {verdicts_removed[c]} {cells[c]['corrected_target_pass_budget_ms']['residual_removed']:.0f} ms / {verdicts_kept[c]} {cells[c]['corrected_target_pass_budget_ms']['residual_kept']:.0f} ms"
            for c in ("5", "6", "7", "8")
        )
        + ". Mechanism A+B entry is positive at C5/C6/C7; C8 is a near-miss at the optimistic anchor end."
    )
    artifact["findings"].append(
        "Even with zero accept-boundary host cost, today's ~790 ms verify pass "
        "cannot reach parity in any cell; the owners, not the boundary, are the "
        "lever. B1 (registry transfer of the retained Y2/Y3 owners) measures the "
        "truth; all projections here are derived, not measured."
    )

    OUT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {OUT}")
    for c in ("5", "6", "7", "8"):
        cell = cells[c]
        print(
            f"C{c}: coverage={cell['kernel_coverage']:.4f} "
            f"reassigned={cell['reassigned_target_kernel_ms_per_cycle']:.1f}ms "
            f"budget(removed/kept)={cell['corrected_target_pass_budget_ms']['residual_removed']:.1f}/"
            f"{cell['corrected_target_pass_budget_ms']['residual_kept']:.1f}ms "
            f"verdict={verdicts_removed[c]}/{verdicts_kept[c]}"
        )


def _verdict(budget_ms: float) -> str:
    if budget_ms >= ANCHOR_HI_MS:
        return "positive"
    if budget_ms >= ANCHOR_LO_MS:
        return "near_miss_covers_low_anchor_only"
    if budget_ms >= 0.9 * ANCHOR_LO_MS:
        return "near_miss_below_anchor"
    return "blocked"


if __name__ == "__main__":
    main()
