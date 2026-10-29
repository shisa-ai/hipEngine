#!/usr/bin/env python3
"""Qualify a Qwen4Exp execution-profile candidate against its strict parent.

The strict and candidate sessions use the same GGUF/BF16-KV configuration and
prompt/teacher-token schedule. Strict runs under the registered strict profile.
T0 exact candidates also use strict plus their post-binder override; T1/T2
candidates use the registered production stack plus their override. The gate records 450
full-vocabulary rows by default (18 prompts x prefill-last plus 24 c1 rows),
three candidate repeats, request-local state fingerprints, and 32-token free
outputs. A cross-route free-output difference requires a separate task review;
it is never silently accepted by this harness.

``--candidate ar_batch`` runs the AR batch adapter instead of a c1 candidate
loop. Prefill stays on the committed c1 path (one request at a time, imported
once per request); every recorded decode row is a genuine N-row
``owner.step`` at ``--batch-width``. The prefill-last row keeps its
``prefill_last``/``prefill_to_c1`` label; decode rows are labeled ``batch_wN``
and the first one ``prefill_to_batch``. The state gate compares batch-slot
fingerprints across repeats, positions, and ownership rather than
byte-for-byte to the c1 layout. ``--protocol-check`` permits an uncommitted
tracked tree and exits 0 with ``status=protocol_check`` when the adapter,
state gate, determinism, and teardown are sound; it never marks the candidate
admitted. A non-protocol batch run outside the normative 18-prompt/24-step/
450-row scope, or one whose profile manifest does not bind the batch path,
reports ``status=ineligible`` with the reasons under ``admission``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.execution_profiles import EvaluationThresholds
from hipengine.runtime.qwen4_exp_runner import (
    gdn_peer_prefill, gdn_prefill_variant, gdn_register_prefill,
    q4_exact_grouped_all, q4_exact_grouped_gate,
    q4_out4, q51_exact_grouped_down, q51_out8, q51_wmma_down,
    qsa_ordered_decode_v2,
    qsa_prefill_head_mode, qsa_h256_prefill_mode,
)
from scripts.execution_profile_gdn_calibration import (
    PromptCalibrationCapture,
    build_candidate_quality,
)
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import prompt_sha256
from scripts.qwen4exp_candidate_dispatch import count_candidate_dispatch, shape_records
from scripts.qwen4exp_canonical_ar_bench import (
    _git_metadata,
    _host_metadata,
    _write_json,
    sha256_path,
    token_ids_sha256,
)

KIND = "qwen4exp_execution_profile_candidate_gate"

#: Normative AR batch admission scope: 18 named prompts x (24 decode steps + 1
#: prefill-last row) = 450 teacher-forced rows, three repeats, and 32-token
#: free outputs over two runs. A shorter run validates the adapter protocol but
#: is not eligible for admission, regardless of a clean tracked tree.
BATCH_NORMATIVE_DECODE_STEPS = 24
BATCH_NORMATIVE_REPEAT_RUNS = 3
BATCH_NORMATIVE_FREE_TOKENS = 32
BATCH_NORMATIVE_FREE_RUNS = 2
BATCH_NORMATIVE_TEACHER_ROWS = 450


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: str
    classification: str
    mechanism: str
    environment: Mapping[str, str]
    base_profile: str
    scenario_id: str
    candidate_key: tuple[str, str, str, str] | None
    fallback_key: tuple[str, str, str, str] | None
    compact_output: bool = False
    count_registered_dispatch: bool = False
    direct_dispatch_target: tuple[str, str] | None = None
    direct_dispatch_reference: tuple[str, str] | None = None
    dispatch_shape_positions: tuple[int, ...] | None = None
    quant: str | None = None
    # Explicit H256 prefill head grouping for this arm. ``None`` keeps whatever
    # the base profile published; the gate binds a non-None value with
    # ``qsa_prefill_head_mode`` around the measured rows and restores it after.
    qsa_prefill_head_mode: str | None = None
    # None preserves the base profile; parent, wave or page256 binds one arm.
    qsa_h256_prefill_mode: str | None = None
    # Explicit GDN serial-prefill owner for this arm, same protocol. ``None``
    # keeps the base profile's published variant (the register parent on a
    # strict base); a wide/narrow name binds that leaf for the measured rows.
    gdn_prefill_variant: str | None = None
    gdn_register_prefill: bool | None = None
    # Explicit QSA ordered-v2 sparse decode selection for this arm, same
    # protocol. ``None`` keeps the base profile's published decision (on for a
    # production base, the ordered v1 parent on a strict one); the gate binds a
    # non-None value around the measured rows and restores it after.
    qsa_ordered_decode_v2: bool | None = None
    # Explicit GDN peer prefill layer scope for this arm, same protocol. ``None``
    # keeps the base profile's published scope (the certified layers 35-47 on a
    # non-recovered production base, no scope on strict); ``False`` binds the
    # serial sigmoid owners for the measured rows, and a scope string or ``True``
    # binds the peer owner over that scope. The published value comes back on
    # unwind.
    gdn_peer_prefill: bool | str | None = None
    # Explicit exact grouped Q5_1 down chain gate for this arm, same protocol.
    # ``None`` keeps the base profile's published decision (on for every plan);
    # ``False`` closes the chain around the measured rows so the dispatch falls
    # through to the sorted/WMMA down owners, and the published value comes back
    # after.
    q51_exact_grouped_down: bool | None = None
    # Explicit admission of the F16 WMMA compact grouped Q5_1 down owner for
    # this arm, same protocol. ``None`` keeps the base profile's published
    # decision (off for every plan); ``True`` admits the owner on the chunks
    # the production grouped MoE route does not own, which is the only way an
    # arm can measure it on a non-production chunk, and the published value
    # comes back after.
    q51_wmma_down: bool | None = None
    # Explicit Q5_1 out8 fallback rung for this arm, same protocol. ``None``
    # keeps the base profile's published decision (the out8 owner on every
    # plan); ``False`` pins the output1 rowbatch8 base around the measured rows.
    # It only governs the exact grouped Q5_1 down chain's non-expertgrid rung,
    # so an arm that also needs the output1 leaf to launch closes
    # ``q51_exact_grouped_down`` and the expert-grid family with it.
    q51_out8: bool | None = None
    # Explicit Q4 out4 fallback rung for this arm, same protocol. ``None`` keeps
    # the base profile's published decision (the out4 owner on every plan);
    # ``False`` pins the strict rowbatch8 base around the measured rows. It only
    # governs the exact grouped Q4_K gate/up chain's non-expertgrid rung, so an
    # arm that also needs the strict base to launch closes the chain's
    # iu8-risk+repair owner and the expert-grid family with it.
    q4_out4: bool | None = None
    # Explicit Q4_K exact grouped gate/up gate and its `_ALL` joint term for this
    # arm, same protocol. ``None`` keeps the base profile's publication (both on
    # every plan). ``False`` on the gate is the route to the plain-iu8 and F16
    # compact gate/up owners below the chain, and it also moves the grouped Q5_1
    # down chain's input layout; ``False`` on `_ALL` only narrows the chain to
    # chunks the exact grouped Q5_1 down chain already owns.
    q4_exact_grouped_gate: bool | None = None
    q4_exact_grouped_all: bool | None = None

    @property
    def requires_dispatch_count(self) -> bool:
        return self.count_registered_dispatch or self.direct_dispatch_target is not None


CONSERVATIVE_ARITHMETIC_FLAGS = (
    "PRODUCTION_MOE_PREFILL", "Q4_IU8_PREFILL", "Q8_0_SELECTED_WMMA_DOWN",
    "Q8_MMQ_PREFILL", "GR_IU8", "GR_IU8_DOWN",
    "GDN_COLWARPS_PREFILL", "Q4_DP4A64",
    "QSA_FLASH_PREFILL",
    "QSA_ORDERED_DECODE", "GR_WIDE",
)
# The QSA prefill head grouping, the ordered-v2 sparse decode leaf and the GDN
# peer prefill owner's layer scope are request-local runner state, so this gate
# arms them through CandidateSpec.qsa_prefill_head_mode,
# CandidateSpec.qsa_ordered_decode_v2 and CandidateSpec.gdn_peer_prefill, not
# through environment switches.

_F16_ROUTES_OFF = {
    "HIPENGINE_QWEN4_EXP_ROUTED_F16": "0",
    "HIPENGINE_QWEN4_EXP_GR_WIDE": "0",
    "HIPENGINE_QWEN4_EXP_GR_F16_DIRECT": "0",
    "HIPENGINE_QWEN4_EXP_GDN_DECODE_ROWSPLIT": "0",
    "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_GATE_UP": "0",
    "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_DOWN": "0",
    "HIPENGINE_QWEN4_EXP_DENSE_F16_GEMM": "0",
    "HIPENGINE_QWEN4_EXP_QSA_WMMA": "0",
    "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS": ",".join(map(str, range(13, 48))),
    "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS": ",".join(map(str, range(27, 48))),
    "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT": "qwen4exp_gdn_tiled16_multi_prefill",
    "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT": "f32_hidden_token_tile4_dpp_exact",
}
_ALL_LAYERS = ",".join(map(str, range(48)))
#: Owners that take the Q5_1 down leg before the F16 compact owner. The routed
#: F16 route owns both legs ahead of the Q5_1 branch (so it preempts the arm's
#: declared gate/up control too). An arm that counts the F16 down owner must
#: close that route or it measures the base profile's routed F16 owner. The
#: candidate environment is applied after the bind and restored on exit.
_F16_DOWN_PREEMPTORS_OFF = {
    "HIPENGINE_QWEN4_EXP_ROUTED_F16": "0",
}
_F16_ROUTE_ARMS = {
    "recovery_plan": {},
    "recovery_plus_routed_f16": {"HIPENGINE_QWEN4_EXP_ROUTED_F16": "1"},
    "recovery_plus_dense_wide_all_layers": {
        "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS": _ALL_LAYERS,
    },
    "recovery_plus_gr_wide": {"HIPENGINE_QWEN4_EXP_GR_WIDE": "1"},
    "recovery_plus_indexer_router_f16": {
        "HIPENGINE_QWEN4_EXP_DENSE_F16_GEMM": "1",
        "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT": "f32_hidden_f16_wmma_gemm",
    },
    "recovery_plus_qsa_wmma": {"HIPENGINE_QWEN4_EXP_QSA_WMMA": "1"},
    "recovery_plus_gdn_tiled_all_layers": {
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS": _ALL_LAYERS,
    },
}


def _f16_route_isolation_candidates() -> dict[str, CandidateSpec]:
    """The arithmetic-recovery plan, alone and with one F16 route added.

    The UD-Q4_K_XL production plan carries the F16 route composition; these
    arms clear it after the binder and restore one route at a time, so a
    drift measured on the composition can be attributed to a route.
    """

    return {
        name: CandidateSpec(
            name=name,
            classification="diagnostic",
            mechanism=(
                "production with the F16 route composition cleared after the "
                "binder" + (", plus " + ", ".join(sorted(extra)) if extra else "")
            ),
            environment={**_F16_ROUTES_OFF, **extra},
            base_profile="production",
            scenario_id="qwen4exp-" + name.replace("_", "-"),
            candidate_key=None,
            fallback_key=None,
        )
        for name, extra in _F16_ROUTE_ARMS.items()
    }


CANDIDATES = {
    "dense_q8_checked_full_rows": CandidateSpec(
        name="dense_q8_checked_full_rows",
        classification="T0",
        mechanism=(
            "checked-int32/full-four-row sibling of the production dense Q8 DPP "
            "owner; same ordered FMA/DPP arithmetic with complete-range admission"
        ),
        environment={
            "HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION": "1",
        },
        base_profile="production",
        scenario_id="qwen4exp-dense-q8-checked-full-rows",
        candidate_key=(
            "hip_gfx1151", "linear", "gguf_q8_0",
            "coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out",
        ),
        fallback_key=(
            "hip_gfx1151", "linear", "gguf_q8_0",
            "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out",
        ),
        count_registered_dispatch=True,
    ),
    "q8_blockscale": CandidateSpec(
        name="q8_blockscale",
        classification="T1",
        mechanism="raw Q8 codes times original BF16 activations; FP32 block scales after WMMA",
        environment={
            "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN": "1",
            "HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT":
                "selected_grouped_blockscale_prefill_bf16_bf16_out",
        },
        base_profile="production",
        scenario_id="qwen4exp-q8-blockscale",
        candidate_key=("hip_gfx1151", "linear", "gguf_q8_0",
                       "selected_grouped_blockscale_prefill_bf16_bf16_out"),
        fallback_key=("hip_gfx1151", "linear", "gguf_q8_0",
                      "selected_grouped_gemv_bf16_bf16_out"),
        count_registered_dispatch=True,
    ),
    "production_conservative": CandidateSpec(
        name="production_conservative",
        classification="diagnostic",
        mechanism="conservative arithmetic fallback retaining independently exact production owners",
        environment={"HIPENGINE_QWEN4_EXP_" + flag: "0"
                     for flag in CONSERVATIVE_ARITHMETIC_FLAGS},
        base_profile="production",
        scenario_id="qwen4exp-ud-q4-k-xl-conservative-repair",
        candidate_key=None,
        fallback_key=None,
        # The conservative arm keeps the ordered v1 parent the env zeroing above
        # selects, so the v2 leaf it rides is bound off explicitly rather than
        # left at the production base's published value. The GDN peer prefill
        # owner is request-local state rather than one of those flags, so the
        # arm closes its scope explicitly: on the UD base that is a no-op, and on
        # any lane that publishes a scope it keeps the arm conservative.
        qsa_ordered_decode_v2=False,
        qsa_h256_prefill_mode="parent",
        gdn_peer_prefill=False,
    ),
    # The risk multiplier as a CONFIGURATION.  The production repair is a
    # bit-parity mechanism, and `HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT` sets the
    # Kahan bound that decides which outputs are queued for it.  Its default of
    # 4.0 is deliberately loose against a docstring-screened floor of 0.5-1, so
    # it repairs 0.7119% of rows where 1.0 repairs 0.2092% and 0.5 repairs
    # 0.1124% (measured, `code-p4096`, 1504 calls each, zero over-capacity
    # calls).  The wall follows the incidence: 17.014 -> 16.655 -> 16.564 s for
    # the same prefill.  This candidate is the 1.0 point, which the multiplier's
    # own screening places INSIDE the range where BF16 flips are still caught -
    # so it is the cheapest point that keeps the criterion, and the one worth a
    # gate.  It is a different change from the rejected `repair_free_gate_up`,
    # which removed the repair entirely (a 0% incidence) and also swapped the
    # owner; here the owner is unchanged and 29% of the flagged rows are still
    # repaired.
    "q4_iu8_risk_mult_1": CandidateSpec(
        name="q4_iu8_risk_mult_1",
        classification="T2",
        mechanism=(
            "tighten the exact-iu8 Kahan risk bound from 4.0 to 1.0 so the "
            "sparse exact repair serves 0.209% of rows instead of 0.712%, "
            "with the gate/up owner and the down leg unchanged"
        ),
        environment={"HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT": "1.0"},
        base_profile="production",
        scenario_id="qwen4exp-q4-iu8-risk-mult-1",
        candidate_key=None,
        fallback_key=None,
    ),
    # MX-1's gate/up half as a CONFIGURATION rather than a build. The
    # production repair is a bit-parity mechanism: the risk kernel queues
    # outputs within a Kahan bound of a BF16 rounding boundary so the sparse
    # exact pass can reproduce the pair2 parent bit-for-bit, and the profile
    # binds it "bit-identical to the pair2 owner". The plain iu8 owner
    # publishes *identical arithmetic* without the risk criterion or the
    # repair pass, and it is already reachable: the dispatch tests
    # exact_grouped_q4_gate BEFORE the q4_iu8 branch, so closing the former
    # selects the latter. The gate is request-local runner state
    # (`q4_exact_grouped_gate`), bound by the candidate spec. The production
    # profile already binds
    # Q4_IU8_PREFILL=1 but only on layers 35-47 while the repair route carries
    # no layer restriction, so the layer set is widened to cover exactly the
    # scope the repair currently serves. Q5_1 down is left at production
    # because no plain non-risk iu8 sibling exists for it.
    "repair_free_gate_up": CandidateSpec(
        name="repair_free_gate_up",
        classification="T1",
        mechanism=(
            "plain iu8-WMMA grouped gate/up on every layer with the "
            "bit-parity repair removed; Q5_1 down unchanged at production"
        ),
        environment={
            "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS": "all",
        },
        # The chain gate is runner state now, so the arm closes it through the
        # spec's field instead of a process key.
        q4_exact_grouped_gate=False,
        base_profile="production",
        scenario_id="qwen4exp-repair-free-gate-up",
        candidate_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                       "selected_dual_wmma_iu8_prefill_bf16_bf16_out"),
        fallback_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                      "selected_dual_wmma_prefill_compact_bf16_bf16_out"),
        # The plain iu8 owner is invoked by a DIRECT module-attribute call in
        # the runner, not through a registry resolve, so counting the registry
        # entry alone reads zero however often the route runs. That is exactly
        # how the first run of this candidate reported
        # candidate_dispatch_calls=0 and measurement_valid=false while the
        # route was engaging: verified independently by patching the runner's
        # module attribute and observing 4 plain-iu8 calls with 0 risk and 0
        # repair calls on one short prefill. The direct target makes the
        # gate's counter patch the same attribute the runner actually calls.
        direct_dispatch_target=("hipengine.runtime.qwen4_exp_runner",
                                "gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out"),
        direct_dispatch_reference=(
            "hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill",
            "gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out"),
        dispatch_shape_positions=(7, 8, 9, 10),
        count_registered_dispatch=True,
    ),
    "f16_compact_gate_up": CandidateSpec(
        name="f16_compact_gate_up",
        classification="T1",
        mechanism=(
            "the F16-WMMA compact grouped gate/up owner (Q4_K decoded to F16, "
            "F16 WMMA, F32 accumulate) with no repair pass; Q5_1 down "
            "unchanged at production"
        ),
        environment={},
        # The ONLY change: close the exact-grouped Q4 chain gate (request-local
        # runner state, so it is the spec's field and not a process key) so the
        # chain falls through to the F16 owner. Q4_IU8_PREFILL stays at its
        # default 0 and the exact grouped Q5_1 down chain stays at its
        # production publication, which isolates the gate/up half (the down
        # leg's route is selected independently at its own dispatch site).
        q4_exact_grouped_gate=False,
        base_profile="production",
        scenario_id="qwen4exp-f16-compact-gate-up",
        candidate_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                       "selected_dual_wmma_prefill_compact_bf16_bf16_out"),
        fallback_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                      "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"),
        # This owner is also invoked by a direct module-attribute call, so the
        # registry-entry counter alone would read zero. MX-1's premise is that
        # the F16 route is admissible with NO repair because F16 decode error
        # sits 8-9.5x below the BF16 floor already in the path - if that holds
        # the 1,219.978 ms repair comes out as a configuration change rather
        # than the from-scratch port the row was priced as.
        direct_dispatch_target=("hipengine.runtime.qwen4_exp_runner",
                                "gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out"),
        direct_dispatch_reference=(
            "hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill",
            "gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out"),
        dispatch_shape_positions=(7, 8, 9, 10),
        count_registered_dispatch=True,
    ),
    # MX-1's down half, gated on its own. The full F16 route and its gate/up
    # half are refuted on numerics, but the down half's main kernel is the
    # fastest registered down owner (-42.29% median vs the production iu8-risk
    # owner, `2026-10-07-qwen4exp-mx1-f16-owner-recheck`), so a down-only F16
    # variant is the only MX-1 configuration whose numerical status is unknown.
    # The overrides isolate exactly that leg:
    #   * `q51_exact_grouped_down=False` on the runner skips the production Q5_1
    #     down branch so the chain falls through to `elif q5_wmma`.
    #   * `q51_wmma_down=True` on the runner admits the F16 down owner there
    #     (request-local state, not a process key).
    #   * `ROUTED_F16=0` closes the route that owns both legs before the Q5_1
    #     branch. Otherwise the production base binds `ROUTED_F16=1` and the arm
    #     counts zero F16 down calls while measuring another owner.
    # The gate/up leg is selected by the independent `exact_grouped_q4_gate`
    # predicate, which stays true because `q4_exact_grouped_all` keeps its
    # production publication of True when only the down chain is closed. No
    # gate/up decision is overridden, so gate/up stays on the production exact
    # iu8-risk+repair route on the layers whose exact chain owns it (0-26); the
    # 27-47 production grouped window keeps the gate/up owner production selects
    # there.
    "gdn_tiled16_multi": CandidateSpec(
        name="gdn_tiled16_multi",
        classification="T1",
        mechanism=(
            "the registered tiled16 multi-column GDN prefill owner for the "
            "certified suffix window: the same recurrence reassociated over "
            "multi-column tiles (NMSE 1.8e-14 vs the sequential chain per the "
            "GD-4 drift screens, fp32 round-off class); the production "
            "wave_norm route stays the registered strict fallback"
        ),
        environment={
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT": "qwen4exp_gdn_tiled16_multi_prefill",
        },
        base_profile="production",
        scenario_id="qwen4exp-gdn-tiled16-multi",
        candidate_key=(
            "hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
            "qwen4exp_gdn_tiled16_multi_prefill",
        ),
        fallback_key=(
            "hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
            "qwen4exp_sigmoid_wave_norm_prefill",
        ),
        count_registered_dispatch=True,
    ),
    "f16_compact_down": CandidateSpec(
        name="f16_compact_down",
        classification="T1",
        mechanism=(
            "the F16-WMMA compact grouped down owner (Q5_1 decoded to F16, F16 "
            "WMMA, F32 accumulate) with no repair pass; routed-F16 composition "
            "is held off and other Q4_K gate/up controls retain their profile values"
        ),
        environment=_F16_DOWN_PREEMPTORS_OFF,
        q51_exact_grouped_down=False,
        q51_wmma_down=True,
        base_profile="production",
        scenario_id="qwen4exp-f16-compact-down",
        candidate_key=("hip_gfx1151", "moe_linear", "gguf_q5_1",
                       "selected_grouped_wmma_prefill_compact_bf16_bf16_out"),
        fallback_key=("hip_gfx1151", "moe_linear", "gguf_q5_1",
                      "selected_wmma_iu8_risk_prefill_bf16_bf16_out"),
        # The down owner is invoked by a direct module-attribute call in the
        # runner, so counting the registry entry alone would read zero however
        # often the route ran. The direct target makes the gate's counter patch
        # the same attribute the runner calls.
        direct_dispatch_target=("hipengine.runtime.qwen4_exp_runner",
                                "qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out"),
        direct_dispatch_reference=(
            "hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1",
            "qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out"),
        # Direct call site ABI: (compact_rows, num_experts, in_features,
        # out_features) are positional arguments 6-9 of
        # qwen4_exp_runner.py's Q5_1 down dispatch.
        dispatch_shape_positions=(6, 7, 8, 9),
        count_registered_dispatch=True,
    ),
    "routed_f16": CandidateSpec(
        name="routed_f16",
        classification="T1",
        mechanism=(
            "the wide-tile routed F16 GEMMs over the raw expert weights (the "
            "fast profile's routed-expert route): Q4_K/Q5_K gate and up in one "
            "launch with the SwiGLU epilogue and the Q5_1/Q8_0 down "
            "projection, weights decoded to F16 once per macro tile of "
            "grouped rows, F16 WMMA with F32 accumulation, no repair pass, "
            "replacing the exact grouped chains for every grouped prefill "
            "dispatch"
        ),
        environment={
            "HIPENGINE_QWEN4_EXP_ROUTED_F16": "1",
        },
        base_profile="production",
        scenario_id="qwen4exp-routed-f16",
        candidate_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                       "selected_routed_f16_dual_silu_wmma_prefill_bf16_bf16_out"),
        fallback_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                      "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"),
        count_registered_dispatch=True,
    ),
    "fast_composition": CandidateSpec(
        name="fast_composition",
        classification="T2",
        mechanism=(
            "the complete fast execution profile composition applied to the "
            "production plan: wide-tile routed F16 GEMMs for every grouped "
            "routed-expert dispatch, the F16 dense wide route at layers 0-47 "
            "(which carries the hyper-connection down projection), the "
            "hyper-connection up projection on the wide route, the "
            "small-matrix F16 WMMA GEMM for the indexer projections, the "
            "F16-operand router-logits producer, the WMMA QSA prefill "
            "attention, and the tiled GDN recurrence at every GDN layer"
        ),
        environment={
            "HIPENGINE_QWEN4_EXP_ROUTED_F16": "1",
            "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS": ",".join(
                str(layer) for layer in range(48)
            ),
            "HIPENGINE_QWEN4_EXP_GR_WIDE": "1",
            "HIPENGINE_QWEN4_EXP_DENSE_F16_GEMM": "1",
            "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT": "f32_hidden_f16_wmma_gemm",
            "HIPENGINE_QWEN4_EXP_QSA_WMMA": "1",
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS": ",".join(
                str(layer) for layer in range(48)
            ),
        },
        base_profile="production",
        scenario_id="qwen4exp-fast-composition",
        candidate_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                       "selected_routed_f16_dual_silu_wmma_prefill_bf16_bf16_out"),
        fallback_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                      "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"),
        count_registered_dispatch=True,
    ),
    "mx1_full_route": CandidateSpec(
        name="mx1_full_route",
        classification="T1",
        mechanism=(
            "MX-1 as specified: F16-WMMA grouped gate/up (Q4_K) AND F16-WMMA "
            "grouped down (Q5_1), both with no repair pass"
        ),
        environment={
            # Two process keys and three runner fields, because the down leg's
            # own chain tests `exact_grouped_down` BEFORE `q5_wmma`:
            #   if routed_f16:              <- production base binds this on
            #   elif down == gguf_q5_1:
            #     if exact_grouped_down:    <- production chain (repair)
            #     elif q5_wmma:             <- F16 down owner
            # So the arm closes the routed F16 owner as well; without
            # that both legs are the routed F16 route and the compact aliases
            # the counters watch never launch. Both chain gates are runner state
            # (`q51_exact_grouped_down`, `q4_exact_grouped_gate`), so they are
            # the spec's fields rather than process keys, and so is the F16 down
            # owner's admission (`q51_wmma_down`).
            **_F16_DOWN_PREEMPTORS_OFF,
        },
        q51_exact_grouped_down=False,
        q51_wmma_down=True,
        q4_exact_grouped_gate=False,
        base_profile="production",
        scenario_id="qwen4exp-mx1-full-route",
        # Both owners are invoked by direct module-attribute calls, so both
        # counters are declared. The candidate key names the gate/up owner; the
        # down owner is counted through the same direct-dispatch mechanism.
        candidate_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                       "selected_dual_wmma_prefill_compact_bf16_bf16_out"),
        fallback_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                      "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"),
        direct_dispatch_target=("hipengine.runtime.qwen4_exp_runner",
                                "gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out"),
        direct_dispatch_reference=(
            "hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill",
            "gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out"),
        dispatch_shape_positions=(7, 8, 9, 10),
        count_registered_dispatch=True,
    ),
    "production_q8_fallback": CandidateSpec(
        name="production_q8_fallback",
        classification="diagnostic",
        mechanism="strict Q8 selected-down and dense Q8 MMQ fallback; other production paths unchanged",
        environment={
            "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN": "0",
            "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "0",
        },
        base_profile="production",
        scenario_id="qwen4exp-ud-q4-k-xl-q8-fallback",
        candidate_key=None,
        fallback_key=None,
    ),
    "gdn_isolated": CandidateSpec(
        name="gdn_isolated",
        classification="diagnostic",
        mechanism="current GDN suffix over strict non-GDN arithmetic",
        environment={
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS": ",".join(map(str, range(27, 48))),
            "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT": "qwen4exp_gdn_tiled16_dpp_prefill",
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS": "",
        },
        base_profile="strict",
        scenario_id="qwen4exp-ud-q4-k-xl-gdn-isolated",
        gdn_register_prefill=True,
        gdn_prefill_variant="qwen4exp_sigmoid_wave_norm_prefill",
        gdn_peer_prefill=",".join(map(str, range(35, 48))),
        candidate_key=("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                       "qwen4exp_gdn_tiled16_dpp_prefill"),
        fallback_key=("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                      "qwen4exp_sigmoid_strict_prefill"),
    ),
    "gdn_multi_column": CandidateSpec(
        name="gdn_multi_column",
        classification="T1",
        mechanism="eight state columns per warp in admitted tiled16 GDN suffix",
        environment={
            "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT": "qwen4exp_gdn_tiled16_multi_prefill",
        },
        base_profile="production",
        scenario_id="qwen4exp-ud-q4-k-xl-gdn-multi-column",
        candidate_key=("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                       "qwen4exp_gdn_tiled16_multi_prefill"),
        fallback_key=("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                      "qwen4exp_sigmoid_strict_prefill"),
    ),
    "production_q8down_gdn_fallback": CandidateSpec(
        name="production_q8down_gdn_fallback",
        classification="diagnostic",
        mechanism="incumbent repair candidate: strict Q8 selected-down and GDN prefill",
        environment={
            "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN": "0",
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "0",
        },
        base_profile="production",
        scenario_id="qwen4exp-ud-q4-k-xl-production-fallback-review",
        candidate_key=None,
        fallback_key=None,
        gdn_peer_prefill=False,
    ),
    **_f16_route_isolation_candidates(),
    "production_baseline": CandidateSpec(
        name="production_baseline",
        classification="diagnostic",
        mechanism="unmodified named production profile versus strict",
        environment={},
        base_profile="production",
        scenario_id="qwen4exp-ud-q4-k-xl-production-baseline",
        candidate_key=None,
        fallback_key=None,
    ),
    "device_argmax": CandidateSpec(
        name="device_argmax",
        classification="T0",
        mechanism="select greedy token on device and copy one int64",
        environment={},
        base_profile="strict",
        scenario_id="qwen4exp-ud-q4-k-xl-device-argmax",
        candidate_key=("hip_gfx1151", "argmax", "f32", "top1_i64"),
        fallback_key=("hip_gfx1151", "argmax", "f32", "top1_i64"),
        compact_output=True,
    ),
    "q8_mmq_attn_gate": CandidateSpec(
        name="q8_mmq_attn_gate",
        classification="T2",
        mechanism="extend guarded Q8 MMQ F32 prefill to GDN attn_gate K2560/N6144",
        environment={"HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE": "1"},
        base_profile="production",
        scenario_id="qwen4exp-ud-q4-k-xl-q8-mmq-attn-gate",
        candidate_key=(
            "hip_gfx1151", "linear", "gguf_q8_0",
            "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out",
        ),
        fallback_key=(
            "hip_gfx1151", "linear", "gguf_q8_0",
            "coltile8_rowbatch4_f32_f32_out",
        ),
    ),
}


CANDIDATES["ar_batch"] = CandidateSpec(
    name="qwen4_exp_ar_batch",
    classification="T2",
    mechanism=(
        "committed c1 prefill (one request at a time) imported once into one "
        "physical batch slot per request, then one genuine N-row AR decode step "
        "per teacher-forced position; width-specific association is the change"
    ),
    environment={},
    base_profile="production",
    scenario_id="qwen4exp-ar-batch",
    candidate_key=None,
    fallback_key=None,
)


CANDIDATES["bounded_kv_reset"] = CandidateSpec(
    name="bounded_kv_reset",
    classification="T1",
    mechanism=(
        "reset() clears only the K/V bytes the request could have written instead "
        "of the full allocated capacity; exact by the written_cache_bytes invariant "
        "(allocation zeroes the whole cache, writes only land below position, so "
        "bytes above the round_up(position, block_size) bound are already zero)"
    ),
    environment={"HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET": "1"},
    base_profile="production",
    scenario_id="qwen4exp-bounded-kv-reset",
    # This changes how much storage reset() zeroes, not a registry key, so there is
    # no swapped key to count.  For a T1 candidate the binding evidence is that the
    # strict and candidate state/logits hashes are IDENTICAL - the bytes skipped are
    # already zero, so skipping them cannot change a result.
    candidate_key=None,
    fallback_key=None,
)

CANDIDATES["pack8_gemv_wide_256"] = CandidateSpec(
    name="pack8_gemv_wide_256",
    classification="T2",
    mechanism=(
        "decode-shaped pack8 GEMV at a 256-wide block instead of the 128-wide "
        "launcher default where out_features <= 640; same kernel body, wider "
        "block, so the K-reduction tree differs and the result is not byte-exact"
    ),
    environment={"HIPENGINE_GGUF_PACK8_GEMV_WIDE": "1"},
    base_profile="production",
    scenario_id="qwen4exp-ud-q4-k-xl-pack8-gemv-wide-256",
    # This candidate changes a launch parameter, not a registry key, so there is
    # no swapped key to count. Engagement is established separately from the
    # kernel trace (the three narrow shapes move from wg=128 to wg=256 with
    # identical workgroup counts) and from strict_and_candidate_bytes_equal,
    # which must be false for this run to mean anything.
    candidate_key=None,
    fallback_key=None,
)

CANDIDATES["q8_blockscale_mmq"] = replace(
    CANDIDATES["q8_blockscale"],
    name="q8_blockscale_mmq", scenario_id="qwen4exp-q8-blockscale-mmq",
    environment={**CANDIDATES["q8_blockscale"].environment,
                 "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "1"},
)
CANDIDATES["q8_blockscale_guarded"] = replace(
    CANDIDATES["q8_blockscale"], name="q8_blockscale_guarded",
    scenario_id="qwen4exp-q8-blockscale-guarded",
    environment={
        **CANDIDATES["q8_blockscale"].environment,
        "HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT":
            "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out",
    },
    candidate_key=("hip_gfx1151", "linear", "gguf_q8_0",
                   "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out"),
)
CANDIDATES["q8_blockscale_guarded_mmq"] = replace(
    CANDIDATES["q8_blockscale_guarded"], name="q8_blockscale_guarded_mmq",
    scenario_id="qwen4exp-q8-blockscale-guarded-mmq",
    environment={**CANDIDATES["q8_blockscale_guarded"].environment,
                 "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "1"},
)
CANDIDATES["q8_blockscale_guarded_quad"] = replace(
    CANDIDATES["q8_blockscale_guarded"], name="q8_blockscale_guarded_quad",
    scenario_id="qwen4exp-q8-blockscale-guarded-qsa-quad",
    qsa_h256_prefill_mode="page256",
    qsa_prefill_head_mode="quad",
)
CANDIDATES["q8_blockscale_guarded_restore"] = replace(
    CANDIDATES["q8_blockscale_guarded_quad"], name="q8_blockscale_guarded_restore",
    scenario_id="qwen4exp-q8-blockscale-guarded-dense-qsa",
    environment={
        **CANDIDATES["q8_blockscale_guarded_quad"].environment,
        "HIPENGINE_QWEN4_EXP_GR_IU8": "1",
        "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "1",
        "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE": "1",
    },
    qsa_ordered_decode_v2=True,
)
CANDIDATES["q8_blockscale_guarded_ordered"] = replace(
    CANDIDATES["q8_blockscale_guarded_quad"], name="q8_blockscale_guarded_ordered",
    scenario_id="qwen4exp-q8-blockscale-guarded-qsa-ordered",
    environment={**CANDIDATES["q8_blockscale_guarded_quad"].environment,
                 "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE": "1"},
    qsa_ordered_decode_v2=True,
)
CANDIDATES["production_gdn_restore"] = replace(
    CANDIDATES["gdn_isolated"],
    name="production_gdn_restore",
    classification="T2",
    mechanism="restore the existing GDN/DPP suffix over corrected Q8/QSA production",
    base_profile="production",
    scenario_id="qwen4exp-production-gdn-restoration",
    count_registered_dispatch=True,
)
CANDIDATES["production_gdn_multi_restore"] = replace(
    CANDIDATES["production_gdn_restore"],
    name="production_gdn_multi_restore",
    mechanism="multi-column GDN suffix over corrected Q8/QSA production",
    scenario_id="qwen4exp-production-gdn-multi-restoration",
    environment={
        **CANDIDATES["production_gdn_restore"].environment,
        "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT": "qwen4exp_gdn_tiled16_multi_prefill",
    },
    candidate_key=("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                   "qwen4exp_gdn_tiled16_multi_prefill"),
)
# The multi-column route is consulted only inside the runner's colwarps gate, and
# that gate is open exactly on the profiles that do NOT run the UD arithmetic
# recovery. Every other candidate above pins the route with an env override on top
# of a profile that either closes the gate (UD) or clears the variant (q4_k_m), so
# none of them can observe the binding the production profiles actually install.
# This candidate has no env override at all: it passes only if the q4_k_m
# production binder by itself dispatches the multi-column owner, with the dispatch
# count recorded so the result cannot be vacuous.
CANDIDATES["production_gdn_multi_q4km"] = CandidateSpec(
    name="production_gdn_multi_q4km",
    classification="T1",
    mechanism="multi-column GDN prefill owner selected by the q4_k_m production "
              "binder alone, on the profile that keeps the colwarps gate open",
    environment={},
    base_profile="production",
    quant="gguf_q4_k_m",
    scenario_id="qwen4exp-q4-k-m-gdn-multi-column",
    candidate_key=("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                   "qwen4exp_gdn_tiled16_multi_prefill"),
    fallback_key=("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                  "qwen4exp_sigmoid_strict_prefill"),
    count_registered_dispatch=True,
)
# Attribution control for `production_gdn_multi_q4km`. It is the same q4_k_m production
# base profile with the GDN prefill owner pinned to `qwen4exp_gdn_tiled16_prefill`,
# which is the callable the binder fell back to before the tile16 binding was
# repaired. Running both candidates separates "the multi-column route is outside
# the envelope" from "the production composition around it is".
CANDIDATES["production_gdn_tiled16_q4km"] = CandidateSpec(
    name="production_gdn_tiled16_q4km",
    classification="T1",
    mechanism="q4_k_m production base profile with the GDN prefill owner pinned to "
              "the tiled16 route the binder selected before the multi-column "
              "binding was repaired - the attribution control for "
              "production_gdn_multi_q4km",
    environment={
        "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT": "qwen4exp_gdn_tiled16_prefill",
    },
    base_profile="production",
    quant="gguf_q4_k_m",
    scenario_id="qwen4exp-q4-k-m-gdn-tiled16-control",
    candidate_key=("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                   "qwen4exp_gdn_tiled16_prefill"),
    fallback_key=("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                  "qwen4exp_sigmoid_strict_prefill"),
    count_registered_dispatch=True,
)
CANDIDATES["production_gr_up_restore"] = CandidateSpec(
    name="production_gr_up_restore", classification="T2",
    mechanism="GR-up three-plane IU8 projection plus existing sigmoid/mean; dense and GR-down off",
    environment={
        "HIPENGINE_QWEN4_EXP_GR_IU8": "1",
        "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "0",
    },
    base_profile="production", scenario_id="qwen4exp-gr-up-restoration",
    candidate_key=("hip_gfx1151", "linear", "gguf_q8_0", "iu8_wmma_prefill_f32_f32_out"),
    fallback_key=("hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                  "coltile2_branch4_rowbatch4_f32_exact"),
    direct_dispatch_target=("hipengine.runtime.qwen4_exp_runner",
                            "gguf_q8_0_iu8_wmma_prefill_f32_f32"),
    direct_dispatch_reference=("hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv",
                               "gguf_q8_0_iu8_wmma_prefill_f32_f32"),
    dispatch_shape_positions=(3, 4, 5),
)
CANDIDATES["production_gr_up_wide"] = CandidateSpec(
    name="production_gr_up_wide", classification="T1",
    mechanism=(
        "GR-up wide-row 256-column tile (F16 operands, F32 accumulate) plus "
        "the retained exact sigmoid/gated-mean epilogue; dense and GR-down off"
    ),
    environment={
        "HIPENGINE_QWEN4_EXP_GR_WIDE": "1",
        "HIPENGINE_QWEN4_EXP_GR_IU8": "0",
        "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "0",
    },
    base_profile="production", scenario_id="qwen4exp-gr-up-wide-tile",
    candidate_key=("hip_gfx1151", "linear", "gguf_q8_0",
                   "dense_wide256_f32_f32_out"),
    fallback_key=("hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                  "coltile2_branch4_rowbatch4_f32_exact"),
    direct_dispatch_target=("hipengine.runtime.qwen4_exp_runner",
                            "gguf_q8_0_dense_wide256_f32_f32_out"),
    direct_dispatch_reference=(
        "hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide",
        "gguf_q8_0_dense_wide256_f32_f32_out"),
    dispatch_shape_positions=(3, 4, 5),
)
CANDIDATES["production_gr_down_restore"] = replace(
    CANDIDATES["production_gr_up_restore"], name="production_gr_down_restore",
    mechanism="GR-down three-plane IU8 projection; dense and GR-up off",
    scenario_id="qwen4exp-gr-down-restoration",
    environment={
        **CANDIDATES["production_gr_up_restore"].environment,
        "HIPENGINE_QWEN4_EXP_GR_IU8": "0",
        "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "1",
    },
    fallback_key=("hip_gfx1151", "linear", "gguf_q8_0", "coltile8_rowbatch4_f32_f32_out"),
)
CANDIDATES["production_moe_restore"] = CandidateSpec(
    name="production_moe_restore",
    classification="T2",
    mechanism=(
        "WMMA-MoE27 prefill restored on layers 27-47 (iu8-WMMA gate/up plus the "
        "compact f16-WMMA down), replacing the exact grouped iu8-risk+repair "
        "route that production runs today. The 2026-09-14 arithmetic recovery "
        "zeroed this flag as part of a blanket sweep of "
        "PRODUCTION_ARITHMETIC_RECOVERY_FLAGS, while its own triggering candidate "
        "(production_q8_fallback) declared 'other production paths unchanged' "
        "and failed on TASK QUALITY rather than on this route. The route has its "
        "own passing acceptance from 2026-08-29 (hard_gates_passed true, "
        "eligible_for_automatic_admission true, mean KL 2.79e-4, max KL 5.98e-3, "
        "top-1 0.99111, determinism and state-repeat exact) and a measured "
        "-4.34%/-5.26% paired prefill at p508/p1012. The lever is the sparse "
        "exact repair this route omits: that family is 5.44% of the prefill "
        "window."
    ),
    environment={
        "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL": "1",
        "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1",
    },
    base_profile="production",
    scenario_id="qwen4exp-ud-q4-k-xl-moe-wmma-restore",
    candidate_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                   "selected_dual_wmma_iu8_prefill_bf16_bf16_out"),
    fallback_key=("hip_gfx1151", "moe_linear", "gguf_q4_k",
                  "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"),
    count_registered_dispatch=True,
)
CANDIDATES["router_warp_reduce"] = CandidateSpec(
    name="router_warp_reduce",
    classification="T1",
    mechanism=(
        "warp-per-token router logits reduction: one warp per token with a "
        "32-lane shuffle reduction and the expert row staged once per block "
        "in LDS, replacing the four-token tile's 256-thread LDS tree over a "
        "global weight row"
    ),
    environment={
        "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT": "f32_hidden_warp_reduce_dense_exact",
    },
    base_profile="production",
    scenario_id="qwen4exp-ud-q4-k-xl-router-warp-reduce",
    candidate_key=("hip_gfx1151", "router_logits", "f32",
                   "f32_hidden_warp_reduce_dense_exact"),
    fallback_key=("hip_gfx1151", "router_logits", "f32",
                  "f32_hidden_token_tile4_dense_exact"),
    count_registered_dispatch=True,
    # The launcher ABI is (hidden, weight, logits, tokens, hidden_size, num_rows).
    dispatch_shape_positions=(3, 4, 5),
)


CANDIDATES["decode_dp4a_linear"] = CandidateSpec(
    name="decode_dp4a_linear",
    classification="T1",
    mechanism=(
        "One-plane Q8_1 DP4A Q4 gate/up on the historical 43-layer scope, "
        "with the warp override disabled so the declared logical128/t64 "
        "kernel is the one measured. Production keeps strict Q4 decode after "
        "the long-context numerical failure; this post-binder candidate is "
        "the explicit route for reevaluating a repair."
    ),
    environment={
        "HIPENGINE_QWEN4_EXP_Q4_DP4A64": "1",
        "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_GATE_UP": "0",
    },
    base_profile="production",
    scenario_id="qwen4exp-ud-q4-k-xl-decode-dp4a-linear",
    candidate_key=("hip_gfx1151", "linear", "gguf_q4_k",
                   "selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out"),
    fallback_key=("hip_gfx1151", "linear", "gguf_q4_k",
                  "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out"),
    count_registered_dispatch=True,
)


class GateError(RuntimeError):
    """Raised when the profile gate cannot be evaluated safely."""


def batch_admission_eligibility(
    *,
    prompt_ids: Sequence[str],
    named_prompt_ids: Sequence[str],
    prompt_selection_unrestricted: bool,
    decode_steps: int,
    repeat_runs: int,
    free_tokens: int,
    free_runs: int,
    tracked_clean: bool,
    manifest_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Decide whether an AR batch run covers the normative admission scope.

    Eligibility is deliberately separate from numerical quality: a run can be
    clean, deterministic, and numerically strong at 8 steps and still be
    ineligible because it measured 162 rows instead of 450. The caller must not
    fold this into a protocol check.
    """

    reasons: list[str] = []
    binding = dict(manifest_binding or {})
    if not prompt_selection_unrestricted:
        reasons.append("prompt_selection_truncated")
    if tuple(str(prompt) for prompt in prompt_ids) != tuple(
        str(prompt) for prompt in named_prompt_ids
    ):
        reasons.append("prompt_set_not_named_fixture")
    if int(decode_steps) < BATCH_NORMATIVE_DECODE_STEPS:
        reasons.append(
            f"decode_steps_below_normative:{int(decode_steps)}"
            f"<{BATCH_NORMATIVE_DECODE_STEPS}"
        )
    if int(repeat_runs) < BATCH_NORMATIVE_REPEAT_RUNS:
        reasons.append(
            f"repeat_runs_below_normative:{int(repeat_runs)}"
            f"<{BATCH_NORMATIVE_REPEAT_RUNS}"
        )
    if int(free_tokens) < BATCH_NORMATIVE_FREE_TOKENS:
        reasons.append(
            f"free_tokens_below_normative:{int(free_tokens)}"
            f"<{BATCH_NORMATIVE_FREE_TOKENS}"
        )
    if int(free_runs) < BATCH_NORMATIVE_FREE_RUNS:
        reasons.append(
            f"free_runs_below_normative:{int(free_runs)}"
            f"<{BATCH_NORMATIVE_FREE_RUNS}"
        )
    if not tracked_clean:
        reasons.append("tracked_tree_not_clean")
    binding_valid = bool(binding.get("valid", binding.get("bound", False)))
    if not binding_valid:
        binding_reasons = tuple(str(reason) for reason in binding.get("reasons", ()))
        if binding_reasons:
            reasons.extend(f"batch_manifest:{reason}" for reason in binding_reasons)
        else:
            reasons.append("batch_manifest_not_bound")
    observed_rows = len(prompt_ids) * (int(decode_steps) + 1)
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "requirements": {
            "prompt_set": "full_named_fixture",
            "prompt_count": len(named_prompt_ids),
            "decode_steps": BATCH_NORMATIVE_DECODE_STEPS,
            "teacher_rows": BATCH_NORMATIVE_TEACHER_ROWS,
            "repeat_runs": BATCH_NORMATIVE_REPEAT_RUNS,
            "free_tokens": BATCH_NORMATIVE_FREE_TOKENS,
            "free_runs": BATCH_NORMATIVE_FREE_RUNS,
            "tracked_clean": True,
            "batch_manifest_bound": True,
        },
        "observed": {
            "prompt_count": len(prompt_ids),
            "decode_steps": int(decode_steps),
            "teacher_rows": observed_rows,
            "repeat_runs": int(repeat_runs),
            "free_tokens": int(free_tokens),
            "free_runs": int(free_runs),
            "tracked_clean": bool(tracked_clean),
            "batch_manifest_bound": binding_valid,
        },
        "manifest_binding": binding,
    }


def batch_admission_decision(
    *,
    eligibility: Mapping[str, Any],
    measurement_valid: bool,
    numerical_pass: bool,
    task_pass: bool,
    protocol_check: bool = False,
) -> dict[str, Any]:
    """Combine scope eligibility with the measurement, numerical, and task gates.

    ``measurement_valid`` normally already includes eligibility. It is passed
    separately so a caller that reports a valid measurement for a short run
    still cannot mark that run admitted.
    """

    reasons = list(eligibility.get("reasons", ()))
    if not measurement_valid:
        reasons.append("measurement_invalid")
    if not numerical_pass:
        reasons.append("numerical_gate_failed")
    if not task_pass:
        reasons.append("task_gate_requires_review")
    if protocol_check:
        reasons.append("protocol_check_only")
    return {
        "admitted": bool(
            eligibility.get("eligible")
            and measurement_valid
            and numerical_pass
            and task_pass
            and not protocol_check
        ),
        "reasons": reasons,
    }


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint16)
    return (values.astype(np.uint32) << 16).view(np.float32)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _state_summary(runner: Any) -> dict[str, Any]:
    snapshot = runner.snapshot()
    digest = hashlib.sha256()
    layout: dict[str, int] = {}
    finite = True
    for name, raw in sorted(snapshot.decode_state.buffers.items()):
        value = np.ascontiguousarray(raw, dtype=np.uint8)
        layout[str(name)] = int(value.nbytes)
        digest.update(str(name).encode())
        digest.update(b"\0")
        digest.update(value)
        if name == "residual":
            finite = finite and bool(np.isfinite(_bf16_to_f32(value.view(np.uint16))).all())
        else:
            finite = finite and bool(np.isfinite(value.view(np.float32)).all())
    ple = {
        str(layer): {
            "tokens": [int(token) for token in state.tokens],
            "next_position": int(state.next_position),
        }
        for layer, state in sorted(snapshot.ple_hash_states.items())
    }
    attention_positions = [
        [int(state.position_host[0]), int(state.context_host[0])]
        for state in runner.attention_states
    ]
    index_counts = [
        [int(state.count), int(state.pooled_count)] for state in runner.index_states
    ]
    metadata = {
        "position": int(snapshot.position),
        "ple": ple,
        "attention_positions": attention_positions,
        "index_counts": index_counts,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    return {
        "state_sha256": digest.hexdigest(),
        "layout_sha256": _hash_json(layout),
        "buffer_bytes": layout,
        "finite": bool(finite),
        **metadata,
    }


def _state_repeat_gate(
    strict_by_prompt: Sequence[Mapping[str, Any]],
    candidate_by_prompt: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if len(strict_by_prompt) != len(candidate_by_prompt):
        raise ValueError("strict and candidate state prompt counts differ")
    prompts: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for strict, candidate_values in zip(
        strict_by_prompt, candidate_by_prompt, strict=True
    ):
        runs = tuple(candidate_values)
        if len(runs) < 3:
            raise ValueError("state repeat gate requires at least three candidate runs")
        first = runs[0]
        finite = bool(strict["finite"]) and all(bool(run["finite"]) for run in runs)
        layout_exact = all(
            run["layout_sha256"] == strict["layout_sha256"] for run in runs
        )
        metadata_exact = all(
            run["position"] == strict["position"]
            and run["attention_positions"] == strict["attention_positions"]
            and run["index_counts"] == strict["index_counts"]
            for run in runs
        )
        repeat_exact = all(
            run["state_sha256"] == first["state_sha256"] for run in runs[1:]
        )
        prompt_id = str(strict["prompt_id"])
        result = {
            "prompt_id": prompt_id,
            "finite": finite,
            "layout_exact": layout_exact,
            "metadata_exact": metadata_exact,
            "repeat_exact": repeat_exact,
            "strict_state_sha256": strict["state_sha256"],
            "candidate_state_sha256": [run["state_sha256"] for run in runs],
            "strict_candidate_state_exact": (
                strict["state_sha256"] == first["state_sha256"]
            ),
        }
        prompts.append(result)
        if not (finite and layout_exact and metadata_exact and repeat_exact):
            mismatches.append(result)
    return {"passed": not mismatches, "mismatches": mismatches, "prompts": prompts}


def _task_gate(
    strict: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    categories: Mapping[str, str],
) -> dict[str, Any]:
    if set(strict) != set(candidate) or set(strict) != set(categories):
        raise ValueError("task result prompt sets differ")
    divergences: list[dict[str, Any]] = []
    repeat_mismatches: list[str] = []
    rows: list[dict[str, Any]] = []
    for prompt_id in strict:
        runs = tuple(candidate[prompt_id])
        if len(runs) < 2:
            raise ValueError("task gate requires at least two candidate runs")
        first = runs[0]
        repeat_exact = all(
            run["ids"] == first["ids"] and run["text"] == first["text"]
            for run in runs[1:]
        )
        strict_exact = (
            strict[prompt_id]["ids"] == first["ids"]
            and strict[prompt_id]["text"] == first["text"]
        )
        if not repeat_exact:
            repeat_mismatches.append(prompt_id)
        if not strict_exact:
            divergences.append(
                {
                    "id": prompt_id,
                    "category": categories[prompt_id],
                    "strict_text": strict[prompt_id]["text"],
                    "candidate_text": first["text"],
                    "strict_ids_sha256": strict[prompt_id]["ids_sha256"],
                    "candidate_ids_sha256": first["ids_sha256"],
                }
            )
        rows.append(
            {
                "id": prompt_id,
                "category": categories[prompt_id],
                "candidate_repeat_exact": repeat_exact,
                "strict_exact": strict_exact,
                "strict_ids_sha256": strict[prompt_id]["ids_sha256"],
                "candidate_ids_sha256": first["ids_sha256"],
            }
        )
    repeat_pass = not repeat_mismatches
    status = (
        "failed_nondeterminism"
        if not repeat_pass
        else "requires_review"
        if divergences
        else "passed_exact"
    )
    return {
        "status": status,
        "candidate_repeat_exact": repeat_pass,
        "repeat_mismatches": repeat_mismatches,
        "strict_exact_count": len(rows) - len(divergences),
        "total": len(rows),
        "divergences": divergences,
        "rows": rows,
    }


def _ids_sha256(ids: Sequence[int]) -> str:
    return token_ids_sha256(int(value) for value in ids)


def _free_trajectory(
    runner: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    steps: int,
    *,
    compact_output: bool = False,
) -> dict[str, Any]:
    runner.reset()
    prefill_kwargs = (
        {"capture_logits": False, "capture_target_hidden": False}
        if compact_output else {}
    )
    step_kwargs = (
        {
            "capture_logits": False,
            "capture_target_hidden": False,
            "token_id_resident": True,
        }
        if compact_output else {}
    )
    result = runner.prefill([int(token) for token in prompt_ids], **prefill_kwargs)
    ids: list[int] = []
    for index in range(int(steps)):
        token = int(result.token_id)
        ids.append(token)
        if index + 1 < steps:
            result = runner.step(token, **step_kwargs)
    return {
        "ids": ids,
        "ids_sha256": _ids_sha256(ids),
        "text": str(tokenizer.decode(ids, skip_special=False)),
        "state": _state_summary(runner),
    }


def _strict_trajectory(runner: Any, prompt_ids: Sequence[int], decode_steps: int) -> tuple[dict[str, Any], ...]:
    runner.reset()
    result = runner.prefill([int(token) for token in prompt_ids])
    trajectory = [
        {
            "token_id": int(result.token_id),
            "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
        }
    ]
    current = int(result.token_id)
    for _ in range(int(decode_steps)):
        result = runner.step(current)
        current = int(result.token_id)
        trajectory.append(
            {
                "token_id": current,
                "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
            }
        )
    return tuple(trajectory)


def _candidate_trajectory(
    runner: Any,
    prompt_ids: Sequence[int],
    forced_input_ids: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    runner.reset()
    result = runner.prefill([int(token) for token in prompt_ids])
    trajectory = [
        {
            "token_id": int(result.token_id),
            "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
        }
    ]
    for token in forced_input_ids:
        result = runner.step(int(token))
        trajectory.append(
            {
                "token_id": int(result.token_id),
                "logits": np.ascontiguousarray(result.logits, dtype=np.float32),
            }
        )
    return tuple(trajectory)


def _make_generator(args: argparse.Namespace, profile: str, quant: str | None = None):
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
    )
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    plugin = resolve_model(index.architecture or "")
    requested = ExecutionProfile(profile)
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=quant or QWEN4_EXP_QUANTS[1],
        profile=requested,
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=args.model_root,
            weight_index=index,
            model_plugin=plugin,
            backend=QWEN4_EXP_BACKEND,
            max_sequence_length=int(args.max_sequence_length),
            prefill_chunk_size=int(args.prefill_chunk_size),
        )

    return resolved.construct_generator(factory), resolved, index


def _run_batch_candidate(
    args: argparse.Namespace,
    prompt_rows: Sequence[Mapping[str, Any]],
    prompt_tokens: Mapping[str, Sequence[int]],
    strict_trajectories: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_generator: Any,
    *,
    candidate_mode: str,
    width: int,
    source_revision: str,
) -> tuple[
    dict[str, list[tuple[Mapping[str, Any], ...]]],
    dict[str, list[Mapping[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    """Drive the AR batch adapter for every prompt, group, and repeat."""

    from scripts.qwen4exp_ar_batch_candidate import (
        BatchCandidateRequest,
        Qwen4ExpARBatchCandidate,
        plan_batch_composition,
    )

    requests = []
    for index, row in enumerate(prompt_rows, 1):
        prompt_id = str(row["id"])
        forced = tuple(
            int(step["token_id"]) for step in strict_trajectories[prompt_id][:-1]
        )
        requests.append(
            BatchCandidateRequest(
                prompt_id=prompt_id,
                request_id=index,
                prompt_ids=tuple(int(token) for token in prompt_tokens[prompt_id]),
                forced_ids=forced,
            )
        )
    groups = plan_batch_composition(requests, width)
    adapter = Qwen4ExpARBatchCandidate(
        candidate_generator, width=width, label=candidate_mode
    )
    runs: dict[str, list[tuple[Mapping[str, Any], ...]]] = {
        request.prompt_id: [] for request in requests
    }
    states: dict[str, list[Mapping[str, Any]]] = {
        request.prompt_id: [] for request in requests
    }
    tasks: dict[str, list[dict[str, Any]]] = {
        request.prompt_id: [] for request in requests
    }
    try:
        adapter.open()
        for _ in range(int(args.repeat_runs)):
            for group in groups:
                result = adapter.run_group(group)
                for prompt_id, trajectory in result.trajectories.items():
                    runs[prompt_id].append(tuple(trajectory))
                for prompt_id, state in result.states.items():
                    states[prompt_id].append(state)
                print(
                    f"candidate batch group {group.index} width {group.width} "
                    f"{list(group.prompt_ids)}",
                    flush=True,
                )
        for _ in range(int(args.free_runs)):
            for group in groups:
                free = adapter.run_free_group(group, steps=int(args.free_tokens))
                for prompt_id, task in free.items():
                    tasks[prompt_id].append(task)
        adapter.build_variant_manifest(source_revision=source_revision)
    finally:
        adapter.close()
    return runs, states, tasks, adapter.evidence


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    if not args.model_root.is_dir():
        raise GateError(f"model root does not exist: {args.model_root}")
    source = _git_metadata(ROOT)
    tracked_clean = bool(source["tracked_clean"])
    protocol_check = bool(getattr(args, "protocol_check", False))
    if not tracked_clean and not protocol_check:
        raise GateError(
            "profile qualification requires a clean committed tracked tree; "
            "use --protocol-check only to validate an uncommitted adapter"
        )
    if args.decode_steps <= 0 or args.repeat_runs < 3 or args.free_runs < 2:
        raise GateError("decode steps must be positive; candidate repeats >=3; free runs >=2")
    prompt_rows = _load_suites(args.prompts)
    limit_per_category = getattr(args, "limit_per_category", None)
    if args.limit is not None and limit_per_category is not None:
        raise GateError("--limit and --limit-per-category are mutually exclusive")
    if limit_per_category is not None:
        if int(limit_per_category) <= 0:
            raise GateError("--limit-per-category must be positive")
        selected = []
        taken: dict[str, int] = {}
        for row in prompt_rows:
            category = str(row["category"])
            if taken.get(category, 0) < int(limit_per_category):
                taken[category] = taken.get(category, 0) + 1
                selected.append(row)
        prompt_rows = selected
    elif args.limit is not None:
        prompt_rows = prompt_rows[: max(0, int(args.limit))]
    if not prompt_rows:
        raise GateError("selected prompt suite is empty")
    complete_suite = args.limit is None and limit_per_category is None
    candidate_spec = CANDIDATES[str(args.candidate)]
    if candidate_spec.compact_output and args.free_runs < 3:
        raise GateError("compact-output candidates require at least three free runs")
    batch_candidate = candidate_spec.name == "qwen4_exp_ar_batch"
    batch_width = getattr(args, "batch_width", None)
    if protocol_check and not batch_candidate:
        raise GateError("--protocol-check is only valid with --candidate ar_batch")
    if batch_candidate:
        if batch_width not in (1, 2, 4, 8):
            raise GateError(
                "--candidate ar_batch requires --batch-width 1, 2, 4, or 8"
            )
    elif batch_width is not None:
        raise GateError("--batch-width is only valid with --candidate ar_batch")
    candidate_mode = (
        f"{candidate_spec.name}_w{int(batch_width)}"
        if batch_candidate
        else candidate_spec.name
    )
    named_prompt_ids = (
        tuple(str(row["id"]) for row in _load_suites(list(DEFAULT_PROMPTS)))
        if batch_candidate
        else ()
    )

    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
        register_gguf_q4_k_selected_prefill_kernels,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve

    register_gfx1151_kernels(replace=True)
    # The selected Q4_K MoE prefill owners live in the gfx1100 quant package and
    # are registered lazily by the runner. The gate resolves its candidate and
    # fallback keys before any generator exists, so register them here too.
    register_gguf_q4_k_selected_prefill_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    candidate_kernel = None if candidate_spec.candidate_key is None else resolve(
        backend=candidate_spec.candidate_key[0],
        layer=candidate_spec.candidate_key[1],
        quant=candidate_spec.candidate_key[2],
        variant=candidate_spec.candidate_key[3],
    )
    strict_fallback = None if candidate_spec.fallback_key is None else resolve(
        backend=candidate_spec.fallback_key[0],
        layer=candidate_spec.fallback_key[1],
        quant=candidate_spec.fallback_key[2],
        variant=candidate_spec.fallback_key[3],
    )

    prompt_tokens: dict[str, list[int]] = {}
    strict_trajectories: dict[str, tuple[Mapping[str, Any], ...]] = {}
    strict_states: list[dict[str, Any]] = []
    strict_tasks: dict[str, dict[str, Any]] = {}
    strict_compact_state_runs: dict[str, list[dict[str, Any]]] = {}
    prompt_manifest: list[dict[str, Any]] = []

    reset_memory_stats()
    strict_generator, strict_profile, index = _make_generator(args, "strict")
    try:
        for number, row in enumerate(prompt_rows, 1):
            prompt_id = str(row["id"])
            tokens = build_chat_prompt(strict_generator.tokenizer, str(row["prompt"]))
            prompt_tokens[prompt_id] = [int(token) for token in tokens]
            strict = _strict_trajectory(
                strict_generator.runner,
                prompt_tokens[prompt_id],
                int(args.decode_steps),
            )
            state = _state_summary(strict_generator.runner)
            state["prompt_id"] = prompt_id
            strict_trajectories[prompt_id] = strict
            strict_states.append(state)
            strict_tasks[prompt_id] = _free_trajectory(
                strict_generator.runner,
                strict_generator.tokenizer,
                prompt_tokens[prompt_id],
                int(args.free_tokens),
            )
            if candidate_spec.compact_output:
                strict_compact_state_runs[prompt_id] = [
                    dict(strict_tasks[prompt_id]["state"])
                ]
                for _ in range(1, int(args.free_runs)):
                    repeat = _free_trajectory(
                        strict_generator.runner,
                        strict_generator.tokenizer,
                        prompt_tokens[prompt_id],
                        int(args.free_tokens),
                    )
                    strict_compact_state_runs[prompt_id].append(
                        dict(repeat["state"])
                    )
            prompt_manifest.append(
                {
                    "id": prompt_id,
                    "category": str(row["category"]),
                    "suite": str(row["suite"]),
                    "prompt_sha256": prompt_sha256(str(row["prompt"])),
                    "prompt_tokens": len(tokens),
                    "prompt_token_ids_sha256": _ids_sha256(tokens),
                }
            )
            print(f"strict {number}/{len(prompt_rows)} {prompt_id}", flush=True)
    finally:
        strict_generator.close()
    strict_after_close = memory_stats()

    candidate_runs: dict[str, list[tuple[Mapping[str, Any], ...]]] = {}
    candidate_states: list[list[dict[str, Any]]] = []
    candidate_tasks: dict[str, list[dict[str, Any]]] = {}
    reset_memory_stats()
    candidate_generator, candidate_profile, _ = _make_generator(
        args, candidate_spec.base_profile, candidate_spec.quant
    )
    batch_manifest_binding_value = None
    bound_environment: dict[str, str | None] = {}
    dispatch_stack = ExitStack()
    if candidate_spec.qsa_h256_prefill_mode is not None:
        dispatch_stack.enter_context(qsa_h256_prefill_mode(
            candidate_generator.runner, candidate_spec.qsa_h256_prefill_mode))
    if candidate_spec.qsa_prefill_head_mode is not None:
        # Request-local state, so the arm is bound around the measured rows and
        # the profile's own decision comes back when the stack unwinds.
        dispatch_stack.enter_context(qsa_prefill_head_mode(
            candidate_generator.runner, candidate_spec.qsa_prefill_head_mode))
    if candidate_spec.gdn_register_prefill is not None:
        dispatch_stack.enter_context(gdn_register_prefill(
            candidate_generator.runner, candidate_spec.gdn_register_prefill))
    if candidate_spec.gdn_prefill_variant is not None:
        # Same for the GDN serial owner: the arm names its leaf explicitly and
        # the base profile's published variant is restored on exit.
        dispatch_stack.enter_context(gdn_prefill_variant(
            candidate_generator.runner, candidate_spec.gdn_prefill_variant))
    if candidate_spec.qsa_ordered_decode_v2 is not None:
        # And for the ordered-v2 sparse decode leaf: request-local state, so the
        # arm binds it and the profile's own decision comes back on unwind.
        dispatch_stack.enter_context(qsa_ordered_decode_v2(
            candidate_generator.runner, candidate_spec.qsa_ordered_decode_v2))
    if candidate_spec.gdn_peer_prefill is not None:
        # The GDN peer owner's layer scope is request-local too, so an arm that
        # restores or closes it binds the scope on this runner and the base
        # profile's published scope comes back on unwind.
        dispatch_stack.enter_context(gdn_peer_prefill(
            candidate_generator.runner, candidate_spec.gdn_peer_prefill))
    if candidate_spec.q51_exact_grouped_down is not None:
        # And for the exact grouped Q5_1 down chain gate: request-local state, so
        # an arm that needs the owners below the chain binds it off for the
        # measured rows and the profile's own decision comes back on unwind.
        dispatch_stack.enter_context(q51_exact_grouped_down(
            candidate_generator.runner,
            candidate_spec.q51_exact_grouped_down))
    if candidate_spec.q51_wmma_down is not None:
        # And for the F16 WMMA down owner's admission: request-local state, so
        # an arm that measures the down-only route admits it for the measured
        # rows and the profile's own decision comes back on unwind.
        dispatch_stack.enter_context(q51_wmma_down(
            candidate_generator.runner, candidate_spec.q51_wmma_down))
    if candidate_spec.q51_out8 is not None:
        # And for that chain's out8 fallback rung: same protocol, so an arm that
        # measures the output1 base binds it and the published out8 owner comes
        # back on unwind.
        dispatch_stack.enter_context(q51_out8(
            candidate_generator.runner, candidate_spec.q51_out8))
    if candidate_spec.q4_out4 is not None:
        # And for the Q4 gate/up ladder's out4 fallback rung: same protocol, so
        # an arm that measures the strict rowbatch8 base binds it and the
        # published out4 owner comes back on unwind.
        dispatch_stack.enter_context(q4_out4(
            candidate_generator.runner, candidate_spec.q4_out4))
    if candidate_spec.q4_exact_grouped_gate is not None:
        # And for the Q4_K exact grouped gate/up chain gate: request-local state,
        # so an arm that measures the plain-iu8 or F16 compact owner below the
        # chain binds it off for the measured rows and the profile's own
        # publication comes back on unwind.
        dispatch_stack.enter_context(q4_exact_grouped_gate(
            candidate_generator.runner, candidate_spec.q4_exact_grouped_gate))
    if candidate_spec.q4_exact_grouped_all is not None:
        # And for that chain's `_ALL` joint term, which can only narrow a chain
        # the gate already opened.
        dispatch_stack.enter_context(q4_exact_grouped_all(
            candidate_generator.runner, candidate_spec.q4_exact_grouped_all))
    counter = {"calls": 0, "shapes": {}}
    batch_evidence = None
    batch_runs: dict[str, list[tuple[Mapping[str, Any], ...]]] = {}
    batch_states: dict[str, list[Mapping[str, Any]]] = {}
    batch_tasks: dict[str, list[dict[str, Any]]] = {}
    try:
        if batch_candidate:
            (
                batch_runs,
                batch_states,
                batch_tasks,
                batch_evidence,
            ) = _run_batch_candidate(
                args,
                prompt_rows,
                prompt_tokens,
                strict_trajectories,
                candidate_generator,
                candidate_mode=candidate_mode,
                width=int(batch_width),
                source_revision=str(source["head"] or ""),
            )
        else:
            bound_environment = {
                key: os.environ.get(key) for key in candidate_spec.environment
            }
            os.environ.update(candidate_spec.environment)
            candidate_runs: dict[str, list[tuple[Mapping[str, Any], ...]]] = {}
            candidate_states: list[list[dict[str, Any]]] = []
            candidate_tasks: dict[str, list[dict[str, Any]]] = {}
            if candidate_spec.environment.get("HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL") == "1":
                candidate_generator.runner.configure_mmq_prefill_resources()
            if candidate_spec.requires_dispatch_count:
                from hipengine.kernels.registry import KernelKey
                counter = dispatch_stack.enter_context(count_candidate_dispatch(
                    key=KernelKey(*candidate_spec.candidate_key) if candidate_spec.candidate_key is not None else None,
                    direct_target=candidate_spec.direct_dispatch_target,
                    direct_reference=candidate_spec.direct_dispatch_reference,
                    shape_positions=candidate_spec.dispatch_shape_positions))
            for number, row in enumerate(prompt_rows, 1):
                prompt_id = str(row["id"])
                strict = strict_trajectories[prompt_id]
                forced = [int(step["token_id"]) for step in strict[:-1]]
                runs: list[tuple[Mapping[str, Any], ...]] = []
                states: list[dict[str, Any]] = []
                for _ in range(int(args.repeat_runs)):
                    run = _candidate_trajectory(
                        candidate_generator.runner,
                        prompt_tokens[prompt_id],
                        forced,
                    )
                    state = _state_summary(candidate_generator.runner)
                    state["prompt_id"] = prompt_id
                    runs.append(run)
                    states.append(state)
                candidate_runs[prompt_id] = runs
                candidate_states.append(states)
                candidate_tasks[prompt_id] = [
                    _free_trajectory(
                        candidate_generator.runner,
                        candidate_generator.tokenizer,
                        prompt_tokens[prompt_id],
                        int(args.free_tokens),
                        compact_output=candidate_spec.compact_output,
                    )
                    for _ in range(int(args.free_runs))
                ]
                print(
                    f"candidate {number}/{len(prompt_rows)} {prompt_id}: "
                    f"{args.repeat_runs} numerical + {args.free_runs} free",
                    flush=True,
                )
    finally:
        for key, value in bound_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            candidate_generator.close()
        finally:
            dispatch_stack.close()
    candidate_after_close = memory_stats()

    if batch_candidate:
        from hipengine.runtime.qwen4_exp_batch_manifest import (
            validate_batch_variant_manifest,
        )

        batch_manifest_binding_value = validate_batch_variant_manifest(
            batch_evidence.get("variant_manifest"),
            expected_profile_manifest_sha256=candidate_profile.manifest_sha256,
            expected_source_revision=str(source["head"] or ""),
            expected_backend=candidate_profile.manifest["backend"],
            expected_model=candidate_profile.manifest["model"],
            expected_quant=candidate_profile.manifest["quant"],
        )
        batch_evidence = {
            **batch_evidence,
            "manifest_binding": batch_manifest_binding_value,
        }
        candidate_runs = batch_runs
        candidate_states = [
            list(batch_states[str(row["id"])]) for row in prompt_rows
        ]
        candidate_tasks = batch_tasks
    captures = tuple(
        PromptCalibrationCapture(
            prompt_id=str(row["id"]),
            category=str(row["category"]),
            strict=strict_trajectories[str(row["id"])],
            candidate_runs={
                candidate_mode: tuple(candidate_runs[str(row["id"])])
            },
        )
        for row in prompt_rows
    )
    thresholds = EvaluationThresholds()
    if batch_candidate:
        from scripts.qwen4exp_ar_batch_candidate import batch_state_gate

        # The final composition chunk keeps its natural width, so a row's scope
        # names the width it actually ran at, not the configured maximum.
        width_by_prompt: dict[str, int] = {}
        for group in batch_evidence["groups"]:
            for prompt_id in group["prompt_ids"]:
                width_by_prompt[str(prompt_id)] = int(group["width"])
        quality = build_candidate_quality(
            captures,
            candidate_mode=candidate_mode,
            scenario_id=candidate_spec.scenario_id,
            thresholds=thresholds,
            shape_for_step=lambda step, prompt_id: (
                "prefill_last"
                if step == 0
                else f"batch_w{width_by_prompt[str(prompt_id)]}"
            ),
            transition_for_step=lambda step, prompt_id: (
                "prefill_to_c1"
                if step == 0
                else "prefill_to_batch"
                if step == 1
                else "steady"
            ),
        )
        state_gate = batch_state_gate(
            strict_states,
            candidate_states,
            groups=batch_evidence["groups"],
        )
    else:
        quality = build_candidate_quality(
            captures,
            candidate_mode=candidate_mode,
            scenario_id=candidate_spec.scenario_id,
            thresholds=thresholds,
        )
        state_gate = _state_repeat_gate(strict_states, candidate_states)
    compact_state_gate = None
    if candidate_spec.compact_output:
        strict_task_states = []
        candidate_task_states = []
        for row in prompt_rows:
            prompt_id = str(row["id"])
            strict_state = dict(strict_tasks[prompt_id]["state"])
            strict_state["prompt_id"] = prompt_id
            strict_task_states.append(strict_state)
            runs = []
            for task in candidate_tasks[prompt_id]:
                run_state = dict(task["state"])
                run_state["prompt_id"] = prompt_id
                runs.append(run_state)
            candidate_task_states.append(runs)
        compact_state_gate = _state_repeat_gate(
            strict_task_states, candidate_task_states
        )
        strict_repeat_mismatches = []
        for prompt_id, runs in strict_compact_state_runs.items():
            hashes = [str(run["state_sha256"]) for run in runs]
            if len(set(hashes)) != 1:
                strict_repeat_mismatches.append(
                    {"prompt_id": prompt_id, "state_sha256": hashes}
                )
        compact_state_gate["strict_repeat_mismatches"] = strict_repeat_mismatches
        compact_state_gate["strict_repeat_exact"] = not strict_repeat_mismatches
        compact_state_gate["all_strict_candidate_state_exact"] = all(
            bool(prompt["strict_candidate_state_exact"])
            for prompt in compact_state_gate["prompts"]
        )
        compact_state_gate["passed"] = bool(
            compact_state_gate["passed"]
            and compact_state_gate["strict_repeat_exact"]
            and compact_state_gate["all_strict_candidate_state_exact"]
        )
    categories = {str(row["id"]): str(row["category"]) for row in prompt_rows}
    task_gate = _task_gate(strict_tasks, candidate_tasks, categories=categories)
    teardown = bool(
        strict_after_close["current_allocated_bytes"] == 0
        and candidate_after_close["current_allocated_bytes"] == 0
    )
    numerical_pass = bool(
        quality["quality"]["hard_gates_passed"]
        and quality["repeat_determinism"]["passed"]
    )
    task_pass = task_gate["status"] == "passed_exact"
    admission = None
    admission_decision = None
    if batch_candidate:
        admission = batch_admission_eligibility(
            prompt_ids=tuple(str(row["id"]) for row in prompt_rows),
            named_prompt_ids=named_prompt_ids,
            prompt_selection_unrestricted=bool(complete_suite),
            decode_steps=int(args.decode_steps),
            repeat_runs=int(args.repeat_runs),
            free_tokens=int(args.free_tokens),
            free_runs=int(args.free_runs),
            tracked_clean=tracked_clean,
            manifest_binding=batch_manifest_binding_value,
        )
    measurement_valid = bool(
        complete_suite
        and tracked_clean
        and teardown
        and state_gate["passed"]
        and (compact_state_gate is None or compact_state_gate["passed"])
        and (not candidate_spec.requires_dispatch_count or counter["calls"] > 0)
        and (admission is None or admission["eligible"])
    )
    passed = bool(measurement_valid and numerical_pass and task_pass)
    if batch_candidate:
        admission_decision = batch_admission_decision(
            eligibility=admission,
            measurement_valid=measurement_valid,
            numerical_pass=numerical_pass,
            task_pass=task_pass,
            protocol_check=protocol_check,
        )
        passed = bool(admission_decision["admitted"])
    protocol_ok = None
    if batch_candidate:
        protocol_ok = bool(
            state_gate["passed"]
            and teardown
            and quality["repeat_determinism"]["passed"]
            and quality["quality"]["finite"]
            and batch_evidence is not None
            and batch_evidence["failure"] is None
            and all(bool(group["state_distinct"]) for group in batch_evidence["groups"])
            and bool(batch_evidence["step_widths"])
        )
    if protocol_check:
        status = "protocol_check" if protocol_ok else "failed"
    elif batch_candidate and admission is not None and not admission["eligible"]:
        status = "ineligible"
    else:
        status = (
            "passed"
            if passed
            else "requires_task_review"
            if measurement_valid and numerical_pass and task_gate["status"] == "requires_review"
            else "failed"
        )
    return {
        "schema": 1,
        "kind": KIND,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "measurement_valid": measurement_valid,
        "protocol_check": protocol_check,
        "batch_candidate": batch_evidence,
        "batch_variant_manifest_sha256": (
            batch_manifest_binding_value.get("manifest_sha256")
            if batch_manifest_binding_value is not None
            else None
        ),
        "admission": admission,
        "candidate_dispatch_calls": counter["calls"] if candidate_spec.requires_dispatch_count else None,
        "candidate_dispatch_shapes": shape_records(counter),
        "source": source,
        "host": _host_metadata(),
        "command": list(command),
        "model": {
            "root": str(args.model_root.resolve()),
            "architecture": index.architecture,
        },
        "candidate": {
            "name": candidate_spec.name,
            "candidate_id": str(args.candidate),
            "classification": candidate_spec.classification,
            "mechanism": candidate_spec.mechanism,
            "environment": dict(candidate_spec.environment),
            "base_profile": candidate_spec.base_profile,
            "bound_environment": bound_environment,
            "direct_dispatch_target": candidate_spec.direct_dispatch_target,
            "direct_dispatch_reference": candidate_spec.direct_dispatch_reference,
            "dispatch_shape_positions": candidate_spec.dispatch_shape_positions,
            "candidate_kernel": (candidate_spec.direct_dispatch_target[1]
                                 if candidate_spec.direct_dispatch_target is not None
                                 else getattr(candidate_kernel, "__name__", None)),
            "registered_candidate_callable": getattr(candidate_kernel, "__qualname__", None),
            "dispatch_count_mode": ("direct_alias" if candidate_spec.direct_dispatch_target
                                    else "registry" if candidate_spec.count_registered_dispatch
                                    else "not_counted"),
            "strict_fallback": getattr(strict_fallback, "__name__", None),
            "candidate_kernel_registered": (
                candidate_kernel is not None and candidate_spec.direct_dispatch_target is None),
            "related_registry_variant_available": candidate_kernel is not None,
            "strict_fallback_registered": strict_fallback is not None,
            "batch_width": int(batch_width) if batch_candidate else None,
        },
        "profiles": {
            "strict_manifest": _json_value(strict_profile.manifest),
            "strict_manifest_sha256": strict_profile.manifest_sha256,
            "candidate_base_profile": candidate_spec.base_profile,
            "candidate_manifest": _json_value(candidate_profile.manifest),
            "candidate_manifest_sha256": candidate_profile.manifest_sha256,
            "candidate_named_profile_intact": not bool(candidate_spec.environment),
        },
        "protocol": {
            "prompt_suites": [str(path.resolve()) for path in args.prompts],
            "prompt_suite_sha256": {
                str(path.resolve()): sha256_path(path.resolve()) for path in args.prompts
            },
            "complete_prompt_and_heldout_suite": complete_suite,
            "prompt_count": len(prompt_rows),
            "decode_steps": int(args.decode_steps),
            "teacher_forced_rows": sum(len(capture.strict) for capture in captures),
            "candidate_repeat_runs": int(args.repeat_runs),
            "free_tokens": int(args.free_tokens),
            "candidate_free_runs": int(args.free_runs),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "kv": "BF16",
            "thresholds": thresholds.to_dict(),
        },
        "prompts": prompt_manifest,
        "quality": quality,
        "state_repeat_gate": state_gate,
        "compact_output_state_repeat_gate": compact_state_gate,
        "task_gate": task_gate,
        "lifecycle": {
            "strict_after_close": strict_after_close,
            "candidate_after_close": candidate_after_close,
            "passed": teardown,
        },
        "decision": {
            "passed": passed,
            "admitted": passed,
            "numerical_passed": numerical_pass,
            "state_passed": state_gate["passed"],
            "task_passed": task_pass,
            "task_status": task_gate["status"],
            "lifecycle_passed": teardown,
            "protocol_check_passed": protocol_ok,
            "admission_eligible": admission["eligible"] if admission else None,
            "admission_reasons": admission["reasons"] if admission else [],
            "admission_decision": admission_decision,
            "next": (
                "protocol validation only; admission requires a clean committed "
                "tree and the complete suite"
                if protocol_check
                else "admission blocked: " + "; ".join(admission["reasons"])
                if batch_candidate
                and admission is not None
                and not admission["eligible"]
                else "complete prompt and heldout suite required"
                if not complete_suite
                else "task review required for deterministic cross-route divergences"
                if status == "requires_task_review"
                else "eligible for manifest/c2/canonical-depth promotion work"
                if status == "passed"
                else "reject candidate and keep strict fallback"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=tuple(CANDIDATES),
        default="production_baseline",
    )
    parser.add_argument("--prompts", action="append", type=Path, default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument("--batch-width", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--protocol-check", action="store_true")
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--free-tokens", type=int, default=32)
    parser.add_argument("--free-runs", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=2051)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--hip-arch", default="gfx1151")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prompts is None:
        args.prompts = list(DEFAULT_PROMPTS)
    os.environ.setdefault("HIPENGINE_HIP_ARCH", args.hip_arch)
    if args.compiler_version_file is not None:
        os.environ.setdefault(
            "HIPENGINE_COMPILER_VERSION_FILE",
            str(args.compiler_version_file.resolve()),
        )
    if args.require_cached_build:
        os.environ.setdefault("HIPENGINE_REQUIRE_CACHED_BUILD", "1")
    payload = run(args, command=[str(Path(sys.argv[0]).name), *sys.argv[1:]])
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return (
        0
        if payload["status"] in ("passed", "protocol_check")
        else 2
        if payload["status"] == "requires_task_review"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
