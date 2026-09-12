"""Frozen Laguna source policy for the WPF source-default contracts.

WPF-H8A, WPF-H8B, and WPF-H7U each promote one ``LAGUNA_*`` capability and must
prove the promotion did not move any *other* Laguna policy in
``hipengine/kernels/hip_gfx1100/__init__.py``. That file is a shared registry
module which every Qwen3.8, Qwen4Exp, SPECDEC2, and execution-profile campaign
edits, so the whole-file hash these tests started with kept failing on changes
that had no bearing on the Laguna contract. Three separate re-baselines of it
(the `_NORMALIZED_PACKAGE_SHA256` constants in the H8A and H8B tests and the
`_POST_MERGE_PACKAGE_SHA256` constants in the H7U tests) are already in the
tree. This module pins the policy itself: the source text of every module-level
``LAGUNA_*`` assignment, minus the one capability each test owns. A failure
names what moved.

Refresh after an audited Laguna policy change::

    .venv/bin/python tests/_laguna_policy_pin.py --refresh
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

PACKAGE = (
    Path(__file__).resolve().parents[1] / "hipengine/kernels/hip_gfx1100/__init__.py"
)

# BEGIN LAGUNA_POLICY
LAGUNA_POLICY: dict[str, str] = {
    "LAGUNA_ACTIVATION_PACK_REUSE": 'LAGUNA_ACTIVATION_PACK_REUSE = True',
    "LAGUNA_GLOBAL_SPLIT_MIN_LIVE": 'LAGUNA_GLOBAL_SPLIT_MIN_LIVE = 127',
    "LAGUNA_GROUPED_GATE_UP_ROLE_VARIANTS": """LAGUNA_GROUPED_GATE_UP_ROLE_VARIANTS = {
    "layer47_iq3_k3072_n1024_e256": _H6C_IQ3_GATE_UP_VARIANT
}""",
    "LAGUNA_GROUPED_GATE_UP_VARIANT_ABIS": """LAGUNA_GROUPED_GATE_UP_VARIANT_ABIS = {
    _H6C_IQ3_GATE_UP_VARIANT: "grouped_raw_iq_dual_silu"
}""",
    "LAGUNA_GROUPED_IQ_DOWN_VARIANTS": """LAGUNA_GROUPED_IQ_DOWN_VARIANTS = {
    "gguf_iq3_xxs": _H6T_IQ3_FUSED_DPP_ADD_VARIANT,
    "gguf_iq4_xs": (
        "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out"
    ),
}""",
    "LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS": """LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS = {
    _H5Q_IQ3_ACTIVE_EXPERT_VARIANT: "grouped_raw_iq_active_experts",
    _H5Z_IQ3_ACTIVATION_RESIDENT_VARIANT: "grouped_raw_iq_active_experts",
    _H6D_IQ3_ROW_INTERLEAVED_VOPD_VARIANT: "grouped_raw_iq_active_experts",
    _H6F_IQ3_PAIRED_OUTPUT_VARIANT: "grouped_raw_iq_active_experts",
    _H6I_IQ3_TRIPLE_OUTPUT_VARIANT: "grouped_raw_iq_active_experts",
    _H6P_IQ3_STAGED_WAVE_PUBLICATION_VARIANT: (
        "grouped_raw_iq_active_experts"
    ),
    _H6Q_IQ3_COMPACT_SHUFFLE_LOOP_VARIANT: (
        "grouped_raw_iq_active_experts"
    ),
    _H6R_IQ3_DPP_PEER_EXCHANGE_VARIANT: (
        "grouped_raw_iq_active_experts"
    ),
    _H6T_IQ3_FUSED_DPP_ADD_VARIANT: "grouped_raw_iq_active_experts",
}""",
    "LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANTS": """LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANTS = {
    "gguf_iq2_xs": _H6L_IQ2_PAIR16_ROWBATCH16_VARIANT
}""",
    "LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANT_ABIS": """LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANT_ABIS = {
    _WPF2B_IQ2_PAIR16_VARIANT: "grouped_raw_iq_dual_silu",
    _H6L_IQ2_PAIR16_ROWBATCH16_VARIANT: "grouped_raw_iq_dual_silu",
}""",
    "LAGUNA_HEAD_KV_FUSION": 'LAGUNA_HEAD_KV_FUSION = True',
    "LAGUNA_IQ2_GRID64": 'LAGUNA_IQ2_GRID64 = True',
    "LAGUNA_IQ3_C1_DOWN_SCHEDULE": 'LAGUNA_IQ3_C1_DOWN_SCHEDULE = "wave4_reduce"',
    "LAGUNA_IQ3_WAVE10_FUSED": 'LAGUNA_IQ3_WAVE10_FUSED = True',
    "LAGUNA_MIXED_ATTENTION_PROJECTIONS": 'LAGUNA_MIXED_ATTENTION_PROJECTIONS = True',
    "LAGUNA_MIXED_LOCAL32_FIXED_METADATA": 'LAGUNA_MIXED_LOCAL32_FIXED_METADATA = True',
    "LAGUNA_MIXED_Q6_FIXED_METADATA": 'LAGUNA_MIXED_Q6_FIXED_METADATA = True',
    "LAGUNA_MOE_GROUP_COMPACT_MODE": 'LAGUNA_MOE_GROUP_COMPACT_MODE = "parallel"',
    "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6A_ROLE_VARIANTS": """LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6A_ROLE_VARIANTS = {
    "global_m128_c4096_first_fill_exact": (
        "global_context_rows_dense_initial_fixed512_cached_exact_spans"
    ),
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
    ),
}""",
    "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6W_ROLE_VARIANTS": """LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6W_ROLE_VARIANTS = {
    "global_m128_c4096_first_fill_exact": (
        "global_context_rows_dense_initial_fixed512_cached_exact_spans"
    ),
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
    ),
}""",
    "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6Z_ROLE_VARIANTS": """LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6Z_ROLE_VARIANTS = {
    "global_m128_c4096_first_fill_exact": (
        "global_context_rows_qrow4_dense_initial_global_score_weight_replay_"
        "exact_spans"
    ),
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
    ),
}""",
    "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H7Y_ROLE_VARIANTS": """LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H7Y_ROLE_VARIANTS = {
    "global_m128_c4096_first_fill_exact": (
        "global_context_rows_qrow4_dense_initial_global_score_weight_replay_"
        "exact_spans"
    ),
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_dense_initial_lane_major_"
        "global_score_replay_exact_spans"
    ),
}""",
    "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS": """LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS = dict(
    LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6Z_ROLE_VARIANTS
)""",
    "LAGUNA_PREFILL_KV_PREAPPEND": 'LAGUNA_PREFILL_KV_PREAPPEND = True',
    "LAGUNA_PREFILL_MATRIX_ROWS": 'LAGUNA_PREFILL_MATRIX_ROWS = 512',
    "LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS": """LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS = {
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_cached_exact_spans"
    ),
}""",
    "LAGUNA_Q4_LM_HEAD_LOCAL32_FIXED_METADATA": 'LAGUNA_Q4_LM_HEAD_LOCAL32_FIXED_METADATA = True',
    "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE": 'LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE = True',
    "LAGUNA_Q5_FIXED_METADATA": 'LAGUNA_Q5_FIXED_METADATA = True',
    "LAGUNA_Q5_SHARED_FIXED_METADATA": 'LAGUNA_Q5_SHARED_FIXED_METADATA = True',
    "LAGUNA_Q5_WAVE32X2_OUTPUT": 'LAGUNA_Q5_WAVE32X2_OUTPUT = True',
    "LAGUNA_Q5_WAVE32X2_QUERY_GATE": 'LAGUNA_Q5_WAVE32X2_QUERY_GATE = True',
    "LAGUNA_SELECTED_DOWN_MODE": 'LAGUNA_SELECTED_DOWN_MODE = "grouped_exact"',
    "LAGUNA_SELECTED_GATE_UP_MODE": 'LAGUNA_SELECTED_GATE_UP_MODE = "grouped_pair16"',
    "LAGUNA_SPLIT_GATE_FUSION": 'LAGUNA_SPLIT_GATE_FUSION = True',
    "LAGUNA_SWA_DECODE_VARIANT": 'LAGUNA_SWA_DECODE_VARIANT = "swa_context_token4_exact_spans"',
    "LAGUNA_SWA_PREFILL_ROLE_VARIANTS": """LAGUNA_SWA_PREFILL_ROLE_VARIANTS = {
    "qrow4_m128_c256_exact": (
        "swa_context_rows_qrow4_sourcequal_exact_spans"
    ),
}""",
    "LAGUNA_SWA_PREFILL_VARIANT": 'LAGUNA_SWA_PREFILL_VARIANT = "swa_context_rows_qrow4_m128_c256_exact_spans"',
    "LAGUNA_SWA_SPLIT_MIN_LIVE": 'LAGUNA_SWA_SPLIT_MIN_LIVE = 65',
    "LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE": 'LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE = 257',
    "LAGUNA_SWA_SPLIT_WAVE_LOCAL": 'LAGUNA_SWA_SPLIT_WAVE_LOCAL = True',
}
# END LAGUNA_POLICY


def laguna_policy(package: Path = PACKAGE) -> dict[str, str]:
    """Source text of every module-level `LAGUNA_*` assignment in `package`."""
    source = package.read_text()
    policy: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.startswith("LAGUNA_"):
                policy[target.id] = ast.get_source_segment(source, node)
    return policy


def assert_laguna_policy_pinned(
    *, exclude: frozenset[str] = frozenset(), package: Path = PACKAGE
) -> None:
    """Fail, naming the assignments, when a Laguna policy moved.

    `exclude` drops the capability under test: the calling contract asserts that
    flag's live value itself, and its whole point is that it may have flipped.
    """
    live = laguna_policy(package)
    for name in exclude:
        live.pop(name, None)
    expected = {
        name: text for name, text in LAGUNA_POLICY.items() if name not in exclude
    }
    added = sorted(set(live) - set(expected))
    removed = sorted(set(expected) - set(live))
    changed = sorted(
        name for name in live if name in expected and live[name] != expected[name]
    )
    assert not (added or removed or changed), (
        f"Laguna source policy moved in {package.name}: added={added} "
        f"removed={removed} changed={changed}. Audit the change against the WPF "
        "source-default contract, then refresh the pin with "
        "`.venv/bin/python tests/_laguna_policy_pin.py --refresh`."
    )


def render() -> str:
    """The generated literal block, markers included."""
    lines = ["# BEGIN LAGUNA_POLICY", "LAGUNA_POLICY: dict[str, str] = {"]
    for name, text in sorted(laguna_policy().items()):
        if "\n" in text:
            lines.append(f'    "{name}": """{text}""",')
        else:
            lines.append(f"""    "{name}": '{text}',""")
    lines.append("}")
    lines.append("# END LAGUNA_POLICY")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="rewrite the frozen literal from the current package",
    )
    args = parser.parse_args(argv)
    path = Path(__file__).resolve()
    if not args.refresh:
        assert_laguna_policy_pinned()
        print(f"Laguna policy pinned: {len(LAGUNA_POLICY)} assignments")
        return 0
    source = path.read_text()
    start = source.index("# BEGIN LAGUNA_POLICY")
    end = source.index("# END LAGUNA_POLICY") + len("# END LAGUNA_POLICY")
    path.write_text(source[:start] + render() + source[end:])
    print(f"refreshed {len(laguna_policy())} assignments in {path.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
