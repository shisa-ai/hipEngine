"""Qwen3.5/PARO model plugin metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from hipengine.models.kv_capabilities import (
    KVCapabilityDeclaration,
    KVCapabilityEvidence,
    KVCapabilityKey,
    KVCapabilityResolution,
    ModelArtifactIdentity,
    resolve_kv_capability,
)
from hipengine.models.registry import register_model
from hipengine.speculative.serving import (
    SpeculativeMTPServingDecision,
    SpeculativeMTPServingEvidence,
    SpeculativeMTPServingKey,
    resolve_max_qualified_candidate_budget,
    resolve_speculative_mtp_serving_plan,
    unsupported_contract,
    SpeculativeMTPServingImplementation,
)


_QWEN36_MOE_Q4KM_MTP_SERVING_EVIDENCE = (
    SpeculativeMTPServingEvidence(
        evidence_key="qwen36-moe-q4km-gfx1100-production-bf16-c1-k2-d24",
        artifact_sha256=(
            "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b"
        ),
        artifact_size_bytes=22_663_387_424,
        artifact_execution_fingerprint=(
            "c65907f4b56a10e8f89dee5860fb2eae4a9f36deb2d60ab53c6b85dcc42e3bf6"
        ),
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=2,
        sampling_modes=("greedy_fast",),
        reason="qualified_automatic_moe_c1_k2_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-08-27-w7900-35b-moe-mtp2-production-quality.json",
            "benchmarks/results/2026-08-27-w7900-35b-moe-mtp2-production-performance.json",
            "benchmarks/results/2026-08-27-w7900-35b-moe-mtp2-production-serving.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen36-moe-q4km-gfx1100-production-bf16-c2-k2-d24",
        artifact_sha256=(
            "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b"
        ),
        artifact_size_bytes=22_663_387_424,
        artifact_execution_fingerprint=(
            "c65907f4b56a10e8f89dee5860fb2eae4a9f36deb2d60ab53c6b85dcc42e3bf6"
        ),
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        sampling_modes=("greedy_fast",),
        reason="qualified_automatic_production_moe_c2_k2_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-08-28-w7900-35b-moe-mtp2-c2-automatic-promotion.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
)


_QWEN36_DENSE_Q4KM_MTP_SERVING_EVIDENCE = (
    SpeculativeMTPServingEvidence(
        evidence_key="qwen36-dense-q4km-gfx1100-production-bf16-c1-k3-d24",
        artifact_sha256=(
            "a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f"
        ),
        artifact_size_bytes=17_106_773_120,
        # Execution identity unresolved: this artifact is not present on
        # this host, so the row cannot admit until it is recorded with
        # python3 scripts/gguf_execution_identity.py <artifact.gguf>.
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        reason="qualified_automatic_dense_c1_k3_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-production-quality.json",
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-production-performance.json",
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-production-serving.json",
            "benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen36-dense-q4km-gfx1100-strict-bf16-c2-k2-d24",
        artifact_sha256=(
            "a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f"
        ),
        artifact_size_bytes=17_106_773_120,
        # Execution identity unresolved: this artifact is not present on
        # this host, so the row cannot admit until it is recorded with
        # python3 scripts/gguf_execution_identity.py <artifact.gguf>.
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        sampling_modes=("greedy_fast",),
        reason="qualified_explicit_dense_c2_k2_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-c2-explicit-ownership.json",
            "benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen36-dense-q4km-gfx1100-production-bf16-c2-k2-d24",
        artifact_sha256=(
            "a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f"
        ),
        artifact_size_bytes=17_106_773_120,
        # Execution identity unresolved: this artifact is not present on
        # this host, so the row cannot admit until it is recorded with
        # python3 scripts/gguf_execution_identity.py <artifact.gguf>.
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        sampling_modes=("greedy_fast",),
        reason="qualified_automatic_production_dense_c2_k2_d24",
        evidence_artifacts=(
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-c2-production-quality.json",
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-c2-explicit-ownership.json",
            "benchmarks/results/2026-08-27-w7900-27b-dense-mtp2-c2-automatic-promotion.json",
            "benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
)


# UD MTP serving evidence.  The pinned UD artifacts share the plain Qwen3.8
# dense production manifest and the ``gguf_q4_k_m``/``gguf_q4_k_s`` file-type
# stamps, so the only axes that distinguish them from the plain lane are the
# artifact sha256 and size.  The declared scope is the measured one: width c1
# only, because c2 measured 1.0418x, c4 has no production physical width cell,
# and c8 runs out of memory on the default GPU; the context bucket stops well
# below the 1023 sentinel where the adapter refuses and the verifier stops
# batching.  Evidence and measurements:
# ``worklog/entries/20260912T214335.416137Z-lhl-ud-mtp-width-cells-8e85ce.md``
# and ``worklog/entries/20260912T220000.000000Z-lhl-ud-u6-pin-automatic-scope-2c7d1e.md``.
_UD_Q4K_MTP_SERVING_EVIDENCE = (
    SpeculativeMTPServingEvidence(
        evidence_key="ud-q4km-gfx1100-production-bf16-c1-k3-d24",
        artifact_sha256=(
            "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482"
        ),
        artifact_size_bytes=16_464_440_224,
        artifact_execution_fingerprint=(
            "93fe11b8ac0f4696567cc123f2c08ea4bc7fcf615ed985d4fb11742df404f3ca"
        ),
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        reason="qualified_automatic_ud_c1_k3_d24",
        evidence_artifacts=(
            "benchmarks/results/paired-ud-plain-mtp-c1-natural25-b3-phase4.json",
            "benchmarks/results/2026-09-12-ud-gfx1100-mtp-width-cells.json",
            "benchmarks/results/2026-09-12-ud-gfx1100-phase5-ar-verify-numerics.json",
            "benchmarks/results/2026-09-12-ud-gfx1100-phase4-ar-verify-numerics.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="ud-q4ks-gfx1100-production-bf16-c1-k3-d24",
        artifact_sha256=(
            "75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017"
        ),
        artifact_size_bytes=15_358_213_024,
        artifact_execution_fingerprint=(
            "018bdb7473fa5a4e4aa70cf3ff07c3d3150939528095379902320d6700f36f09"
        ),
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_s",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        reason="qualified_automatic_ud_c1_k3_d24",
        evidence_artifacts=(
            "benchmarks/results/paired-ud-plain-mtp-c1-natural25-b3-phase4.json",
            "benchmarks/results/2026-09-12-ud-gfx1100-mtp-width-cells.json",
            "benchmarks/results/2026-09-12-ud-gfx1100-phase5-ar-verify-numerics.json",
            "benchmarks/results/2026-09-12-ud-gfx1100-phase4-ar-verify-numerics.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
)


_QWEN38_Q4KM_MTP_SERVING_EVIDENCE = (
    # W7900 C1 evidence is withdrawn: its runs used the legacy singleton
    # target after preparation. Requalify the packed target independently
    # before adding C1 serving rows (see the better-MTP campaign review).
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1151-strict-bf16-c1-b3-natural25-s0",
        artifact_sha256=(
            "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
        ),
        artifact_size_bytes=17_106_775_008,
        artifact_execution_fingerprint=(
            "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"
        ),
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        reason="qualified_automatic_c1_b3",
        evidence_artifacts=(
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s0.json",
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s0-openai.json",
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s1.json",
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s2.json",
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s3.json",
            "benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1151-strict-bf16-cap4-realized-c1-b3",
        artifact_sha256=(
            "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
        ),
        artifact_size_bytes=17_106_775_008,
        artifact_execution_fingerprint=(
            "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"
        ),
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=4,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        reason="qualified_automatic_realized_singleton_c1_b3",
        evidence_artifacts=(
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s4-auto.json",
            "benchmarks/results/2026-08-26-gfx1151-qwen38-q4km-mtp-serving-s5-closure.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-realized-singleton-auto.json",
            "benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=True,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1151-production-bf16-c1-b3-context128",
        artifact_sha256=(
            "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
        ),
        artifact_size_bytes=17_106_775_008,
        artifact_execution_fingerprint=(
            "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"
        ),
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=1,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        reason="qualified_explicit_production_c1_b3_context128",
        evidence_artifacts=(
            "benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t04-production-suite.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t11-t13-ownership.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-postcampaign-mtp-c1-c8.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-c68-c128-production-explicit.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1151-production-bf16-cap4-c1-intent-k3-d24",
        artifact_sha256=(
            "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
        ),
        artifact_size_bytes=17_106_775_008,
        artifact_execution_fingerprint=(
            "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"
        ),
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=1,
        resident_capacity=4,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        reason="diagnostic_production_cap4_c1_or_c2_after_ar_rebase",
        evidence_artifacts=(
            "benchmarks/results/2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-realized-singleton-auto.json",
            "benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json",
        ),
        max_realized_group_rows=2,
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1151-production-bf16-c2-k3-d24",
        artifact_sha256=(
            "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
        ),
        artifact_size_bytes=17_106_775_008,
        artifact_execution_fingerprint=(
            "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"
        ),
        backend="hip_gfx1151",
        target_arch="gfx1151",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=2,
        resident_capacity=4,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        reason="diagnostic_production_c2_after_ar_rebase",
        evidence_artifacts=(
            "benchmarks/results/2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json",
            "benchmarks/results/2026-08-27-gfx1151-qwen38-dynamic-admission-d3-lifecycle.json",
            "benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json",
        ),
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1100-production-bf16-c2-k2-d24",
        artifact_sha256=(
            "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
        ),
        artifact_size_bytes=17_106_773_984,
        artifact_execution_fingerprint=(
            "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"
        ),
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=2,
        sampling_modes=("greedy_fast",),
        reason="qualified_automatic_gfx1100_production_c2_k2_d24_measured_slower_than_ar_2026_09_06",
        evidence_artifacts=(
            "benchmarks/results/2026-08-29-w7900-qwen38-q4km-p8-c2-correctness-closure.json",
            "benchmarks/results/2026-08-30-w7900-qwen38-q4km-p11-integrated-explicit-c2.json",
            "benchmarks/results/2026-08-30-w7900-qwen38-q4km-p12-c2-automatic-promotion.json",
            "benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json",
        ),
        max_realized_group_rows=2,
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1100-production-bf16-c2-k3-d24",
        artifact_sha256=(
            "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
        ),
        artifact_size_bytes=17_106_773_984,
        artifact_execution_fingerprint=(
            "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"
        ),
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=2,
        resident_capacity=2,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        reason="qualified_explicit_gfx1100_production_c2_k3_d24_packet6_grid_selection_2026_09_06",
        evidence_artifacts=(
            "benchmarks/results/2026-09-06-w7900-q4km-mtp-packet6-grid-and-c2k3.json",
        ),
        max_realized_group_rows=2,
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
    SpeculativeMTPServingEvidence(
        evidence_key="qwen38-q4km-gfx1100-production-bf16-c8-k3-d24",
        artifact_sha256=(
            "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
        ),
        artifact_size_bytes=17_106_773_984,
        artifact_execution_fingerprint=(
            "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"
        ),
        backend="hip_gfx1100",
        target_arch="gfx1100",
        weight_quant="gguf_q4_k_m",
        kv_storage="bf16",
        kv_layout="uniform",
        realized_group_rows=8,
        resident_capacity=8,
        candidate_budget=3,
        sampling_modes=("greedy_fast",),
        reason="qualified_automatic_gfx1100_production_c8_k3_d24_measured_slower_than_ar_2026_09_06",
        evidence_artifacts=(
            "benchmarks/results/2026-09-04-w7900-q4km-k3-c8-p4-q6-dp4a-l4-numerics.json",
            "benchmarks/results/2026-09-04-w7900-q4km-k3-c8-p4-q6-dp4a-retention-e2e.json",
            "benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json",
        ),
        max_realized_group_rows=8,
        strict_fallback_key="gguf_target_ar",
        automatic_eligible=False,
    ),
)


# Sampled selection is independent of the older greedy C2 timing policy.
_QWEN38_Q4KM_MTP_SERVING_EVIDENCE += tuple(
    replace(
        next(row for row in _QWEN38_Q4KM_MTP_SERVING_EVIDENCE
             if row.evidence_key == "qwen38-q4km-gfx1151-production-bf16-cap4-c1-intent-k3-d24"),
        evidence_key=f"qwen38-q4km-gfx1151-native-sampled-c{width}-k3",
        realized_group_rows=width,
        sampling_modes=("sampled",),
        max_realized_group_rows=4,
        automatic_eligible=True,
        reason="automatic_native_sampled_c1_c4",
        evidence_artifacts=(
            "worklog/entries/20260920T200240.750746Z-sampling-mtp-mtp-sampled-concurrency-6667be.md",
        ),
    )
    for width in (1, 2, 3, 4)
)


_QWEN38_GGUF_KV_CAPABILITY_DECLARATIONS = tuple(
    KVCapabilityDeclaration(
        backend=backend,
        target_arch=target_arch,
        kv_storage="int8_per_token_head",
        storage_layout="uniform",
        scale_dtype="fp32",
        scale_granularity="per_token_head",
        max_direct_rows=4,
        max_serial_resident_rows=4,
        persistent_bf16_mirror=False,
        decode_batch_variant=(
            "per_token_head_gqa_splitk_gate_bf16_batch_strided_spans"
        ),
        reason=(
            "registered int8 per-token-head paged decode chain; the row-batched "
            "producer and its strided reducer are implemented to physical c4"
        ),
    )
    for backend, target_arch in (
        ("hip_gfx1100", "gfx1100"),
        ("hip_gfx1151", "gfx1151"),
    )
)


_QWEN38_GGUF_KV_CAPABILITY_EVIDENCE = (
    KVCapabilityEvidence(
        key=KVCapabilityKey(
            artifact_sha256="7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b",
            artifact_size_bytes=17_106_773_984,
            artifact_execution_fingerprint=(
                "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"
            ),
            backend="hip_gfx1100",
            target_arch="gfx1100",
            weight_quant="gguf_q4_k_m",
            kv_storage="int8_per_token_head",
            storage_layout="uniform",
            scale_dtype="fp32",
            scale_granularity="per_token_head",
        ),
        decision="qualified",
        scope="explicit_no_mirror_direct_c4",
        quality_artifact=(
            "benchmarks/results/"
            "2026-08-16-qwen38-27b-actual-context-quality-w7900.json"
        ),
        reason=(
            "complete 512/8 and 4K/16 plus bounded 129024/16 quality pass on "
            "gfx1100; direct compact row-batched decode is qualified to physical "
            "c4 because the row-batched 24Q/4KV/D256 producer and its strided "
            "reducer are bit-identical to the already-qualified c1 leaf at fixed "
            "width and across retire/admit transitions, so the retained quality "
            "artifact stays the applicable quality basis"
        ),
        max_direct_rows=4,
        max_serial_resident_rows=4,
        persistent_bf16_mirror=False,
        decode_batch_variant=(
            "per_token_head_gqa_splitk_gate_bf16_batch_strided_spans"
        ),
    ),
    KVCapabilityEvidence(
        key=KVCapabilityKey(
            artifact_sha256="7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
            artifact_size_bytes=17_106_775_008,
            artifact_execution_fingerprint=(
                "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"
            ),
            backend="hip_gfx1151",
            target_arch="gfx1151",
            weight_quant="gguf_q4_k_m",
            kv_storage="int8_per_token_head",
            storage_layout="uniform",
            scale_dtype="fp32",
            scale_granularity="per_token_head",
        ),
        decision="rejected",
        scope="native_no_mirror_quality",
        quality_artifact=(
            "benchmarks/results/"
            "2026-08-15-gfx1151-qwen38-27b-int8-kv-quality-rejected.json"
        ),
        reason=(
            "complete 1K/8 transfer rejected: minimum-prompt top-1 agreement "
            "0.7778 is below the 0.90 gate"
        ),
        max_direct_rows=0,
        max_serial_resident_rows=0,
        persistent_bf16_mirror=False,
    ),
)


@dataclass(frozen=True)
class Qwen35ParoMoeModel:
    """Qwen3.5 MoE decode metadata for the PARO/W4A16 path.

    This plugin is intentionally metadata-only: it gives the planner stable layer keys and
    records the canonical HF architecture/weight-name shape without loading tensors or
    importing torch. Config-driven layer repetition and attention-specific parameters will
    live in the loader/model-spec layer.
    """

    name: str = "qwen3_5_moe_paro"
    architectures: tuple[str, ...] = (
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
    )
    default_quant: str = "w4_paro"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        "model.embed_tokens.weight",
        "model.layers.{layer}.input_layernorm.weight",
        "model.layers.{layer}.self_attn.{proj}.qweight",
        "model.layers.{layer}.self_attn.{proj}.qzeros",
        "model.layers.{layer}.self_attn.{proj}.scales",
        "model.layers.{layer}.post_attention_layernorm.weight",
        "model.layers.{layer}.mlp.gate.weight",
        "model.layers.{layer}.mlp.experts.{expert}.{proj}.qweight",
        "model.layers.{layer}.mlp.experts.{expert}.{proj}.qzeros",
        "model.layers.{layer}.mlp.experts.{expert}.{proj}.scales",
        "model.layers.{layer}.mlp.shared_expert.{proj}.weight",
        "model.layers.{layer}.mlp.shared_expert_gate.weight",
        "model.norm.weight",
        "lm_head.weight",
    )

    def layer_sequence(self) -> tuple[str, ...]:
        """Return a representative decode sequence for registry/fusion planning."""

        return (
            "embed",
            *self.decode_layer_sequence(attention_kind="full_attention"),
            "final_rmsnorm",
            "lm_head",
        )

    def decode_layer_sequence(self, *, attention_kind: str) -> tuple[str, ...]:
        """Return primitive layer keys for one Qwen3.5 decode layer.

        ``attention_kind`` mirrors Qwen3.5's config-level ``layer_types`` entries.
        """

        if attention_kind == "full_attention":
            attention_layers = (
                "rmsnorm",
                "full_attention_qkv_proj",
                "rope",
                "paged_kv_write",
                "full_attention_decode",
                "full_attention_o_proj",
            )
        elif attention_kind == "linear_attention":
            attention_layers = (
                "rmsnorm",
                "linear_attention_qkvz_proj",
                "linear_attention_conv_decode",
                "linear_attention_recurrence",
                "linear_attention_o_proj",
            )
        else:
            raise ValueError("attention_kind must be 'full_attention' or 'linear_attention'")

        return (
            *attention_layers,
            "add_rmsnorm",
            "router_topk_shared",
            "selected_dual_pack8_gemv",
            "silu_mul_dual_rotate",
            "selected_pack8_gemv",
            "w8a16_linear",
            "weighted_sum+shared_gate+residual",
        )


def select_speculative_mtp_serving_implementation(
    implementations: Sequence[SpeculativeMTPServingImplementation],
    *,
    key: SpeculativeMTPServingKey,
) -> SpeculativeMTPServingImplementation | None:
    """The declaration covering this key's KV storage, backend, and width.

    Storage is matched first because that is the contract a declaration
    describes, then the backend that offers the path.  Among the backend's
    declarations the narrowest one that still covers the realized width wins, so
    a single-row request may use the C1 chain's full structural depth while a
    wider group takes the multi-row cell's measured depth.  A storage match on
    the wrong backend is still returned so the refusal names the real axis
    (``mtp_backend_unsupported``) instead of a storage mismatch, and a width no
    declaration covers returns the widest one so the refusal names the group
    axis.
    """

    candidates = tuple(
        implementation
        for implementation in implementations
        if implementation.kv_storage == key.kv_storage
    )
    if not candidates:
        return None
    offered = tuple(
        implementation
        for implementation in candidates
        if (key.backend, key.target_arch) in implementation.backends
    )
    if not offered:
        return candidates[0]
    covering = tuple(
        implementation
        for implementation in offered
        if implementation.max_group_rows >= key.realized_group_rows
    )
    if covering:
        return min(covering, key=lambda implementation: implementation.max_group_rows)
    return max(offered, key=lambda implementation: implementation.max_group_rows)


@dataclass(frozen=True)
class Qwen35GGUFModel:
    """Qwen3.5 dense GGUF model plugin metadata."""

    name: str = "qwen3_5_gguf"
    architectures: tuple[str, ...] = ("qwen35",)
    default_quant: str = "gguf_q4_k_m"
    default_backend: str = "auto"
    weight_name_templates: tuple[str, ...] = (
        "token_embd.weight",
        "output_norm.weight",
        "blk.{layer}.attn_norm.weight",
        "blk.{layer}.post_attention_norm.weight",
        "blk.{layer}.attn_gate.weight",
        "blk.{layer}.attn_qkv.weight",
        "blk.{layer}.attn_q.weight",
        "blk.{layer}.attn_k.weight",
        "blk.{layer}.attn_v.weight",
        "blk.{layer}.attn_output.weight",
        "blk.{layer}.ffn_gate.weight",
        "blk.{layer}.ffn_up.weight",
        "blk.{layer}.ffn_down.weight",
    )
    kv_capability_evidence: tuple[KVCapabilityEvidence, ...] = _QWEN38_GGUF_KV_CAPABILITY_EVIDENCE
    kv_capability_declarations: tuple[KVCapabilityDeclaration, ...] = (
        _QWEN38_GGUF_KV_CAPABILITY_DECLARATIONS
    )
    speculative_mtp_serving_evidence: tuple[SpeculativeMTPServingEvidence, ...] = (
        _QWEN38_Q4KM_MTP_SERVING_EVIDENCE
        + _QWEN36_DENSE_Q4KM_MTP_SERVING_EVIDENCE
        # Appended, not prepended: the declaration order is the resolver's
        # tie-break, and the existing rows' positions are part of the retained
        # evidence ordering that the serving tests assert.
        + _UD_Q4K_MTP_SERVING_EVIDENCE
    )
    speculative_mtp2_adapter: str = "dense_nextn"
    speculative_mtp_serving_implementations: tuple[SpeculativeMTPServingImplementation, ...] = (
        # Capability declarations for the dense NextN chain.  They are what lets
        # an explicitly requested run execute on an artifact no evidence row
        # describes, and they carry no performance claim.  Depth is the
        # adapter's structural maximum (MTP2_MAX_CANDIDATE_DEPTH) for a
        # single-row chain, where each candidate is the same single-row kernel
        # repeated, and the widest depth the backend package offers once the
        # verifier covers a multi-row group.  Group width is the widest physical
        # cell that backend offers, so a plan never forms a group the adapter
        # would decline; the backend package's own cell table stays the
        # group-formation gate, and a cell it does not list falls through to
        # whole-group AR rather than running a cell nothing measured.  The declared sampling mode lists `sampled` for
        # int8 storage because the accept path is storage-agnostic: it samples from
        # the target's own law over the verifier's row logits and never reads KV.
        # Declaring it does not widen the effective scope, because the runtime
        # qualification gate (`_sampled_route_qualified`) still requires an
        # evidence row matching this backend, target architecture, weight quant,
        # and artifact size -- and that check is storage-blind, so it already
        # treats both storages alike.  What remains open on the route itself is
        # the autoregressive finish rule: the cycle commit ends a row only when
        # its last visible token is the row's EOS, and a stochastic accept has no
        # `greedy_chain_eos_limit` bound, so a stop token or EOS can land
        # mid-cycle.  That is contained per request by the servable-blocker set
        # rather than fixed, and tracked in docs/REFACTOR.md "Sampled MTP
        # acceptance route".  The device-side accept that used to be the second
        # precondition landed 2026-09-19 (`_device_sampled_accept_plan`).
        #
        # Every declaration here is automatic-eligible: a contract the kernels
        # implement is the automatic scope for any artifact that routes through
        # it, and a missing measurement never withholds it.  The operative gate
        # is the physical cell table -- which (width, depth) pairs the active
        # backend package offers, capped by resident capacity -- so automatic
        # intent still forms no cell the adapter would decline.  A retained
        # evidence row is consulted first and keeps its own promotion decision;
        # the declaration is the scope an artifact with no row of its own
        # inherits.
        SpeculativeMTPServingImplementation(
            name="gguf_dense_bf16_gfx1100_c1_native_chain",
            kv_storage="bf16",
            backends=(("hip_gfx1100", "gfx1100"),),
            max_candidate_count=7,
            max_group_rows=1,
            group_rejection_reason="dense_group_above_offered_width",
            automatic_eligible=True,
        ),
        SpeculativeMTPServingImplementation(
            name="gguf_dense_bf16_gfx1100_group_native_chain",
            kv_storage="bf16",
            backends=(("hip_gfx1100", "gfx1100"),),
            max_candidate_count=3,
            max_group_rows=8,
            group_rejection_reason="dense_group_above_offered_width",
            automatic_eligible=True,
        ),
        SpeculativeMTPServingImplementation(
            name="gguf_dense_bf16_gfx1151_c1_native_chain",
            kv_storage="bf16",
            backends=(("hip_gfx1151", "gfx1151"),),
            max_candidate_count=7,
            max_group_rows=1,
            group_rejection_reason="dense_group_above_offered_width",
            automatic_eligible=True,
        ),
        SpeculativeMTPServingImplementation(
            name="gguf_dense_bf16_gfx1151_group_native_chain",
            kv_storage="bf16",
            backends=(("hip_gfx1151", "gfx1151"),),
            max_candidate_count=3,
            max_group_rows=4,
            group_rejection_reason="dense_group_above_offered_width",
            automatic_eligible=True,
        ),
        SpeculativeMTPServingImplementation(
            name="gguf_dense_int8_native_chain",
            kv_storage="int8_per_token_head",
            backends=(("hip_gfx1100", "gfx1100"), ("hip_gfx1151", "gfx1151")),
            max_candidate_count=7,
            max_group_rows=1,
            group_rejection_reason="dense_group_above_offered_width",
            sampling_modes=("greedy_fast", "sampled"),
            # The int8 chain has no evidence rows of its own, so its
            # declaration is the automatic scope it has always been.
            automatic_eligible=True,
        ),
        # A packed INT8 verify group runs the same row-bulk pass as BF16, but its
        # full-attention layers bind the retained INT8 payload planes and their
        # per-token-head scale metadata and attend through the retained-decode
        # split-K leaf instead of the BF16 context-batch decoder. Group width is
        # therefore bounded by the KV capability's qualified direct width, which
        # is physical c4 on both backends, and not by the backend cell table the
        # BF16 split follows (gfx1100 offers 8 cells but qualifies direct INT8 to
        # c4). The prefill and decode classes carry the same limit because they
        # write the same packed physical cell.
        SpeculativeMTPServingImplementation(
            name="gguf_dense_int8_gfx1151_group_native_chain",
            kv_storage="int8_per_token_head",
            backends=(("hip_gfx1151", "gfx1151"),),
            max_candidate_count=3,
            max_group_rows=4,
            group_rejection_reason="dense_group_above_offered_width",
            sampling_modes=("greedy_fast", "sampled"),
            automatic_eligible=True,
        ),
        SpeculativeMTPServingImplementation(
            name="gguf_dense_int8_gfx1100_group_native_chain",
            kv_storage="int8_per_token_head",
            backends=(("hip_gfx1100", "gfx1100"),),
            max_candidate_count=3,
            max_group_rows=4,
            group_rejection_reason="dense_group_above_offered_width",
            sampling_modes=("greedy_fast", "sampled"),
            automatic_eligible=True,
        ),
    )

    def _speculative_mtp_serving_implementation(
        self,
        key: SpeculativeMTPServingKey,
    ) -> SpeculativeMTPServingImplementation | None:
        """The declaration covering this key's storage and backend."""

        return select_speculative_mtp_serving_implementation(
            self.speculative_mtp_serving_implementations,
            key=key,
        )

    def resolve_speculative_mtp_serving_plan(
        self,
        *,
        key: SpeculativeMTPServingKey,
        request_mode: str = "automatic",
    ) -> SpeculativeMTPServingDecision:
        """Resolve the exact Qwen dense serving scope before mutation.

        A retained evidence row is consulted first because it carries the
        measured scope and the promotion decision.  When no row covers the cell,
        the implementation declaration admits it in every request mode: a
        missing measurement never withholds a path the kernels support, and only
        a capability gap or a recorded bad cell keeps it off.
        """

        evidence_decision = resolve_speculative_mtp_serving_plan(
            self.speculative_mtp_serving_evidence,
            key=key,
            request_mode=request_mode,
        )
        if evidence_decision.admitted:
            return evidence_decision
        implementation = self._speculative_mtp_serving_implementation(key)
        if implementation is not None:
            return implementation.resolve(key, request_mode=request_mode)
        return unsupported_contract(evidence_decision)

    def max_qualified_candidate_budget(
        self,
        *,
        key: SpeculativeMTPServingKey,
    ) -> int | None:
        """Return the deepest speculative depth this artifact's evidence qualifies."""

        return resolve_max_qualified_candidate_budget(
            self.speculative_mtp_serving_evidence,
            key=key,
        )

    def resolve_kv_capability(
        self,
        *,
        key: KVCapabilityKey,
        artifact: ModelArtifactIdentity,
    ) -> KVCapabilityResolution:
        """Resolve artifact/backend-specific KV evidence for this plugin."""

        return resolve_kv_capability(
            self.kv_capability_evidence,
            key=key,
            artifact=artifact,
            declarations=self.kv_capability_declarations,
        )


@dataclass(frozen=True)
class Qwen35MoeGGUFModel:
    """Qwen3.6/Qwen3.5 MoE GGUF model plugin metadata."""

    name: str = "qwen3_5_moe_gguf"
    architectures: tuple[str, ...] = ("qwen35moe",)
    default_quant: str = "gguf_q4_k_m"
    default_backend: str = "auto"
    speculative_mtp2_adapter: str = "moe_nextn"
    speculative_mtp_serving_evidence: tuple[SpeculativeMTPServingEvidence, ...] = (
        _QWEN36_MOE_Q4KM_MTP_SERVING_EVIDENCE
    )
    # The MoE NextN chain is declared for the backend its kernels are built and
    # exercised on.  gfx1151 has no MoE MTP evidence and no MoE prompt-streaming
    # policy entry, so an explicit run there stays evidence-gated until the
    # capability is established rather than assumed.  Depth and width are what
    # the adapter serves rather than a wider claim: it clamps candidate depth to
    # two, and its prompt streaming covers capacities 1 and 2 only.
    speculative_mtp_serving_implementations: tuple[SpeculativeMTPServingImplementation, ...] = (
        SpeculativeMTPServingImplementation(
            name="gguf_moe_bf16_gfx1100_native_chain",
            kv_storage="bf16",
            backends=(("hip_gfx1100", "gfx1100"),),
            max_candidate_count=2,
            max_group_rows=2,
            group_rejection_reason="moe_group_above_offered_width",
            automatic_eligible=True,
        ),
    )
    weight_name_templates: tuple[str, ...] = (
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.{layer}.attn_norm.weight",
        "blk.{layer}.post_attention_norm.weight",
        "blk.{layer}.attn_gate.weight",
        "blk.{layer}.attn_qkv.weight",
        "blk.{layer}.attn_q.weight",
        "blk.{layer}.attn_k.weight",
        "blk.{layer}.attn_v.weight",
        "blk.{layer}.attn_output.weight",
        "blk.{layer}.ffn_gate_inp.weight",
        "blk.{layer}.ffn_gate_inp_shexp.weight",
        "blk.{layer}.ffn_gate_exps.weight",
        "blk.{layer}.ffn_up_exps.weight",
        "blk.{layer}.ffn_down_exps.weight",
        "blk.{layer}.ffn_gate_shexp.weight",
        "blk.{layer}.ffn_up_shexp.weight",
        "blk.{layer}.ffn_down_shexp.weight",
    )

    def resolve_speculative_mtp_serving_plan(
        self,
        *,
        key: SpeculativeMTPServingKey,
        request_mode: str = "automatic",
    ) -> SpeculativeMTPServingDecision:
        """Resolve the exact Qwen MoE serving scope before mutation.

        Same rule as the dense plugin: retained evidence is consulted first for
        the measured scope, and the implementation declaration admits the cell
        in every request mode so a missing measurement cannot refuse a path the
        kernels support.
        """

        evidence_decision = resolve_speculative_mtp_serving_plan(
            self.speculative_mtp_serving_evidence,
            key=key,
            request_mode=request_mode,
        )
        if evidence_decision.admitted:
            return evidence_decision
        implementation = select_speculative_mtp_serving_implementation(
            self.speculative_mtp_serving_implementations,
            key=key,
        )
        if implementation is not None:
            return implementation.resolve(key, request_mode=request_mode)
        return unsupported_contract(evidence_decision)

    def max_qualified_candidate_budget(
        self,
        *,
        key: SpeculativeMTPServingKey,
    ) -> int | None:
        """Return the deepest speculative depth this artifact's evidence qualifies."""

        return resolve_max_qualified_candidate_budget(
            self.speculative_mtp_serving_evidence,
            key=key,
        )


QWEN35_PARO_MOE = register_model(Qwen35ParoMoeModel())
QWEN35_GGUF = register_model(Qwen35GGUFModel())
QWEN35_MOE_GGUF = register_model(Qwen35MoeGGUFModel())
