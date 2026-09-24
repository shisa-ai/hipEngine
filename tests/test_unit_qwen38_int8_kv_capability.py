from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hipengine.models.kv_capabilities import (
    KVCapabilityKey,
    ModelArtifactIdentity,
    model_artifact_identity,
)
from hipengine.kernels.hip_gfx1100.attention import (
    qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_batch_strided_spans,
    register_qwen35_paged_attn_decode_kernels,
)
from hipengine.models.qwen35 import Qwen35GGUFModel
from hipengine.runtime import qwen35_gguf_runner as gguf_runner


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PASS_SHA256 = "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
_REJECT_SHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
# Execution identity of both Qwen3.8-27B-Q4_K_M builds (see models/qwen35.py).
_PLAIN_FINGERPRINT = "4c4268886f225fba3675e32a521fba1d6ff1db4562bd06416c89fd22b6f90faa"


def _key(
    *,
    sha256: str,
    size_bytes: int,
    backend: str,
    target_arch: str | None = None,
    scale_dtype: str = "fp32",
    execution_fingerprint: str | None = None,
) -> KVCapabilityKey:
    return KVCapabilityKey(
        artifact_sha256=sha256,
        artifact_size_bytes=size_bytes,
        artifact_execution_fingerprint=(
            _PLAIN_FINGERPRINT if execution_fingerprint is None else execution_fingerprint
        ),
        backend=backend,
        target_arch=target_arch or backend.removeprefix("hip_"),
        weight_quant="gguf_q4_k_m",
        kv_storage="int8_per_token_head",
        storage_layout="uniform",
        scale_dtype=scale_dtype,
        scale_granularity="per_token_head",
    )


def _artifact(*, sha256: str, size_bytes: int) -> ModelArtifactIdentity:
    return ModelArtifactIdentity(
        path="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
        size_bytes=size_bytes,
        sha256=sha256,
        content_verified=True,
    )


def _declaration(**changes) -> dict[str, object]:
    """The gfx1151 declaration's execution bounds, as ``as_dict`` reports them."""

    base: dict[str, object] = {
        "max_direct_rows": 4,
        "persistent_bf16_mirror": False,
        "decode_batch_variant": (
            "per_token_head_gqa_splitk_gate_bf16_batch_strided_spans"
        ),
    }
    base.update(changes)
    return base


def _forced_payload(
    *,
    declaration: dict[str, object] | None = _declaration(),
    effective_kv_storage: str = "int8_per_token_head",
    requested: dict[str, str] | None = None,
) -> dict[str, object]:
    """A ``diagnostic_override`` resolution for a quality-rejected artifact.

    The rejected evidence row supplies ``max_direct_rows=0``, no decode variant,
    and ``persistent_bf16_mirror=False``; the declaration supplies the bounds a
    forced session executes.  The top-level fields are kept at their evidence
    values so a test that starts reading them by mistake fails here rather than
    passing for the wrong reason.
    """

    return {
        "runtime_action": "diagnostic_override",
        "effective_kv_storage": effective_kv_storage,
        "persistent_bf16_mirror": False,
        "max_direct_rows": 0,
        "decode_batch_variant": None,
        "declaration": declaration,
        "requested": requested
        or {
            "kv_storage": "int8_per_token_head",
            "storage_layout": "uniform",
        },
    }


def test_registered_capabilities_match_retained_artifact_model_identities() -> None:
    passing_path = (
        _REPO_ROOT
        / "benchmarks/results/2026-08-16-qwen38-27b-actual-context-quality-w7900.json"
    )
    rejected_path = (
        _REPO_ROOT
        / "benchmarks/results/2026-08-15-gfx1151-qwen38-27b-int8-kv-quality-rejected.json"
    )
    passing_artifact = json.loads(passing_path.read_text())
    rejected_artifact = json.loads(rejected_path.read_text())
    passing = passing_artifact["model"]
    rejected = rejected_artifact["model"]

    assert (passing["size_bytes"], passing["sha256"], passing["quant"]) == (
        17_106_773_984,
        _PASS_SHA256,
        "gguf_q4_k_m",
    )
    assert (rejected["size_bytes"], rejected["sha256"], rejected["quant"]) == (
        17_106_775_008,
        _REJECT_SHA256,
        "gguf_q4_k_m",
    )
    assert (
        passing_artifact["source"]["backend"],
        passing_artifact["source"]["target_arch"],
        passing_artifact["kv"]["storage"],
        passing_artifact["kv"]["scale_dtype"],
        passing_artifact["kv"]["scale_granularity"],
    ) == (
        "hip_gfx1100",
        "gfx1100",
        "int8_per_token_head",
        "fp32",
        "per_token_head",
    )
    assert (
        rejected_artifact["software"]["backend"],
        rejected_artifact["hardware"]["arch"],
        rejected_artifact["protocol"]["native_invocations"][0]["storage"],
        rejected_artifact["protocol"]["native_invocations"][0]["scale_dtype"],
    ) == ("hip_gfx1151", "gfx1151", "int8_per_token_head", "fp32")


def test_qwen38_gfx1100_exact_artifact_int8_capability_is_qualified() -> None:
    plugin = Qwen35GGUFModel()
    resolution = plugin.resolve_kv_capability(
        key=_key(
            sha256=_PASS_SHA256,
            size_bytes=17_106_773_984,
            backend="hip_gfx1100",
        ),
        artifact=_artifact(sha256=_PASS_SHA256, size_bytes=17_106_773_984),
    )

    payload = resolution.as_dict()
    # The capability id covers the key's execution identity, so it changed when
    # the artifact axis stopped being a byte digest.  The two 2026-09-11 W7900
    # int8-KV artifacts under benchmarks/results/ record the id this contract
    # resolved to before that field existed; they are frozen measurements.
    assert payload["capability_id"] == (
        "aea991f631ef5c1661a43b705660793592cfc5b3e70f1df90d503ed0d7b0e234"
    )
    assert payload["status"] == "qualified"
    assert payload["runtime_action"] == "admit"
    assert payload["promotion_eligible"] is True
    assert payload["effective_kv_storage"] == "int8_per_token_head"
    assert payload["evidence"]["max_direct_rows"] == 4
    assert payload["evidence"]["max_serial_resident_rows"] == 4
    assert payload["evidence"]["scope"] == "explicit_no_mirror_direct_c4"
    assert payload["evidence"]["decode_batch_variant"] == (
        "per_token_head_gqa_splitk_gate_bf16_batch_strided_spans"
    )
    assert payload["evidence"]["persistent_bf16_mirror"] is False
    assert payload["evidence"]["quality_artifact"].endswith(
        "2026-08-16-qwen38-27b-actual-context-quality-w7900.json"
    )


def test_qwen38_gfx1100_capability_resolves_exact_batch_kernel_at_direct_c4() -> None:
    plugin = Qwen35GGUFModel()
    resolution = plugin.resolve_kv_capability(
        key=_key(
            sha256=_PASS_SHA256,
            size_bytes=17_106_773_984,
            backend="hip_gfx1100",
        ),
        artifact=_artifact(sha256=_PASS_SHA256, size_bytes=17_106_773_984),
    )
    register_qwen35_paged_attn_decode_kernels()

    max_rows, kernel = gguf_runner._qualified_kv_decode_batch_route(
        "hip_gfx1100",
        resolution.as_dict(),
    )

    assert max_rows == 4
    assert kernel is qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_batch_strided_spans


def test_qwen38_gfx1151_exact_artifact_int8_capability_remains_rejected() -> None:
    plugin = Qwen35GGUFModel()
    resolution = plugin.resolve_kv_capability(
        key=_key(
            sha256=_REJECT_SHA256,
            size_bytes=17_106_775_008,
            backend="hip_gfx1151",
        ),
        artifact=_artifact(sha256=_REJECT_SHA256, size_bytes=17_106_775_008),
    )

    payload = resolution.as_dict()
    assert payload["status"] == "rejected"
    assert payload["runtime_action"] == "fallback_bf16"
    assert payload["promotion_eligible"] is False
    assert payload["effective_kv_storage"] == "bf16"
    assert "0.7778" in payload["reason"]


def test_gfx1151_rejected_artifact_binds_the_direct_leaf_under_diagnostic_override() -> None:
    """A recorded rejection governs selection; an explicit override governs execution.

    The rejected artifact falls closed to BF16 on its own, so no direct leaf is
    bound and a session cannot silently run mirror-free INT8.  Once the operator
    forces INT8 through the documented override, the selection question is
    answered and the remaining gate is the declaration, which the registry
    lookup verifies.  Excluding the override would leave a configuration no
    command could open.
    """

    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    plugin = Qwen35GGUFModel()
    rejected = plugin.resolve_kv_capability(
        key=_key(
            sha256=_REJECT_SHA256,
            size_bytes=17_106_775_008,
            backend="hip_gfx1151",
        ),
        artifact=_artifact(sha256=_REJECT_SHA256, size_bytes=17_106_775_008),
    )
    register_gfx1151_kernels()

    # The engine's own BF16 fallback binds nothing, before and after the change.
    assert gguf_runner._qualified_kv_decode_batch_route(
        "hip_gfx1151",
        rejected.as_dict(),
    ) == (1, None)

    forced = rejected.with_runtime_outcome(
        effective_kv_storage="int8_per_token_head",
        runtime_action="diagnostic_override",
        reason="explicit unverified INT8 KV diagnostic override is enabled",
    )
    payload = forced.as_dict()
    assert payload["effective_kv_storage"] == "int8_per_token_head"
    # Allocation still reads the strict predicate; only the route widens.
    assert gguf_runner._admitted_no_mirror_int8_capability(payload) is False
    assert gguf_runner._runnable_no_mirror_int8_capability(payload) is True

    max_rows, kernel = gguf_runner._qualified_kv_decode_batch_route(
        "hip_gfx1151",
        payload,
    )
    assert max_rows == 4
    assert kernel is qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_batch_strided_spans


def test_direct_int8_route_refuses_a_contract_no_kernel_registers() -> None:
    """The declaration covers the contract; the registry decides execution.

    A capability that names an unregistered variant is still a running
    mirror-free INT8 mode, so the predicate admits it, and route resolution
    independently reports that there is no kernel.  Callers fail closed on the
    missing kernel rather than on the predicate.
    """

    payload = _forced_payload(
        declaration={
            "max_direct_rows": 4,
            "persistent_bf16_mirror": False,
            "decode_batch_variant": "no_such_registered_variant",
        },
    )

    assert gguf_runner._runnable_no_mirror_int8_capability(payload) is True
    assert gguf_runner._qualified_kv_decode_batch_route(
        "hip_gfx1151",
        payload,
    ) == (1, None)


def test_direct_int8_route_refuses_every_non_running_resolution() -> None:
    """Only a deliberate INT8 selection reaches the direct leaf."""

    for label, capability in (
        # The engine's own BF16 fallback keeps its BF16 storage.
        ("bf16 fallback", _forced_payload(effective_kv_storage="bf16")),
        # A contract no declaration covers has no execution bounds at all.
        ("no declaration", _forced_payload(declaration=None)),
        # A mirror means the layer is not on direct INT8 at all.
        (
            "mirror retained",
            _forced_payload(declaration=_declaration(persistent_bf16_mirror=True)),
        ),
        # Direct width zero is not a runnable width.
        ("zero direct width", _forced_payload(declaration=_declaration(max_direct_rows=0))),
        # The packed physical cell is uniform-only.
        (
            "non-uniform layout",
            _forced_payload(
                requested={
                    "kv_storage": "int8_per_token_head",
                    "storage_layout": "tail4_hadamard_group32",
                }
            ),
        ),
        # A BF16 request cannot resolve an INT8 route.
        (
            "bf16 requested",
            _forced_payload(
                requested={"kv_storage": "bf16", "storage_layout": "uniform"}
            ),
        ),
        ("absent", None),
    ):
        assert gguf_runner._admitted_no_mirror_int8_capability(capability) is False, label
        assert gguf_runner._runnable_no_mirror_int8_capability(capability) is False, label
        assert gguf_runner._qualified_kv_decode_batch_route(
            "hip_gfx1151",
            capability,
        ) == (1, None), label


def test_identity_never_gates_admission_but_capability_does() -> None:
    plugin = Qwen35GGUFModel()
    # No identity gates execution.  An artifact nobody has measured runs on the
    # declared kernel chain and is simply not promotable.  A contract no kernel
    # implements is the only thing refused.
    unknown = plugin.resolve_kv_capability(
        key=_key(
            sha256="f" * 64,
            size_bytes=17_106_773_984,
            backend="hip_gfx1100",
            execution_fingerprint="e" * 64,
        ),
        artifact=_artifact(sha256="f" * 64, size_bytes=17_106_773_984),
    )
    wrong_scale = plugin.resolve_kv_capability(
        key=_key(
            sha256=_PASS_SHA256,
            size_bytes=17_106_773_984,
            backend="hip_gfx1100",
            scale_dtype="fp16",
        ),
        artifact=_artifact(sha256=_PASS_SHA256, size_bytes=17_106_773_984),
    )
    wrong_target = plugin.resolve_kv_capability(
        key=_key(
            sha256=_PASS_SHA256,
            size_bytes=17_106_773_984,
            backend="hip_gfx1100",
            target_arch="gfx1151",
        ),
        artifact=_artifact(sha256=_PASS_SHA256, size_bytes=17_106_773_984),
    )

    # Unmeasured artifact: admitted on kernel capability, not promotable.
    assert unknown.status == "unmeasured"
    assert unknown.runtime_action == "admit"
    assert unknown.effective_kv_storage == "int8_per_token_head"
    assert unknown.promotion_eligible is False
    assert unknown.max_direct_rows >= 1

    # Real capability misses: no kernel implements these contracts.
    assert wrong_scale.status == "unsupported"
    assert wrong_scale.runtime_action == "fallback_bf16"
    assert "no registered kernel implements" in wrong_scale.reason
    assert wrong_target.status == "unsupported"
    assert wrong_target.runtime_action == "fallback_bf16"


def test_unmatched_scale_axis_names_the_declared_verdict_it_overrides() -> None:
    """A key one axis off a declared contract names that axis and its verdict.

    This is the refused cell the INT8 MTP readiness row records: the request
    resolves to fp16 scales while every INT8 declaration is keyed fp32, so the
    refusal reads as an absent implementation even though the artifact's real
    verdict is a retained quality rejection at the declared fp32 contract.
    """

    plugin = Qwen35GGUFModel()

    rejected = plugin.resolve_kv_capability(
        key=_key(
            sha256=_REJECT_SHA256,
            size_bytes=17_106_775_008,
            backend="hip_gfx1151",
            scale_dtype="fp16",
        ),
        artifact=_artifact(sha256=_REJECT_SHA256, size_bytes=17_106_775_008),
    )
    assert rejected.status == "unsupported"
    assert rejected.runtime_action == "fallback_bf16"
    assert "no registered kernel implements" in rejected.reason
    assert "scale_dtype (declared 'fp32', requested 'fp16')" in rejected.reason
    assert "the retained verdict for that contract is rejected" in rejected.reason
    assert "0.7778" in rejected.reason

    # The same axis on the artifact whose retained row is qualified.
    qualified = plugin.resolve_kv_capability(
        key=_key(
            sha256=_PASS_SHA256,
            size_bytes=17_106_773_984,
            backend="hip_gfx1100",
            scale_dtype="fp16",
        ),
        artifact=_artifact(sha256=_PASS_SHA256, size_bytes=17_106_773_984),
    )
    assert qualified.status == "unsupported"
    assert "scale_dtype (declared 'fp32', requested 'fp16')" in qualified.reason
    assert "the retained verdict for that contract is qualified" in qualified.reason


def test_model_artifact_identity_hashes_content_and_invalidates_on_change(tmp_path: Path) -> None:
    path = tmp_path / "same-name.gguf"
    path.write_bytes(b"first-artifact")

    first = model_artifact_identity(path)
    repeated = model_artifact_identity(path)
    assert first == repeated
    assert first.content_verified is True
    assert first.sha256 == hashlib.sha256(b"first-artifact").hexdigest()

    path.write_bytes(b"different-artifact")
    changed = model_artifact_identity(path)
    assert changed.content_verified is True
    assert changed.sha256 == hashlib.sha256(b"different-artifact").hexdigest()
    assert changed.sha256 != first.sha256


def test_missing_model_artifact_identity_is_unverified(tmp_path: Path) -> None:
    identity = model_artifact_identity(tmp_path / "missing.gguf")

    assert identity.content_verified is False
    assert identity.sha256 is None
    assert identity.size_bytes is None
    assert "FileNotFoundError" in str(identity.error)
