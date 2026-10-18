#!/usr/bin/env python3
"""Emit the artifact-scoped strict manifest for a UD certification artifact.

The U6 certification artifact records *rates and exactness* for a named GGUF
artifact. That claim is only meaningful if the strict execution-profile variant
set behind it is named in the artifact itself. This script derives that set from
two sources of truth rather than from a hand-written list:

* the artifact's own GGUF tensor types, read from the file, which say which
  quantized layouts the artifact actually contains; and
* ``CERTIFIED_OPERATION_COVERAGE``, the U6 admission table, whose records carry
  ``source_ggml_types`` (the GGML types a record serves), ``kernel_layer``,
  ``kernel_quant``, ``kernel_variant`` and ``rows_scope``.

A record is in scope when its ``source_ggml_types`` intersects the types present
in the artifact. The resulting selections are then validated with the same
``build_variant_manifest`` the profile gate uses, and every selected and
rows-many variant is checked against the live kernel registry, so a manifest
that names an unregistered owner fails here instead of at gate time.

Usage:

    python3 scripts/ud_strict_manifest.py --artifact ud-q4-k-m --artifact ud-q4-k-s \\
        --output benchmarks/results/ud-mtp-certification-u6.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKEND = "hip_gfx1100"
KV_POLICY = "paged_bf16"
GRAPH_POLICY = "serial_c1"


def _artifacts() -> dict[str, tuple[Path, str]]:
    """Reuse the certification script's artifact table so the two cannot drift."""

    from scripts.ud_mtp_certification import ARTIFACTS

    return ARTIFACTS


def _register_kernels() -> None:
    """Register every owner the admission table can name.

    The runtime registers these through the materialization and speculative
    packages; a manifest check has to resolve *all* of them, including owners
    the artifact does not currently reach, so it imports the same registrar set
    the UD admission test uses rather than a subset.
    """

    import hipengine.kernels.hip_gfx1100.speculative  # noqa: F401
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import register_gguf_ops
    from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
        register_dense_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
        register_qwen35_linear_attn_conv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
        register_qwen35_linear_attn_gdn_kernels,
    )
    from hipengine.kernels.hip_gfx1100.moe.router import (
        register_qwen35_router_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import (
        register_gguf_iq_dense_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_gemv import (
        register_gguf_iq_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        register_gguf_k_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
        register_gguf_k_t16_selected_prefill_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q3_k_gemv import (
        register_gguf_q3_k_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        register_gguf_q4_k_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import (
        register_gguf_q6_k_embedding_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        register_gguf_q6_k_t16_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
        register_gguf_q8_0_t16_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
        register_gguf_t16_selected_gemv_kernels,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_x8_selected_gemv import (
        register_gguf_x8_selected_gemv_kernels,
    )
    from hipengine.kernels.registry import DuplicateKernelError

    for registrar in (
        register_gguf_iq_dense_kernels,
        register_gguf_iq_gemv_kernels,
        register_dense_gemv_kernels,
        register_gguf_k_gemv_kernels,
        register_gguf_q4_k_gemv_kernels,
        register_gguf_q3_k_gemv_kernels,
        register_gguf_x8_selected_gemv_kernels,
        register_gguf_q8_0_t16_gemv_kernels,
        register_gguf_t16_selected_gemv_kernels,
        register_gguf_q6_k_t16_gemv_kernels,
        register_gguf_k_t16_selected_prefill_kernels,
        register_gguf_q6_k_embedding_kernels,
        register_gguf_ops,
        register_qwen35_linear_attn_conv_kernels,
        register_qwen35_linear_attn_gdn_kernels,
        register_qwen35_router_kernels,
    ):
        try:
            registrar(replace=False)
        except DuplicateKernelError:
            pass


def _present_ggml_types(path: Path) -> tuple[set[str], int]:
    """Return the GGML type names present in the artifact and its tensor count."""

    from hipengine.loading.gguf import GGUFReader, ggml_type_name

    reader = GGUFReader(path)
    names: set[str] = set()
    for info in reader.info.tensors:
        names.add(ggml_type_name(info.ggml_type))
    return names, len(reader.info.tensors)


def _scoped_records(present: set[str]):
    """Coverage records the artifact's declared GGML types put in scope.

    Records split two ways, matching how the admission table itself is checked:
    a record with ``consumer_module`` is certified by the presence of a symbol in
    that module rather than by a registry key, so it contributes provenance but
    cannot appear as a manifest selection.
    """

    from hipengine.loading.qwen35_gguf_admission import CERTIFIED_OPERATION_COVERAGE

    registry_records = []
    consumer_records = []
    for record in CERTIFIED_OPERATION_COVERAGE:
        declared = record.source_ggml_types
        if not declared or not (set(declared) & present):
            continue
        if record.consumer_module is not None:
            consumer_records.append(record)
        else:
            registry_records.append(record)
    return registry_records, consumer_records


SERVING_OUTPUT_SUFFIX = "_bf16_out"
# Two naming conventions are in the table: most layers mark the serving owner
# explicitly (``..._bf16_out``), while others leave bf16 unmarked and qualify
# only the alternates (``out`` versus ``f32_out``). Both are handled by
# preferring owners with no foreign output-dtype token.
FOREIGN_OUTPUT_TOKENS = ("f32", "fp16")


def _owner_class(variant: str) -> str:
    """Which certified dispatch path an owner belongs to.

    ``linear/prefill_rows`` in ``gguf_q5_k`` carries both ``gemv_bf16_bf16_out``
    (the dense owner) and ``selected_gemv_bf16_bf16_out`` (the owner the
    verifier's selected-row shape dispatches to). Those are two certified
    consumers under one (layer, scope), which a single manifest cannot hold, so
    the bundle is split by owner class.
    """

    if "selected" in variant:
        return "selected"
    if "pack8" in variant:
        return "pack8"
    if variant.startswith("gemv") or variant.endswith("out") or "gemv" in variant:
        return "dense"
    return "other"


def _selections(records):
    """One strict selection per (layer, scope) *within one registry quant*.

    The manifest schema keys selections by (layer, scope) alone, and the
    admission table legitimately carries several registry quants under one
    (layer, scope) -- ``dense_gemv/prefill_rows`` is both bf16 and f32. So the
    artifact's strict scope is a *bundle* of per-quant manifests rather than one
    manifest that would have to collapse real dispatch distinctions to fit.

    Within a (layer, scope) the table also lists one owner per output dtype,
    because the F32-logits comparison surface needs an f32-output variant of the
    same kernel. This artifact serves bf16 activations, so the selection is the
    ``_bf16_out`` owner and the alternates are recorded alongside it rather than
    dropped.
    """

    from hipengine.execution_profiles import VariantSelection

    by_quant: dict[str, list] = collections.defaultdict(list)
    for record in records:
        by_quant[record.kernel_quant].append(record)

    manifests = []
    for quant in sorted(by_quant):
        by_class: dict[str, list] = collections.defaultdict(list)
        for record in by_quant[quant]:
            by_class[_owner_class(record.kernel_variant)].append(record)
        for owner_class in sorted(by_class):
            grouped: dict[tuple[str, str], list] = collections.defaultdict(list)
            for record in by_class[owner_class]:
                grouped[(record.kernel_layer, record.rows_scope or record.operation)].append(
                    record
                )

            selections = []
            variants: list[tuple[str, str, str]] = []
            alternates: dict[str, list[str]] = {}
            for (layer, scope) in sorted(grouped):
                group = grouped[(layer, scope)]
                owners = sorted(
                    {
                        (record.kernel_variant, record.kernel_variant_rows_many)
                        for record in group
                        if record.kernel_variant
                    }
                )
                serving = [
                    owner
                    for owner in owners
                    if not any(token in owner[0] for token in FOREIGN_OUTPUT_TOKENS)
                ]
                chosen = serving or owners
                selected = chosen[0][0]
                if len(chosen) > 1:
                    raise SystemExit(
                        f"scope {layer}/{scope} in {quant}/{owner_class} has several "
                        f"serving-output owners {[owner[0] for owner in chosen]}; the "
                        "manifest cannot express more than one owner per (layer, scope)"
                    )
                selections.append(
                    VariantSelection(
                        layer=layer,
                        scope=scope,
                        selected_variant=selected,
                        strict_fallback_variant=selected,
                        registry_quant=quant,
                    )
                )
                for owner in owners:
                    for variant in owner:
                        if variant:
                            variants.append((layer, quant, variant))
                if len(owners) > 1:
                    alternates[f"{layer}/{scope}"] = [
                        owner[0] for owner in owners if owner[0] != selected
                    ]
            manifests.append((quant, owner_class, selections, variants, alternates))
    return manifests


def _verify_registered(variants) -> None:
    from hipengine.kernels.registry import KernelKey, registered_keys

    registered = set(registered_keys())
    missing = [
        (layer, quant, variant)
        for (layer, quant, variant) in variants
        if KernelKey(BACKEND, layer, quant, variant) not in registered
    ]
    if missing:
        detail = "\n".join(
            f"  {KernelKey(BACKEND, l, q, v).display()}" for (l, q, v) in missing
        )
        raise SystemExit(
            "artifact-scoped strict manifest names unregistered owners:\n" + detail
        )


def build(artifact: str) -> dict[str, object]:
    from hipengine.execution_profiles import build_variant_manifest, manifest_sha256

    table = _artifacts()
    if artifact not in table:
        raise SystemExit(f"unknown artifact {artifact!r}; known: {sorted(table)}")
    path, quant_identity = table[artifact]

    present, tensor_count = _present_ggml_types(path)
    records, consumer_records = _scoped_records(present)
    if not records and not consumer_records:
        raise SystemExit(f"{artifact}: no coverage record declares any of {sorted(present)}")

    per_quant = []
    variants_checked = 0
    for quant, owner_class, selections, variants, alternates in _selections(records):
        _verify_registered(variants)
        variants_checked += len(variants)
        manifest = build_variant_manifest(
            profile="strict",
            backend=BACKEND,
            model=quant_identity,
            quant=quant_identity,
            kv_policy=KV_POLICY,
            graph_policy=GRAPH_POLICY,
            selections=selections,
        )
        per_quant.append(
            {
                "registry_quant": quant,
                "owner_class": owner_class,
                "coverage_records": sum(
                    1
                    for record in records
                    if record.kernel_quant == quant
                    and _owner_class(record.kernel_variant) == owner_class
                ),
                "output_dtype_preference": SERVING_OUTPUT_SUFFIX,
                "alternate_output_dtype_owners": alternates,
                "strict_manifest": manifest,
                "strict_manifest_sha256": manifest_sha256(manifest),
            }
        )

    bundle_encoded = json.dumps(
        [
            (item["registry_quant"], item["owner_class"], item["strict_manifest_sha256"])
            for item in per_quant
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    return {
        "artifact": artifact,
        "gguf_path": str(path),
        "quant_identity": quant_identity,
        "scope": {
            "source": "artifact GGUF tensor types x CERTIFIED_OPERATION_COVERAGE",
            "tensors": tensor_count,
            "ggml_types_present": sorted(present),
            "coverage_records_in_scope": len(records) + len(consumer_records),
            "coverage_records_named_by_registry_key": len(records),
            "coverage_records_named_by_consumer_module": len(consumer_records),
            "consumer_modules": sorted(
                {str(record.consumer_module) for record in consumer_records}
            ),
            "coverage_records_total": len(
                __import__(
                    "hipengine.loading.qwen35_gguf_admission",
                    fromlist=["CERTIFIED_OPERATION_COVERAGE"],
                ).CERTIFIED_OPERATION_COVERAGE
            ),
            "registry_quants": sorted({item["registry_quant"] for item in per_quant}),
            "owner_classes": sorted({item["owner_class"] for item in per_quant}),
            "selections": sum(
                len(item["strict_manifest"]["selections"]) for item in per_quant
            ),
            "variants_verified_registered": variants_checked,
        },
        "strict_manifests": per_quant,
        "strict_manifest_bundle_sha256": hashlib.sha256(bundle_encoded).hexdigest(),
    }


def _stamp_certification(cert_path: Path, built: list[dict[str, object]]) -> dict[str, object]:
    """Write the manifest hash into the certification artifact's entries."""

    cert = json.loads(cert_path.read_text())
    by_name = {str(item["artifact"]): item for item in built}
    stamped = 0
    for entry in cert.get("artifacts", []):
        record = by_name.get(str(entry.get("artifact")))
        if record is None:
            continue
        entry["strict_manifest_bundle_sha256"] = record["strict_manifest_bundle_sha256"]
        entry["strict_manifests"] = record["strict_manifests"]
        entry["strict_manifest_scope"] = record["scope"]
        stamped += 1
    if not stamped:
        raise SystemExit(f"{cert_path}: no artifact entry matched {sorted(by_name)}")
    cert_path.write_text(json.dumps(cert, indent=2, sort_keys=True) + "\n")
    return {"stamped_entries": stamped, "certification": str(cert_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results",
        help="where per-artifact strict manifests are written",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="certification artifact to stamp with the manifest hash",
    )
    parser.add_argument("--check", action="store_true", help="fail if files differ")
    args = parser.parse_args(argv)

    _register_kernels()

    built = []
    for artifact in args.artifact:
        record = build(artifact)
        built.append(record)
        out = args.manifest_dir / f"{artifact}.strict-manifest.json"
        payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
        if args.check:
            if not out.exists() or out.read_text() != payload:
                raise SystemExit(f"{out} is stale; re-run without --check")
        else:
            out.write_text(payload)
        print(
            f"{artifact}: {len(record['strict_manifests'])} manifests over "
            f"{len(record['scope']['registry_quants'])} registry quants, "
            f"{record['scope']['selections']} selections, "
            f"{record['scope']['variants_verified_registered']} variants registered, "
            f"bundle {record['strict_manifest_bundle_sha256'][:16]}…"
        )

    if args.output is not None:
        summary = _stamp_certification(args.output, built)
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
