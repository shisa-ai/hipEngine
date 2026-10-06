#!/usr/bin/env python3
"""Qualify the Qwen4Exp heterogeneous Q5_1/Q8_0 expert-down path.

The natural-input population census established the model's quant split, but
that census did not prove that the production runtime dispatches each family
through its intended down owner.  This gate counts the registered down-owner
calls during real prefill requests and checks their per-layer order, complete
operation pairing, and coverage over the canonical categories plus heldouts.

This is an engagement/ownership gate, not a throughput claim.  The named
production profile remains unchanged and no quantization is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    QWEN4_EXP_QUANTS,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.gguf_gdn_semantic_gate import _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE,
    _git_metadata,
    _host_metadata,
    load_fixture,
)
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity

Q51_RISK_KEY = KernelKey(
    "hip_gfx1151", "moe_linear", "gguf_q5_1",
    "selected_wmma_iu8_risk_prefill_bf16_bf16_out",
)
Q51_REPAIR_KEY = KernelKey(
    "hip_gfx1151", "moe_linear", "gguf_q5_1",
    "selected_sparse_exact_repair_row_publish_bf16",
)
Q8_GUARDED_KEY = KernelKey(
    "hip_gfx1151", "linear", "gguf_q8_0",
    "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out",
)
Q8_LAYERS = (2, 4, 30, 46, 47)
Q51_LAYERS = tuple(layer for layer in range(48) if layer not in Q8_LAYERS)
EXPECTED_ORDER = tuple("q8" if layer in Q8_LAYERS else "q51" for layer in range(48))


def expected_chunks(tokens: int, chunk: int) -> int:
    """Return the number of non-empty prefill chunks."""
    if tokens < 1 or chunk < 1:
        raise ValueError("tokens and chunk must be positive")
    return (tokens + chunk - 1) // chunk


def expected_owner_calls(tokens: int, chunk: int) -> dict[str, int]:
    """Return expected Q5_1 and Q8_0 down-owner calls for a request."""
    chunks = expected_chunks(tokens, chunk)
    return {"q51": len(Q51_LAYERS) * chunks, "q8": len(Q8_LAYERS) * chunks}


def validate_quant_split(quant_triplets: list[tuple[str, str, str]]) -> dict[str, Any]:
    """Validate the immutable model quant triplet for all 48 MoE layers."""
    if len(quant_triplets) != 48:
        raise ValueError(f"expected 48 layer quant triplets, got {len(quant_triplets)}")
    normalized = [tuple(map(str, row)) for row in quant_triplets]
    bad = []
    for layer, triplet in enumerate(normalized):
        expected_gate_up = ("gguf_q5_k", "gguf_q5_k") if layer == 2 else ("gguf_q4_k", "gguf_q4_k")
        expected_down = "gguf_q8_0" if layer in Q8_LAYERS else "gguf_q5_1"
        if triplet != (*expected_gate_up, expected_down):
            bad.append({"layer": layer, "actual": list(triplet),
                        "expected_gate_up": list(expected_gate_up),
                        "expected_down": expected_down})
    if bad:
        raise ValueError(f"unexpected model quant split: {bad}")
    return {
        "layers": 48,
        "q51_layers": list(Q51_LAYERS),
        "q8_layers": list(Q8_LAYERS),
        "quant_triplets": [list(row) for row in normalized],
    }


def _case_rows(fixture: dict[str, Any], heldout_path: Path, tokenizer: Any) -> list[dict[str, Any]]:
    cases = []
    for case in fixture["cases"]:
        cases.append({
            "id": str(case["id"]),
            "category": str(case["category"]),
            "prompt_token_ids": [int(value) for value in case["prompt_token_ids"]],
            "source": "canonical",
        })
    for row in _load_suites([heldout_path]):
        tokens = list(build_chat_prompt(tokenizer, str(row["prompt"])))
        cases.append({
            "id": str(row["id"]),
            "category": str(row["category"]),
            "prompt_token_ids": [int(value) for value in tokens],
            "source": "heldout",
        })
    return cases


def _install_owner_counters() -> tuple[dict[str, Any], dict[KernelKey, Any]]:
    """Wrap the three registered leaves and return counters plus originals."""
    counters: dict[str, Any] = {"q51_risk": 0, "q51_repair": 0, "q8_guarded": 0, "order": []}
    originals: dict[KernelKey, Any] = {}
    for label, key in (
        ("q51_risk", Q51_RISK_KEY),
        ("q51_repair", Q51_REPAIR_KEY),
        ("q8_guarded", Q8_GUARDED_KEY),
    ):
        original = resolve(backend=key.backend, layer=key.layer,
                           quant=key.quant, variant=key.variant)
        originals[key] = original

        def counted(*args: Any, _label: str = label, _original: Any = original, **kwargs: Any) -> Any:
            counters[_label] += 1
            if _label == "q51_risk":
                counters["order"].append("q51")
            elif _label == "q8_guarded":
                counters["order"].append("q8")
            return _original(*args, **kwargs)

        register(key, counted, replace=True)
    return counters, originals


def _restore_owner_counters(originals: dict[KernelKey, Any]) -> None:
    for key, original in originals.items():
        register(key, original, replace=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-file", type=Path,
                        default=ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl")
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--allow-jit-build", action="store_true",
                        help="Allow the preparation run to build uncached kernels")
    args = parser.parse_args()
    if args.prefill_chunk_size < 1:
        parser.error("--prefill-chunk-size must be positive")

    check_host()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if args.allow_jit_build:
        os.environ.pop("HIPENGINE_REQUIRE_CACHED_BUILD", None)
    else:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION,
    )
    fixture, fixture_sha256 = load_fixture(DEFAULT_FIXTURE)
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root,
        weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend="hip_gfx1151",
        max_sequence_length=4352,
        prefill_chunk_size=args.prefill_chunk_size,
    ))
    counters, originals = _install_owner_counters()
    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_ed3_mixed_quant_owner_gate",
        "status": "running",
        "performance_claim": False,
        "command": sys.argv,
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "model_identity": model_identity(args.model_root),
        "fixture_sha256": fixture_sha256,
        "heldout_sha256": hashlib.sha256(args.heldout_file.read_bytes()).hexdigest(),
        "manifest_sha256": profile.manifest_sha256,
        "strict_manifest_sha256": profile.strict_manifest_sha256,
        "execution_profile": profile.profile.value,
        "prefill_chunk_size": args.prefill_chunk_size,
        "owners": {
            "q51_risk": Q51_RISK_KEY.variant,
            "q51_repair": Q51_REPAIR_KEY.variant,
            "q8_guarded": Q8_GUARDED_KEY.variant,
        },
        "cases": [],
    }
    try:
        cases = _case_rows(fixture, args.heldout_file, generator.tokenizer)
        quant_triplets = []
        for layer in range(48):
            prefix = f"layers.{layer}."
            quant_triplets.append(tuple(
                str(generator.runner.resident.weight(prefix + slot).spec.quant_key)
                for slot in ("expert_gate", "expert_up", "expert_down")
            ))
        report["quant_split"] = validate_quant_split(quant_triplets)
        expected_order = list(EXPECTED_ORDER)
        for case in cases:
            tokens = case["prompt_token_ids"]
            counters.update(q51_risk=0, q51_repair=0, q8_guarded=0)
            counters["order"] = []
            before = {key: counters[key] for key in ("q51_risk", "q51_repair", "q8_guarded")}
            result = generator.runner.prefill(tokens)
            generator.runner.runtime.device_synchronize()
            deltas = {key: counters[key] - before[key] for key in before}
            chunks = expected_chunks(len(tokens), args.prefill_chunk_size)
            expected = expected_owner_calls(len(tokens), args.prefill_chunk_size)
            observed_order = list(counters["order"])
            expected_case_order = expected_order * chunks
            if deltas["q51_risk"] != expected["q51"]:
                raise AssertionError((case["id"], "q51_risk", deltas, expected))
            if deltas["q51_repair"] != expected["q51"]:
                raise AssertionError((case["id"], "q51_repair", deltas, expected))
            if deltas["q8_guarded"] != expected["q8"]:
                raise AssertionError((case["id"], "q8_guarded", deltas, expected))
            if observed_order != expected_case_order:
                raise AssertionError((case["id"], "owner order", observed_order, expected_case_order))
            if not bool(result.logits is not None) or not all(
                map(bool, __import__("numpy").isfinite(result.logits))
            ):
                raise AssertionError((case["id"], "non-finite logits"))
            report["cases"].append({
                "id": case["id"], "category": case["category"], "source": case["source"],
                "prompt_tokens": len(tokens), "chunks": chunks,
                "expected_calls": expected, "observed_calls": deltas,
                "owner_order_sha256": hashlib.sha256(
                    ("".join(observed_order)).encode("ascii")
                ).hexdigest(),
                "token_id": int(result.token_id),
                "logits_finite": True,
            })
        report["status"] = "passed"
        report["coverage"] = {
            "canonical_cases": sum(row["source"] == "canonical" for row in report["cases"]),
            "heldout_cases": sum(row["source"] == "heldout" for row in report["cases"]),
            "categories": sorted({row["category"] for row in report["cases"]}),
        }
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        _restore_owner_counters(originals)
        generator.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
