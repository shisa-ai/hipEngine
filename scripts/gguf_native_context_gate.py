#!/usr/bin/env python3
"""Diagnostic context qualification using the existing dense AR/MTP suite.

Overrides are process-local and restored on failure. Synthetic fixed-length
prompts are correctness diagnostics, never acceptance/speed promotion evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextmanager
def native_context_override(package, limit):
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("native context limit must be a positive integer")
    names = (
        "GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT",
        "GGUF_SPECDEC2_NATIVE_TARGET_GRAPH_MAX_CONTEXT",
    )
    prior = {name: getattr(package, name) for name in names}
    try:
        if limit is not None:
            for name in names:
                setattr(package, name, limit)
        yield prior
    finally:
        for name, value in prior.items():
            setattr(package, name, value)


def fixed_prompt(tokens, length):
    tokens = list(tokens)
    if not tokens:
        raise ValueError("prompt must not be empty")
    if length is None:
        return tokens
    if type(length) is not int or length <= 0:
        raise ValueError("fixed prompt length must be positive")
    return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]


def main():
    from scripts import qwen36_dense_gguf_suite as suite
    from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier
    from hipengine.util.amdgpu_vram import VramSampler, select_card

    parser = suite.build_parser()
    parser.description = __doc__
    parser.add_argument("--native-context-limit", type=int)
    parser.add_argument("--fixed-prompt-length", type=int)
    parser.add_argument("--native-eager", action="store_true")
    parser.add_argument("--backend", choices=("hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument("--pci", default="0000:10:00.0")
    args = parser.parse_args()
    if args.native_context_limit is not None and args.native_context_limit <= 0:
        parser.error("native context limit must be positive")
    if args.fixed_prompt_length is not None and args.fixed_prompt_length <= 0:
        parser.error("fixed prompt length must be positive")
    if args.output.exists():
        parser.error("output already exists; preserve previous attempts")
    ctypes.CDLL("libamdhip64.so")
    card = select_card(pci_id=args.pci)
    if int(card.vram_used_path.read_text()) > 128 * (1 << 20):
        raise RuntimeError("selected GPU is not idle")
    package = importlib.import_module(f"hipengine.kernels.{args.backend}")
    original_prompt = suite.build_chat_prompt
    original_prepare = Qwen35GGUFTransactionalVerifier.prepare
    prompts = {}
    actual_modes = {}
    graph_submissions = 0
    graph_extents = set()
    identity_checked = False
    device_metadata = {}

    def prompt(*positional, **keywords):
        tokens = fixed_prompt(original_prompt(*positional, **keywords), args.fixed_prompt_length)
        prompts[str(positional[1])] = list(map(int, tokens))
        return tokens

    def prepare(self, *positional, **keywords):
        nonlocal graph_submissions, identity_checked
        if not identity_checked:
            if self.backend != args.backend:
                raise RuntimeError("target backend differs from diagnostic override")
            from hipengine.benchmark.provenance import collect_artifact_provenance
            identity = collect_artifact_provenance(
                repo_root=ROOT, configured_backend=args.backend, resolved_backend=self.backend,
                target_arch=self.target.runner.target_arch, model_path=args.model, quant=args.quant,
                kv_dtype="bf16", command=tuple(sys.argv), timing_protocol="context qualification",
                warmups=int(args.warmup), repetitions=args.runs,
            )
            device_metadata.update(identity)
            identity_checked = True
        if args.native_eager:
            keywords["allow_graph"] = False
        result = original_prepare(self, *positional, **keywords)
        actual_modes[result.target_verify_mode] = actual_modes.get(result.target_verify_mode, 0) + 1
        graph_submissions += int(result.native_graph_submitted)
        for graph in getattr(self.target, "_native_spec_target_graphs", {}).values():
            graph_extents.add(int(graph.context_limit))
        return result

    original_ready = Qwen35GGUFTransactionalVerifier.device_proposal_ready

    def ready(self, *positional, **keywords):
        if args.native_eager:
            return False
        return original_ready(self, *positional, **keywords)

    payload = {}
    sampler = VramSampler(card=card, interval_ms=20)
    sampler.start()
    error = None
    prior = {}
    try:
        with (
            native_context_override(package, args.native_context_limit) as prior,
            patch.object(suite, "build_chat_prompt", prompt),
            patch.object(Qwen35GGUFTransactionalVerifier, "prepare", prepare),
            patch.object(Qwen35GGUFTransactionalVerifier, "device_proposal_ready", ready),
        ):
            payload = suite.run(args)
    except Exception:
        error = traceback.format_exc()
        payload = {"status": "failed", "error": error}
    finally:
        sampler.stop()
        payload["performance_claim"] = False
        payload["speed_claim_eligible"] = False
        payload["context_gate"] = {
            "schema": 1, "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "environment": {k: v for k, v in os.environ.items()
                            if k.startswith(("HIP", "ROCR", "GPU_MAX", "HSA"))},
            "native_context_limit": args.native_context_limit, "prior_limits": prior,
            "backend": args.backend, "fixed_prompt_length": args.fixed_prompt_length,
            "native_eager": args.native_eager, "prompt_ids": prompts,
            "actual_verify_modes": actual_modes, "graph_submissions": graph_submissions,
            "graph_context_extents": sorted(graph_extents),
            "memory": sampler.result().to_dict(), "identity": device_metadata,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "modes": actual_modes,
                      "graphs": graph_submissions, "extents": sorted(graph_extents),
                      "error": error}, indent=2))
    return 0 if payload["status"] == "complete_exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())
