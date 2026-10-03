#!/usr/bin/env python3
"""Observe attention launcher invocations during a public Gemma LLM.generate request.

Selection telemetry counts successful wrapper invocations, not merely resolution
or generated-ID parity. GPU completion is checked separately with synchronize.
This probe observes the product route; it does not override variant selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@contextmanager
def observe_attention_launches():
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer

    original = layer.select_prefill_attention
    records = []

    def observed(**geometry):
        selection = original(**geometry)
        launch = selection.launcher

        def invoked(*args, **kwargs):
            result = launch(*args, **kwargs)
            records.append({
                "variant": selection.variant,
                "reason": selection.reason,
                **{name: geometry.get(name) for name in
                   ("tokens", "keys", "head_dim", "num_heads", "num_kv_heads")},
            })
            return result

        return replace(selection, launcher=invoked)

    layer.select_prefill_attention = observed
    try:
        yield records
    finally:
        layer.select_prefill_attention = original


def summarize_attention_launches(records):
    """Count by query width, not semantic phase: prefill can end in one row."""
    counts = Counter((r["variant"], "singleton" if r["tokens"] == 1 else "multi_token")
                     for r in records)
    return [{"variant": variant, "query_width": width, "invocations": count}
            for (variant, width), count in sorted(counts.items())]


def main(argv=None):
    from scripts.gemma4_campaign_bench import DEFAULT_ARTIFACT, exact_prompt_ids

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--prompt", type=int, default=8192)
    parser.add_argument("--output", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.prompt < 1 or args.output < 1:
        parser.error("prompt and output must be positive")

    import hipengine
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.core.hip import get_hip_runtime
    from hipengine.llm import SamplingParams

    llm = hipengine.LLM(model=str(args.artifact), max_sequence_length=args.prompt + args.output)
    generator = llm._get_text_generator()
    ids = exact_prompt_ids(generator.tokenize, args.prompt)
    try:
        with observe_attention_launches() as records:
            outputs = llm.generate(ids, SamplingParams(max_tokens=args.output,
                                                       temperature=0.0, ignore_eos=True))
            get_hip_runtime().device_synchronize()
        counts = summarize_attention_launches(records)
        profile = getattr(llm, "_resolved_execution_profile", None)
        sources = {}
        for path in sorted((_ROOT / "hipengine/kernels/hip_gfx1100/gemma4").glob("gemma4_attention*")):
            if path.suffix in (".hip", ".py"):
                sources[str(path.relative_to(_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = {
            "kind": "gemma4_public_attention_route_observation",
            "performance_claim": False,
            "prompt_tokens": len(ids), "requested_output_tokens": args.output,
            "surface": "hipengine.LLM.generate", "device_synchronize_passed": True,
            "execution_profile_manifest": getattr(profile, "manifest", None),
            "manifest_sha256": getattr(profile, "manifest_sha256", None),
            "source_sha256": sources,
            "counts": counts,
            "launches": records,
            "outputs": outputs,
            "provenance": collect_artifact_provenance(
                repo_root=_ROOT, model_path=args.artifact, quant="UD-Q4_K_XL", kv_dtype="bf16"),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=dict) + "\n")
        print(json.dumps({"counts": report["counts"], "manifest_sha256": report["manifest_sha256"]}))
        return 0 if records and outputs else 1
    finally:
        llm.close()


if __name__ == "__main__":
    raise SystemExit(main())
