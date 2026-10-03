"""Does the default path actually run the MMQ route, and does =0 roll it back?

Exercises hipengine.LLM.generate(), the surface a user reaches, rather than the
kernel entry point the unit tests call. Loads once and generates twice, because
the route predicate is read per projection rather than at load time.
"""
import json
import os
import sys

import hipengine
from hipengine.llm import SamplingParams
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as experts
from scripts.gemma4_campaign_bench import resolve_artifact

PROMPT = ("def merge_intervals(intervals):\n"
          '    """Merge overlapping intervals and return the disjoint union."""\n')


def routes() -> dict:
    return dict(experts.gemma4_moe_expert_route_counts())


def main() -> int:
    os.environ.pop("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", None)
    print("default at import:", experts.gemma4_moe_gate_up_mmq_enabled(), flush=True)
    llm = hipengine.LLM(model=str(resolve_artifact()))
    print("loaded", flush=True)

    report = {}
    for label, value in (("default_unset", None), ("rollback_0", "0")):
        if value is None:
            os.environ.pop("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", None)
        else:
            os.environ["HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ"] = value
        before = routes()
        out = llm.generate(PROMPT, SamplingParams(max_tokens=8))
        text = out[0] if isinstance(out, list) else out
        text = getattr(text, "text", text)
        after = routes()
        delta = {k: v - before.get(k, 0) for k, v in after.items() if v - before.get(k, 0)}
        report[label] = {
            "env": value,
            "enabled": experts.gemma4_moe_gate_up_mmq_enabled(),
            "route_delta": delta,
            "mmq_ran": any("mmq" in k for k in delta),
            "text_head": text[:60].replace("\n", "\\n"),
        }
        print(f"{label}: enabled={report[label]['enabled']} mmq_ran={report[label]['mmq_ran']} "
              f"delta={delta}", flush=True)

    ok = report["default_unset"]["mmq_ran"] and not report["rollback_0"]["mmq_ran"]
    report["verdict"] = "PASS" if ok else "FAIL"
    print(json.dumps(report, indent=2), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
