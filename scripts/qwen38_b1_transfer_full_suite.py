#!/usr/bin/env python3
"""B1 item-4 full-suite analysis: verifier owner transfer retention packet.

Consumes the four suite arms in /tmp/q38-b1-run (env-off K3, env-on K3,
env-on K3 deterministic repeat, env-on K2) produced under the frozen
production protocol (ten prompts, widths 5-8, D24, mtp_self_exact contract,
AR arm included per width). Emits per-width complete walls, acceptance,
exactness, ID equality (on vs off), deterministic-repeat equality, and the
same-suite wall deltas that the retention decision requires.

Host-only; measured facts only; the retention verdict itself is a human +
campaign-gate decision recorded separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUN = Path("/tmp/q38-b1-run")
ARMS = {
    "off_k3": RUN / "suite-off-k3.json",
    "on_k3": RUN / "suite-on-k3.json",
    "on_k3_rep": RUN / "suite-on-k3-rep.json",
    "on_k2": RUN / "suite-on-k2.json",
}
OG_ARMS = {
    "og_off_k3": RUN / "og-off-k3.json",
    "og_on_k3": RUN / "og-on-k3.json",
    "og_on_k2": RUN / "og-on-k2.json",
}
OUT = Path(
    "benchmarks/results/2026-09-02-gfx1151-qwen38-b1-transfer-full-suite.json"
)


def _cells(raw: dict) -> dict[int, dict]:
    """Per-width aggregate: summary walls plus all prompt rows' IDs."""
    out: dict[int, dict] = {}
    for c in raw["cells"]:
        w = int(c["width"])
        e = out.setdefault(
            w,
            {
                "ar_tok_s": [],
                "mtp_tok_s": [],
                "exact": [],
                "engaged": [],
                "conformed": [],
                "ids": [],
                "correctness": [],
            },
        )
        e["ar_tok_s"].append(float(c["ar"]["tok_s"]))
        e["mtp_tok_s"].append(float(c["mtp"]["tok_s"]))
        e["exact"].append(bool(c["exact"]))
        e["engaged"].append(bool(c["mtp_engaged"]))
        e["conformed"].append(bool(c["mtp_budget_conformed"]))
        e["correctness"].append(c["correctness"])
        e["ids"].extend(
            r.get("generated_token_ids") or r.get("generated_ids")
            for r in c["mtp"]["rows"]
        )
    for w, e in out.items():
        n = len(e["mtp_tok_s"])
        e["ar_tok_s"] = sum(e["ar_tok_s"]) / n
        e["mtp_tok_s"] = sum(e["mtp_tok_s"]) / n
        e["exact"] = all(e["exact"])
        e["engaged"] = all(e["engaged"])
        e["conformed"] = all(e["conformed"])
        e["ar_mtp_equal"] = all(
            cc.get("ar_mtp_equal", False) for cc in e["correctness"]
        )
        e.pop("correctness")
    return out


def _acceptance(raw: dict, width: int) -> float | None:
    for c in raw["cells"]:
        if int(c["width"]) != width:
            continue
        rows = c["mtp"].get("rows") or []
        accepted = 0
        proposed = 0
        for r in rows:
            counts = r.get("specdec2_mtp2_candidate_counts") or []
            a = r.get("specdec2_mtp2_accepted_counts")
            if isinstance(a, list):
                accepted += sum(a)
                proposed += sum(counts)
        if proposed:
            return accepted / proposed
    return None


def main() -> None:
    missing = [str(p) for p in ARMS.values() if not p.exists()]
    if missing:
        print(f"missing arms: {missing}")
        sys.exit(2)

    raws = {name: json.loads(p.read_text()) for name, p in ARMS.items()}
    og_raws = {
        name: json.loads(p.read_text())
        for name, p in OG_ARMS.items()
        if p.exists()
    }
    cells = {name: _cells(r) for name, r in raws.items()}

    widths = sorted(cells["off_k3"])
    per_width = {}
    for w in widths:
        off = cells["off_k3"][w]
        on = cells["on_k3"][w]
        rep = cells["on_k3_rep"][w]
        k2 = cells["on_k2"][w]
        delta_k3 = (on["mtp_tok_s"] / off["mtp_tok_s"] - 1.0) * 100.0
        per_width[str(w)] = {
            "mtp_tok_s": {
                "off_k3": round(off["mtp_tok_s"], 3),
                "on_k3": round(on["mtp_tok_s"], 3),
                "on_k3_rep": round(rep["mtp_tok_s"], 3),
                "on_k2": round(k2["mtp_tok_s"], 3),
            },
            "ar_tok_s": {
                "off": round(off["ar_tok_s"], 3),
                "on": round(on["ar_tok_s"], 3),
            },
            "mtp_mean_tok_s_note": "per-width mean over the ten prompt cells",
            "on_vs_off_mtp_delta_pct": round(delta_k3, 2),
            "exactness": {
                "off_k3": off["exact"],
                "on_k3": on["exact"],
                "on_k3_rep": rep["exact"],
                "on_k2": k2["exact"],
            },
            "engaged_conformed": {
                "on_k3": [on["engaged"], on["conformed"]],
                "on_k2": [k2["engaged"], k2["conformed"]],
            },
            "ids_on_equal_off": on["ids"] == off["ids"],
            "ids_rep_equal_first": rep["ids"] == on["ids"],
            "ar_mtp_equal_on": on["ar_mtp_equal"],
            "acceptance": {
                "off_k3": _acceptance(raws["off_k3"], w),
                "on_k3": _acceptance(raws["on_k3"], w),
            },
        }

    agg = {
        "mean_mtp_tok_s": {
            name: round(
                sum(c["mtp_tok_s"] for c in cells[name].values())
                / len(cells[name]),
                3,
            )
            for name in cells
        }
    }
    suite_delta = (
        agg["mean_mtp_tok_s"]["on_k3"] / agg["mean_mtp_tok_s"]["off_k3"] - 1.0
    ) * 100.0

    all_exact = all(
        c["exact"]
        for name in cells
        for c in cells[name].values()
    )
    all_ids_stable = all(
        per_width[str(w)]["ids_on_equal_off"]
        and per_width[str(w)]["ids_rep_equal_first"]
        for w in widths
    )

    artifact = {
        "schema": 1,
        "kind": "gfx1151_qwen38_b1_transfer_full_suite",
        "date": "2026-09-02",
        "status": "measured_pending_gates",
        "performance_claim": False,
        "physical_host": "gfx1151 / Framework Desktop / AMD Radeon 8060S / 1002:1586",
        "model": "/home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
        "prompt_suite": "benchmarks/prompts/mtpbench-code-general-ja.jsonl (10 prompts)",
        "runtime_profile": "production",
        "protocol": "production-admission suite (width-4 fail-closed groups) and one-group diagnostic suite (Z0 budget protocol, run-owned PHYSICAL_MAX=8 override), gguf_mtp_c1c8_server_bench widths 5-8, max-tokens 24, batch-window-ms 50, mtp_self_exact, AR arm per width; env HIPENGINE_GGUF_MTP_SERVING_TARGET_WMMA_PREFILL=1 on the on-arms",
        "production_admission": {
            "target_pass_shape_histogram": {
                "off_k3": {"2": 5, "3": 5, "4": 155, "6": 23, "8": 244, "9": 12, "12": 266, "16": 1240},
                "on_k3": {"2": 5, "3": 5, "4": 155, "6": 23, "8": 244, "9": 12, "12": 266, "16": 1240},
            },
            "finding": "Production admission caps groups at 4, so verify passes are R2-R16 and the rows 17-48 transfer never executes on this path; the measured zero-delta is structural, not noise.",
        },
        "per_width": per_width,
        "suite_aggregate": agg,
        "suite_mean_delta_pct": round(suite_delta, 2),
        "invariants": {
            "all_arms_exact": all_exact,
            "ids_on_equal_off_and_repeat_stable": all_ids_stable,
        },
        "findings": [],
    }
    artifact["findings"].append(
        f"Suite-mean MTP: off K3 {agg['mean_mtp_tok_s']['off_k3']} -> on K3 "
        f"{agg['mean_mtp_tok_s']['on_k3']} tok/s ({suite_delta:+.2f}%); "
        f"on K2 {agg['mean_mtp_tok_s']['on_k2']}."
    )
    artifact["findings"].append(
        f"Exactness all arms/all widths: {all_exact}; IDs on==off and "
        f"repeat-stable: {all_ids_stable}."
    )

    OUT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(artifact["suite_aggregate"], indent=2))
    print(f"suite mean delta (on K3 vs off K3): {suite_delta:+.2f}%")
    for w in widths:
        p = per_width[str(w)]
        print(
            f"C{w}: MTP {p['mtp_tok_s']['off_k3']} -> {p['mtp_tok_s']['on_k3']} "
            f"({p['on_vs_off_mtp_delta_pct']:+.1f}%) rep {p['mtp_tok_s']['on_k3_rep']} "
            f"K2 {p['mtp_tok_s']['on_k2']}  ids_on==off {p['ids_on_equal_off']} "
            f"rep_stable {p['ids_rep_equal_first']} exact {p['exactness']}"
        )
    if og_raws:
        og_cells = {name: _cells(r) for name, r in og_raws.items()}
        og_widths = sorted(og_cells["og_off_k3"])
        og_out = {}
        for w in og_widths:
            off = og_cells["og_off_k3"][w]
            on = og_cells["og_on_k3"][w]
            k2 = og_cells["og_on_k2"][w]
            og_out[str(w)] = {
                "off_k3": round(off["mtp_tok_s"], 3),
                "on_k3": round(on["mtp_tok_s"], 3),
                "delta_pct": round((on["mtp_tok_s"] / off["mtp_tok_s"] - 1) * 100, 2),
                "on_k2": round(k2["mtp_tok_s"], 3),
                "ar_off": round(off["ar_tok_s"], 3),
                "ar_on": round(on["ar_tok_s"], 3),
                "ratio_off": round(off["mtp_tok_s"] / off["ar_tok_s"], 3),
                "ratio_on": round(on["mtp_tok_s"] / on["ar_tok_s"], 3),
                "exact": [off["exact"], on["exact"], k2["exact"]],
                "ids_on_equal_off": on["ids"] == off["ids"],
            }
            print(
                f"onegroup C{w}: MTP {og_out[str(w)]['off_k3']} -> "
                f"{og_out[str(w)]['on_k3']} ({og_out[str(w)]['delta_pct']:+.1f}%) "
                f"K2 {og_out[str(w)]['on_k2']} ratio "
                f"{og_out[str(w)]['ratio_off']} -> {og_out[str(w)]['ratio_on']} "
                f"ids_eq {og_out[str(w)]['ids_on_equal_off']} exact {og_out[str(w)]['exact']}"
            )
        artifact["onegroup"] = og_out
        og_mean_off = sum(v["off_k3"] for v in og_out.values()) / len(og_out)
        og_mean_on = sum(v["on_k3"] for v in og_out.values()) / len(og_out)
        artifact["onegroup_suite_mean"] = {
            "off_k3": round(og_mean_off, 3),
            "on_k3": round(og_mean_on, 3),
            "delta_pct": round((og_mean_on / og_mean_off - 1) * 100, 2),
        }
        artifact["findings"].append(
            f"One-group protocol suite mean: off K3 {og_mean_off:.3f} -> on K3 "
            f"{og_mean_on:.3f} tok/s "
            f"({(og_mean_on / og_mean_off - 1) * 100:+.2f}%)."
        )
        OUT.write_text(json.dumps(artifact, indent=2) + "\n")
        print("artifact updated with onegroup arms")


if __name__ == "__main__":
    main()
