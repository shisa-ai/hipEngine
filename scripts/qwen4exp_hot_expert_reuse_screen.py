#!/usr/bin/env python3
"""#35 hot-expert weight reuse / next-K pipelining: bounded SCREEN.

Captures per-(layer, chunk) and per-(layer, decode-step) expert selections
for the canonical fixture by patching the router select call, then
quantifies:

PREFILL (hot-expert weight reuse potential):
  - per-layer expert frequency concentration: pair share of the top-K
    hottest experts (K in {8,16,32,64,128});
  - weight-traffic reuse bound: mean fraction of DISTINCT selected experts
    per capture that fall in the layer's hot set (L2-resident reuse);
  - temporal stability: consecutive-chunk expert-set Jaccard (p4096),
    cross-case hot-set (top-32) Jaccard;
  - hot-set weight bytes vs device L2 capacity.

DECODE (next-K prediction potential, arithmetic-preserving prefetch only):
  - consecutive-step expert-set overlap and exact-repeat rate per layer.

No runtime default changes; no arithmetic changes. Screen only.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--case-id", action="append")
    p.add_argument("--decode-steps", type=int, default=48)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not hip_available():
        p.error("HIP runtime unavailable")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        register_qwen4_exp_gfx1151_profiles, QWEN4_EXP_MODEL,
        QWEN4_EXP_BACKEND, QWEN4_EXP_QUANTS,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model
    from scripts.qwen4exp_canonical_ar_bench import (
        DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata,
    )
    import hipengine.runtime.qwen4_exp_runner as runner_module

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
    fixture, digest = load_fixture(DEFAULT_FIXTURE)
    index = load_gguf_index(discover_gguf_files(a.model_root)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=a.model_root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND, max_sequence_length=4352,
        prefill_chunk_size=1024))

    runtime = get_hip_runtime()
    orig_qsa = runner_module.run_qwen4_exp_dense_qsa_layer
    orig_gdn = runner_module.run_qwen4_exp_gdn_layer
    orig_qsa_prefill = runner_module.run_qwen4_exp_qsa_prefill_layer
    # per capture: (phase, case, layer, rows, selected int64 array rows*top_k)
    captures: list[tuple[str, str, int, int, np.ndarray]] = []
    phase = {"name": "prefill", "case": ""}

    def _snap(scratch, layer: int, rows: int, top_k: int) -> None:
        # Between layer calls: no graph capture can be active (captures are
        # confined inside MoeGraphCache.run calls), so a device sync is safe.
        runtime.device_synchronize()
        buf = scratch.moe.selected
        host = np.empty(int(rows) * int(top_k), dtype=np.int64)
        runtime.memcpy(host.ctypes.data, buf.ptr, host.nbytes, 2)  # D2H
        captures.append((phase["name"], phase["case"], int(layer), int(rows),
                         host.copy()))

    def wrap_qsa(residual_ptr, weights, **kw):
        result = orig_qsa(residual_ptr, weights, **kw)
        _snap(kw["scratch"], getattr(weights, "layer_id"),
              int(kw["rows"]), int(kw["top_k"]))
        return result

    def wrap_gdn(residual_ptr, weights, **kw):
        result = orig_gdn(residual_ptr, weights, **kw)
        _snap(kw["scratch"], getattr(weights, "layer_id"),
              int(kw["rows"]), int(kw["top_k"]))
        return result

    def wrap_qsa_prefill(residual_ptr, weights, **kw):
        result = orig_qsa_prefill(residual_ptr, weights, **kw)
        _snap(kw["scratch"], getattr(weights, "layer_id"),
              int(kw["rows"]), int(kw["top_k"]))
        return result

    runner_module.run_qwen4_exp_dense_qsa_layer = wrap_qsa
    runner_module.run_qwen4_exp_gdn_layer = wrap_gdn
    runner_module.run_qwen4_exp_qsa_prefill_layer = wrap_qsa_prefill
    try:
        for case in fixture["cases"]:
            if a.case_id and case["id"] not in a.case_id:
                continue
            phase["name"] = "prefill"
            phase["case"] = case["id"]
            generator.runner.reset()
            first = generator.runner.prefill(case["prompt_token_ids"])
            generator.runner.runtime.device_synchronize()
            token = int(first.token_id)
            phase["name"] = "decode"
            for _ in range(a.decode_steps):
                nxt = generator.runner.step(token)
                generator.runner.runtime.device_synchronize()
                token = int(nxt.token_id)
            print(f"{case['id']}: {len(captures)} router captures so far",
                  flush=True)
    finally:
        runner_module.run_qwen4_exp_dense_qsa_layer = orig_qsa
        runner_module.run_qwen4_exp_gdn_layer = orig_gdn
        runner_module.run_qwen4_exp_qsa_prefill_layer = orig_qsa_prefill

    # ---- reconstruct (layer, chunk) / (layer, step) structure ------------
    cfg = generator.runner.config
    n_layers = int(cfg.block_count)
    experts = int(cfg.expert_count)
    top_k = int(cfg.expert_used_count)

    prefill_cap = [c for c in captures if c[0] == "prefill"]
    decode_cap = [c for c in captures if c[0] == "decode"]

    # captures carry (phase, case, layer, rows, selected); per (case, layer)
    # the call order gives the chunk (prefill) / step (decode) index.
    per_case_layer_seq: dict[tuple[str, int], list] = {}
    for c in prefill_cap:
        per_case_layer_seq.setdefault((c[1], c[2]), []).append(c)

    layer_chunk: dict[tuple[str, int, int], np.ndarray] = {}  # (case,layer,chunk)
    for (case_id, layer), caps in per_case_layer_seq.items():
        for chunk, (_, _, _, rows, sel) in enumerate(caps):
            if len(sel) != rows * top_k:
                raise RuntimeError(f"{case_id} L{layer}: rows/top_k mismatch")
            counts = np.bincount(sel, minlength=experts)
            layer_chunk[(case_id, layer, chunk)] = counts

    per_case_step_seq: dict[tuple[str, int], list] = {}
    for c in decode_cap:
        if c[3] != 1:
            raise RuntimeError("decode capture with rows != 1")
        per_case_step_seq.setdefault((c[1], c[2]), []).append(c)
    layer_step: dict[tuple[str, int, int], np.ndarray] = {}
    for (case_id, layer), caps in sorted(per_case_step_seq.items()):
        for step, (_, _, _, _, sel) in enumerate(caps):
            layer_step[(case_id, layer, step)] = sel.copy()
    n_steps = (max(len(v) for v in per_case_step_seq.values())
               if per_case_step_seq else 0)

    # ---- prefill concentration metrics -----------------------------------
    KS = [8, 16, 32, 64, 128]
    layer_pairs: dict[int, np.ndarray] = {}
    for (case_id, layer, chunk), counts in layer_chunk.items():
        layer_pairs.setdefault(layer, np.zeros(experts, dtype=np.int64))
        layer_pairs[layer] += counts
    total_pairs = int(sum(v.sum() for v in layer_pairs.values()))

    pair_coverage = {K: 0 for K in KS}
    traffic_coverage = {K: [] for K in KS}   # distinct-selected-in-hotset fraction
    hot_sets: dict[int, dict[int, np.ndarray]] = {}
    for layer, counts in layer_pairs.items():
        order = np.argsort(counts)[::-1]
        for K in KS:
            hot = order[:K]
            hot_sets.setdefault(layer, {})[K] = set(hot.tolist())
            pair_coverage[K] += int(counts[hot].sum())
            hotset = set(hot.tolist())
            for (case_id, lyr, chunk), cnt in layer_chunk.items():
                if lyr != layer:
                    continue
                sel = set(np.nonzero(cnt)[0].tolist())
                if sel:
                    traffic_coverage[K].append(len(sel & hotset) / len(sel))

    # temporal stability: consecutive chunks (p4096, 4 chunks), per layer
    jaccards = []
    for (case_id, layer, chunk), counts in layer_chunk.items():
        nxt = layer_chunk.get((case_id, layer, chunk + 1))
        if nxt is None:
            continue
        s0 = set(np.nonzero(counts)[0].tolist())
        s1 = set(np.nonzero(nxt)[0].tolist())
        if s0 or s1:
            jaccards.append(len(s0 & s1) / len(s0 | s1))

    # cross-case hot-set stability (top-32)
    cc_jaccards = []
    case_ids = sorted({k[0] for k in per_case_layer_seq})
    case_hot: dict[str, dict[int, set]] = {}
    for case_id in case_ids:
        for layer in range(n_layers):
            counts = np.zeros(experts, dtype=np.int64)
            for chunk in range(4):
                c = layer_chunk.get((case_id, layer, chunk))
                if c is not None:
                    counts += c
            case_hot.setdefault(case_id, {})[layer] = set(
                np.argsort(counts)[::-1][:32].tolist())
    for i, ca in enumerate(case_ids):
        for cb in case_ids[i + 1:]:
            vals = [
                len(case_hot[ca][l] & case_hot[cb][l]) /
                len(case_hot[ca][l] | case_hot[cb][l])
                for l in range(n_layers)
            ]
            cc_jaccards.append(float(np.mean(vals)))

    # ---- weight bytes -----------------------------------------------------
    resident = generator.runner.resident
    def slot_nbytes(layer, slot):
        tensor = resident.weight(f"layers.{layer}.{slot}").allocation("raw").tensor
        count = 1
        for dim in tensor.shape:
            count *= int(dim)
        return count * tensor.dtype.itemsize
    per_expert_bytes = {}
    for layer in range(n_layers):
        per_expert_bytes[layer] = sum(
            slot_nbytes(layer, s)
            for s in ("expert_gate", "expert_up", "expert_down")) / experts

    # L2 exact size unquerable on this stack (enum unavailable); hot-set
    # bytes are orders of magnitude above any plausible L2 (2-8 MB class).
    l2_bytes = None

    hot_bytes = {K: int(sum(per_expert_bytes[l] * K for l in range(n_layers)))
                 for K in KS}

    # saturation / per-capture weight traffic
    distinct_per_capture = [
        int(np.count_nonzero(counts)) for counts in layer_chunk.values()]
    traffic_per_capture_bytes = [
        float(distinct_per_capture[i] * per_expert_bytes[key[1]])
        for i, key in enumerate(layer_chunk.keys())]

    # within-expert tile re-read ceiling (R13 ladder item 3 mechanism):
    # worst case, an expert's weight tile is re-read once per 16-row padded
    # fragment; perfect within-expert reuse would read it once per selection.
    tile_rows = 16
    rereads_today = sum(
        int(np.ceil(counts[counts > 0] / tile_rows).sum())
        for counts in layer_chunk.values())
    rereads_ideal = sum(distinct_per_capture)

    # ---- decode next-K prediction metrics --------------------------------
    overlaps, exact = [], 0
    total_steps = 0
    for (case_id, layer, step), sel in layer_step.items():
        nxt = layer_step.get((case_id, layer, step + 1))
        if nxt is None:
            continue
        total_steps += 1
        s0, s1 = set(sel.tolist()), set(nxt.tolist())
        overlaps.append(len(s0 & s1) / top_k)
        if s0 == s1:
            exact += 1

    report = {
        "schema": 1,
        "kind": "qwen4exp_hot_expert_reuse_screen",
        "source": _git_metadata(ROOT), "host": _host_metadata(),
        "command": sys.argv, "fixture_sha256": digest,
        "model": {
            "layers": n_layers, "experts": experts, "top_k": top_k,
            "per_expert_bytes_mean": float(np.mean(list(per_expert_bytes.values()))),
            "per_layer_expert_weights_bytes_mean": float(
                np.mean([per_expert_bytes[l] * experts for l in range(n_layers)])),
            "l2_cache_bytes": l2_bytes,
            "l2_note": ("exact enum unquerable; hot-set bytes below are "
                        "orders of magnitude above any plausible L2"),
        },
        "prefill": {
            "captures": len(prefill_cap),
            "total_pairs": total_pairs,
            "distinct_experts_per_capture_mean": float(
                np.mean(distinct_per_capture)),
            "selected_expert_weight_traffic_per_capture_gb_mean": float(
                np.mean(traffic_per_capture_bytes)) / 1e9,
            "within_expert_tile_reread_ratio": {
                "tile_rows": tile_rows,
                "fragments_today": int(rereads_today),
                "ideal_reads": int(rereads_ideal),
                "ratio": float(rereads_today / rereads_ideal),
                "assumption": ("worst case: expert weight tile re-read once per "
                                "16-row padded fragment; ideal: once per "
                                "selection (perfect within-expert L2 reuse)"),
            },
            "pair_coverage_of_topK": {str(K): pair_coverage[K] / total_pairs
                                      for K in KS},
            "distinct_expert_weight_traffic_in_hotset": {
                str(K): float(np.mean(traffic_coverage[K])) for K in KS},
            "consecutive_chunk_jaccard_mean": float(np.mean(jaccards))
            if jaccards else None,
            "consecutive_chunk_jaccard_n": len(jaccards),
            "cross_case_top32_jaccard_mean": float(np.mean(cc_jaccards))
            if cc_jaccards else None,
            "hotset_bytes_per_K": {str(K): hot_bytes[K] for K in KS},
        },
        "decode": {
            "steps": n_steps,
            "consecutive_step_overlap_mean": float(np.mean(overlaps)),
            "exact_repeat_rate": exact / total_steps if total_steps else None,
        },
        "note": (
            "Screen only. Pair coverage answers 'what fraction of token-expert "
            "pairs hit a small hot set'; distinct-expert traffic coverage bounds "
            "the weight-read reuse from keeping the hot set resident; decode "
            "overlap bounds next-K prefetch (same-set prediction) hit rate. "
            "No arithmetic changes were made or proposed by this screen."
        ),
    }
    a.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("prefill", "decode", "model")},
                     indent=1))


if __name__ == "__main__":
    main()
