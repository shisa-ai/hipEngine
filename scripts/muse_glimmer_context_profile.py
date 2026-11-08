#!/usr/bin/env python3
"""Short-vs-long context DFlash/AR phase breakdown for Muse Glimmer (diagnostic).

Per context: builds a coding-agent prompt (repo source + instruction, optional
needle at depth 0.25) trimmed to N tokens, runs AR and DFlash greedy through
the public generator / provider, checks DFlash IDs == AR, and times each DFlash
cycle phase with device syncs (draft_block, verify_with_taps, inject; inject
before the first verify is prefill). ``--ar-steps`` times graph decode steps
directly after the DFlash run. Contexts above the artifact's trained positions
run with a warning.

``--context-dir DIR`` keeps a KV snapshot per prompt (target + drafter, via
``save_context``/``load_context``). The first run of a context saves it after
the DFlash pass; later runs load it, and the prefix cache then prefills only
the last partial chunk, so a 256K profile skips its ~30 min prefill. Rows
report ``*_cached_tokens``; prefill times from a loaded snapshot are not
prefill measurements. A 256K snapshot is ~14 GB of target KV plus ~1.4 GB of
drafter KV.

Kernel breakdown: run under rocprofv3 with ``ROCTX_LIB`` pointing at
librocprofiler-sdk-roctx.so; each phase becomes a ``<ctx>:<phase>`` marker
range. Prebuild caches first (one unprofiled run), then profile with
HIPENGINE_REQUIRE_CACHED_BUILD=1 and HIPENGINE_COMPILER_VERSION_FILE
(docs/OPTIMIZATION.md §7):

  rocprofv3 --kernel-trace --marker-trace --output-format csv -d OUT -o run -- \
    .venv/bin/python scripts/muse_glimmer_context_profile.py \
    --contexts 512,32768,126000 --max-tokens 64 --no-ar --ar-steps 16 --output OUT/run.json
  python3 scripts/rocprof_marker_breakdown.py OUT/run_kernel_trace.csv OUT/run_marker_api_trace.csv
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MD = Path("/models/gguf/Muse-Glimmer-30B-GGUF")  # override with --model-dir

INSTR = ("\n\n---\nAbove is source code from the hipEngine repository. Task: write a new "
         "Python module `hipengine/tools/kv_report.py` with a function "
         "`summarize_kv(session)` that reports per-layer KV-cache bytes, live length and "
         "SWA vs full layers for a MuseGlimmerResidentSession, plus pytest tests. Return only code.")


def corpus() -> str:
    parts = []
    for p in sorted((REPO / "hipengine").rglob("*.py")):
        parts.append(f"\n# ===== {p.relative_to(REPO)} =====\n" + p.read_text(errors="ignore"))
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="512,8192,32768,126000")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-context", type=int, default=131072)
    ap.add_argument("--skip-ar", default="", help="comma list of contexts to skip AR")
    ap.add_argument("--no-ar", action="store_true")
    ap.add_argument("--ar-steps", type=int, default=0)
    ap.add_argument("--needle", action="store_true",
                    help="insert a passphrase at depth 0.25 and ask for it before the code")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=MD)
    ap.add_argument("--context-dir", type=Path, default=None,
                    help="save/load a KV snapshot per prompt to skip repeat prefills")
    a = ap.parse_args()

    from hipengine.generation.muse_glimmer_dflash import make_muse_glimmer_dflash_text_provider
    from hipengine.generation.muse_glimmer_gguf import make_muse_glimmer_gguf_generator_gfx1151
    from hipengine.generation.registry import GenerationRequest
    from hipengine.loading.gguf import scan_gguf
    from hipengine.models.muse_glimmer import MUSE_GLIMMER_GGUF
    from hipengine.speculative.registry import SpeculativeProviderConfig

    model = a.model_dir / "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
    gen = make_muse_glimmer_gguf_generator_gfx1151(
        model_path=model, weight_index=scan_gguf(model), model_plugin=MUSE_GLIMMER_GGUF,
        max_sequence_length=a.max_context)
    prov = make_muse_glimmer_dflash_text_provider(target_generator=gen, config=SpeculativeProviderConfig(
        provider="dflash", draft_model=a.model_dir / "dflash-Muse-Glimmer-30B-Q4_K_M.gguf", candidate_budget=4))
    gen.attach_speculative_provider(prov)

    needle_text = "\n# NOTE: the secret deployment passphrase is 'cobalt-heron-4417'.\n"
    instr = INSTR
    if a.needle:
        instr = INSTR.replace("Return only code.", "First line: state the secret deployment passphrase "
                              "mentioned in a NOTE comment above. Then return only code.")
    instr_ids = list(gen.tokenize(instr))[1:]  # drop BOS
    needle_ids = list(gen.tokenize(needle_text))[1:]
    body_ids = list(gen.tokenize(corpus()))
    print(f"corpus tokens {len(body_ids)}", flush=True)

    def ids_for(n: int) -> tuple[int, ...]:
        if n <= 600:  # short: natural mtpbench-like prompt
            return tuple(gen.tokenize("Write a Python function merge_intervals(intervals) that merges "
                                      "overlapping closed integer intervals. Include a compact pytest-style "
                                      "test block. Return only code."))
        need = n - len(instr_ids)
        if a.needle:
            need -= len(needle_ids)
        body = (body_ids * (need // len(body_ids) + 1))[:need]
        if a.needle:
            cut = len(body) // 4
            body = body[:cut] + needle_ids + body[cut:]
        return tuple(body + instr_ids)

    def req(ids):
        return GenerationRequest(prompts=(ids,), max_tokens=a.max_tokens, temperature=0.0,
                                 top_p=1.0, ignore_eos=True)

    sess = gen._ensure_session()
    drafter = prov._ensure_drafter()
    rt = sess.runtime
    T: dict[str, list[float]] = {}
    CTX = ["warm"]

    import ctypes, os
    _rtx = ctypes.CDLL(os.environ["ROCTX_LIB"]) if os.environ.get("ROCTX_LIB") else None
    if _rtx:
        _rtx.roctxRangePushA.argtypes = [ctypes.c_char_p]

    def push(label):
        if _rtx: _rtx.roctxRangePushA(label.encode())

    def pop():
        if _rtx: _rtx.roctxRangePop()

    def wrap(obj, name, key):
        orig = getattr(obj, name)

        def f(*args, **kw):
            rt.device_synchronize()
            push(f"{CTX[0]}:{key}")
            t = time.perf_counter()
            out = orig(*args, **kw)
            rt.device_synchronize()
            pop()
            k = key
            if key == "inject" and not T.get("verify"):
                k = "inject_prefill"
            T.setdefault(k, []).append(time.perf_counter() - t)
            return out
        setattr(obj, name, f)

    wrap(sess, "verify_with_taps", "verify")
    wrap(drafter, "draft_block", "draft")
    wrap(drafter, "inject", "inject")

    # warmup (kernel builds, graph capture)
    w = ids_for(512)
    gen.generate_detailed(GenerationRequest(prompts=(w,), max_tokens=16, temperature=0.0, top_p=1.0, ignore_eos=True))
    gen.generate_speculative_detailed(GenerationRequest(prompts=(w,), max_tokens=16, temperature=0.0, top_p=1.0, ignore_eos=True))

    skip_ar = {int(x) for x in a.skip_ar.split(",") if x}
    rows = []
    for n in [int(x) for x in a.contexts.split(",")]:
        ids = ids_for(n)
        CTX[0] = str(n)
        row = {"ctx_target": n, "prompt_tokens": len(ids)}
        snap = None
        if a.context_dir is not None:
            import numpy as np
            digest = hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:12]
            snap = a.context_dir / f"ctx{len(ids)}-{digest}"
            if (snap / "context.json").exists():
                t = time.perf_counter()
                gen.load_context(snap)
                row["snapshot_load_s"] = time.perf_counter() - t
                print(f"[{n}] loaded snapshot {snap} in {row['snapshot_load_s']:.1f}s", flush=True)
                snap = None  # already on disk
        if not a.no_ar and n not in skip_ar:
            out = gen.generate_detailed(req(ids))[0]
            row.update(ar_prefill_s=gen.last_prefill_seconds, ar_decode_s=gen.last_decode_seconds,
                       ar_tokens=len(out.generated_token_ids),
                       ar_tok_s=len(out.generated_token_ids) / gen.last_decode_seconds,
                       ar_cached_tokens=out.telemetry.diagnostics["prefix_cache"]["reused_tokens"])
            ar_ids = tuple(out.generated_token_ids)
            row["ar_text_head"] = gen.decode(ar_ids[:160]); row["needle_found"] = "cobalt-heron-4417" in gen.decode(ar_ids)
            print(f"[{n}] needle_found={row['needle_found']} AR text head: {row['ar_text_head']!r}", flush=True)
            print(f"[{n}] AR prefill {row['ar_prefill_s']:.1f}s decode {row['ar_tok_s']:.2f} tok/s", flush=True)
        else:
            ar_ids = None
        T.clear()
        prov.counters.__init__()
        out = gen.generate_speculative_detailed(req(ids))[0]
        c = prov.counters
        cyc = len(T.get("verify", []))
        dec = prov.last_decode_seconds
        sums = {k: sum(v) for k, v in T.items()}
        verify_s, draft_s = sums.get("verify", 0.0), sums.get("draft", 0.0)
        row.update(
            spec_prefill_s=prov.last_prefill_seconds, spec_decode_s=dec,
            spec_tokens=len(out.generated_token_ids),
            spec_tok_s=len(out.generated_token_ids) / dec,
            exact=(None if ar_ids is None else tuple(out.generated_token_ids) == ar_ids),
            cycles=cyc, accepted=c.accepted, proposed=c.proposed, committed=c.committed,
            tokens_per_cycle=c.committed / max(cyc, 1),
            ms_per_cycle=1e3 * dec / max(cyc, 1),
            verify_ms=1e3 * verify_s / max(cyc, 1),
            draft_ms=1e3 * draft_s / max(cyc, 1),
            inject_ms=1e3 * sums.get("inject", 0.0) / max(cyc, 1),
            inject_prefill_s=sums.get("inject_prefill", 0.0),
            spec_cached_tokens=out.telemetry.diagnostics["prefix_cache"]["reused_tokens"],
        )
        if snap is not None:
            t = time.perf_counter()
            gen.save_context(snap)
            row["snapshot_save_s"] = time.perf_counter() - t
            print(f"[{n}] saved snapshot {snap} in {row['snapshot_save_s']:.1f}s", flush=True)
        print(f"[{n}] DFlash {row['spec_tok_s']:.2f} tok/s exact={row['exact']} "
              f"cyc={cyc} tok/cyc={row['tokens_per_cycle']:.2f} ms/cyc={row['ms_per_cycle']:.1f} "
              f"verify={row['verify_ms']:.1f} draft={row['draft_ms']:.1f} "
              f"inject={row['inject_ms']:.1f} inject_prefill={row['inject_prefill_s']:.1f}s", flush=True)
        if a.ar_steps:
            tok = int(out.generated_token_ids[-1])
            ts = []
            for _ in range(a.ar_steps):
                rt.device_synchronize(); push(f"{n}:ar_step"); t = time.perf_counter()
                lg = sess.decode_step(tok)
                rt.device_synchronize(); pop(); ts.append(time.perf_counter() - t)
                tok = int(lg.argmax())
            row["ar_step_ms_median"] = 1e3 * sorted(ts)[len(ts) // 2]
            print(f"[{n}] AR step median {row['ar_step_ms_median']:.1f} ms", flush=True)
        rows.append(row)
        a.output.write_text(json.dumps(rows, indent=2))
    gen.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
