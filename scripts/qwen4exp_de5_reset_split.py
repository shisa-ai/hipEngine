"""Split ``Qwen4ExpDenseRunner.reset()`` into its components and time each one.

DE-5's re-derivation localized a ~1.5-1.8 s per-prefill transient to the first
~32 decode steps and excluded queued device work, clock throttling, and a pure
host block, while showing a 5 s idle removes it.  ``prefill`` calls ``reset()``
first, so the remaining question is which part of the reset path sets up that
transient.

This probe answers it directly rather than by inference.  It builds the runner,
then times the whole ``reset()`` and each component separately with a device
synchronize on both sides, and reports the byte sizes of every buffer those
components touch.  A component that is cheap here cannot be the 1.5 s owner; a
component that is expensive here is the owner.

Run:

    D=/home/lhl/miniforge3/envs/vllm/lib/python3.12/site-packages/_rocm_sdk_devel
    PATH=$D/bin:$PATH LD_LIBRARY_PATH=$D/lib python3 scripts/qwen4exp_de5_reset_split.py \
        --case code-p1024 --steps 128 --repeats 2 \
        --json-out benchmarks/results/2026-09-30-qwen4exp-de5-wrapper-wall/reset-split.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = pathlib.Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_FIXTURE = REPO_ROOT / "benchmarks" / "fixtures" / "qwen4exp_canonical_ar_p512_p1024_p4096.json"

BUSY_PATH = pathlib.Path("/sys/class/drm/card1/device/gpu_busy_percent")


def _mem(runtime) -> dict:
    try:
        free_bytes, total_bytes = runtime.memory_info()
        return {"free": int(free_bytes), "total": int(total_bytes)}
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"error": repr(exc)}


def _timed(runtime, fn):
    """Run ``fn`` and return (milliseconds, result) with a sync on both sides."""
    runtime.device_synchronize()
    started = time.perf_counter_ns()
    result = fn()
    runtime.device_synchronize()
    return (time.perf_counter_ns() - started) / 1e6, result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-root", type=pathlib.Path, default=DEFAULT_MODEL)
    ap.add_argument("--fixture", type=pathlib.Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--case", default="code-p1024")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--capacity", type=int, default=None,
                    help="Override max_sequence_length.  RT-4's case is a SHORT request at "
                         "native capacity, where the full-capacity clear spends nearly all "
                         "its time on storage the request never touched.")
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

    from dataclasses import replace as _replace

    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models.registry import resolve_model
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile

    fixture = json.loads(args.fixture.read_text())
    cases = {str(row["id"]): row for row in fixture["cases"]}
    if args.case not in cases:
        raise SystemExit(f"unknown case {args.case!r}; have {sorted(cases)}")
    case = cases[args.case]
    transitions = int(fixture["decode_transitions"])

    model_root = args.model_root.resolve()
    token_ids = [int(t) for t in case["prompt_token_ids"]]
    prompt_tokens = int(case["prompt_tokens"])
    max_sequence_length = prompt_tokens + max(int(args.steps), transitions) + 8
    if args.capacity is not None:
        capacity = int(args.capacity)
        if capacity <= prompt_tokens + int(args.steps):
            raise SystemExit("capacity must exceed prompt_tokens + steps")
        max_sequence_length = capacity

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile("production"),
    )

    def factory():
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=512,
        )

    generator = resolved.construct_generator(factory)
    rows: list[dict] = []
    try:
        runner = generator.runner
        runtime = runner.runtime

        # Sizes of everything reset() touches, so an expensive component can be
        # read against the bytes it actually moves.
        sizes = {
            "max_sequence_length": max_sequence_length,
            "prompt_tokens": prompt_tokens,
            "attention_states": len(runner.attention_states),
            "key_cache_bytes_each": [
                int(a.key_cache.nbytes) for a in runner.attention_states
            ],
            "value_cache_bytes_each": [
                int(a.value_cache.nbytes) for a in runner.attention_states
            ],
            "index_states": len(runner.index_states),
        }
        sizes["key_cache_bytes_total"] = sum(sizes["key_cache_bytes_each"])
        sizes["value_cache_bytes_total"] = sum(sizes["value_cache_bytes_each"])
        sizes["kv_bytes_total"] = (
            sizes["key_cache_bytes_total"] + sizes["value_cache_bytes_total"]
        )
        # RT-4's measured hardware memset rate; anything far under this is not
        # explained by memset time.
        sizes["kv_memset_ms_at_220GBs"] = sizes["kv_bytes_total"] / 220e9 * 1e3

        index_sizes = []
        for index in runner.index_states:
            entry = {"type": type(index).__name__}
            for name in ("key_cache", "value_cache", "capacity", "max_entries",
                         "positions", "slot_table", "index"):
                obj = getattr(index, name, None)
                nbytes = getattr(obj, "nbytes", None)
                if isinstance(obj, int):
                    entry[name] = obj
                elif nbytes is not None:
                    entry[name] = int(nbytes)
            index_sizes.append(entry)
        sizes["index_states_detail"] = index_sizes
        print(f"[sizes] {json.dumps(sizes, indent=2)}", flush=True)

        for rep in range(int(args.repeats) + 1):
            tag = "warmup" if rep == 0 else f"rep{rep}"
            mem_before = _mem(runtime)
            # A real prefill first, so the reset is measured in the same state a
            # repeat would see.
            runtime.device_synchronize()
            prefill_started = time.perf_counter_ns()
            runner.prefill(token_ids, capture_logits=False, capture_target_hidden=False)
            runtime.device_synchronize()
            prefill_ms = (time.perf_counter_ns() - prefill_started) / 1e6
            mem_after_prefill = _mem(runtime)

            # Whole reset.
            reset_ms, _ = _timed(runtime, runner.reset)

            # Components, in the order reset() runs them.
            components: dict[str, float] = {}
            if runner.state is not None:
                components["state_zero"], _ = _timed(runtime, runner.state.zero)
            kv_ms = 0.0
            for attention in runner.attention_states:
                a, _ = _timed(
                    runtime,
                    lambda a=attention: runtime.memset(a.key_cache.ptr, 0, a.key_cache.nbytes),
                )
                b, _ = _timed(
                    runtime,
                    lambda a=attention: runtime.memset(a.value_cache.ptr, 0, a.value_cache.nbytes),
                )
                kv_ms += a + b
            components["kv_memsets_total"] = kv_ms
            idx_ms = 0.0
            for index in runner.index_states:
                a, _ = _timed(runtime, index.reset)
                idx_ms += a
            components["index_resets_total"] = idx_ms
            components["ple_hash_clear"], _ = _timed(
                runtime, lambda: setattr(runner, "_ple_hash_states", {})
            )

            # Now the decode loop, to confirm the transient is present in this
            # same process/state and can be compared against the reset cost.
            runtime.device_synchronize()
            result = runner.prefill(
                token_ids, capture_logits=False, capture_target_hidden=False
            )
            runtime.device_synchronize()
            current = int(result.token_id)
            mem_before_decode = _mem(runtime)
            step_ns: list[int] = []
            mem_during: list[dict] = []
            for index in range(int(args.steps)):
                t0 = time.perf_counter_ns()
                result = runner.step(
                    current,
                    capture_logits=False,
                    capture_target_hidden=False,
                    token_id_resident=True,
                )
                current = int(result.token_id)
                step_ns.append(time.perf_counter_ns() - t0)
                if index < 40:
                    m0 = time.perf_counter_ns()
                    sample = _mem(runtime)
                    sample["call_ms"] = (time.perf_counter_ns() - m0) / 1e6
                    mem_during.append(sample)
            runtime.device_synchronize()
            mem_after_decode = _mem(runtime)

            n = len(step_ns)
            q = max(1, n // 8)
            bands = [statistics.fmean(step_ns[i:i + q]) / 1e6 for i in range(0, n, q)]
            row = {
                "repetition": rep,
                "warmup": rep == 0,
                "steps": int(args.steps),
                "prefill_ms": prefill_ms,
                "reset_ms": reset_ms,
                "memory": {
                    "before": mem_before,
                    "after_prefill": mem_after_prefill,
                    "before_decode": mem_before_decode,
                    "after_decode": mem_after_decode,
                    "free_during_first_40_steps": [m.get("free") for m in mem_during],
                    "mem_call_ms_first_40": [m.get("call_ms") for m in mem_during],
                },
                "components": components,
                "step_ms_mean": statistics.fmean(step_ns) / 1e6,
                "step_ms_first_eighth": bands[0],
                "step_ms_last_eighth": bands[-1],
                "step_ms_bands": bands,
            }
            rows.append(row)
            print(
                f"[{tag}] prefill {prefill_ms:8.2f} ms | reset {reset_ms:8.2f} ms"
                f" | first-eighth {bands[0]:7.2f} last-eighth {bands[-1]:7.2f} ms",
                flush=True,
            )
            print(
                "        memory: before=%s after_prefill=%s before_decode=%s after_decode=%s"
                % (
                    mem_before.get("free"),
                    mem_after_prefill.get("free"),
                    mem_before_decode.get("free"),
                    mem_after_decode.get("free"),
                ),
                flush=True,
            )
            print(
                "        reset components: "
                + "  ".join(f"{k}={v:7.2f}" for k, v in components.items()),
                flush=True,
            )
    finally:
        try:
            generator.close()
        except Exception:
            pass

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "case": args.case,
                    "model_root": str(args.model_root),
                    "bounded_kv_reset": os.environ.get("HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET", "0"),
                    "sizes": sizes,
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
