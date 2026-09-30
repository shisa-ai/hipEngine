#!/usr/bin/env python3
"""Measure the end-to-end prefill value of the DP-8 staged-f32 layer scope.

DP-8's composition (staged 13-15 + the wide route at 16-47) passes every hard
gate, but its **value** has never been measured. The ledger carries
``~0.66% of prefill``, derived as ``3 layers x 8.985 ms per layer-step x 4
chunks of 1024 rows``.

That arithmetic is the one thing the source artifact explicitly forbids. Its
``extrapolation_warning`` reads:

    Do NOT scale the per-layer-step total by a layer count to get a prefill
    share. ... 4553.007/38.906 = 117 and 3262.5/38.906 = 84, neither of which
    matches 16, 48 or 192 layer-steps. The per-step multiplicity or row count
    in the profiled forward pass is therefore NOT established, and the prefill
    share remains underivable from this data.

So this harness measures the share directly instead of deriving it: one
resident generator, one prompt, both arms interleaved, and the wall time of a
real prefill on each side.

Design notes
------------
**Construction follows the production gate, and that is load-bearing.** The
first version of this harness built a ``Qwen35GGUFResidentSession``, which never
applies the profile plan, so its baseline would have been a non-shipping
configuration rather than the one this measurement is meant to price. The gate
path is used instead: ``register_*``, ``load_gguf_index``, ``resolve_model``,
``resolve_runtime_profile``, then ``resolved.construct_generator(factory)``. The
binder writes the production selectors itself (``Q8_WMMA_LAYERS=""`` off,
``Q8_DENSE_WIDE=1`` at 16-47), so the baseline arm is the real default and the
candidate arm differs from it by exactly the staged scope. The bound values are
recorded in the artifact so a reader can confirm that without re-deriving it.

**In-process arm switching is safe here, and the reason is specific.** The
staged scope is read inside the dispatch function (``_q8_staged_f32_layers()``)
and it is part of the dispatch memoization key (``_staged_f32_key``). Without
that key entry the candidate arm would replay the baseline arm's exact-chain
dispatch and this harness would measure a 0.00% difference that read as a clean
null result. That is not hypothetical: it is the documented false pass in
``benchmarks/results/2026-09-29-q8-staged-f32-dense-prefill-layers-gate/``
(bit-identical arms, KL exactly zero, ``measurement_valid: true``). This harness
therefore *counts the launches each arm actually performed* rather than trusting
the env var, reports the counts per arm, and refuses to report a timing verdict
if the candidate arm never launched the staged variant.

**The counter wraps the launched kernel, not the resolver, and that is not a
style choice.** The first version of this harness wrapped
``resolve_gguf_linear_dispatch``. That function is *inside* the memoization it
is supposed to observe: ``launch_gguf_linear`` caches the whole dispatch chain -
every selector, including the staged-f32 rewrite - in
``_DISPATCH_RESOLVE_CACHE``, and the warmups fill that cache for both arms. The
measured repetitions therefore hit the cache and never call the resolver, so the
counter read zero in both arms and the harness refused a run whose route had in
fact engaged. The instrument was blind; the refusal message blamed the route.

So the counter is installed on the *cache*, which is also the fix: after the
warmups it holds the resolved kernel callable for every (arm, weight, shape)
key, and ``launch_gguf_linear`` invokes that callable once per launch. Wrapping
the callables counts real launches without touching the dispatch chain, so the
timed region is unchanged - which matters here, because the quantity being
measured is 0.15% of prefill.

**Alternating arm order.** Each repetition runs the arms in the opposite order
from its predecessor, so a monotonic thermal or clock drift cannot favour one
arm.

**Warmups are discarded.** The first prefill of a shape JIT-builds and warms the
caches.

Usage
-----
    bash scripts/run_qwen4exp_dp8_staged_scope_prefill_ab.sh

The ROCm SDK library path and ``hipcc`` must be on the environment, or the run
fails at ``libamdhip64.so`` / the first JIT build. The wrapper pins both.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_ROOT = Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)

STAGED_ENV = "HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS"
STAGED_VARIANT = "staged_f32_bm128_bn64_bk32"

BASELINE_ARM = "baseline"
CANDIDATE_ARM = "staged_13_15"

# The attribution anchor this harness exists to stop extrapolating from. Kept as
# a constant so the report can express the measured delta as a share of the
# prefill it belongs to, and so a reader can see which number is measured and
# which is the anchor.
PREFILL_ANCHOR_MS = 15741.963
DENSE_FAMILY_ANCHOR_MS = 4553.007


class BenchmarkError(RuntimeError):
    """A measurement precondition failed."""


@contextlib.contextmanager
def staged_scope(layers: str | None) -> Iterator[None]:
    """Set the staged-f32 layer scope for the duration of the block.

    ``None`` unsets the variable, which is the baseline arm. An empty string is
    deliberately not used to mean off: the route's own parser treats an empty
    value as unset, but ``=0`` selects layer 0, so the two spellings are kept
    distinct rather than relying on that.
    """

    previous = os.environ.get(STAGED_ENV)
    if layers is None:
        os.environ.pop(STAGED_ENV, None)
    else:
        os.environ[STAGED_ENV] = layers
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(STAGED_ENV, None)
        else:
            os.environ[STAGED_ENV] = previous


def _wrap_cached_launches(
    by_variant: Counter,
    by_key: Counter,
    key_identity: dict[Any, tuple[int | None, str]],
) -> tuple[int, int]:
    """Wrap every resolved linear kernel callable in the dispatch cache.

    Idempotent: an entry already holding a counted callable is left alone. The
    return value is ``(wrapped_this_call, total_wrapped)``; a non-zero first
    element *after* a measured arm means that arm created a dispatch-cache entry
    the arm's own launches could not be counted through, and the caller treats
    that as an incomplete count rather than a zero.
    """

    from hipengine.runtime import gguf_linear

    cache = gguf_linear._DISPATCH_RESOLVE_CACHE
    wrapped_now = 0
    total = 0
    for cache_key, entry in list(cache.items()):
        if not isinstance(entry, tuple) or len(entry) != 5:
            raise BenchmarkError(
                "gguf_linear dispatch cache entry is not the expected "
                f"(abi, fn, layer, quant, variant) 5-tuple: {entry!r}"
            )
        abi, fn, layer, quant, variant = entry
        total += 1
        if getattr(fn, "_qwen4exp_launch_counted", False):
            continue

        def counting(
            *args: Any,
            _fn: Any = fn,
            _variant: str = str(variant),
            _cache_key: Any = cache_key,
            **kwargs: Any,
        ) -> Any:
            by_variant[_variant] += 1
            by_key[_cache_key] += 1
            return _fn(*args, **kwargs)

        counting._qwen4exp_launch_counted = True  # type: ignore[attr-defined]
        cache[cache_key] = (abi, counting, layer, quant, variant)
        key_identity[cache_key] = (_staged_layer_of(cache_key), str(variant))
        wrapped_now += 1
    return wrapped_now, total


def _staged_layer_of(cache_key: Any) -> int | None:
    """Recover the model layer index a dispatch cache key was built for.

    ``launch_gguf_linear`` assembles its key as ``(generation, dense_wide_key,
    staged_f32_key, wmma_scope_key, ...)``, and ``staged_f32_key`` is
    ``(scope, layer_index)`` with the layer index present only while a scope is
    bound. Reading it back is what lets a counted launch be attributed to a
    layer without re-deriving the key or trusting the env var. Anything that
    does not match that shape returns ``None`` and is reported as unattributed
    rather than guessed at.
    """

    if not isinstance(cache_key, tuple) or len(cache_key) < 3:
        return None
    staged_key = cache_key[2]
    if not isinstance(staged_key, tuple) or len(staged_key) != 2:
        return None
    layer = staged_key[1]
    return int(layer) if isinstance(layer, int) and not isinstance(layer, bool) else None


def _layer_census(
    counts: Counter, key_identity: dict[Any, tuple[int | None, str]]
) -> dict[str, int]:
    """Render a per-cache-key launch counter as ``layer:variant`` rows."""

    census: Counter = Counter()
    for cache_key, launches in counts.items():
        layer, variant = key_identity.get(cache_key, (None, "<unknown>"))
        name = "unattributed" if layer is None else str(layer)
        census[f"{name}:{variant}"] += int(launches)
    return dict(sorted(census.items()))


def _timed_prefill(runner: Any, *, prompt_ids: Sequence[int]) -> dict[str, Any]:
    runtime = getattr(runner, "runtime", None)
    if runtime is None:
        raise BenchmarkError("resident runner has no HIP runtime")
    runtime.device_synchronize()
    started_ns = time.perf_counter_ns()
    result = runner.prefill(
        [int(token) for token in prompt_ids],
        capture_logits=True,
        capture_target_hidden=False,
    )
    runtime.device_synchronize()
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
        raise BenchmarkError(f"invalid prefill wall: {elapsed_ms!r}")
    return {"wall_ms": float(elapsed_ms), "token_id": int(result.token_id)}


def _statistics(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in samples]
    if not values or any(not math.isfinite(v) or v <= 0.0 for v in values):
        raise BenchmarkError("timing samples must be finite and positive")
    return {
        "samples_ms": values,
        "count": len(values),
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.fmean(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "stdev_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _summarize(measured: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not measured:
        raise BenchmarkError("no measured repetitions")
    arms = (BASELINE_ARM, CANDIDATE_ARM)
    stats = {
        arm: _statistics([float(row["arms"][arm]["wall_ms"]) for row in measured])
        for arm in arms
    }
    paired_delta = [
        float(row["arms"][CANDIDATE_ARM]["wall_ms"])
        - float(row["arms"][BASELINE_ARM]["wall_ms"])
        for row in measured
    ]
    baseline_median = stats[BASELINE_ARM]["median_ms"]
    candidate_median = stats[CANDIDATE_ARM]["median_ms"]
    # Positive means the candidate saved time.
    saving_ms = baseline_median - candidate_median
    return {
        "arms": list(arms),
        "statistics": stats,
        "paired_candidate_minus_baseline_ms": paired_delta,
        "paired_candidate_minus_baseline_median_ms": float(statistics.median(paired_delta)),
        "candidate_speedup_vs_baseline": float(baseline_median / candidate_median),
        "candidate_wall_delta_percent": float(
            (candidate_median - baseline_median) / baseline_median * 100.0
        ),
        "measured_saving_ms": float(saving_ms),
        "measured_saving_percent_of_prefill": float(saving_ms / PREFILL_ANCHOR_MS * 100.0),
        "prefill_anchor_ms": float(PREFILL_ANCHOR_MS),
        "token_ids": {
            arm: [int(row["arms"][arm]["token_id"]) for row in measured] for arm in arms
        },
    }


def _provenance() -> dict[str, Any]:
    import platform
    import subprocess

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # pragma: no cover - defensive
            return ""

    def read(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:  # pragma: no cover - defensive
            return ""

    # Physical host identity is part of the evidence policy, and two machines
    # with the same GPU architecture are independent lanes: without this a
    # reader cannot tell whether a comparison was same-host. ``machine_id`` is
    # the identifier the other gfx1151 packets carry, so this artifact can be
    # matched to them directly.
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "machine_id": read("/etc/machine-id"),
        "hostname": platform.node(),
        "product_name": read("/sys/class/dmi/id/product_name"),
        "command": " ".join(["python3", *sys.argv]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--candidate-layers", default="13,14,15")
    parser.add_argument(
        "--execution-profile", choices=("production",), default="production"
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    model_root = args.model_root.resolve()
    if not model_root.exists():
        raise BenchmarkError(f"model root does not exist: {model_root}")

    candidate_layers = str(args.candidate_layers).strip()
    if not candidate_layers:
        raise BenchmarkError("--candidate-layers must name at least one layer")

    from hipengine.execution_profiles import (
        ExecutionProfile,
        resolve_runtime_profile,
    )
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile(str(args.execution_profile)),
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend=QWEN4_EXP_BACKEND,
            max_sequence_length=int(args.prompt_length) + 2,
            prefill_chunk_size=int(args.prefill_chunk_size),
        )

    by_variant, by_key = Counter(), Counter()
    key_identity: dict[Any, tuple[int | None, str]] = {}

    prompt_ids = [int(args.prompt_token_id)] * int(args.prompt_length)

    # 4096 rows at the 1024-row chunk cap is the four chunks the estimate
    # assumed, so this measures the configuration the estimate describes.
    payload: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_dp8_staged_scope_prefill_ab",
        "model_root": str(model_root),
        "prompt_length": int(args.prompt_length),
        "prompt_token_id": int(args.prompt_token_id),
        "candidate_layers": candidate_layers,
        "staged_variant": STAGED_VARIANT,
        "quant": str(QWEN4_EXP_QUANTS[1]),
        "execution_profile": str(args.execution_profile),
        "prefill_chunk_size": int(args.prefill_chunk_size),
        "warmups": int(args.warmups),
        "repetitions": int(args.repetitions),
        "prefill_anchor_ms": float(PREFILL_ANCHOR_MS),
        "dense_family_anchor_ms": float(DENSE_FAMILY_ANCHOR_MS),
        "provenance": _provenance(),
        "engagement": {},
    }

    # The binder writes the production selectors during its own pass, so
    # construction must complete before any arm's scope is set. Nothing below
    # writes an env var except ``staged_scope``.
    generator = resolved.construct_generator(factory)
    runner = generator.runner
    if runner is None:
        raise BenchmarkError("generator is not resident")
    payload["resolved_backend"] = str(getattr(runner, "backend", None))
    payload["target_arch"] = str(os.environ.get("HIPENGINE_HIP_ARCH", "gfx1151"))
    # Record what the binder actually bound, so a reader can confirm the
    # baseline arm is the shipping configuration without re-deriving it.
    payload["bound_environment"] = {
        name: os.environ.get(name)
        for name in (
            "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS",
            "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE",
            "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS",
            STAGED_ENV,
        )
    }

    # Construction and the binder resolve a great many dispatches; only the
    # launches below are measured, and the wrap is installed after the warmups
    # have populated the cache for both arms. It is re-installed before each
    # measured arm so a key created by a previous arm is covered too.
    _wrap_cached_launches(by_variant, by_key, key_identity)

    try:
        for warmup in range(int(args.warmups)):
            arms = (BASELINE_ARM, CANDIDATE_ARM)
            order = arms if warmup % 2 == 0 else tuple(reversed(arms))
            row: dict[str, Any] = {"warmup": warmup, "order": list(order), "arms": {}}
            for arm in order:
                scope = None if arm == BASELINE_ARM else candidate_layers
                with staged_scope(scope):
                    row["arms"][arm] = _timed_prefill(runner, prompt_ids=prompt_ids)
            print(
                f"[warmup {warmup}] "
                + " ".join(f"{arm}={row['arms'][arm]['wall_ms']:.1f}ms" for arm in order),
                flush=True,
            )

        # Install on the now-populated cache, then clear the counters so the
        # recorded engagement describes the measured runs only.
        wrapped_entries = _wrap_cached_launches(by_variant, by_key, key_identity)[1]
        by_variant.clear()
        by_key.clear()

        measured: list[dict[str, Any]] = []
        per_arm_variants: dict[str, Counter] = {
            BASELINE_ARM: Counter(),
            CANDIDATE_ARM: Counter(),
        }
        per_arm_keys: dict[str, Counter] = {
            BASELINE_ARM: Counter(),
            CANDIDATE_ARM: Counter(),
        }
        uncounted: dict[str, int] = {BASELINE_ARM: 0, CANDIDATE_ARM: 0}
        for repetition in range(int(args.repetitions)):
            arms = (BASELINE_ARM, CANDIDATE_ARM)
            order = arms if repetition % 2 == 0 else tuple(reversed(arms))
            row = {"repetition": repetition, "order": list(order), "arms": {}}
            for arm in order:
                scope = None if arm == BASELINE_ARM else candidate_layers
                _wrap_cached_launches(by_variant, by_key, key_identity)
                before = Counter(by_variant)
                before_keys = Counter(by_key)
                with staged_scope(scope):
                    row["arms"][arm] = _timed_prefill(runner, prompt_ids=prompt_ids)
                per_arm_variants[arm].update(by_variant - before)
                per_arm_keys[arm].update(by_key - before_keys)
                # A key created *inside* the timed region was launched before it
                # could be wrapped, so its launches are missing from the count.
                # Record it instead of reporting a clean zero.
                uncounted[arm] += _wrap_cached_launches(
                    by_variant, by_key, key_identity
                )[0]
            measured.append(row)
            print(
                f"[rep {repetition}] "
                + " ".join(f"{arm}={row['arms'][arm]['wall_ms']:.1f}ms" for arm in order),
                flush=True,
            )

        payload["summary"] = _summarize(measured)
        payload["engagement"] = {
            "launches_by_arm": {
                arm: dict(sorted(counts.items()))
                for arm, counts in per_arm_variants.items()
            },
            "launches_by_layer": {
                arm: _layer_census(counts, key_identity)
                for arm, counts in per_arm_keys.items()
            },
            "cached_entries_wrapped": int(wrapped_entries),
            "uncounted_cache_entries_by_arm": dict(uncounted),
        }
    finally:
        close = getattr(generator, "close", None)
        if callable(close):
            close()

    # Engagement is asserted, not reported-and-ignored. A candidate arm that
    # never launched the staged variant is the documented false pass this packet
    # family has already produced once, and a null result from it would be read
    # as "the route is worth nothing" rather than "the route did not run".
    staged_launches = int(per_arm_variants[CANDIDATE_ARM].get(STAGED_VARIANT, 0))
    baseline_staged = int(per_arm_variants[BASELINE_ARM].get(STAGED_VARIANT, 0))
    incomplete = bool(uncounted[BASELINE_ARM] or uncounted[CANDIDATE_ARM])
    payload["engagement"]["candidate_staged_launches"] = staged_launches
    payload["engagement"]["baseline_staged_launches"] = baseline_staged
    payload["engagement"]["engaged"] = bool(
        staged_launches > 0 and baseline_staged == 0 and not incomplete
    )
    payload["engagement"]["incomplete"] = incomplete
    print(
        "launches         "
        + "  ".join(
            f"{arm}={dict(sorted(per_arm_variants[arm].items()))}"
            for arm in (BASELINE_ARM, CANDIDATE_ARM)
        )
    )

    summary = payload["summary"]
    print()
    print(
        f"baseline median  {summary['statistics'][BASELINE_ARM]['median_ms']:.2f} ms "
        f"(stdev {summary['statistics'][BASELINE_ARM]['stdev_ms']:.2f})"
    )
    print(
        f"candidate median {summary['statistics'][CANDIDATE_ARM]['median_ms']:.2f} ms "
        f"(stdev {summary['statistics'][CANDIDATE_ARM]['stdev_ms']:.2f})"
    )
    print(
        f"saving           {summary['measured_saving_ms']:.2f} ms "
        f"({summary['candidate_wall_delta_percent']:.3f}%), "
        f"speedup {summary['candidate_speedup_vs_baseline']:.4f}x"
    )
    print(
        f"share of prefill {summary['measured_saving_percent_of_prefill']:.3f}% "
        f"of the {PREFILL_ANCHOR_MS:.0f} ms anchor"
    )
    print(
        f"engagement       staged variant launched "
        f"{staged_launches}x candidate / {baseline_staged}x baseline"
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json}")

    if incomplete:
        print(
            "REFUSING: the candidate arm created a dispatch-cache entry inside "
            "the timed region, so its launches were counted only from the next "
            "repetition onward and this arm's census is incomplete. The timings "
            "are not wrong, but the engagement evidence is, and a run whose "
            "route cannot be shown to have engaged is not a measurement of it.",
            file=sys.stderr,
        )
        return 2
    if not payload["engagement"]["engaged"]:
        print(
            "REFUSING: the staged variant was launched "
            f"{staged_launches}x in the candidate arm and {baseline_staged}x in "
            "the baseline arm, so this run does not measure the route. Any "
            "timing difference here is not the route's. The census above names "
            "what did launch, which distinguishes a route that declined its "
            "guard from a launch path this instrument does not see.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
